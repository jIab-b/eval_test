"""
PTX-first IR + artifact + benchmark recording

Design goals:
- Full generality at PTX level (no loss): exact mnemonic + qualifier chain + operands + raw form escape hatch
- Fits as an "overcomplicated step" (dialect) in a broader CUDA IR (so kernels/modules exist)
- Evidence/learning layer: compile/run/correctness/benchmark records attachable to any node
- Schema evolution: stable core + flexible annotations + versioning
- Working examples registry: "golden" kernels + reproducible harness metadata

This is intentionally explicit and verbose so you can grow it into a DB-backed system later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import json
import time
import hashlib


# -----------------------------
# Utilities
# -----------------------------

def now_ns() -> int:
    return time.time_ns()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def content_hash(obj: Any) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))

def clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo or x > hi:
        raise ValueError(f"Immediate out of range: {x} not in [{lo},{hi}]")
    return x


# -----------------------------
# Core IR: PTX types, operands, qualifiers, instructions
# -----------------------------

class PtxAddrSpace(str, Enum):
    reg = "reg"
    sreg = "sreg"          # special registers
    const = "const"
    global_ = "global"
    shared = "shared"
    local = "local"
    param = "param"
    tex = "tex"
    surf = "surf"
    tmem = "tmem"          # tensor memory (Blackwell tcgen05)
    generic = "generic"

class PtxScalarType(str, Enum):
    # Keep open-ended; you can add more without breaking IR.
    pred = "pred"
    b8 = "b8"
    b16 = "b16"
    b32 = "b32"
    b64 = "b64"
    u8 = "u8"
    u16 = "u16"
    u32 = "u32"
    u64 = "u64"
    s8 = "s8"
    s16 = "s16"
    s32 = "s32"
    s64 = "s64"
    f16 = "f16"
    f32 = "f32"
    f64 = "f64"
    tf32 = "tf32"
    # FP8/FP4 family names are often encoded via instruction/descriptor rather than scalar types,
    # but we still allow them as "spellings" if you want.
    f8e4m3 = "f8e4m3"
    f8e5m2 = "f8e5m2"
    nvfp4_e2m1 = "nvfp4_e2m1"
    fp4 = "fp4"  # generic


@dataclass(frozen=True)
class PtxType:
    """A PTX type descriptor. Keep this minimal and permissive."""
    scalar: PtxScalarType
    lanes: int = 1  # vector width (e.g., .v4) where applicable

    def __post_init__(self):
        if self.lanes < 1:
            raise ValueError("lanes must be >= 1")


class OperandKind(str, Enum):
    reg = "reg"
    imm = "imm"
    mem = "mem"
    label = "label"
    barrier = "barrier"   # e.g. mbarrier address-ish
    raw = "raw"           # escape hatch: raw PTX operand text


@dataclass(frozen=True)
class Reg:
    name: str               # e.g. "%r17" or "UR4" (you can standardize later)
    ty: Optional[PtxType] = None
    space: PtxAddrSpace = PtxAddrSpace.reg


@dataclass(frozen=True)
class Imm:
    value: Union[int, float, str]  # allow symbolic immediates too
    ty: Optional[PtxType] = None


@dataclass(frozen=True)
class MemRef:
    """A memory operand; address expression can be structured or raw."""
    space: PtxAddrSpace
    base: Union[Reg, str]           # base register or symbol
    offset: int = 0
    # Optional richer addressing:
    index: Optional[Reg] = None
    scale: int = 1
    raw_expr: Optional[str] = None  # escape hatch to preserve exact text


@dataclass(frozen=True)
class LabelRef:
    name: str


@dataclass(frozen=True)
class RawOperand:
    text: str


PtxOperand = Union[Reg, Imm, MemRef, LabelRef, RawOperand]


@dataclass(frozen=True)
class QualifierAtom:
    """
    Single qualifier piece:
      ".cta_group::2"  -> name="cta_group", value="2", namespace="cta_group"
      ".block_scale{.scale_vec::4X}" -> name="block_scale", params=[...]
    """
    text: str  # store exact spelling for no-loss generality

@dataclass(frozen=True)
class QualifierChain:
    """
    Preserve exact order and spelling:
      tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale{.scale_vec::4X}.collector::a::fill
    """
    atoms: Tuple[QualifierAtom, ...] = ()

    def __str__(self) -> str:
        return "".join(a.text for a in self.atoms)


@dataclass
class PtxInstruction:
    """
    No-loss PTX instruction representation.

    - mnemonic: base opcode without qualifiers (e.g., "tcgen05.mma")
    - qualifiers: ordered chain storing exact spelling
    - operands: structured operands; can mix RawOperand for exact preservation
    - pred: optional predicate guard (e.g., "@%p0")
    - raw: optional raw PTX line as escape hatch (when parsing unknown forms)
    """
    mnemonic: str
    qualifiers: QualifierChain = field(default_factory=QualifierChain)
    operands: List[PtxOperand] = field(default_factory=list)
    pred: Optional[str] = None  # keep exact predicate text for no-loss
    raw: Optional[str] = None

    # Optional semantic hints (not required for expressivity):
    comment: Optional[str] = None

    def canonical_key(self) -> str:
        """
        A stable identity for grouping statistics/evidence:
        uses mnemonic + qualifiers + operand "kinds" (not exact values).
        """
        opk = []
        for o in self.operands:
            if isinstance(o, Reg): opk.append(f"reg:{o.space}")
            elif isinstance(o, Imm): opk.append("imm")
            elif isinstance(o, MemRef): opk.append(f"mem:{o.space}")
            elif isinstance(o, LabelRef): opk.append("label")
            elif isinstance(o, RawOperand): opk.append("raw")
            else: opk.append("unk")
        shape = {"m": self.mnemonic, "q": str(self.qualifiers), "opk": opk}
        return content_hash(shape)


# -----------------------------
# Broader CUDA IR containers: module/kernel/basic blocks
# -----------------------------

@dataclass
class PtxBasicBlock:
    label: Optional[str] = None
    insts: List[PtxInstruction] = field(default_factory=list)

@dataclass
class PtxKernel:
    """
    PTX-level kernel body (inside a wider CUDA IR pipeline, you'd have higher IR too).
    """
    name: str
    target_sm: Optional[str] = None        # e.g., "sm_110a"
    ptx_version: Optional[str] = None      # e.g., "8.8"
    entry_params: List[Tuple[str, PtxType]] = field(default_factory=list)
    blocks: List[PtxBasicBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # extensible

    def kernel_hash(self) -> str:
        return content_hash(asdict(self))

@dataclass
class PtxModule:
    """
    A module may contain multiple kernels and global directives.
    """
    name: str
    kernels: List[PtxKernel] = field(default_factory=list)
    directives_raw: List[str] = field(default_factory=list)  # keep exact for no-loss
    metadata: Dict[str, Any] = field(default_factory=dict)

    def module_hash(self) -> str:
        return content_hash(asdict(self))


# -----------------------------
# Evidence / Learning Layer (attachable records)
# -----------------------------

class ArtifactKind(str, Enum):
    ptx = "ptx"
    cubin = "cubin"
    fatbin = "fatbin"
    log = "log"
    report = "report"
    source = "source"
    other = "other"

@dataclass
class Artifact:
    kind: ArtifactKind
    name: str
    bytes_sha256: str
    size_bytes: int
    # Store content optionally; in practice you'd put it in object storage and keep a URI.
    uri: Optional[str] = None
    inline_bytes_b64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolchainInfo:
    """
    Pin reproducibility. Keep this as data; do not overfit schema.
    """
    nvcc_version: Optional[str] = None
    ptxas_version: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    libcudacxx_version: Optional[str] = None
    other: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareInfo:
    gpu_name: Optional[str] = None
    sm: Optional[str] = None  # e.g., "sm_110a"
    sm_count: Optional[int] = None
    clocks: Dict[str, Any] = field(default_factory=dict)
    other: Dict[str, Any] = field(default_factory=dict)

class BuildStatus(str, Enum):
    success = "success"
    failed = "failed"
    timeout = "timeout"

class RunStatus(str, Enum):
    success = "success"
    failed = "failed"
    timeout = "timeout"
    incorrect = "incorrect"

@dataclass
class BuildRecord:
    status: BuildStatus
    started_ns: int
    ended_ns: int
    toolchain: ToolchainInfo
    cmdline: List[str] = field(default_factory=list)
    stderr_artifact: Optional[Artifact] = None
    stdout_artifact: Optional[Artifact] = None
    outputs: List[Artifact] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrectnessRecord:
    ok: bool
    method: str                    # e.g. "ref_gemm_fp32", "unit_test"
    tolerance: Optional[Dict[str, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkMetrics:
    """
    Metrics are intentionally flexible; add raw counters when you have them.
    """
    time_us_min: Optional[float] = None
    time_us_median: Optional[float] = None
    time_us_p95: Optional[float] = None
    achieved_tflops: Optional[float] = None
    dram_gbps: Optional[float] = None
    occupancy: Optional[float] = None
    # Attach any Nsight metrics, custom counters, etc.
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkCase:
    """
    "Case" describes inputs and configuration. Separate from results.
    """
    name: str
    shapes: Dict[str, Any]                 # e.g. {"M":7168,"K":16384,"L":1}
    dtypes: Dict[str, str]                 # e.g. {"a":"nvfp4_e2m1", ...}
    layouts: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkRecord:
    status: RunStatus
    started_ns: int
    ended_ns: int
    hardware: HardwareInfo
    toolchain: ToolchainInfo
    case: BenchmarkCase
    correctness: Optional[CorrectnessRecord] = None
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    stdout_artifact: Optional[Artifact] = None
    stderr_artifact: Optional[Artifact] = None
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidenceTag:
    """
    Human/automation labels for learning:
      - "golden" (known-good)
      - "regression"
      - "fast_path"
      - "illegal_but_parses" etc.
    """
    tag: str
    value: Optional[Any] = None

@dataclass
class EvidenceBundle:
    """
    Attach evidence to:
    - a whole kernel
    - a kernel variant (same kernel w different params)
    - an instruction canonical_key
    - or any node in a bigger IR graph (not shown here)
    """
    schema_version: str = "1.0"
    created_ns: int = field(default_factory=now_ns)
    tags: List[EvidenceTag] = field(default_factory=list)
    build: Optional[BuildRecord] = None
    benchmarks: List[BenchmarkRecord] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    # Extensible for “learned legality rules”, heuristics, policy, etc.
    learned: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Linking: an index that maps IR nodes -> evidence
# -----------------------------

@dataclass
class EvidenceIndex:
    """
    Simple in-memory index.
    In production you’d back this with SQLite/Postgres + blob store.
    """
    # key -> EvidenceBundle
    by_kernel_hash: Dict[str, EvidenceBundle] = field(default_factory=dict)
    by_inst_key: Dict[str, EvidenceBundle] = field(default_factory=dict)
    # curated examples registry:
    examples: Dict[str, "WorkingExample"] = field(default_factory=dict)

    def attach_kernel_evidence(self, kernel: PtxKernel, ev: EvidenceBundle) -> None:
        self.by_kernel_hash[kernel.kernel_hash()] = ev

    def attach_instruction_evidence(self, inst: PtxInstruction, ev: EvidenceBundle) -> None:
        self.by_inst_key[inst.canonical_key()] = ev


# -----------------------------
# Working Examples ("API-like")
# -----------------------------

@dataclass
class WorkingExample:
    """
    A reproducible, runnable, *blessed* artifact that demonstrates something working.

    Think of it like an API example:
    - references one or more kernels (PTX-level IR or higher-level IR elsewhere)
    - includes harness config for build/run
    - stores at least one passing benchmark/correctness record
    """
    name: str
    description: str
    module: PtxModule
    harness: Dict[str, Any] = field(default_factory=dict)  # build/run scripts, env, etc.
    # Pointers into EvidenceIndex (or embed evidence directly)
    kernel_hashes: List[str] = field(default_factory=list)
    tags: List[EvidenceTag] = field(default_factory=list)

    def example_hash(self) -> str:
        return content_hash({
            "name": self.name,
            "module_hash": self.module.module_hash(),
            "harness": self.harness,
            "kernel_hashes": self.kernel_hashes,
            "tags": [asdict(t) for t in self.tags],
        })


# -----------------------------
# Serialization (JSON) - no-loss-ish
# -----------------------------

def to_json(obj: Any) -> str:
    return stable_json(asdict(obj))

def from_json_evidence_bundle(s: str) -> EvidenceBundle:
    # Minimal loader example; in practice use pydantic for robust schema evolution.
    d = json.loads(s)
    # For brevity, only reconstruct shallowly; you can build full loaders later.
    return EvidenceBundle(**d)


# -----------------------------
# Example: encode a tcgen05.mma instruction instance
# -----------------------------

def example_tcgen05_mma_inst() -> PtxInstruction:
    # tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale{.scale_vec::4X}
    q = QualifierChain(atoms=(
        QualifierAtom(".cta_group::2"),
        QualifierAtom(".kind::mxf4nvf4"),
        QualifierAtom(".block_scale{.scale_vec::4X}"),
    ))

    # Operands (illustrative):
    # [d-tmem], a-desc, b-desc, idesc, [scale-A-tmem], [scale-B-tmem], enable-input-d;
    inst = PtxInstruction(
        mnemonic="tcgen05.mma",
        qualifiers=q,
        operands=[
            MemRef(space=PtxAddrSpace.tmem, base=Reg("tmem_d"), offset=0, raw_expr="[d_tmem]"),
            Reg("a_desc", ty=PtxType(PtxScalarType.b64), space=PtxAddrSpace.reg),
            Reg("b_desc", ty=PtxType(PtxScalarType.b64), space=PtxAddrSpace.reg),
            Reg("idesc", ty=PtxType(PtxScalarType.b32), space=PtxAddrSpace.reg),
            MemRef(space=PtxAddrSpace.tmem, base=Reg("tmem_sfa"), raw_expr="[scale_A_tmem]"),
            MemRef(space=PtxAddrSpace.tmem, base=Reg("tmem_sfb"), raw_expr="[scale_B_tmem]"),
            Reg("p_enable_d", ty=PtxType(PtxScalarType.pred), space=PtxAddrSpace.reg),
        ],
        comment="Example tcgen05.mma (block_scale). Operand shapes/idesc not shown.",
    )
    return inst


# -----------------------------
# Example: kernel/module + evidence + "golden example" registration
# -----------------------------

def build_demo_working_example() -> Tuple[WorkingExample, EvidenceIndex]:
    inst = example_tcgen05_mma_inst()
    bb = PtxBasicBlock(label="BB0", insts=[inst])
    k = PtxKernel(
        name="demo_tcgen05_kernel",
        target_sm="sm_110a",
        ptx_version="8.8",
        entry_params=[("a", PtxType(PtxScalarType.b64)), ("b", PtxType(PtxScalarType.b64))],
        blocks=[bb],
        metadata={"note": "This is a skeletal PTX kernel body for schema demo."},
    )
    m = PtxModule(name="demo_module", kernels=[k])

    # Evidence bundle (pretend we compiled and benchmarked)
    tool = ToolchainInfo(cuda_version="12.x", nvcc_version="12.x", ptxas_version="12.x")
    hw = HardwareInfo(gpu_name="B200 (demo)", sm="sm_110a", sm_count=120)

    # Artifacts: just placeholders with hashes; in practice compute from file bytes.
    build_log = Artifact(kind=ArtifactKind.log, name="build.stderr", bytes_sha256="0"*64, size_bytes=1234)
    cubin = Artifact(kind=ArtifactKind.cubin, name="demo.cubin", bytes_sha256="1"*64, size_bytes=56789)

    build_rec = BuildRecord(
        status=BuildStatus.success,
        started_ns=now_ns(),
        ended_ns=now_ns(),
        toolchain=tool,
        cmdline=["nvcc", "-arch=sm_110a", "..."],
        stderr_artifact=build_log,
        outputs=[cubin],
    )

    case = BenchmarkCase(
        name="gemv_case_7168x16384x1",
        shapes={"M": 7168, "K": 16384, "L": 1},
        dtypes={"a": "nvfp4(e2m1)", "b": "nvfp4(e2m1)", "sfa": "fp8(e4m3fnuz)", "sfb": "fp8(e4m3fnuz)", "c": "fp16"},
        layouts={"a": "K-major", "b": "K-major", "sfa": "K-major", "sfb": "K-major"},
        seed=123,
        params={"cta_group": 2, "k_tile": 128},
    )

    corr = CorrectnessRecord(ok=True, method="ref_fp32", tolerance={"atol": 1e-2, "rtol": 1e-2})
    metrics = BenchmarkMetrics(time_us_min=8.6, time_us_median=8.7, time_us_p95=9.1, raw={"note": "demo numbers"})
    bench = BenchmarkRecord(
        status=RunStatus.success,
        started_ns=now_ns(),
        ended_ns=now_ns(),
        hardware=hw,
        toolchain=tool,
        case=case,
        correctness=corr,
        metrics=metrics,
    )

    ev = EvidenceBundle(
        tags=[EvidenceTag("golden", True), EvidenceTag("covers", "tcgen05.mma.block_scale")],
        build=build_rec,
        benchmarks=[bench],
        artifacts=[cubin, build_log],
        learned={
            # Example of "learned legality": not authoritative, but useful for sampling/search.
            "notes": "Compiles and runs on sm_110a with PTX 8.8. Mark as seed example.",
            "constraint_hints": {
                "tcgen05.kernel_level": ["all tcgen05 must share same .cta_group (observed)"],
            },
        },
    )

    idx = EvidenceIndex()
    idx.attach_kernel_evidence(k, ev)
    idx.attach_instruction_evidence(inst, EvidenceBundle(tags=[EvidenceTag("inst_works", True)], benchmarks=[bench]))

    ex = WorkingExample(
        name="golden_tcgen05_mma_block_scale",
        description="Minimal working example exercising tcgen05.mma block_scale path (schema demo).",
        module=m,
        harness={
            "build": {"cmd": ["nvcc", "-arch=sm_110a", "demo.cu", "-o", "demo"]},
            "run": {"cmd": ["./demo", "--case", "gemv_case_7168x16384x1"]},
            "notes": "In real usage, store exact source + env in artifacts.",
        },
        kernel_hashes=[k.kernel_hash()],
        tags=[EvidenceTag("golden", True), EvidenceTag("dialect", "ptx")],
    )
    idx.examples[ex.name] = ex

    return ex, idx


# -----------------------------
# If you run this file directly, dump a demo JSON
# -----------------------------

if __name__ == "__main__":
    ex, idx = build_demo_working_example()

    out = {
        "example": asdict(ex),
        "kernel_evidence": {k: asdict(v) for k, v in idx.by_kernel_hash.items()},
        "inst_evidence": {k: asdict(v) for k, v in idx.by_inst_key.items()},
    }
    print(stable_json(out))
