from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


# -----------------------------
# Generic helpers
# -----------------------------


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def slice(self, s: str) -> str:
        return s[self.start : self.end]


def _line_col(s: str, idx: int) -> tuple[int, int]:
    line = s.count("\n", 0, idx) + 1
    last_nl = s.rfind("\n", 0, idx)
    col = idx + 1 if last_nl < 0 else (idx - last_nl)
    return line, col


# -----------------------------
# PTX IR (minimal + extensible)
# -----------------------------


PtxKind = Literal["inst", "directive", "label", "brace", "comment", "raw"]


@dataclass(frozen=True)
class PtxStmt:
    kind: PtxKind
    text: str  # canonical statement text (no trailing ';')
    pred: str | None = None  # e.g. "@%p0" / "@!%p0"
    op: str | None = None  # full opcode token, e.g. "ld.global.cs.v4.u32"
    op_parts: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()

    def render(self) -> str:
        if self.kind == "brace":
            return self.text
        if self.kind in ("comment", "label", "raw"):
            return self.text
        if self.kind == "directive":
            return f"{self.text};"
        # inst
        pred = f"{self.pred} " if self.pred else ""
        ops = ", ".join(self.operands) if self.operands else ""
        if ops:
            return f"{pred}{self.op} {ops};"
        return f"{pred}{self.op};"


@dataclass(frozen=True)
class PtxProgram:
    stmts: tuple[PtxStmt, ...]

    def render(self) -> str:
        # Keep it simple: one stmt per line.
        out = []
        for st in self.stmts:
            out.append(st.render())
        return "\n".join(out) + ("\n" if out and not out[-1].endswith("\n") else "")


def _split_operands(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    pairs = {"{": "}", "[": "]", "(": ")"}
    opens = set(pairs)
    closes = set(pairs.values())
    for ch in s:
        if ch in opens:
            depth += 1
        elif ch in closes and depth:
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _strip_trailing_comment(s: str) -> tuple[str, str | None]:
    # Split on '//' if present. PTX also allows '//' comments; keep them as a separate stmt.
    i = s.find("//")
    if i < 0:
        return s, None
    return s[:i].rstrip(), s[i:].rstrip()


def parse_ptx(template: str) -> PtxProgram:
    template = template.replace("\r\n", "\n").replace("\r", "\n")
    stmts_raw: list[str] = []
    buf: list[str] = []
    for ch in template:
        buf.append(ch)
        if ch == ";":
            stmts_raw.append("".join(buf))
            buf = []
        elif ch == "\n":
            if "".join(buf).strip() in ("{", "}"):
                stmts_raw.append("".join(buf))
                buf = []
    tail = "".join(buf)
    if tail.strip():
        stmts_raw.append(tail)

    stmts: list[PtxStmt] = []
    for raw in stmts_raw:
        s = raw.strip()
        if not s:
            continue
        if s in ("{", "}"):
            stmts.append(PtxStmt(kind="brace", text=s))
            continue
        s = s[:-1].rstrip() if s.endswith(";") else s
        code, comment = _strip_trailing_comment(s)
        if comment and not code:
            stmts.append(PtxStmt(kind="comment", text=comment))
            continue
        code = code.strip()
        if not code:
            if comment:
                stmts.append(PtxStmt(kind="comment", text=comment))
            continue
        if code.startswith("//"):
            stmts.append(PtxStmt(kind="comment", text=code))
            continue
        if code.endswith(":") and " " not in code and "\t" not in code:
            stmts.append(PtxStmt(kind="label", text=code))
            continue
        if code.startswith("."):
            stmts.append(PtxStmt(kind="directive", text=code))
            continue

        pred = None
        rest = code
        if rest.startswith("@"):
            parts = rest.split(None, 1)
            pred = parts[0]
            rest = parts[1] if len(parts) == 2 else ""
        if not rest:
            stmts.append(PtxStmt(kind="raw", text=code))
            continue
        parts = rest.split(None, 1)
        op = parts[0]
        ops_str = parts[1] if len(parts) == 2 else ""
        operands = tuple(_split_operands(ops_str))
        stmts.append(
            PtxStmt(
                kind="inst",
                text=code,
                pred=pred,
                op=op,
                op_parts=tuple(op.split(".")),
                operands=operands,
            )
        )
        if comment:
            stmts.append(PtxStmt(kind="comment", text=comment))

    return PtxProgram(stmts=tuple(stmts))


# -----------------------------
# CUDA IR: extract inline asm + ptx:: calls
# -----------------------------


@dataclass(frozen=True)
class InlineAsm:
    volatile: bool
    template_span: Span  # span of the template expression inside the CUDA/C++ source
    template: str  # decoded, concatenated template string
    ptx: PtxProgram
    outputs: str | None
    inputs: str | None
    clobbers: str | None
    gotos: str | None


@dataclass(frozen=True)
class PtxCall:
    name: str  # e.g. "mbarrier_try_wait_parity"
    call_span: Span
    args: str


def _decode_c_escapes(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(s):
            out.append("\\")
            break
        esc = s[i]
        i += 1
        if esc == "n":
            out.append("\n")
        elif esc == "t":
            out.append("\t")
        elif esc == "r":
            out.append("\r")
        elif esc == "0":
            out.append("\0")
        elif esc in ('\\', '"', "'"):
            out.append(esc)
        elif esc == "x":
            hx = []
            while i < len(s) and len(hx) < 2 and s[i] in "0123456789abcdefABCDEF":
                hx.append(s[i])
                i += 1
            out.append(chr(int("".join(hx or ["0"]), 16)))
        else:
            out.append(esc)
    return "".join(out)


def _concat_c_string_literals(expr: str) -> str:
    # Best-effort: concatenate all normal C/C++ string literals found in the expression.
    out: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i] != '"':
            i += 1
            continue
        i += 1
        buf: list[str] = []
        while i < len(expr):
            ch = expr[i]
            if ch == "\\" and i + 1 < len(expr):
                buf.append(expr[i])
                buf.append(expr[i + 1])
                i += 2
                continue
            if ch == '"':
                i += 1
                break
            buf.append(ch)
            i += 1
        out.append(_decode_c_escapes("".join(buf)))
    return "".join(out)


def _encode_c_string_expr(text: str) -> str:
    # Emit as concatenated per-line string literals to avoid extreme single-line literals.
    def esc_line(s: str) -> str:
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t")
        return s

    lines = text.splitlines(keepends=True)
    if not lines:
        return '""'
    out = []
    for ln in lines:
        if ln.endswith("\n"):
            core = ln[:-1]
            out.append(f"\"{esc_line(core)}\\\\n\"")
        else:
            out.append(f"\"{esc_line(ln)}\"")
    return "\n".join(out)


def _find_matching_paren(src: str, open_idx: int) -> int | None:
    # open_idx points at '('
    i = open_idx
    depth = 0
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < len(src):
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_char:
            if ch == "\\" and i + 1 < len(src):
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _split_asm_colon_sections(src: str, start: int, end: int) -> list[Span]:
    # Split src[start:end] on ':' at top level (not in strings/comments/paren/bracket/brace).
    spans: list[Span] = []
    seg_start = start
    par = br = bc = 0
    in_str = in_char = in_line_comment = in_block_comment = False
    i = start
    while i < end:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < end else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < end:
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_char:
            if ch == "\\" and i + 1 < end:
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue

        if ch == "(":
            par += 1
        elif ch == ")" and par:
            par -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br:
            br -= 1
        elif ch == "{":
            bc += 1
        elif ch == "}" and bc:
            bc -= 1

        if ch == ":" and par == br == bc == 0:
            spans.append(Span(seg_start, i))
            seg_start = i + 1
        i += 1
    spans.append(Span(seg_start, end))
    return spans


def extract_inline_asm(cuda_src: str) -> list[InlineAsm]:
    out: list[InlineAsm] = []
    i = 0
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < len(cuda_src):
        ch = cuda_src[i]
        nxt = cuda_src[i + 1] if i + 1 < len(cuda_src) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < len(cuda_src):
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_char:
            if ch == "\\" and i + 1 < len(cuda_src):
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue

        if cuda_src.startswith("asm", i) and (i == 0 or not (cuda_src[i - 1].isalnum() or cuda_src[i - 1] == "_")):
            end_tok = i + 3
            if end_tok < len(cuda_src) and (cuda_src[end_tok].isalnum() or cuda_src[end_tok] == "_"):
                i += 1
                continue
            j = end_tok
        # Skip whitespace
            while j < len(cuda_src) and cuda_src[j].isspace():
                j += 1
            volatile = False
            if cuda_src.startswith("volatile", j):
                volatile = True
                j += len("volatile")
                while j < len(cuda_src) and cuda_src[j].isspace():
                    j += 1
            if j >= len(cuda_src) or cuda_src[j] != "(":
                i += 1
                continue
            close = _find_matching_paren(cuda_src, j)
            if close is None:
                i += 1
                continue

            args_start, args_end = j + 1, close
            sections = _split_asm_colon_sections(cuda_src, args_start, args_end)
            if not sections:
                i = close + 1
                continue
            template_span = sections[0]
            template_expr = template_span.slice(cuda_src)
            template = _concat_c_string_literals(template_expr)
            ptx = parse_ptx(template)

            def sec(k: int) -> str | None:
                if k >= len(sections):
                    return None
                return sections[k].slice(cuda_src).strip()

            out.append(
                InlineAsm(
                    volatile=volatile,
                    template_span=template_span,
                    template=template,
                    ptx=ptx,
                    outputs=sec(1),
                    inputs=sec(2),
                    clobbers=sec(3),
                    gotos=sec(4),
                )
            )
            i = close + 1
            continue
        i += 1
    return out


def extract_ptx_calls(cuda_src: str) -> list[PtxCall]:
    out: list[PtxCall] = []
    for m in re.finditer(r"\bptx::([A-Za-z_][A-Za-z0-9_]*)\s*\(", cuda_src):
        name = m.group(1)
        open_idx = cuda_src.find("(", m.end() - 1)
        close = _find_matching_paren(cuda_src, open_idx)
        if close is None:
            continue
        args = cuda_src[open_idx + 1 : close].strip()
        out.append(PtxCall(name=name, call_span=Span(m.start(), close + 1), args=args))
    return out


@dataclass(frozen=True)
class CudaSourceIR:
    src: str
    asm: tuple[InlineAsm, ...]
    ptx_calls: tuple[PtxCall, ...]

    def patched(self, replacements: dict[int, str]) -> str:
        # replacements: asm-index -> new template string (decoded)
        out = self.src
        edits: list[tuple[int, int, str]] = []
        for idx, new_template in replacements.items():
            block = self.asm[idx]
            edits.append((block.template_span.start, block.template_span.end, _encode_c_string_expr(new_template)))
        for a, b, rep in sorted(edits, key=lambda t: t[0], reverse=True):
            out = out[:a] + rep + out[b:]
        return out


# -----------------------------
# Python submission parsing (extract cpp/cuda strings + load_inline args)
# -----------------------------


@dataclass(frozen=True)
class LoadInlineIR:
    name: Any | None
    cpp_sources: tuple[str, ...]
    cuda_sources: tuple[str, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class KernelIR:
    path: str
    load_inline: LoadInlineIR
    cuda: tuple[CudaSourceIR, ...]


def _py_eval(node: ast.AST, env: dict[str, Any]) -> Any | None:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = []
        for el in node.elts:
            v = _py_eval(el, env)
            if v is None:
                return None
            vals.append(v)
        return vals
    if isinstance(node, ast.Dict):
        d: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values, strict=False):
            kk = _py_eval(k, env) if k is not None else None
            vv = _py_eval(v, env)
            if kk is None or vv is None:
                return None
            d[kk] = vv
        return d
    return None


def parse_kernel_file(path: str | Path) -> KernelIR:
    path = str(path)
    src = Path(path).read_text()
    tree = ast.parse(src, filename=path)

    env: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = _py_eval(node.value, env)
            if v is not None:
                env[node.targets[0].id] = v
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            v = _py_eval(node.value, env)
            if v is not None:
                env[node.target.id] = v

    load_call: ast.Call | None = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name) and fn.id == "load_inline":
                load_call = n
                break
            if isinstance(fn, ast.Attribute) and fn.attr == "load_inline":
                load_call = n
                break
    if load_call is None:
        raise ValueError(f"No load_inline() call found in {path}")

    kw_nodes = {k.arg: k.value for k in load_call.keywords if k.arg}

    def get_kw(name: str) -> Any | None:
        n = kw_nodes.get(name)
        return _py_eval(n, env) if n is not None else None

    name_val = get_kw("name")
    if name_val is None and load_call.args:
        name_val = _py_eval(load_call.args[0], env)

    cpp_sources_val = get_kw("cpp_sources")
    cuda_sources_val = get_kw("cuda_sources")
    if cpp_sources_val is None:
        cpp_sources_val = ""
    if cuda_sources_val is None:
        raise ValueError(f"load_inline() missing cuda_sources in {path}")

    def as_str_list(v: Any) -> tuple[str, ...]:
        if v is None:
            return tuple()
        if isinstance(v, str):
            return (v,)
        if isinstance(v, list):
            if not all(isinstance(x, str) for x in v):
                raise TypeError(f"Expected list[str], got {type(v)}")
            return tuple(v)
        raise TypeError(f"Expected str|list[str], got {type(v)}")

    cpp_list = as_str_list(cpp_sources_val)
    cuda_list = as_str_list(cuda_sources_val)

    kwargs_eval: dict[str, Any] = {}
    for k, v in kw_nodes.items():
        vv = _py_eval(v, env)
        kwargs_eval[k] = vv if vv is not None else ast.get_source_segment(src, v)

    li = LoadInlineIR(
        name=name_val,
        cpp_sources=cpp_list,
        cuda_sources=cuda_list,
        kwargs=kwargs_eval,
    )
    cuda_irs = []
    for cu in cuda_list:
        asm = extract_inline_asm(cu)
        calls = extract_ptx_calls(cu)
        cuda_irs.append(CudaSourceIR(src=cu, asm=tuple(asm), ptx_calls=tuple(calls)))

    return KernelIR(path=path, load_inline=li, cuda=tuple(cuda_irs))


def parse_dir(dir_path: str | Path) -> list[KernelIR]:
    p = Path(dir_path)
    return [parse_kernel_file(x) for x in sorted(p.glob("*.py"))]
