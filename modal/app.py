import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable

import modal


APP_NAME = "nv_app"
VOLUME_NAME = "nv_vol"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
#BASE_IMAGE = "nvidia/cuda:13.1.0-devel-ubuntu24.04"
#BASE_IMAGE = "nvcr.io/nvidia/pytorch:25.11-py3"
#BASE_IMAGE  = "nvcr.io/nvidia/pytorch:25.09-py3"

IMAGE_ENV = {
    "HF_HOME": "/modal_data/hf",
    "HUGGINGFACE_HUB_CACHE": "/modal_data/hf",
}
RUN_COMMANDS = [
    "apt-get update && apt-get install -y curl ca-certificates gnupg",
    "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
    "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
    "apt-get update",
]
APT_PACKAGES = [
    "graphviz",
]
UV_PACKAGES = [
 #   "torch==2.9.1",
    "nvidia-cutlass-dsl",
    "nvtx",
    "pydot",
]
GPU_DEFAULT = "B200"
GPU_ALIASES = {
    "L4": "L4",
    "L40S": "L40S",
    "A100": "A100",
    "H100": "H100",
    "B200": "B200"
}

LOCAL_WORKSPACE = Path("nvfp4")
LOCAL_OUTPUTS = Path("out_local")
VOLUME_MOUNT_PATH = PurePosixPath("/modal_data")


def _to_volume_path(container_path: PurePosixPath) -> PurePosixPath:
    try:
        relative = container_path.relative_to(VOLUME_MOUNT_PATH)
    except ValueError:
        return container_path

    if str(relative) in {"", "."}:
        return PurePosixPath("/")

    return PurePosixPath("/") / relative


REMOTE_WORKSPACE = PurePosixPath("/modal_data/nvfp4")
REMOTE_OUTPUTS = PurePosixPath("/modal_data/nvfp4/out_local")
VOLUME_WORKSPACE_PATH = _to_volume_path(REMOTE_WORKSPACE)
VOLUME_OUTPUTS_PATH = _to_volume_path(REMOTE_OUTPUTS)


def _build_image() -> modal.Image:
    image = modal.Image.from_registry(BASE_IMAGE).env(IMAGE_ENV)

    for command in RUN_COMMANDS:
        image = image.run_commands(command)

    if APT_PACKAGES:
        image = image.apt_install(*APT_PACKAGES)

    image = image.run_commands("pip install --upgrade pip uv")
    if UV_PACKAGES:
        image = image.run_commands("uv pip install --system " + " ".join(UV_PACKAGES))

    return image


def _gpu_type(name: str | None = None) -> str:
    alias = name or GPU_DEFAULT
    return GPU_ALIASES.get(alias, alias)


def _cpu_model_name() -> str:
    import platform

    cpu = platform.processor()
    if cpu:
        return cpu

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return "Unknown"


def _system_info() -> dict[str, str | int]:
    import platform
    import torch

    has_cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if has_cuda and torch.cuda.device_count() > 0 else "CPU"
    device_count = torch.cuda.device_count() if has_cuda else 0

    return {
        "gpu": gpu,
        "cpu": _cpu_model_name(),
        "device_count": device_count,
        "runtime": "CUDA" if has_cuda else "CPU",
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "hostname": platform.node(),
    }


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = _build_image()


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type())
def build_image_remote() -> None:
    """Build the container image on Modal."""


def _download_entries(entries: Iterable[modal.volume.FileEntry]) -> int:
    downloaded = 0
    for entry in entries:
        remote_path = PurePosixPath(entry.path)
        try:
            rel_path = remote_path.relative_to(VOLUME_OUTPUTS_PATH)
        except ValueError:
            rel_path = remote_path

        local_target = LOCAL_OUTPUTS / Path(rel_path.as_posix())

        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            local_target.mkdir(parents=True, exist_ok=True)
            continue

        if entry.type != modal.volume.FileEntryType.FILE:
            continue

        local_target.parent.mkdir(parents=True, exist_ok=True)
        with local_target.open("wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        downloaded += 1

    return downloaded


def _sync_outputs_impl(verbose: bool = True) -> int:
    try:
        entries = volume.listdir(VOLUME_OUTPUTS_PATH.as_posix(), recursive=True)
    except FileNotFoundError:
        if verbose:
            print(f"No outputs at {REMOTE_OUTPUTS}; nothing to download.")
        return 0

    if not entries:
        if verbose:
            print(f"No outputs at {REMOTE_OUTPUTS}; nothing to download.")
        return 0

    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    downloaded = _download_entries(entries)
    if verbose:
        print(f"Downloaded {downloaded} file(s) from {REMOTE_OUTPUTS} into {LOCAL_OUTPUTS}")
    return downloaded


@app.local_entrypoint()
def sync_workspace() -> None:
    if not LOCAL_WORKSPACE.is_dir():
        raise FileNotFoundError(f"Missing local workspace directory: {LOCAL_WORKSPACE}")

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_WORKSPACE), VOLUME_WORKSPACE_PATH.as_posix())

    print(f"Uploaded {LOCAL_WORKSPACE} -> {REMOTE_WORKSPACE} (volume {volume.name})")


@app.local_entrypoint()
def sync_outputs() -> None:
    _sync_outputs_impl(verbose=True)


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type())
def run_profile_script() -> None:
    subprocess.run(["/bin/bash", "profile_test.sh"], cwd=str(REMOTE_WORKSPACE), check=True)


@app.local_entrypoint()
def profile_and_fetch() -> None:
    run_profile_script.call()
    downloaded = _sync_outputs_impl(verbose=False)
    if downloaded:
        print(f"Profile artifacts pulled into {LOCAL_OUTPUTS}")
    else:
        print("Profile completed, but no outputs were found to download.")


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume})
def volume_shell() -> None:
    """Dummy function spec for opening a Modal shell with the workspace volume mounted."""


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type())
def gpu_shell() -> None:
    """Dummy function spec for opening a Modal shell with both the volume and GPU attached."""


# --- Hash-based sync ---
PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = "/manifest.json"
EVAL_DIR = PROJECT_ROOT / "eval"

# Task directories to sync (each becomes /workspace/{dir_name}/)
# common/ contains shared utilities used by all tasks
TASK_DIRS = ["common", "nvfp4_gemv", "nvfp4_gemm", "nvfp4_dual_gemm", "nvfp4_grouped_gemm"]


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _get_sync_mapping() -> dict[str, tuple[Path, str]]:
    """Build mapping of manifest_key -> (local_path, remote_path) for all task files."""
    mapping = {}

    for task_dir in TASK_DIRS:
        task_local_dir = EVAL_DIR / task_dir

        # Sync all files from task directory to /workspace/{task_dir}/
        if task_local_dir.exists():
            for f in task_local_dir.iterdir():
                if f.is_file() and not f.name.endswith(":Zone.Identifier"):
                    manifest_key = f"{task_dir}/{f.name}"
                    remote_path = f"/{task_dir}/{f.name}"
                    mapping[manifest_key] = (f, remote_path)

    return mapping


def _get_local_manifest() -> dict[str, str]:
    """Build manifest with hashes for all files to sync."""
    mapping = _get_sync_mapping()
    return {key: _file_hash(local_path) for key, (local_path, _) in mapping.items()}


def _get_remote_manifest() -> dict[str, str]:
    try:
        data = b"".join(volume.read_file(MANIFEST_PATH))
        return json.loads(data)
    except Exception:
        return {}


def sync_project() -> int:
    """Sync project files to volume, uploading only changed files."""
    mapping = _get_sync_mapping()
    local = {key: _file_hash(local_path) for key, (local_path, _) in mapping.items()}
    remote = _get_remote_manifest()

    changed = [k for k, v in local.items() if remote.get(k) != v]
    deleted = [k for k in remote if k not in local]

    if not changed and not deleted:
        print("No changes to sync.")
        return 0

    # Write manifest to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(local, f)
        manifest_tmp = f.name

    with volume.batch_upload(force=True) as batch:
        for key in changed:
            local_path, remote_path = mapping[key]
            batch.put_file(str(local_path), remote_path)
        batch.put_file(manifest_tmp, MANIFEST_PATH)

    Path(manifest_tmp).unlink()
    print(f"Synced {len(changed)} file(s), {len(deleted)} removed.")
    return len(changed)


# --- Remote eval ---
# Eval script name per task
EVAL_SCRIPTS = {
    "nvfp4_gemv": "eval.py",
    "nvfp4_gemm": "eval.py",
    "nvfp4_dual_gemm": "eval.py",
    "nvfp4_grouped_gemm": "eval.py",
}


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type(), timeout=600)
def run_eval(submission_code: str, tests_content: str, mode: str = "test", workspace_name: str = "nvfp4_gemv") -> dict:
    """Run eval remotely with given submission and tests."""
    import sys
    work = Path(f"{VOLUME_MOUNT_PATH}/{workspace_name}")
    work.mkdir(exist_ok=True)

    # Write submission and tests
    (work / "submission.py").write_text(submission_code)
    (work / "tests.txt").write_text(tests_content)

    # Get the correct eval script for this task
    eval_script = EVAL_SCRIPTS.get(workspace_name, "eval.py")

    # Set up output capture via pipe
    r, w = os.pipe()
    os.set_inheritable(w, True)
    env = os.environ.copy()
    env["POPCORN_FD"] = str(w)

    proc = subprocess.Popen(
        [sys.executable, eval_script, mode, "tests.txt"],
        cwd=str(work), env=env, pass_fds=(w,),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    os.close(w)

    stdout, stderr = proc.communicate()
    output = os.read(r, 1 << 20).decode()
    os.close(r)

    return {
        "popcorn": output,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "mode": mode,
        "system": _system_info(),
    }


@app.function(image=image, volumes={str(VOLUME_MOUNT_PATH): volume}, gpu=_gpu_type(), timeout=1800)
def run_eval_batch(
    submissions: dict[str, str],
    tests_content: str,
    mode: str = "test",
    workspace_name: str = "nvfp4_gemv",
    workers: int = 1,
) -> dict[str, dict]:
    """Run many submissions inside one GPU container (optional in-container parallelism)."""
    import sys
    import concurrent.futures

    work = Path(f"{VOLUME_MOUNT_PATH}/{workspace_name}")
    work.mkdir(exist_ok=True)
    eval_script = EVAL_SCRIPTS.get(workspace_name, "eval.py")

    # Keep eval.py's local imports intact; provide submission.py via PYTHONPATH per worker.
    stale_submission = work / "submission.py"
    if stale_submission.exists():
        stale_submission.unlink()

    tests_path = work / "__batch_tests__.txt"
    tests_path.write_text(tests_content)

    submit_root = work / "__batch_submissions__"
    submit_root.mkdir(parents=True, exist_ok=True)

    system_info = _system_info()

    def _run_one(item: tuple[str, str], idx: int) -> tuple[str, dict]:
        name, code = item
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
        subdir = submit_root / f"{idx:04d}_{safe}"
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "submission.py").write_text(code)

        r, w = os.pipe()
        os.set_inheritable(w, True)
        env = os.environ.copy()
        env["POPCORN_FD"] = str(w)
        old_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{subdir}:{old_pp}" if old_pp else str(subdir)

        proc = subprocess.Popen(
            [sys.executable, eval_script, mode, str(tests_path)],
            cwd=str(work),
            env=env,
            pass_fds=(w,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.close(w)

        stdout, stderr = proc.communicate()
        output = os.read(r, 1 << 20).decode()
        os.close(r)

        return name, {
            "popcorn": output,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "mode": mode,
            "system": system_info,
        }

    ordered = list(submissions.items())
    worker_count = max(1, int(workers))
    if worker_count > len(ordered):
        worker_count = len(ordered)

    results: dict[str, dict] = {}
    if worker_count == 1:
        for idx, item in enumerate(ordered):
            name, result = _run_one(item, idx)
            results[name] = result
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futs = [pool.submit(_run_one, item, idx) for idx, item in enumerate(ordered)]
        for fut in concurrent.futures.as_completed(futs):
            try:
                name, result = fut.result()
                results[name] = result
            except Exception as e:
                err = {
                    "popcorn": "",
                    "stdout": "",
                    "stderr": f"run_eval_batch worker exception: {e}",
                    "mode": mode,
                    "system": system_info,
                }
                # Keep deterministic key if we can, otherwise use synthetic name.
                key = f"batch_worker_error_{len(results)}"
                results[key] = err

    return results


__all__ = [
    "app",
    "image",
    "volume",
    "build_image_remote",
    "run_profile_script",
    "profile_and_fetch",
    "volume_shell",
    "gpu_shell",
    "sync_workspace",
    "sync_outputs",
    "sync_project",
    "run_eval",
    "run_eval_batch",
]
