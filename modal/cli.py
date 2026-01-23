#!/usr/bin/env python3
"""CLI for running kernel eval on Modal."""
import argparse
import sys
from pathlib import Path

import modal
from app import app, sync_project, run_eval

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval"

# Task configurations: task_name -> workspace directory name
TASKS = {
    "gemv": "nvfp4_gemv",
    "gemm": "nvfp4_gemm",
    "dual_gemm": "nvfp4_dual_gemm",
    "grouped_gemm": "nvfp4_grouped_gemm",
}

CHECK = "\u2705"
CROSS = "\u274c"
STOPWATCH = "\u23f1"
ZAP = "\u26a1"
SNAIL = "\U0001F40C"
MICRO = "\u00b5"


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _looks_formatted(text: str) -> bool:
    trimmed = text.lstrip()
    return (
        "## Benchmarks:" in text
        or "## Program stdout:" in text
        or trimmed.startswith('"**')
        or "NVIDIA on GitHub" in text
    )


def _unwrap_quotes(text: str) -> tuple[str, bool]:
    trimmed = text.strip()
    if trimmed.startswith('"') and trimmed.endswith('"'):
        return trimmed[1:-1], True
    return text, False


def _reorder_spec(spec: str) -> str:
    parts = [part.strip() for part in spec.split(";") if part.strip()]
    kv = {}
    for part in parts:
        key, sep, value = part.partition(":")
        if not sep:
            continue
        kv[key.strip()] = value.strip()
    if not kv:
        return spec.strip()

    order = ["k", "l", "m", "n", "g", "seed"]
    ordered = [f"{key}: {kv[key]}" for key in order if key in kv]
    remaining = sorted(k for k in kv.keys() if k not in order)
    ordered.extend(f"{key}: {kv[key]}" for key in remaining)
    return "; ".join(ordered)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_popcorn_log(log_text: str) -> dict:
    data: dict = {"benchmarks": {}, "tests": {}}
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()

        if key == "compile":
            data["compile"] = value
            continue
        if key == "compile.error":
            data["compile_error"] = value
            continue
        if key == "check":
            data["check"] = value
            continue
        if key == "benchmark-count":
            try:
                data["benchmark_count"] = int(value)
            except ValueError:
                data["benchmark_count"] = value
            continue
        if key.startswith("benchmark."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                idx = int(parts[1])
                field = parts[2]
                data["benchmarks"].setdefault(idx, {})[field] = value
            continue
        if key == "test-count":
            try:
                data["test_count"] = int(value)
            except ValueError:
                data["test_count"] = value
            continue
        if key.startswith("test."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                idx = int(parts[1])
                field = parts[2]
                data["tests"].setdefault(idx, {})[field] = value
            continue

    return data


def _format_benchmarks(benchmarks: dict[int, dict[str, str]]) -> list[str]:
    lines = ["## Benchmarks:", "```"]
    items = sorted(benchmarks.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))

        status = info.get("status")
        error = info.get("error")
        mean = _to_float(info.get("mean"))
        err = _to_float(info.get("err"))
        best = _to_float(info.get("best"))
        worst = _to_float(info.get("worst"))

        # Show timing if available (even if correctness failed)
        if None not in (mean, err, best, worst):
            lines.append(f" {STOPWATCH} {mean / 1000:.1f} \u00b1 {err / 1000:.2f} {MICRO}s")
            lines.append(f" {ZAP} {best / 1000:.1f} {MICRO}s {SNAIL} {worst / 1000:.1f} {MICRO}s")
        # Show error if present
        if status == "fail" or error:
            if error:
                lines.append(f" {CROSS} {error}")
            else:
                lines.append(f" {CROSS} failed")

        if idx != items[-1][0]:
            lines.append("")

    lines.append("```")
    return lines


def _format_tests(tests: dict[int, dict[str, str]]) -> list[str]:
    lines = ["## Tests:", "```"]
    items = sorted(tests.items())
    for idx, info in items:
        spec = info.get("spec", "")
        if spec:
            lines.append(_reorder_spec(spec))
        status = info.get("status", "")
        error = info.get("error")
        if status == "pass":
            lines.append(f" {CHECK} pass")
        elif status == "fail":
            lines.append(f" {CROSS} fail")
        if error:
            lines.append(f"   {error}")
        if idx != items[-1][0]:
            lines.append("")
    lines.append("```")
    return lines


def _format_program_output(stdout: str, stderr: str) -> list[str]:
    combined = stdout or ""
    if stderr:
        if combined:
            combined = combined.rstrip("\n") + "\n--- stderr ---\n" + stderr
        else:
            combined = "--- stderr ---\n" + stderr

    lines = ["## Program stdout:", "```"]
    if combined:
        lines.extend(combined.rstrip("\n").splitlines())
    lines.append("```")
    return lines


def _phase_label(mode: str, data: dict) -> str:
    if data.get("benchmarks"):
        return "Benchmarking"
    if data.get("tests"):
        return "Testing"
    if mode == "profile":
        return "Profiling"
    if mode == "test":
        return "Testing"
    return "Run"


def _build_formatted_output(
    data: dict,
    stdout: str,
    stderr: str,
    system: dict | None,
    mode: str,
) -> str:
    compile_status = data.get("compile")
    check_status = data.get("check")
    success = check_status == "pass"
    if compile_status == "fail":
        success = False

    header_status = "success" if success else "failure"
    lines = [f"\"**Modal {CHECK if success else CROSS} {header_status}**"]

    if compile_status:
        lines.append(
            f"> {CHECK if compile_status == 'pass' else CROSS} "
            f"Compilation {'successful' if compile_status == 'pass' else 'failed'}"
        )

    if check_status:
        phase = _phase_label(mode, data)
        lines.append(
            f"> {CHECK if check_status == 'pass' else CROSS} "
            f"{phase} {'successful' if check_status == 'pass' else 'failed'}"
        )

    lines.append("")

    if system:
        lines.append("Running on:")
        lines.append(f"* GPU: `{system.get('gpu', 'Unknown')}`")
        lines.append(f"* CPU: `{system.get('cpu', 'Unknown')}`")
        if "device_count" in system:
            lines.append(f"* Device count: `{system.get('device_count')}`")
        lines.append(f"* Runtime: `{system.get('runtime', 'Unknown')}`")
        lines.append(f"* Platform: `{system.get('platform', 'Unknown')}`")
        lines.append(f"* Torch: `{system.get('torch', 'Unknown')}`")
        lines.append(f"* Hostname: `{system.get('hostname', 'Unknown')}`")
        lines.append("")
        lines.append("")

    benchmarks = data.get("benchmarks", {})
    tests = data.get("tests", {})

    if benchmarks:
        lines.extend(_format_benchmarks(benchmarks))
        lines.append("")
    elif tests:
        lines.extend(_format_tests(tests))
        lines.append("")

    lines.extend(_format_program_output(stdout, stderr))
    return "\n".join(lines) + "\""


def _format_result(result, mode: str) -> str:
    if isinstance(result, str):
        if _looks_formatted(result):
            return result
        data = _parse_popcorn_log(result)
        return _build_formatted_output(data, "", "", None, mode)

    popcorn = result.get("popcorn", "")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    system = result.get("system")
    result_mode = result.get("mode", mode)

    if _looks_formatted(popcorn):
        body, wrapped = _unwrap_quotes(popcorn)
        if "## Program stdout:" not in body and (stdout or stderr):
            body = body.rstrip("\n") + "\n\n" + "\n".join(_format_program_output(stdout, stderr))
        return f"\"{body}\"" if wrapped else body

    data = _parse_popcorn_log(popcorn)
    return _build_formatted_output(data, stdout, stderr, system, result_mode)


def get_task_dir(task: str) -> Path:
    """Get the eval directory for a task."""
    return EVAL_ROOT / TASKS[task]


def run_single(
    submission: Path,
    output: Path,
    mode: str,
    task: str,
    no_sync: bool,
    suppress_stdout: bool,
):
    """Run a single submission."""
    task_dir = get_task_dir(task)
    workspace_name = TASKS[task]

    # Use benchmarks.txt for benchmark/leaderboard modes
    tests_file = task_dir / ("benchmarks.txt" if mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    if not no_sync:
        sync_project()

    submission_code = submission.read_text()
    tests_content = tests_file.read_text()

    _log(f"Running {mode} for task '{task}' on Modal...")
    with app.run():
        result = run_eval.remote(submission_code, tests_content, mode, workspace_name)

    formatted = _format_result(result, mode)
    output.write_text(formatted)
    if not suppress_stdout:
        print(formatted)
    _log(f"Output saved to {output}")


def run_batch(
    submissions_dir: Path,
    output_dir: Path,
    mode: str,
    task: str,
    suppress_stdout: bool,
):
    """Run all submissions in a directory in a single container."""
    task_dir = get_task_dir(task)
    workspace_name = TASKS[task]

    # Use benchmarks.txt for benchmark/leaderboard modes
    tests_file = task_dir / ("benchmarks.txt" if mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    # Find all .py files (excluding Zone.Identifier files)
    submissions = sorted([
        f for f in submissions_dir.glob("*.py")
        if not f.name.endswith(":Zone.Identifier")
    ])

    if not submissions:
        _log(f"No submissions found in {submissions_dir}")
        return

    # Sync once before running all
    sync_project()

    tests_content = tests_file.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Running {len(submissions)} submissions in batch mode (task: {task}, mode: {mode})")

    # Run all submissions in a single app.run() context
    with app.run():
        for submission in submissions:
            basename = submission.stem
            output_file = output_dir / f"{basename}.txt"

            _log(f"  [{submissions.index(submission)+1}/{len(submissions)}] {basename}...")

            try:
                submission_code = submission.read_text()
                result = run_eval.remote(submission_code, tests_content, mode, workspace_name)
                formatted = _format_result(result, mode)
                output_file.write_text(formatted)
                if not suppress_stdout:
                    print(formatted)
            except Exception as e:
                _log(f"FAILED: {e}")
                output_file.write_text(f"Error: {e}")

    _log(f"Batch complete. Results in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run kernel eval on Modal")
    parser.add_argument("submission", nargs="?", help="Submission file path (or use -b for batch)")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file (or directory for batch)")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="gemv", choices=list(TASKS.keys()),
                        help="Task to run (default: gemv)")
    parser.add_argument("-b", "--batch", metavar="DIR", help="Run all .py files in DIR in a single container")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing project files")
    parser.add_argument(
        "--supress",
        "--suppress",
        dest="suppress_stdout",
        action="store_true",
        help="Suppress printing results to stdout",
    )
    args = parser.parse_args()

    if args.batch:
        # Batch mode
        submissions_dir = Path(args.batch)
        if not submissions_dir.exists():
            submissions_dir = PROJECT_ROOT / args.batch
        if not submissions_dir.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {args.batch}")

        output_dir = Path(args.output)
        run_batch(submissions_dir, output_dir, args.mode, args.task, args.suppress_stdout)
    else:
        # Single submission mode
        if not args.submission:
            parser.error("submission is required (or use -b for batch mode)")

        submission = Path(args.submission)
        if not submission.exists():
            submission = PROJECT_ROOT / args.submission
        if not submission.exists():
            raise FileNotFoundError(f"Submission not found: {args.submission}")

        output = Path(args.output)
        run_single(submission, output, args.mode, args.task, args.no_sync, args.suppress_stdout)


if __name__ == "__main__":
    main()
