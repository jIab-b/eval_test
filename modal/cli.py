#!/usr/bin/env python3
"""CLI for running kernel eval on Modal."""
import argparse
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
}


def get_task_dir(task: str) -> Path:
    """Get the eval directory for a task."""
    return EVAL_ROOT / TASKS[task]


def main():
    parser = argparse.ArgumentParser(description="Run kernel eval on Modal")
    parser.add_argument("submission", help="Submission file path")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="gemv", choices=list(TASKS.keys()),
                        help="Task to run (default: gemv)")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing project files")
    args = parser.parse_args()

    # Get task directory
    task_dir = get_task_dir(args.task)
    workspace_name = TASKS[args.task]

    # Try CWD first, then project root
    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")
    output = Path(args.output)

    # Use benchmarks.txt for benchmark/leaderboard modes
    tests_file = task_dir / ("benchmarks.txt" if args.mode in ("benchmark", "leaderboard") else "tests.txt")
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    if not args.no_sync:
        sync_project()

    submission_code = submission.read_text()
    tests_content = tests_file.read_text()

    print(f"Running {args.mode} for task '{args.task}' on Modal...")
    with app.run():
        result = run_eval.remote(submission_code, tests_content, args.mode, workspace_name)

    output.write_text(result)
    print(f"Output saved to {output}")


if __name__ == "__main__":
    main()
