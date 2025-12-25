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


def run_single(submission: Path, output: Path, mode: str, task: str, no_sync: bool):
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

    print(f"Running {mode} for task '{task}' on Modal...")
    with app.run():
        result = run_eval.remote(submission_code, tests_content, mode, workspace_name)

    output.write_text(result)
    print(f"Output saved to {output}")


def run_batch(submissions_dir: Path, output_dir: Path, mode: str, task: str):
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
        print(f"No submissions found in {submissions_dir}")
        return

    # Sync once before running all
    sync_project()

    tests_content = tests_file.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(submissions)} submissions in batch mode (task: {task}, mode: {mode})")

    # Run all submissions in a single app.run() context
    with app.run():
        for submission in submissions:
            basename = submission.stem
            output_file = output_dir / f"{basename}.txt"

            print(f"  [{submissions.index(submission)+1}/{len(submissions)}] {basename}...", end=" ", flush=True)

            try:
                submission_code = submission.read_text()
                result = run_eval.remote(submission_code, tests_content, mode, workspace_name)
                output_file.write_text(result)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                output_file.write_text(f"Error: {e}")

    print(f"Batch complete. Results in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run kernel eval on Modal")
    parser.add_argument("submission", nargs="?", help="Submission file path (or use -b for batch)")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file (or directory for batch)")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("-t", "--task", default="gemv", choices=list(TASKS.keys()),
                        help="Task to run (default: gemv)")
    parser.add_argument("-b", "--batch", metavar="DIR", help="Run all .py files in DIR in a single container")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing project files")
    args = parser.parse_args()

    if args.batch:
        # Batch mode
        submissions_dir = Path(args.batch)
        if not submissions_dir.exists():
            submissions_dir = PROJECT_ROOT / args.batch
        if not submissions_dir.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {args.batch}")

        output_dir = Path(args.output)
        run_batch(submissions_dir, output_dir, args.mode, args.task)
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
        run_single(submission, output, args.mode, args.task, args.no_sync)


if __name__ == "__main__":
    main()
