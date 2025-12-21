#!/usr/bin/env python3
"""CLI for running kernel eval on Modal."""
import argparse
from pathlib import Path

import modal
from app import app, sync_project, run_eval

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_FILE = PROJECT_ROOT / "eval" / "tests.txt"
BENCHMARKS_FILE = PROJECT_ROOT / "eval" / "benchmarks.txt"


def main():
    parser = argparse.ArgumentParser(description="Run kernel eval on Modal")
    parser.add_argument("submission", help="Submission file path")
    parser.add_argument("-o", "--output", default="out.txt", help="Output file")
    parser.add_argument("-m", "--mode", default="benchmark", choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing project files")
    args = parser.parse_args()

    # Try CWD first, then project root
    submission = Path(args.submission)
    if not submission.exists():
        submission = PROJECT_ROOT / args.submission
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {args.submission}")
    output = Path(args.output)

    # Use benchmarks.txt for benchmark/leaderboard modes
    tests_file = BENCHMARKS_FILE if args.mode in ("benchmark", "leaderboard") else TESTS_FILE
    if not tests_file.exists():
        raise FileNotFoundError(f"Tests not found: {tests_file}")

    if not args.no_sync:
        sync_project()

    submission_code = submission.read_text()
    tests_content = tests_file.read_text()

    print(f"Running {args.mode} on Modal...")
    with app.run():
        result = run_eval.remote(submission_code, tests_content, args.mode)

    output.write_text(result)
    print(f"Output saved to {output}")


if __name__ == "__main__":
    main()
