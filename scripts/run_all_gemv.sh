#!/bin/bash
# Run all gemv tests from gemv_subs in a single container
# Usage: ./run_all_gemv.sh [mode] [-o output_dir]
# Example: ./run_all_gemv.sh benchmark
# Example: ./run_all_gemv.sh benchmark -o /path/to/output/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

# Parse arguments
MODE="benchmark"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            MODE="$1"
            shift
            ;;
    esac
done

# Default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/out"
fi

echo "Running all gemv tests (mode: $MODE)"
echo "Output dir: $OUTPUT_DIR"

cd "$PROJECT_ROOT/modal"
python cli.py -b "$PROJECT_ROOT/gemv_subs" -o "$OUTPUT_DIR" -m "$MODE" -t gemv

echo "Done!"
