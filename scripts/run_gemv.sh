#!/bin/bash
# Run a single gemv test from gemv_subs
# Usage: ./run_gemv.sh <submission_name> [mode] [-o output_file]
# Example: ./run_gemv.sh 1 test
# Example: ./run_gemv.sh 2 benchmark
# Example: ./run_gemv.sh 1 test -o /path/to/output.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

# Parse arguments
SUBMISSION=""
MODE="benchmark"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            if [ -z "$SUBMISSION" ]; then
                SUBMISSION="$1"
            else
                MODE="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$SUBMISSION" ]; then
    echo "Usage: $0 <submission_name> [mode] [-o output_file]"
    echo "Available submissions:"
    ls "$PROJECT_ROOT/gemv_subs/" | sed 's/\.py$//'
    exit 1
fi

# Add .py if not present
if [[ ! "$SUBMISSION" == *.py ]]; then
    SUBMISSION="${SUBMISSION}.py"
fi

SUBMISSION_PATH="$PROJECT_ROOT/gemv_subs/$SUBMISSION"

if [ ! -f "$SUBMISSION_PATH" ]; then
    echo "Error: Submission not found: $SUBMISSION_PATH"
    exit 1
fi

# Generate output filename with timestamp if not specified
BASENAME="${SUBMISSION%.py}"
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$PROJECT_ROOT/out/gemv_${BASENAME}_${MODE}_${TIMESTAMP}.txt"
fi

echo "Running gemv test: $SUBMISSION (mode: $MODE)"
echo "Output: $OUTPUT_FILE"

cd "$PROJECT_ROOT/modal"
python cli.py "$SUBMISSION_PATH" -o "$OUTPUT_FILE" -m "$MODE" -t gemv

echo "Done! Results saved to: $OUTPUT_FILE"
