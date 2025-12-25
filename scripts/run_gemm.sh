#!/bin/bash
# Run a single gemm test from gemm_subs
# Usage: ./run_gemm.sh <submission_name> [mode]
# Example: ./run_gemm.sh 1mm test
# Example: ./run_gemm.sh 2mm benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

if [ -z "$1" ]; then
    echo "Usage: $0 <submission_name> [mode]"
    echo "Available submissions:"
    ls "$PROJECT_ROOT/gemm_subs/" | sed 's/\.py$//'
    exit 1
fi

SUBMISSION="$1"
MODE="${2:-benchmark}"

# Add .py if not present
if [[ ! "$SUBMISSION" == *.py ]]; then
    SUBMISSION="${SUBMISSION}.py"
fi

SUBMISSION_PATH="$PROJECT_ROOT/gemm_subs/$SUBMISSION"

if [ ! -f "$SUBMISSION_PATH" ]; then
    echo "Error: Submission not found: $SUBMISSION_PATH"
    exit 1
fi

# Generate output filename with timestamp
BASENAME="${SUBMISSION%.py}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$PROJECT_ROOT/out/gemm_${BASENAME}_${MODE}_${TIMESTAMP}.txt"

echo "Running gemm test: $SUBMISSION (mode: $MODE)"
echo "Output: $OUTPUT_FILE"

cd "$PROJECT_ROOT/modal"
python cli.py "$SUBMISSION_PATH" -o "$OUTPUT_FILE" -m "$MODE" -t gemm

echo "Done! Results saved to: $OUTPUT_FILE"
