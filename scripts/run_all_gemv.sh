#!/bin/bash
# Async run all gemv tests from gemv_subs
# Usage: ./run_all_gemv.sh [mode]
# Example: ./run_all_gemv.sh benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

MODE="${1:-benchmark}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting all gemv tests async (mode: $MODE)"
echo "Timestamp: $TIMESTAMP"

PIDS=()
SUBMISSIONS=()

for submission in "$PROJECT_ROOT/gemv_subs"/*.py; do
    if [ -f "$submission" ]; then
        BASENAME=$(basename "$submission" .py)
        OUTPUT_FILE="$PROJECT_ROOT/out/gemv_${BASENAME}_${MODE}_${TIMESTAMP}.txt"

        echo "Launching: $BASENAME -> $OUTPUT_FILE"

        (cd "$PROJECT_ROOT/modal" && python cli.py "$submission" -o "$OUTPUT_FILE" -m "$MODE" -t gemv) &
        PIDS+=($!)
        SUBMISSIONS+=("$BASENAME")
    fi
done

echo ""
echo "Launched ${#PIDS[@]} tests. Waiting for completion..."
echo ""

# Wait for all and report results
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${SUBMISSIONS[$i]}"
    else
        echo "[FAIL] ${SUBMISSIONS[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "All gemv tests completed. Failed: $FAILED/${#PIDS[@]}"
echo "Results in: $PROJECT_ROOT/out/gemv_*_${TIMESTAMP}.txt"
