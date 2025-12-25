#!/bin/bash
# Async run all gemm tests from gemm_subs
# Usage: ./run_all_gemm.sh [mode]
# Example: ./run_all_gemm.sh benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

MODE="${1:-benchmark}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting all gemm tests async (mode: $MODE)"
echo "Timestamp: $TIMESTAMP"

PIDS=()
SUBMISSIONS=()

for submission in "$PROJECT_ROOT/gemm_subs"/*.py; do
    if [ -f "$submission" ]; then
        BASENAME=$(basename "$submission" .py)
        OUTPUT_FILE="$PROJECT_ROOT/out/gemm_${BASENAME}_${MODE}_${TIMESTAMP}.txt"

        echo "Launching: $BASENAME -> $OUTPUT_FILE"

        (cd "$PROJECT_ROOT/modal" && python cli.py "$submission" -o "$OUTPUT_FILE" -m "$MODE" -t gemm) &
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
echo "All gemm tests completed. Failed: $FAILED/${#PIDS[@]}"
echo "Results in: $PROJECT_ROOT/out/gemm_*_${TIMESTAMP}.txt"
