#!/bin/bash
# Test IR reconstruction by:
# 1. Parsing a kernel file with ir.py
# 2. Reconstructing it to a temp location
# 3. Running both original and reconstructed with the eval scripts
# 4. Comparing results
#
# Usage: ./test_ir_reconstruction.sh [gemv_file] [gemm_file]
# Example: ./test_ir_reconstruction.sh 1 1mm

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source venv
source "$PROJECT_ROOT/venv/bin/activate"

GEMV_FILE="${1:-1}"
GEMM_FILE="${2:-1mm}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create temp dir for reconstructed files
TEMP_DIR="$PROJECT_ROOT/out/ir_test_$TIMESTAMP"
mkdir -p "$TEMP_DIR"

echo "=== IR Reconstruction Test ==="
echo "GEMV file: $GEMV_FILE"
echo "GEMM file: $GEMM_FILE"
echo "Temp dir: $TEMP_DIR"
echo ""

# Run Python script to do reconstruction and comparison
python3 "$SCRIPT_DIR/test_ir_reconstruction.py" \
    --gemv "$PROJECT_ROOT/gemv_subs/${GEMV_FILE}.py" \
    --gemm "$PROJECT_ROOT/gemm_subs/${GEMM_FILE}.py" \
    --temp-dir "$TEMP_DIR" \
    --project-root "$PROJECT_ROOT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== IR Reconstruction Test PASSED ==="
else
    echo ""
    echo "=== IR Reconstruction Test FAILED ==="
fi

exit $EXIT_CODE
