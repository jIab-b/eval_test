#!/usr/bin/env python3
"""
Test IR reconstruction correctness.

1. Parse kernel files using ir.py
2. Reconstruct them to temp location
3. Run both original and reconstructed via Modal eval
4. Compare test results
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test IR reconstruction")
    parser.add_argument("--gemv", required=True, help="Path to gemv submission file")
    parser.add_argument("--gemm", required=True, help="Path to gemm submission file")
    parser.add_argument("--temp-dir", required=True, help="Temp directory for reconstructed files")
    parser.add_argument("--project-root", required=True, help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Add ir directory to path
    sys.path.insert(0, str(project_root / "ir"))
    from ir import parse_kernel_file, reconstruct_from_ir

    results = {"gemv": {}, "gemm": {}}
    all_passed = True

    for task_type, src_path in [("gemv", args.gemv), ("gemm", args.gemm)]:
        src_path = Path(src_path)
        if not src_path.exists():
            print(f"[SKIP] {task_type}: {src_path} not found")
            continue

        print(f"\n--- Testing {task_type}: {src_path.name} ---")

        # Step 1: Parse with IR
        print(f"[1] Parsing {src_path.name}...")
        try:
            kernel_ir = parse_kernel_file(src_path)
            print(f"    load_inline name: {kernel_ir.load_inline.name}")
            print(f"    cuda_sources: {len(kernel_ir.load_inline.cuda_sources)}")
            print(f"    inline asm blocks: {sum(len(c.asm) for c in kernel_ir.cuda)}")
        except Exception as e:
            print(f"    [FAIL] Parse error: {e}")
            all_passed = False
            continue

        # Step 2: Reconstruct
        reconstructed_path = temp_dir / src_path.name
        print(f"[2] Reconstructing to {reconstructed_path}...")
        try:
            reconstruct_from_ir(kernel_ir, reconstructed_path)
            print(f"    Written: {reconstructed_path.stat().st_size} bytes")
        except Exception as e:
            print(f"    [FAIL] Reconstruction error: {e}")
            all_passed = False
            continue

        # Step 3: Verify content matches (reconstruct_from_ir copies original)
        original_content = src_path.read_text()
        reconstructed_content = reconstructed_path.read_text()

        if original_content == reconstructed_content:
            print(f"[3] Content match: EXACT")
        else:
            print(f"[3] Content match: DIFFER (orig={len(original_content)}, recon={len(reconstructed_content)})")
            # For now, reconstruct_from_ir just copies, so this should always match

        # Step 4: Run both through Modal eval
        print(f"[4] Running original via Modal...")
        orig_output = temp_dir / f"orig_{task_type}.txt"
        recon_output = temp_dir / f"recon_{task_type}.txt"

        cli_path = project_root / "modal" / "cli.py"

        def run_eval(submission_path: Path, output_path: Path) -> bool:
            cmd = [
                sys.executable, str(cli_path),
                str(submission_path),
                "-o", str(output_path),
                "-m", "test",
                "-t", task_type
            ]
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(project_root / "modal"),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                print(f"    [TIMEOUT]")
                return False
            except Exception as e:
                print(f"    [ERROR] {e}")
                return False

        orig_ok = run_eval(src_path, orig_output)
        print(f"    Original: {'PASS' if orig_ok else 'FAIL'}")

        print(f"[5] Running reconstructed via Modal...")
        recon_ok = run_eval(reconstructed_path, recon_output)
        print(f"    Reconstructed: {'PASS' if recon_ok else 'FAIL'}")

        # Step 6: Compare outputs
        if orig_output.exists() and recon_output.exists():
            orig_lines = orig_output.read_text().strip().split('\n')
            recon_lines = recon_output.read_text().strip().split('\n')

            # Check if both have "check: pass"
            orig_check = any("check: pass" in line for line in orig_lines)
            recon_check = any("check: pass" in line for line in recon_lines)

            print(f"[6] Results comparison:")
            print(f"    Original check: {'pass' if orig_check else 'fail'}")
            print(f"    Reconstructed check: {'pass' if recon_check else 'fail'}")

            if orig_check and recon_check:
                print(f"    [OK] Both passed correctness check")
                results[task_type] = {"status": "pass", "orig": orig_check, "recon": recon_check}
            else:
                print(f"    [FAIL] Mismatch or failure")
                all_passed = False
                results[task_type] = {"status": "fail", "orig": orig_check, "recon": recon_check}
        else:
            print(f"[6] Results comparison: SKIPPED (missing output files)")
            all_passed = False

    # Summary
    print("\n=== Summary ===")
    for task_type, result in results.items():
        if result:
            print(f"{task_type}: {result.get('status', 'skipped')}")
        else:
            print(f"{task_type}: skipped")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
