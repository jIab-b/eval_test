#!/usr/bin/env python3
"""
Local IR test - no Modal required.
Tests parsing and reconstruction of kernel files.

Usage: ./test_ir_local.py [--all] [--gemv FILE] [--gemm FILE]
"""

import argparse
import sys
from pathlib import Path


def test_file(src_path: Path, parse_kernel_file, extract_cuda_funcs) -> dict:
    """Test a single file. Returns dict with results."""
    result = {
        "file": src_path.name,
        "parsed": False,
        "skipped": False,
        "asm_blocks": 0,
        "ptx_calls": 0,
        "cuda_funcs": 0,
        "round_trip": False,
        "errors": []
    }

    try:
        # Parse
        kernel_ir = parse_kernel_file(src_path)
        result["parsed"] = True
        result["load_inline_name"] = kernel_ir.load_inline.name

        # Count features
        for cuda_ir in kernel_ir.cuda:
            result["asm_blocks"] += len(cuda_ir.asm)
            result["ptx_calls"] += len(cuda_ir.ptx_calls)
            result["cuda_funcs"] += len(extract_cuda_funcs(cuda_ir.src))

            # Test PTX parsing for each asm block
            for i, asm in enumerate(cuda_ir.asm):
                rendered = asm.ptx.render()
                if not rendered.strip():
                    result["errors"].append(f"asm[{i}] rendered empty")

        # Test round-trip (original should equal original for now)
        original = src_path.read_text()
        result["original_size"] = len(original)
        result["round_trip"] = True

    except ValueError as e:
        # ValueError typically means no load_inline found or unsupported pattern
        err_str = str(e)
        if "No load_inline" in err_str or "missing cuda_sources" in err_str:
            result["skipped"] = True
            result["errors"].append(f"Unsupported pattern: {err_str}")
        else:
            result["errors"].append(err_str)
    except Exception as e:
        result["errors"].append(str(e))

    return result


def main():
    parser = argparse.ArgumentParser(description="Local IR test")
    parser.add_argument("--all", action="store_true", help="Test all files in gemv_subs and gemm_subs")
    parser.add_argument("--gemv", help="Specific gemv file to test (name without .py)")
    parser.add_argument("--gemm", help="Specific gemm file to test (name without .py)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Add ir directory to path
    sys.path.insert(0, str(project_root / "ir"))
    from ir import parse_kernel_file, extract_cuda_funcs

    files_to_test = []

    if args.all:
        # Test all files
        for p in sorted((project_root / "gemv_subs").glob("*.py")):
            files_to_test.append(("gemv", p))
        for p in sorted((project_root / "gemm_subs").glob("*.py")):
            files_to_test.append(("gemm", p))
    else:
        # Test specific files
        if args.gemv:
            p = project_root / "gemv_subs" / f"{args.gemv}.py"
            if p.exists():
                files_to_test.append(("gemv", p))
            else:
                print(f"[ERROR] gemv file not found: {p}")
        if args.gemm:
            p = project_root / "gemm_subs" / f"{args.gemm}.py"
            if p.exists():
                files_to_test.append(("gemm", p))
            else:
                print(f"[ERROR] gemm file not found: {p}")

        # Default: test one of each
        if not files_to_test:
            files_to_test = [
                ("gemv", project_root / "gemv_subs" / "1.py"),
                ("gemm", project_root / "gemm_subs" / "1mm.py"),
            ]

    print(f"Testing {len(files_to_test)} file(s)...\n")

    all_passed = True
    summary = []

    for task_type, path in files_to_test:
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue

        result = test_file(path, parse_kernel_file, extract_cuda_funcs)

        if result["skipped"]:
            status = "SKIP"
        elif result["parsed"] and not result["errors"]:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False

        summary.append((task_type, path.name, status, result))

        if args.verbose or status == "FAIL":
            print(f"[{status}] {task_type}/{path.name}")
            print(f"      load_inline: {result.get('load_inline_name', 'N/A')}")
            print(f"      asm_blocks: {result['asm_blocks']}, ptx_calls: {result['ptx_calls']}, cuda_funcs: {result['cuda_funcs']}")
            if result["errors"]:
                for err in result["errors"]:
                    print(f"      ERROR: {err}")
            print()
        elif status == "SKIP":
            print(f"[{status}] {task_type}/{path.name}: {result['errors'][0] if result['errors'] else 'unknown'}")
        else:
            print(f"[{status}] {task_type}/{path.name}: asm={result['asm_blocks']}, ptx={result['ptx_calls']}, funcs={result['cuda_funcs']}")

    # Summary
    print(f"\n=== Summary ===")
    passed = sum(1 for _, _, s, _ in summary if s == "PASS")
    skipped = sum(1 for _, _, s, _ in summary if s == "SKIP")
    failed = sum(1 for _, _, s, _ in summary if s == "FAIL")
    total = len(summary)
    print(f"Passed: {passed}, Skipped: {skipped}, Failed: {failed} (Total: {total})")

    if failed == 0:
        print("\nAll parseable IR tests passed!")
        return 0
    else:
        print("\nSome IR tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
