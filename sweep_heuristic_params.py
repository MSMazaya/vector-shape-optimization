#!/usr/bin/env python3
"""
Sweep (a, b) parameters for the simple heuristic cost model:

    cost = M * a + N * b

where:
  - M: number of masking "events" (for remainder, M = 1 when masking the remainder)
  - N: number of scalar elements handled by unrolling (for remainder, N = remainder length)
  - a: cost for masking (HEURISTIC_MASK_COST)
  - b: cost for unrolling (HEURISTIC_UNROLL_COST)

This script:
  - Picks a subset of existing matmul tests (same format used by benchmark_heuristic.py)
  - For each (a, b) pair:
      * runs scalar / unrolled / masked / heuristic strategies locally
      * compares the heuristic's speedup to the best blind strategy for each test
  - Reports aggregate metrics per (a, b):
      * median % gap between heuristic and best
      * fraction of tests where heuristic is within a given tolerance of best

Usage examples (from repo root):

  # Simple AVX sweep on first 8 tests with small grids
  python3 sweep_heuristic_params.py --avx --max-tests 8

  # Custom grids, AVX2, 4 runs per configuration (slower but smoother)
  python3 sweep_heuristic_params.py --avx2 --max-tests 8 \\
      --a-values 0.25 0.5 1.0 \\
      --b-values 0.1 0.25 0.5 \\
      --num-runs 4
"""

import argparse
import glob
import json
import os
from statistics import median
from typing import Dict, List, Tuple

import numpy as np  # Used elsewhere in this repo; assumed available

import benchmark_heuristic as bh
import benchmark_performance as bp


def discover_tests(tests_dir: str, max_tests: int, max_dim: int) -> List[str]:
    """
    Return up to max_tests MLIR test files, matching benchmark_heuristic's filtering,
    and restricting to small sizes (all dims <= max_dim).
    """
    all_files = sorted(glob.glob(os.path.join(tests_dir, "test*.mlir")))
    filtered: List[str] = []
    for f in all_files:
        if any(
            suffix in os.path.basename(f)
            for suffix in [
                "_lowered.mlir",
                "_vectorized.mlir",
                "benchmark_template.mlir",
                "test_with_timing.mlir",
            ]
        ):
            continue

        dims = bh.get_matrix_size(f)
        if not dims:
            continue
        m, n = dims
        # Matmul is MxK * KxN; in current tests K == M. We just bound max(m, n).
        if max(m, n) > max_dim:
            continue
        filtered.append(f)

    if max_tests > 0:
        filtered = filtered[:max_tests]
    return filtered


def run_single_config(
    a: float,
    b: float,
    test_files: List[str],
    vector_isa: str,
    machine_type: str,
    machine_config: Dict,
    llvm_path_override: str,
    num_runs: int,
    tolerance_pct: float = 3.0,
) -> Dict:
    """
    Run benchmark_heuristic's local benchmark for a single (a, b) pair.

    For each test:
      - Compute best speedup among blind strategies
      - Compare heuristic speedup to that best
    """
    os.environ["HEURISTIC_MASK_COST"] = str(a)
    os.environ["HEURISTIC_UNROLL_COST"] = str(b)

    per_test_results = []

    blind_strategies = ["scalar_remainder", "unrolled_remainder", "masked_remainder"]

    for test_file in test_files:
        test_name = os.path.basename(test_file).replace(".mlir", "")

        # Choose local vs remote execution strategy.
        if machine_type == "remote":
            # For any remote machine (ARM SVE/SME or x86 AVX/AVX2/AVX-512),
            # use the cross-compile + ship-binaries pipeline from benchmark_performance.py.
            # This matches the behavior of benchmark_performance.py, including its logs.
            results, heuristic_strategy, _ = bp.cross_compile_and_benchmark_remote(
                test_file, vector_isa, machine_config, num_runs=num_runs
            )
        else:
            # Local execution using benchmark_heuristic's helper.
            results, heuristic_strategy, _ = bh.run_benchmark_local(
                test_file,
                vector_isa,
                num_runs=num_runs,
                llvm_path_override=llvm_path_override,
            )

        # Extract blind strategy best
        best_blind = 0.0
        best_blind_name = None
        for s in blind_strategies:
            val = results.get(s)
            if val is not None and val > best_blind:
                best_blind = val
                best_blind_name = s

        heur_speedup = results.get("heuristic")

        if best_blind <= 0 or heur_speedup is None:
            gap_pct = None
            within_tol = False
        else:
            # Ensure we always store plain Python floats/bools (JSON friendly),
            # not numpy scalar types.
            gap_pct = float((heur_speedup - best_blind) / best_blind * 100.0)
            within_tol = bool(gap_pct >= -tolerance_pct)

        per_test_results.append(
            {
                "test": test_name,
                "best_blind": best_blind,
                "best_blind_name": best_blind_name,
                "heuristic": heur_speedup,
                "heuristic_strategy": heuristic_strategy,
                "gap_pct": gap_pct,
                "within_tol": within_tol,
            }
        )

    # Aggregate metrics
    gaps = [r["gap_pct"] for r in per_test_results if r["gap_pct"] is not None]
    if gaps:
        median_gap = median(gaps)
        mean_gap = float(np.mean(gaps))
    else:
        median_gap = None
        mean_gap = None

    within = [r["within_tol"] for r in per_test_results if r["gap_pct"] is not None]
    frac_within = float(sum(within) / len(within)) if within else 0.0

    summary = {
        "a": float(a),
        "b": float(b),
        "median_gap_pct": float(median_gap) if median_gap is not None else None,
        "mean_gap_pct": float(mean_gap) if mean_gap is not None else None,
        "frac_within_tol": float(frac_within),
        "tolerance_pct": float(tolerance_pct),
        "per_test": per_test_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Sweep (a, b) parameters for heuristic remainder cost model."
    )
    isa_group = parser.add_mutually_exclusive_group()
    isa_group.add_argument("--avx", action="store_true", help="Use AVX (default).")
    isa_group.add_argument("--avx2", action="store_true", help="Use AVX2.")
    isa_group.add_argument("--avx512", action="store_true", help="Use AVX-512.")
    isa_group.add_argument("--sve", action="store_true", help="Use ARM SVE.")
    isa_group.add_argument("--sme", action="store_true", help="Use ARM SME.")

    parser.add_argument(
        "--max-tests",
        type=int,
        default=8,
        help="Maximum number of tests to include in the sweep (default: 8).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=32,
        help="Maximum matrix dimension (M or N) for tests used in the sweep (default: 32).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of timing runs per configuration (default: 3).",
    )
    parser.add_argument(
        "--a-values",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="List of 'a' (mask cost) values to sweep.",
    )
    parser.add_argument(
        "--b-values",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="List of 'b' (unroll cost) values to sweep.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine name from machine_config.json (default: use benchmark_heuristic default).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write full sweep results as JSON.",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=3.0,
        help="Tolerance (%%) within which heuristic is considered 'good enough' vs best (default: 3%%).",
    )

    args = parser.parse_args()

    # Determine ISA string
    if args.avx512:
        vector_isa = "avx512"
    elif args.avx2:
        vector_isa = "avx2"
    elif args.sve:
        vector_isa = "sve"
    elif args.sme:
        vector_isa = "sme"
    else:
        vector_isa = "avx"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "tests")

    config = bh.load_machine_config()
    machines = config["machines"]
    default_machine = config.get("default_machine", "local")
    machine_name = args.machine or default_machine
    if machine_name not in machines:
        raise SystemExit(f"Unknown machine '{machine_name}' in machine_config.json")
    machine_config = machines[machine_name]
    machine_type = machine_config.get("type", "local")
    llvm_path = machine_config.get("llvm_project_path", None)

    test_files = discover_tests(tests_dir, args.max_tests, args.max_dim)
    if not test_files:
        raise SystemExit(f"No tests found in {tests_dir}")

    print(f"Machine: {machine_name} (type={machine_type})")
    print(f"Vector ISA: {vector_isa.upper()}")
    print(f"Tests: {len(test_files)}, a values: {args.a_values}, b values: {args.b_values}, runs: {args.num_runs}")
    print()

    all_summaries: List[Dict] = []

    for a in args.a_values:
        for b in args.b_values:
            summary = run_single_config(
                a=a,
                b=b,
                test_files=test_files,
                vector_isa=vector_isa,
                machine_type=machine_type,
                machine_config=machine_config,
                llvm_path_override=llvm_path,
                num_runs=args.num_runs,
                tolerance_pct=args.tolerance_pct,
            )
            all_summaries.append(summary)

    # Print high-level ranking by median gap
    print("\n===== SWEEP SUMMARY (sorted by median gap) =====")
    sortable = [
        s for s in all_summaries if s["median_gap_pct"] is not None
    ]
    sortable.sort(key=lambda s: s["median_gap_pct"], reverse=True)

    for s in sortable:
        print(
            f"a={s['a']:5.2f}, b={s['b']:5.2f}: "
            f"median_gap={s['median_gap_pct']:6.2f}%, "
            f"frac_within_{s['tolerance_pct']:.1f}%={s['frac_within_tol']:.2f}"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nFull sweep results written to: {args.output_json}")


if __name__ == "__main__":
    main()


