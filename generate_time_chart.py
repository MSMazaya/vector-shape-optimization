#!/usr/bin/env python3
"""
Generate bar chart of absolute execution times for scalar and vectorized versions
"""

import sys
import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import progressbar

widgets = [
    'Running: ', progressbar.Percentage(),
    ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
    ' ', progressbar.ETA(),
    ' ', progressbar.FileTransferSpeed(),
]


def run_benchmark(test_file, iterations=1000, num_runs=10):
    """Run actual runtime benchmark multiple times and average the results"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "get_real_speedup.sh")

    scalar_times = []
    avx_times = []
    avx_unrolled_times = []

    # Run benchmark multiple times and collect results
    bar = progressbar.ProgressBar(widgets=widgets, max_value=num_runs).start()
    for run in range(num_runs):
        result = subprocess.run(
            [benchmark_script, test_file, str(iterations)],
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + result.stderr

        # Extract scalar, AVX, and AVX unrolled times
        # The output now has two separate benchmark sections
        scalar_time = None
        avx_time = None
        avx_unrolled_time = None

        # Split output into two sections (before and after the second benchmark header)
        lines = output.split('\n')
        in_first_benchmark = True
        in_second_benchmark = False

        for line in lines:
            # Detect start of second benchmark
            if "Running benchmark: Scalar vs AVX (with unrolled remainder)" in line:
                in_first_benchmark = False
                in_second_benchmark = True
                continue

            # Look for: "Scalar time: X.XXXXXX seconds" (handles variable whitespace)
            # Use the first scalar time we find (they should be similar)
            match = re.search(r'Scalar time:\s+(\d+\.\d+)\s+seconds', line)
            if match and scalar_time is None:
                scalar_time = float(match.group(1))

            # Look for: "AVX time:    X.XXXXXX seconds" (handles variable whitespace)
            match = re.search(r'AVX time:\s+(\d+\.\d+)\s+seconds', line)
            if match:
                if in_first_benchmark:
                    avx_time = float(match.group(1))
                elif in_second_benchmark:
                    avx_unrolled_time = float(match.group(1))

        if scalar_time is not None:
            scalar_times.append(scalar_time)
        if avx_time is not None:
            avx_times.append(avx_time)
        if avx_unrolled_time is not None:
            avx_unrolled_times.append(avx_unrolled_time)

        bar.update(run + 1)

    bar.finish()

    # Calculate averages
    avg_scalar = sum(scalar_times) / \
        len(scalar_times) if scalar_times else None
    avg_avx = sum(avx_times) / len(avx_times) if avx_times else None
    avg_avx_unrolled = sum(avx_unrolled_times) / \
        len(avx_unrolled_times) if avx_unrolled_times else None

    return avg_scalar, avg_avx, avg_avx_unrolled


def get_matrix_size(test_file):
    """Extract matrix size from MLIR file, returns (M, N) tuple"""
    try:
        with open(test_file, 'r') as f:
            content = f.read()
            # Look for result memref dimensions: memref<MxNxf32> or memref<MxNxf64>
            # The result memref is the output (C), which gives us M x N
            # Try to find it in the outs clause first, then fall back to any memref
            match = re.search(
                r'outs\([^:]+:\s*memref<(\d+)x(\d+)x(f32|f64)>', content)
            if match:
                m = int(match.group(1))
                n = int(match.group(2))
                return (m, n)

            # Fall back to finding the last memref (usually the output)
            matches = list(re.finditer(
                r'memref<(\d+)x(\d+)x(f32|f64)>', content))
            if matches:
                last_match = matches[-1]
                m = int(last_match.group(1))
                n = int(last_match.group(2))
                return (m, n)
    except:
        pass
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "tests")

    # Find test files
    test_files = []
    if len(sys.argv) > 1:
        test_files = [f for f in sys.argv[1:] if os.path.isfile(f)]
    else:
        import glob
        all_files = sorted(glob.glob(os.path.join(tests_dir, "test*.mlir")))
        # Filter out intermediate files
        test_files = [f for f in all_files if not any(
            suffix in os.path.basename(f)
            for suffix in ['_lowered.mlir', '_vectorized.mlir', 'benchmark_template.mlir',
                           'test_with_timing.mlir', '_f64_vectorized.mlir']
        )]

    if not test_files:
        print("No test files found")
        return

    print("Running runtime benchmarks to measure absolute execution times...")
    print()

    results = []
    for test_file in test_files:
        test_name = os.path.basename(test_file).replace('.mlir', '')
        size = get_matrix_size(test_file)

        if size is None:
            print(f"Skipping {test_name} (could not extract matrix size)")
            continue

        m, n = size
        print(f"Benchmarking {test_name} ({m}x{n})...")
        print(f"  Running {10} iterations to average results...")
        scalar_time, avx_time, avx_unrolled_time = run_benchmark(
            test_file, iterations=1000, num_runs=10)

        if scalar_time is not None and avx_time is not None and avx_unrolled_time is not None:
            results.append({
                'name': test_name,
                'size': size,
                'scalar_time': scalar_time,
                'avx_time': avx_time,
                'avx_unrolled_time': avx_unrolled_time
            })
            print(
                f"  Average Scalar: {scalar_time:.6f}s, Average AVX: {avx_time:.6f}s, Average AVX Unrolled: {avx_unrolled_time:.6f}s")
        else:
            print(f"  Failed to extract times")
        print()

    if not results:
        print("No successful benchmarks")
        return

    # Prepare data for plotting
    names = [r['name'] for r in results]
    sizes = [r['size'] for r in results]
    scalar_times = [r['scalar_time'] for r in results]
    avx_times = [r['avx_time'] for r in results]
    avx_unrolled_times = [r['avx_unrolled_time'] for r in results]

    # Create bar chart with grouped bars
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create grouped bars
    x = np.arange(len(names))
    width = 0.25

    bars1 = ax.bar(x - width, scalar_times, width, label='Scalar (Non-Vectorized)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, avx_times, width, label='AVX (Vectorized)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, avx_unrolled_times, width, label='AVX (Unrolled)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize
    ax.set_xlabel('Test Case (Matrix Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Execution Time Comparison: Scalar vs Vectorized',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{m}x{n_val}" for n, (m, n_val) in zip(names, sizes)],
                       fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    rotation=90)

    plt.tight_layout()

    # Save
    output_png = os.path.join(script_dir, "execution_time_chart.png")
    output_pdf = os.path.join(script_dir, "execution_time_chart.pdf")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Test Case':<20} {'Size':<10} {'Scalar (s)':<12} {'AVX (s)':<12} {'AVX Unrolled (s)':<18} {'Speedup':<10} {'Unrolled Speedup':<15}")
    print("-" * 100)
    for r in results:
        m, n = r['size']
        speedup = r['scalar_time'] / r['avx_time'] if r['avx_time'] > 0 else 0
        speedup_unrolled = r['scalar_time'] / \
            r['avx_unrolled_time'] if r['avx_unrolled_time'] > 0 else 0
        print(f"{r['name']:<20} {m}x{n:<6} {r['scalar_time']:<12.6f} {r['avx_time']:<12.6f} {r['avx_unrolled_time']:<18.6f} {speedup:.2f}x{'':<5} {speedup_unrolled:.2f}x")
    print("-" * 100)
    avg_scalar = sum(scalar_times) / len(scalar_times)
    avg_avx = sum(avx_times) / len(avx_times)
    avg_avx_unrolled = sum(avx_unrolled_times) / len(avx_unrolled_times)
    avg_speedup = avg_scalar / avg_avx if avg_avx > 0 else 0
    avg_speedup_unrolled = avg_scalar / \
        avg_avx_unrolled if avg_avx_unrolled > 0 else 0
    print(f"{'Average':<20} {'':<10} {avg_scalar:<12.6f} {avg_avx:<12.6f} {avg_avx_unrolled:<18.6f} {avg_speedup:.2f}x{'':<5} {avg_speedup_unrolled:.2f}x")
    print()
    print(f"Chart saved to: {output_png}")
    print(f"PDF saved to: {output_pdf}")

    plt.show()


if __name__ == "__main__":
    main()
