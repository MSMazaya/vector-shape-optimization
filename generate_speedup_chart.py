#!/usr/bin/env python3
"""
Generate bar chart of actual runtime speedups
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


def run_benchmark(test_file):
    """Run actual runtime benchmark and extract speedups"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "get_real_speedup.sh")

    # Run benchmark multiple times and average
    bar = progressbar.ProgressBar(widgets=widgets, max_value=100).start()
    speedups_avx = []
    speedups_unrolled = []
    for i in range(100):  # Run 100 times for average
        result = subprocess.run(
            [benchmark_script, test_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout + result.stderr

        # Extract speedups from two separate benchmark sections
        lines = output.split('\n')
        in_first_benchmark = True
        in_second_benchmark = False

        for line in lines:
            # Detect start of second benchmark
            if "Running benchmark: Scalar vs AVX (with unrolled remainder)" in line:
                in_first_benchmark = False
                in_second_benchmark = True
                continue

            # Look for speedup in first benchmark (AVX)
            if in_first_benchmark and 'Speedup:' in line:
                match = re.search(r'(\d+\.\d+)x', line)
                if match:
                    speedups_avx.append(float(match.group(1)))

            # Look for speedup in second benchmark (unrolled)
            if in_second_benchmark and 'Speedup:' in line:
                match = re.search(r'(\d+\.\d+)x', line)
                if match:
                    speedups_unrolled.append(float(match.group(1)))
        bar.update(i + 1)

    bar.finish()

    avg_speedup_avx = sum(speedups_avx) / \
        len(speedups_avx) if speedups_avx else None
    avg_speedup_unrolled = sum(
        speedups_unrolled) / len(speedups_unrolled) if speedups_unrolled else None

    return avg_speedup_avx, avg_speedup_unrolled


def get_matrix_size(test_file):
    """Extract matrix size from MLIR file, returns (M, N) tuple"""
    try:
        with open(test_file, 'r') as f:
            content = f.read()
            # Look for result memref dimensions: memref<MxNxf32> or memref<MxNxf64>
            # The result memref is the output (C), which gives us M x N
            # Try to find it in the outs clause first, then fall back to any memref
            # Pattern: outs(%C : memref<MxNxf32>) or func.func @name(..., %C: memref<MxNxf32>)

            # First try to find in outs clause
            match = re.search(
                r'outs\([^:]+:\s*memref<(\d+)x(\d+)x(f32|f64)>', content)
            if match:
                m = int(match.group(1))
                n = int(match.group(2))
                return (m, n)

            # Fall back to finding the last memref (usually the output)
            # Find all memref patterns and take the last one
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
            for suffix in ['_lowered.mlir', '_vectorized.mlir', 'benchmark_template.mlir', 'test_with_timing.mlir']
        )]

    if not test_files:
        print("No test files found")
        return

    print("Running runtime benchmarks (this may take a while)...")
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
        speedup_avx, speedup_unrolled = run_benchmark(test_file)

        if speedup_avx and speedup_unrolled:
            results.append({
                'name': test_name,
                'size': size,
                'speedup_avx': speedup_avx,
                'speedup_unrolled': speedup_unrolled
            })
            print(
                f"  Speedup (AVX): {speedup_avx:.2f}x, Speedup (Unrolled): {speedup_unrolled:.2f}x")
        else:
            print(f"  Failed")
        print()

    if not results:
        print("No successful benchmarks")
        return

    # Prepare data for plotting
    names = [r['name'] for r in results]
    sizes = [r['size'] for r in results]  # List of (M, N) tuples
    speedups_avx = [r['speedup_avx'] for r in results]
    speedups_unrolled = [r['speedup_unrolled'] for r in results]

    # Create bar chart with grouped bars
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create grouped bars
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, speedups_avx, width, label='AVX (Vectorized)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, speedups_unrolled, width, label='AVX (Unrolled)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add baseline
    ax.axhline(y=1.0, color='red', linestyle='--',
               linewidth=2, label='Baseline (1.0x)', zorder=0)

    # Customize
    ax.set_xlabel('Test Case (Matrix Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Scalar Time / Vectorized Time)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Vectorization Speedup: AVX vs AVX Unrolled',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n{m}x{n_val}" for n, (m, n_val) in zip(names, sizes)], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add average lines
    avg_speedup_avx = np.mean(speedups_avx)
    avg_speedup_unrolled = np.mean(speedups_unrolled)
    ax.axhline(y=avg_speedup_avx, color='blue', linestyle=':', linewidth=2,
               label=f'AVX Average: {avg_speedup_avx:.2f}x', zorder=0, alpha=0.7)
    ax.axhline(y=avg_speedup_unrolled, color='green', linestyle=':', linewidth=2,
               label=f'Unrolled Average: {avg_speedup_unrolled:.2f}x', zorder=0, alpha=0.7)
    ax.legend(fontsize=11)

    plt.tight_layout()

    # Save
    output_png = os.path.join(script_dir, "runtime_speedup_chart.png")
    output_pdf = os.path.join(script_dir, "runtime_speedup_chart.pdf")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Test Case':<20} {'Size':<10} {'AVX Speedup':<15} {'Unrolled Speedup':<18}")
    print("-" * 80)
    for r in results:
        m, n = r['size']
        print(
            f"{r['name']:<20} {m}x{n:<6} {r['speedup_avx']:.2f}x{'':<10} {r['speedup_unrolled']:.2f}x")
    print("-" * 80)
    print(f"{'Average Speedup':<20} {'':<10} {avg_speedup_avx:.2f}x{'':<10} {avg_speedup_unrolled:.2f}x")
    print()
    print(f"Chart saved to: {output_png}")
    print(f"PDF saved to: {output_pdf}")

    plt.show()


if __name__ == "__main__":
    main()
