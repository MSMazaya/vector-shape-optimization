#!/usr/bin/env python3
"""
Generate training data with CORRECTED input space definition.

Loop structure:
  for (x_iter to X)
     for (y_iter to Y)
        for (k_iter to K)
            vectorized

Input Features (7 features):
1. instruction_type (1=AVX, 2=AVX2, 3=AVX512, etc.)
2. LS [1, VS]
3. LS == VS (binary)
4. K % LS (numeric remainder) - where K is vectorized dimension size
5. remainder_strategy (binary: 0=masking, 1=unrolling)
6. X * Y (numeric) - product of repetition dimensions
7. LS // K (integer division)
"""

import sys
import os
import subprocess
import re
import json
import csv
import shutil
import tempfile
import argparse
from pathlib import Path
import progressbar
import numpy as np  # For median calculation

# Import base functionality
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dl_training_data import (
    VectorISAConfig, VECTOR_ISAS, create_test_case,
    get_matrix_size_from_mlir, compile_and_benchmark_configuration
)

# Instruction type mapping
INSTRUCTION_TYPE_MAP = {
    'AVX': 1,
    'AVX2': 2,
    'AVX512': 3,
    'SVE': 4,
    'SVE2': 5,
    'SME': 6,
}


def get_vector_size(isa_name, element_type='f32'):
    """Get VS (vector size) for given ISA and element type"""
    if isa_name not in VECTOR_ISAS:
        return None
    return VECTOR_ISAS[isa_name].get_vector_len(element_type)


def generate_corrected_configurations(K, X, Y, isa_name, element_type='f32'):
    """
    Generate all valid configurations using corrected input space.
    
    Args:
        K: Size of vectorized dimension (inner loop)
        X: Size of outer repetition dimension
        Y: Size of middle repetition dimension
        isa_name: Instruction type name
        element_type: Element type ('f32' or 'f64')
    
    Returns:
        List of (LS, remainder_strategy, ls_equals_vs, k_remainder) tuples
    """
    if isa_name not in VECTOR_ISAS:
        return []
    
    VS = get_vector_size(isa_name, element_type)
    if VS is None:
        return []
    
    configurations = []
    
    # LS (Logical Size) in [1, VS]
    for LS in range(1, VS + 1):
        # Feature 3: LS == VS (no masking overhead)
        ls_equals_vs = 1 if (LS == VS) else 0
        
        # Feature 4: K % LS (numeric remainder, not binary)
        k_remainder = K % LS
        
        # Feature 7: LS // K (integer division)
        ls_div_k = LS // K if K > 0 else 0
        
        # Feature 5: Test both strategies
        for remainder_strategy in ['masking', 'unrolling']:
            configurations.append((LS, remainder_strategy, ls_equals_vs, k_remainder, ls_div_k))
    
    return configurations


def map_isa_to_body_stride(LS, K, VS):
    """
    Map LS (logical size) to body masking stride for compilation.
    
    Logic:
    - If LS < VS: Body MUST be masked with stride LS (this is what LS means!)
      When LS < VS, there may still be a remainder (K % LS). The remainder is
      automatically handled by masking in fully masked mode - the loop uses stride LS
      and the mask handles any leftover elements in the last iteration.
    - If LS == VS: No body masking (use full vector width, only remainder matters)
    
    Note: When LS < VS, we always mask the body with stride LS regardless of
    whether it divides K or not. The remainder is automatically handled by masking
    (not by separate remainder_strategy - fully masked mode always masks remainders).
    """
    # If LS < VS, body MUST be masked with stride LS
    if LS < VS:
        return LS
    # If LS == VS, no body masking (use full vector width, only remainder matters)
    else:
        return 0


def generate_corrected_training_data(
    matrix_sizes,  # List of (M, K, N) tuples
    # For matmul M×K × K×N = M×N:
    # - Vectorization happens along N dimension
    # - So: K_vec = N (vectorized dimension size)
    # - X = M (outer repetition)
    # - Y = 1 (or could be K? Let's assume Y=1 for now, or make it configurable)
    vector_isas,
    element_types=['f32'],
    Y_dimension=1,  # Default Y=1, can be overridden
    output_dir='dl_training_data_corrected',
    num_runs=5,
    llvm_path=None,
    max_configs_per_size=None
):
    """
    Generate training data using corrected input space.
    
    For matmul M×K × K×N = M×N:
    - Vectorization is along N dimension
    - So K_vec (vectorized dim size) = N
    - X = M (outer repetition)
    - Y = Y_dimension (middle repetition, default 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    test_files_dir = os.path.join(output_dir, 'test_files')
    os.makedirs(test_files_dir, exist_ok=True)
    
    all_data = []
    
    # Initialize CSV file for incremental writes (batch insertion - 1 new data point at a time)
    csv_file = os.path.join(output_dir, 'training_data.csv')
    csv_initialized = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    csv_fieldnames = None
    
    # If CSV exists, read existing data to avoid duplicates and get fieldnames
    if csv_initialized:
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                csv_fieldnames = list(reader.fieldnames)
                existing_count = sum(1 for _ in reader)
                
                # Check if CSV has new feature (LS_div_K) - if not, need to migrate
                expected_features = ['instruction_type', 'LS', 'LS_equals_VS', 'K_remainder', 
                                   'remainder_strategy', 'X_times_Y', 'LS_div_K']
                
                if 'LS_div_K' not in csv_fieldnames:
                    print(f"Found existing CSV with old format (missing LS_div_K).")
                    print(f"Backing up old CSV and creating new one with updated headers.")
                    
                    # Backup old CSV
                    import shutil
                    from datetime import datetime
                    backup_name = f"training_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    backup_path = os.path.join(output_dir, backup_name)
                    shutil.copy2(csv_file, backup_path)
                    print(f"   Old CSV backed up to: {backup_name}")
                    
                    # Reset to create new CSV
                    csv_initialized = False
                    csv_fieldnames = None
                else:
                    print(f"Found existing CSV with {existing_count} rows. Will append new data.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}. Starting fresh.")
            csv_initialized = False
    
    # Generate test files
    print("Generating test files...")
    test_files = []
    for m, k_matmul, n in matrix_sizes:
        # For matmul: vectorization happens along N dimension
        K_vec = n  # Vectorized dimension size (the "K" in the input space)
        X = m      # Outer repetition dimension
        Y = Y_dimension  # Middle repetition dimension (default 1)
        
        for element_type in element_types:
            test_content = create_test_case(m, k_matmul, n, element_type, 'matmul')
            test_file = os.path.join(test_files_dir,
                                   f"test_{m}x{k_matmul}x{n}_{element_type}_matmul.mlir")
            with open(test_file, 'w') as f:
                f.write(test_content)
            test_files.append((test_file, m, k_matmul, n, K_vec, X, Y, element_type))
    
    print(f"Generated {len(test_files)} test files")
    
    # Generate all configurations
    total_configs = 0
    for test_file, m, k_matmul, n, K_vec, X, Y, element_type in test_files:
        for isa_name in vector_isas:
            configs = generate_corrected_configurations(K_vec, X, Y, isa_name, element_type)
            if max_configs_per_size:
                configs = configs[:max_configs_per_size]
            total_configs += len(configs)
    
    print(f"Total configurations to test: {total_configs}")
    print()
    
    # Progress bar
    widgets = [
        'Progress: ', progressbar.Percentage(),
        ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
        ' ', progressbar.ETA(),
        ' ', progressbar.Counter(),
        f'/{total_configs}'
    ]
    bar = progressbar.ProgressBar(max_value=total_configs, widgets=widgets)
    bar.start()
    
    config_count = 0
    
    # Test all configurations
    for test_file, m, k_matmul, n, K_vec, X, Y, element_type in test_files:
        for isa_name in vector_isas:
            if isa_name not in VECTOR_ISAS:
                continue
            
            VS = get_vector_size(isa_name, element_type)
            if VS is None:
                continue
            
            configs = generate_corrected_configurations(K_vec, X, Y, isa_name, element_type)
            if max_configs_per_size:
                configs = configs[:max_configs_per_size]
            
            for LS, remainder_strategy, ls_equals_vs, k_remainder, ls_div_k in configs:
                config_count += 1
                bar.update(config_count)
                
                # Map to compilation parameters
                body_stride = map_isa_to_body_stride(LS, K_vec, VS)
                remainder_stride = 0  # Max it out as specified
                
                # Compile and benchmark
                speedup, success = compile_and_benchmark_configuration(
                    test_file,
                    isa_name,
                    body_stride,
                    remainder_strategy,
                    remainder_stride,
                    num_runs=num_runs,
                    llvm_path=llvm_path
                )
                
                if success and speedup is not None:
                    # Create feature vector (7 features - with LS // K)
                    feature_row = {
                        # Feature 1: Instruction type (numeric)
                        'instruction_type': INSTRUCTION_TYPE_MAP[isa_name],
                        
                        # Feature 2: LS (Logical Size)
                        'LS': LS,
                        
                        # Feature 3: LS == VS (binary)
                        'LS_equals_VS': ls_equals_vs,
                        
                        # Feature 4: K % LS (numeric remainder)
                        'K_remainder': k_remainder,
                        
                        # Feature 5: Remainder strategy (binary: 0=masking, 1=unrolling)
                        'remainder_strategy': 0 if remainder_strategy == 'masking' else 1,
                        
                        # Feature 6: X * Y (numeric)
                        'X_times_Y': X * Y,
                        
                        # Feature 7: LS // K (integer division)
                        'LS_div_K': ls_div_k,
                        
                        # Additional metadata (for reference)
                        'VS': VS,
                        'K': K_vec,
                        'X': X,
                        'Y': Y,
                        'isa_name': isa_name,
                        'element_type': element_type,
                        'm': m,
                        'k': k_matmul,
                        'n': n,
                        
                        # Output: speedup
                        'speedup': speedup
                    }
                    all_data.append(feature_row)
                    
                    # Append to CSV immediately (batch insertion - 1 new data point at a time)
                    if not csv_initialized:
                        # First time: write header
                        csv_fieldnames = list(feature_row.keys())
                        with open(csv_file, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                            writer.writeheader()
                            writer.writerow(feature_row)
                        csv_initialized = True
                    else:
                        # Subsequent times: append row
                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                            writer.writerow(feature_row)
    
    bar.finish()
    
    # Save final results
    print(f"\nSaving results...")
    # CSV is already written incrementally, just save JSON
    json_file = os.path.join(output_dir, 'training_data.json')
    
    if all_data:
        # CSV already written incrementally, just verify it exists
        if os.path.exists(csv_file):
            print(f"CSV already saved incrementally: {csv_file} ({len(all_data)} rows)")
        else:
            # Fallback: write CSV if for some reason it wasn't written
            print(f"Warning: CSV not found, writing now...")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
        
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'input_space': 'corrected',
                    'loop_structure': 'for X -> for Y -> for K (vectorized)',
                    'total_samples': len(all_data),
                    'matrix_sizes': matrix_sizes,
                    'vector_isas': vector_isas,
                    'element_types': element_types,
                    'Y_dimension': Y_dimension,
                    'num_runs_per_config': num_runs,
                    'input_features': [
                        'instruction_type',  # 1: AVX, 2: AVX2, 3: AVX512, etc.
                        'LS',  # Logical Size [1, VS]
                        'LS_equals_VS',  # Binary: no masking overhead
                        'K_remainder',  # Numeric: K % LS (remainder value)
                        'remainder_strategy',  # Binary: 0=masking, 1=unrolling
                        'X_times_Y',  # Numeric: product of repetition dimensions
                        'LS_div_K'  # Numeric: LS // K (integer division)
                    ],
                    'note': 'X_times_Y should be treated as having linear trend in model'
                },
                'data': all_data
            }, f, indent=2)
        
        print(f"Saved {len(all_data)} samples to:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")
        
        # Print feature statistics
        print("\nFeature Statistics:")
        print("=" * 60)
        df_dict = {key: [row[key] for row in all_data] for key in all_data[0].keys()}
        print(f"Instruction types: {set(df_dict['instruction_type'])}")
        print(f"LS range: [{min(df_dict['LS'])}, {max(df_dict['LS'])}]")
        print(f"K range: [{min(df_dict['K'])}, {max(df_dict['K'])}]")
        print(f"X_times_Y range: [{min(df_dict['X_times_Y'])}, {max(df_dict['X_times_Y'])}]")
        print(f"Speedup range: [{min(df_dict['speedup']):.2f}x, {max(df_dict['speedup']):.2f}x]")
        print(f"Average speedup: {sum(df_dict['speedup'])/len(df_dict['speedup']):.2f}x")
    else:
        print("No data collected!")
    
    return all_data


def generate_diverse_matrix_sizes():
    """
    Generate diverse matrix sizes for comprehensive training data:
    - Square matrices (various sizes)
    - Rectangular matrices (different aspect ratios)
    - Sizes with interesting remainders
    - Prime sizes
    - Edge cases
    """
    matrix_sizes = []
    
    # Small square matrices (powers of 2 and non-powers)
    for size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        matrix_sizes.append((size, size, size))
    
    # Medium square matrices
    for size in [17, 20, 24, 32, 33, 40, 48, 64]:
        matrix_sizes.append((size, size, size))
    
    # Rectangular matrices - different aspect ratios
    rectangular = [
        # Wide outputs (many columns)
        (4, 4, 8), (4, 4, 12), (4, 4, 16),
        (8, 8, 16), (8, 8, 24), (8, 8, 32),
        (16, 16, 32), (16, 16, 48), (16, 16, 64),
        
        # Narrow outputs (few columns)
        (8, 8, 4), (16, 16, 8), (32, 32, 16),
        
        # Different K dimensions
        (8, 4, 8), (8, 12, 8),  # M=8, varying K, N=8
        (16, 8, 16), (16, 24, 16),  # M=16, varying K, N=16
        (4, 8, 4), (4, 16, 4),  # Small M, large K
        
        # More rectangular
        (4, 8, 16), (8, 16, 32),
        (16, 8, 4), (32, 16, 8),
    ]
    matrix_sizes.extend(rectangular)
    
    # Sizes with interesting remainders for AVX (VS=4 for f32)
    # These create remainders of 1, 2, 3
    for base in [4, 8, 12, 16, 20, 24, 32]:
        for remainder in [1, 2, 3]:
            n = base + remainder
            matrix_sizes.append((base, base, n))
    
    # Sizes with interesting remainders for AVX2 (VS=8 for f32)
    # Remainders 1-7
    for base in [8, 16, 24, 32, 40, 48, 64]:
        for remainder in [1, 2, 3, 4, 5, 6, 7]:
            n = base + remainder
            matrix_sizes.append((base, base, n))
    
    # Prime sizes (create worst-case remainders)
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in primes:
        if p <= 64:  # Keep reasonable for training
            matrix_sizes.append((p, p, p))
    
    # Sizes around cache line boundaries (16 f32 elements)
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        n = 16 + offset
        if n > 0:
            matrix_sizes.append((16, 16, n))
    
    # Very small edge cases
    edge_cases = [
        (1, 1, 1), (2, 2, 2),
        (1, 2, 2), (2, 1, 2), (2, 2, 1),
        (3, 3, 3),
    ]
    matrix_sizes.extend(edge_cases)
    
    # Remove duplicates and sort
    matrix_sizes = sorted(list(set(matrix_sizes)))
    
    return matrix_sizes


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data with CORRECTED input space'
    )
    parser.add_argument('--output-dir', default='dl_training_data_corrected',
                       help='Output directory')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of benchmark runs per configuration (default: 5). Median speedup is used.')
    parser.add_argument('--llvm-path', default=None,
                       help='Path to LLVM project')
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Max configs per matrix size')
    parser.add_argument('--matrix-sizes', nargs='+', default=None,
                       help='Matrix sizes as M,K,N tuples (default: auto-generate diverse set)')
    parser.add_argument('--vector-isas', nargs='+',
                       default=['AVX', 'AVX2'],  # AVX512 needs external machine
                       help='Vector ISAs to test (default: AVX, AVX2)')
    parser.add_argument('--Y-dimension', type=int, default=1,
                       help='Y dimension size (middle repetition, default: 1)')
    parser.add_argument('--auto-sizes', action='store_true', default=True,
                       help='Auto-generate diverse matrix sizes (default: True)')
    parser.add_argument('--small-set', action='store_true',
                       help='Use smaller set of matrix sizes for quick testing')
    
    args = parser.parse_args()
    
    # Parse matrix sizes
    matrix_sizes = []
    
    if args.matrix_sizes:
        # Use provided matrix sizes
        for size_str in args.matrix_sizes:
            parts = size_str.split(',')
            if len(parts) == 3:
                matrix_sizes.append((int(parts[0]), int(parts[1]), int(parts[2])))
    elif args.small_set:
        # Small set for quick testing
        matrix_sizes = [
            (3, 3, 3), (4, 4, 4), (8, 8, 8), (16, 16, 16),
            (4, 4, 5), (4, 4, 7), (8, 8, 9), (8, 8, 15),
        ]
    else:
        # Auto-generate diverse set
        matrix_sizes = generate_diverse_matrix_sizes()
    
    valid_isas = [isa for isa in args.vector_isas if isa in VECTOR_ISAS]
    if not valid_isas:
        print("Error: No valid vector ISAs specified")
        return 1
    
    print("Generating DL Training Data (CORRECTED Input Space)")
    print("=" * 60)
    print(f"Loop structure: for X -> for Y -> for K (vectorized)")
    print(f"\nMatrix sizes: {len(matrix_sizes)} unique sizes")
    
    # Show sample of matrix sizes
    print("\nSample matrix sizes:")
    square_sizes = [s for s in matrix_sizes if s[0] == s[1] == s[2]]
    rectangular_sizes = [s for s in matrix_sizes if not (s[0] == s[1] == s[2])]
    
    print(f"  Square matrices: {len(square_sizes)}")
    if square_sizes:
        print(f"    Examples: {square_sizes[:10]}")
        if len(square_sizes) > 10:
            print(f"    ... and {len(square_sizes) - 10} more")
    
    print(f"  Rectangular matrices: {len(rectangular_sizes)}")
    if rectangular_sizes:
        print(f"    Examples: {rectangular_sizes[:10]}")
        if len(rectangular_sizes) > 10:
            print(f"    ... and {len(rectangular_sizes) - 10} more")
    
    print(f"\nMapping:")
    print(f"  K_vec (vectorized dim) = N")
    print(f"  X (outer repetition) = M")
    print(f"  Y (middle repetition) = {args.Y_dimension}")
    print(f"Vector ISAs: {valid_isas}")
    print(f"Output directory: {args.output_dir}")
    print(f"Runs per config: {args.num_runs} (using MEDIAN speedup)")
    print()
    
    all_data = generate_corrected_training_data(
        matrix_sizes=matrix_sizes,
        vector_isas=valid_isas,
        Y_dimension=args.Y_dimension,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        llvm_path=args.llvm_path,
        max_configs_per_size=args.max_configs
    )
    
    print(f"\nGenerated {len(all_data)} training samples")
    print("\nNote: X_times_Y should be feature-engineered to indicate linear trend in model")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

