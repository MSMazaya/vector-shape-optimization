#!/usr/bin/env python3
"""
End-to-end benchmark for a BERT-like transformer encoder block.

This script treats `tests/transformer_block_bert_like.mlir` as the starting
Linalg-on-memref IR for a transformer-style block dominated by GEMMs. It then:

  - Compiles a scalar (non-vectorized) version as a reference.
  - Compiles vectorized versions using our `vector-shape-opt` pass under
    different strategy flags.
  - Links each with a small C driver that:
      * allocates and initializes input / weight / buffer tensors
      * calls scalar and vectorized versions many times
      * reports Speedup: <x.xx>x
  - Collects median speedups and prints a summary table + JSON.

This is analogous to `benchmark_heir_mlp.py` but for a transformer-style GEMM
workload instead of the HEIR MLP.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np


def get_default_test_file() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "tests", "transformer_block_bert_like.mlir")


def compile_scalar_block(test_file: str, llvm_path: str, temp_dir: str, vector_isa: str) -> str:
    """Compile scalar (non-vectorized) version of transformer_block, return object file."""
    scalar_mlir = os.path.join(temp_dir, "transformer_scalar.mlir")
    scalar_ll = os.path.join(temp_dir, "transformer_scalar.ll")
    scalar_o = os.path.join(temp_dir, "transformer_scalar.o")

    subprocess.run(
        [
            f"{llvm_path}/build/bin/mlir-opt",
            test_file,
            "--linalg-generalize-named-ops",
            "--expand-strided-metadata",
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--memref-expand",
            "--finalize-memref-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-ub-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            scalar_mlir,
        ],
        check=True,
        capture_output=True,
    )

    subprocess.run(
        [
            f"{llvm_path}/build/bin/mlir-translate",
            "--mlir-to-llvmir",
            scalar_mlir,
            "-o",
            scalar_ll,
        ],
        check=True,
        capture_output=True,
    )

    # Determine architecture based on vector ISA
    if vector_isa == "sve":
        march = "aarch64"
    else:
        march = "x86-64"

    subprocess.run(
        [
            f"{llvm_path}/build/bin/llc",
            f"-march={march}",
            "-O3",
            "-filetype=obj",
            scalar_ll,
            "-o",
            scalar_o,
        ],
        check=True,
        capture_output=True,
    )

    return scalar_o


def compile_vectorized_block(
    test_file: str,
    llvm_path: str,
    temp_dir: str,
    vector_isa: str,
    strategy_type: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Compile vectorized transformer_block with a given strategy. Returns (object_file, chosen_strategy)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if vector_isa == "avx512":
        llc_attrs = "-mattr=+avx512f,+avx512vl"
        vector_width_flag = "--linalg-to-vector-vector-width=512"
        march = "x86-64"
    elif vector_isa == "avx2":
        llc_attrs = "-mattr=+avx2,+fma"
        vector_width_flag = "--linalg-to-vector-vector-width=256"
        march = "x86-64"
    elif vector_isa == "sve":
        llc_attrs = "-mattr=+sve"
        vector_width_flag = "--linalg-to-vector-vector-width=256"
        march = "aarch64"
    else:  # avx
        llc_attrs = "-mattr=+avx,+fma"
        vector_width_flag = "--linalg-to-vector-vector-width=128"
        march = "x86-64"

    vectorized_mlir = os.path.join(
        temp_dir, f"transformer_{strategy_type}.mlir")
    vectorized_lowered = os.path.join(
        temp_dir, f"transformer_{strategy_type}_lowered.mlir"
    )
    vectorized_ll = os.path.join(temp_dir, f"transformer_{strategy_type}.ll")
    vectorized_o = os.path.join(temp_dir, f"transformer_{strategy_type}.o")

    # Preprocess to ensure `module { ... }` and `func.return`.
    preprocessed_file = os.path.join(
        temp_dir, f"transformer_{strategy_type}_pre.mlir")
    with open(test_file, "r") as f_in:
        content = f_in.read()
    if "func.return" not in content:
        content = re.sub(r"(?<!func\.)\breturn\b", "func.return", content)
    if "module {" not in content:
        content = f"module {{\n{content}\n}}\n"
    with open(preprocessed_file, "w") as f_out:
        f_out.write(content)

    # List of all strategy selections (one per matmul)
    chosen_strategies: list[str] = []

    vector_opt_cmd = [
        os.path.join(script_dir, "build", "tools",
                     "vector-shape-opt", "vector-shape-opt"),
        "--linalg-to-vector",
        vector_width_flag,
    ]

    if strategy_type == "scalar_remainder":
        # Default configuration inside the pass.
        pass
    elif strategy_type == "unrolled_remainder":
        vector_opt_cmd.append("--linalg-to-vector-unroll-scalar-k")
    elif strategy_type == "masked_remainder":
        vector_opt_cmd.append("--linalg-to-vector-use-masked-remainder")
    elif strategy_type == "heuristic":
        vector_opt_cmd.append("--linalg-to-vector-use-heuristic")
        vector_opt_cmd.append("--linalg-to-vector-debug-strategy")
    elif strategy_type == "model":
        vector_opt_cmd.append("--linalg-to-vector-use-model")
        vector_opt_cmd.append("--linalg-to-vector-debug-strategy")

    result = subprocess.run(
        vector_opt_cmd + [preprocessed_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if strategy_type in ("heuristic", "model") and result.stderr:
        for line in result.stderr.splitlines():
            # Look for strategy debug output from the pass - try multiple patterns
            strategy_name = None
            line_upper = line.upper()

            # Primary pattern: "[Strategy Debug] Selected:" or "Selected strategy:"
            if "[Strategy Debug] Selected:" in line or "Selected strategy:" in line or "[Strategy]" in line:
                if "NO_MASKING" in line_upper:
                    strategy_name = "NO_MASKING"
                elif "UNROLL_REMAINDER" in line_upper:
                    strategy_name = "UNROLL_REMAINDER"
                elif "MASK_REMAINDER" in line_upper:
                    strategy_name = "MASK_REMAINDER"
                elif "MASK_BODY" in line_upper:
                    strategy_name = "MASK_BODY"
            # Also check for any line containing strategy names (more flexible)
            elif any(keyword in line_upper for keyword in ["NO_MASKING", "UNROLL_REMAINDER", "MASK_REMAINDER", "MASK_BODY"]):
                # Only capture if it looks like a strategy selection line
                if any(indicator in line_upper for indicator in ["SELECTED", "CHOSE", "CHOICE", "STRATEGY", "USING"]):
                    if "NO_MASKING" in line_upper:
                        strategy_name = "NO_MASKING"
                    elif "UNROLL_REMAINDER" in line_upper:
                        strategy_name = "UNROLL_REMAINDER"
                    elif "MASK_REMAINDER" in line_upper:
                        strategy_name = "MASK_REMAINDER"
                    elif "MASK_BODY" in line_upper:
                        strategy_name = "MASK_BODY"

            if strategy_name:
                chosen_strategies.append(strategy_name)

        # If we still didn't find any strategies but stderr exists, show a sample for debugging
        if not chosen_strategies and strategy_type == "model" and result.stderr:
            # Look for any lines that might contain strategy info
            debug_lines = [l.strip() for l in result.stderr.splitlines()
                           if any(kw in l.upper() for kw in ["STRATEGY", "MASK", "UNROLL", "SELECT", "MODEL", "CHOOSE"])]
            if debug_lines:
                # Show first few relevant lines for debugging
                sample = "\n      ".join(debug_lines[:5])
                print(
                    f"    Debug: Could not parse strategies, but found relevant stderr lines:")
                print(f"      {sample[:400]}")
            else:
                # Show a sample of all stderr if no relevant lines found
                all_lines = [l.strip()
                             for l in result.stderr.splitlines() if l.strip()][:5]
                if all_lines:
                    sample = "\n      ".join(all_lines)
                    print(
                        f"    Debug: No strategy keywords found. Sample stderr (first 5 non-empty lines):")
                    print(f"      {sample[:400]}")

    if result.returncode != 0:
        error_msg = result.stderr[:500] if result.stderr else "Unknown error"
        print(f"    vector-shape-opt error ({strategy_type}): {error_msg}")
        if result.stdout:
            print(f"    stdout: {result.stdout[:200]}")
        return None, chosen_strategies

    with open(vectorized_mlir, "w") as f:
        f.write(result.stdout)

    # Lower vector IR to LLVM dialect.
    subprocess.run(
        [
            f"{llvm_path}/build/bin/mlir-opt",
            vectorized_mlir,
            "--expand-strided-metadata",
            "--memref-expand",
            "--lower-vector-multi-reduction",
            "--convert-vector-to-scf",
            "--convert-vector-to-llvm",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-ub-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            vectorized_lowered,
        ],
        check=True,
        capture_output=True,
    )

    # Rename `@transformer_block` to `@vectorized_transformer_block` for the
    # vectorized entry so that the scalar and vectorized objects can be linked
    # together. Also rename @approx_gelu to avoid duplicate definitions.
    with open(vectorized_lowered, "r") as f:
        lowered_content = f.read()
    lowered_content = lowered_content.replace(
        "@transformer_block", "@vectorized_transformer_block"
    )
    lowered_content = lowered_content.replace(
        "@approx_gelu", "@approx_gelu_vec")
    with open(vectorized_lowered, "w") as f:
        f.write(lowered_content)

    subprocess.run(
        [
            f"{llvm_path}/build/bin/mlir-translate",
            "--mlir-to-llvmir",
            vectorized_lowered,
            "-o",
            vectorized_ll,
        ],
        check=True,
        capture_output=True,
    )

    subprocess.run(
        [
            f"{llvm_path}/build/bin/llc",
            f"-march={march}",
            llc_attrs,
            "-O3",
            "-filetype=obj",
            vectorized_ll,
            "-o",
            vectorized_o,
        ],
        check=True,
        capture_output=True,
    )

    return vectorized_o, chosen_strategies


def write_wrapper_c(path: str, c_type: str, alignment: int) -> None:
    """Emit C driver that calls scalar and vectorized transformer_block repeatedly."""
    with open(path, "w") as f:
        f.write(
            f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Memref signatures follow MLIR's lowered form for rank-2 memrefs.

void transformer_block(
    {c_type}* input_q_alloc, {c_type}* input_q_aligned, int64_t input_q_offset,
    int64_t input_q_size0, int64_t input_q_size1,
    int64_t input_q_stride0, int64_t input_q_stride1,
    {c_type}* input_k_alloc, {c_type}* input_k_aligned, int64_t input_k_offset,
    int64_t input_k_size0, int64_t input_k_size1,
    int64_t input_k_stride0, int64_t input_k_stride1,
    {c_type}* input_v_alloc, {c_type}* input_v_aligned, int64_t input_v_offset,
    int64_t input_v_size0, int64_t input_v_size1,
    int64_t input_v_stride0, int64_t input_v_stride1,
    {c_type}* input_ff1_alloc, {c_type}* input_ff1_aligned, int64_t input_ff1_offset,
    int64_t input_ff1_size0, int64_t input_ff1_size1,
    int64_t input_ff1_stride0, int64_t input_ff1_stride1,
    {c_type}* wq_alloc, {c_type}* wq_aligned, int64_t wq_offset,
    int64_t wq_size0, int64_t wq_size1, int64_t wq_stride0, int64_t wq_stride1,
    {c_type}* wk_alloc, {c_type}* wk_aligned, int64_t wk_offset,
    int64_t wk_size0, int64_t wk_size1, int64_t wk_stride0, int64_t wk_stride1,
    {c_type}* wv_alloc, {c_type}* wv_aligned, int64_t wv_offset,
    int64_t wv_size0, int64_t wv_size1, int64_t wv_stride0, int64_t wv_stride1,
    {c_type}* wo_q_alloc, {c_type}* wo_q_aligned, int64_t wo_q_offset,
    int64_t wo_q_size0, int64_t wo_q_size1, int64_t wo_q_stride0, int64_t wo_q_stride1,
    {c_type}* wo_v_alloc, {c_type}* wo_v_aligned, int64_t wo_v_offset,
    int64_t wo_v_size0, int64_t wo_v_size1, int64_t wo_v_stride0, int64_t wo_v_stride1,
    {c_type}* wff1_alloc, {c_type}* wff1_aligned, int64_t wff1_offset,
    int64_t wff1_size0, int64_t wff1_size1, int64_t wff1_stride0, int64_t wff1_stride1,
    {c_type}* wff2_alloc, {c_type}* wff2_aligned, int64_t wff2_offset,
    int64_t wff2_size0, int64_t wff2_size1, int64_t wff2_stride0, int64_t wff2_stride1,
    {c_type}* bufq_alloc, {c_type}* bufq_aligned, int64_t bufq_offset,
    int64_t bufq_size0, int64_t bufq_size1,
    int64_t bufq_stride0, int64_t bufq_stride1,
    {c_type}* bufk_alloc, {c_type}* bufk_aligned, int64_t bufk_offset,
    int64_t bufk_size0, int64_t bufk_size1,
    int64_t bufk_stride0, int64_t bufk_stride1,
    {c_type}* bufv_alloc, {c_type}* bufv_aligned, int64_t bufv_offset,
    int64_t bufv_size0, int64_t bufv_size1,
    int64_t bufv_stride0, int64_t bufv_stride1,
    {c_type}* bufq_o_alloc, {c_type}* bufq_o_aligned, int64_t bufq_o_offset,
    int64_t bufq_o_size0, int64_t bufq_o_size1,
    int64_t bufq_o_stride0, int64_t bufq_o_stride1,
    {c_type}* bufv_o_alloc, {c_type}* bufv_o_aligned, int64_t bufv_o_offset,
    int64_t bufv_o_size0, int64_t bufv_o_size1,
    int64_t bufv_o_stride0, int64_t bufv_o_stride1,
    {c_type}* attn_alloc, {c_type}* attn_aligned, int64_t attn_offset,
    int64_t attn_size0, int64_t attn_size1,
    int64_t attn_stride0, int64_t attn_stride1,
    {c_type}* ff1_alloc, {c_type}* ff1_aligned, int64_t ff1_offset,
    int64_t ff1_size0, int64_t ff1_size1,
    int64_t ff1_stride0, int64_t ff1_stride1,
    {c_type}* ff1gelu_alloc, {c_type}* ff1gelu_aligned, int64_t ff1gelu_offset,
    int64_t ff1gelu_size0, int64_t ff1gelu_size1,
    int64_t ff1gelu_stride0, int64_t ff1gelu_stride1,
    {c_type}* ff2_alloc, {c_type}* ff2_aligned, int64_t ff2_offset,
    int64_t ff2_size0, int64_t ff2_size1,
    int64_t ff2_stride0, int64_t ff2_stride1);

void vectorized_transformer_block(
    {c_type}* input_q_alloc, {c_type}* input_q_aligned, int64_t input_q_offset,
    int64_t input_q_size0, int64_t input_q_size1,
    int64_t input_q_stride0, int64_t input_q_stride1,
    {c_type}* input_k_alloc, {c_type}* input_k_aligned, int64_t input_k_offset,
    int64_t input_k_size0, int64_t input_k_size1,
    int64_t input_k_stride0, int64_t input_k_stride1,
    {c_type}* input_v_alloc, {c_type}* input_v_aligned, int64_t input_v_offset,
    int64_t input_v_size0, int64_t input_v_size1,
    int64_t input_v_stride0, int64_t input_v_stride1,
    {c_type}* input_ff1_alloc, {c_type}* input_ff1_aligned, int64_t input_ff1_offset,
    int64_t input_ff1_size0, int64_t input_ff1_size1,
    int64_t input_ff1_stride0, int64_t input_ff1_stride1,
    {c_type}* wq_alloc, {c_type}* wq_aligned, int64_t wq_offset,
    int64_t wq_size0, int64_t wq_size1, int64_t wq_stride0, int64_t wq_stride1,
    {c_type}* wk_alloc, {c_type}* wk_aligned, int64_t wk_offset,
    int64_t wk_size0, int64_t wk_size1, int64_t wk_stride0, int64_t wk_stride1,
    {c_type}* wv_alloc, {c_type}* wv_aligned, int64_t wv_offset,
    int64_t wv_size0, int64_t wv_size1, int64_t wv_stride0, int64_t wv_stride1,
    {c_type}* wo_q_alloc, {c_type}* wo_q_aligned, int64_t wo_q_offset,
    int64_t wo_q_size0, int64_t wo_q_size1, int64_t wo_q_stride0, int64_t wo_q_stride1,
    {c_type}* wo_v_alloc, {c_type}* wo_v_aligned, int64_t wo_v_offset,
    int64_t wo_v_size0, int64_t wo_v_size1, int64_t wo_v_stride0, int64_t wo_v_stride1,
    {c_type}* wff1_alloc, {c_type}* wff1_aligned, int64_t wff1_offset,
    int64_t wff1_size0, int64_t wff1_size1, int64_t wff1_stride0, int64_t wff1_stride1,
    {c_type}* wff2_alloc, {c_type}* wff2_aligned, int64_t wff2_offset,
    int64_t wff2_size0, int64_t wff2_size1, int64_t wff2_stride0, int64_t wff2_stride1,
    {c_type}* bufq_alloc, {c_type}* bufq_aligned, int64_t bufq_offset,
    int64_t bufq_size0, int64_t bufq_size1,
    int64_t bufq_stride0, int64_t bufq_stride1,
    {c_type}* bufk_alloc, {c_type}* bufk_aligned, int64_t bufk_offset,
    int64_t bufk_size0, int64_t bufk_size1,
    int64_t bufk_stride0, int64_t bufk_stride1,
    {c_type}* bufv_alloc, {c_type}* bufv_aligned, int64_t bufv_offset,
    int64_t bufv_size0, int64_t bufv_size1,
    int64_t bufv_stride0, int64_t bufv_stride1,
    {c_type}* bufq_o_alloc, {c_type}* bufq_o_aligned, int64_t bufq_o_offset,
    int64_t bufq_o_size0, int64_t bufq_o_size1,
    int64_t bufq_o_stride0, int64_t bufq_o_stride1,
    {c_type}* bufv_o_alloc, {c_type}* bufv_o_aligned, int64_t bufv_o_offset,
    int64_t bufv_o_size0, int64_t bufv_o_size1,
    int64_t bufv_o_stride0, int64_t bufv_o_stride1,
    {c_type}* attn_alloc, {c_type}* attn_aligned, int64_t attn_offset,
    int64_t attn_size0, int64_t attn_size1,
    int64_t attn_stride0, int64_t attn_stride1,
    {c_type}* ff1_alloc, {c_type}* ff1_aligned, int64_t ff1_offset,
    int64_t ff1_size0, int64_t ff1_size1,
    int64_t ff1_stride0, int64_t ff1_stride1,
    {c_type}* ff1gelu_alloc, {c_type}* ff1gelu_aligned, int64_t ff1gelu_offset,
    int64_t ff1gelu_size0, int64_t ff1gelu_size1,
    int64_t ff1gelu_stride0, int64_t ff1gelu_stride1,
    {c_type}* ff2_alloc, {c_type}* ff2_aligned, int64_t ff2_offset,
    int64_t ff2_size0, int64_t ff2_size1,
    int64_t ff2_stride0, int64_t ff2_stride1);

int main() {{
  // Dimensions chosen to exercise diverse K-dimension remainders for 256-bit vectors (8 elements for f32):
  // S = 135: sequence length
  // H = 770: hidden dimension (base)
  // H_q = 775: Q projection K dimension (775 % 8 = 7, very far remainder)
  // H_k = 771: K projection K dimension (771 % 8 = 3, small remainder)
  // H_v = 770: V projection K dimension (770 % 8 = 2, small remainder)
  // H_o = 769: Output projection K dimension (769 % 8 = 1, very small remainder)
  // I = 3075: intermediate FFN dimension (3075 % 8 = 3, small remainder)
  const int S = 135;   // sequence length
  const int H = 770;   // hidden dimension (base)
  const int H_input = 775; // Input buffer size (large enough for largest K dimension)
  const int H_q = 775; // Q projection K dimension (outputs to 772)
  const int H_k = 771; // K projection K dimension
  const int H_v = 770; // V projection K dimension (outputs to 774)
  const int H_ff1 = 769; // FFN layer 1 K dimension
  const int H_o_q = 772; // Output projection for Q (K dimension)
  const int H_o_v = 774; // Output projection for V (K dimension)
  const int I = 3075;  // intermediate FFN dimension

  const int iterations = 20;

  // Allocate buffers.
  // Input buffer is [S x H_input] = [135 x 775] to accommodate largest K dimension
  {c_type}* input = ({c_type}*)aligned_alloc({alignment}, S * H_input * sizeof({c_type}));
  {c_type}* wq = ({c_type}*)aligned_alloc({alignment}, H_q * H_o_q * sizeof({c_type}));  // [775x772]
  {c_type}* wk = ({c_type}*)aligned_alloc({alignment}, H_k * H * sizeof({c_type}));
  {c_type}* wv = ({c_type}*)aligned_alloc({alignment}, H_v * H_o_v * sizeof({c_type}));  // [770x774]
  {c_type}* wo_q = ({c_type}*)aligned_alloc({alignment}, H_o_q * H * sizeof({c_type}));  // [772x770]
  {c_type}* wo_v = ({c_type}*)aligned_alloc({alignment}, H_o_v * H * sizeof({c_type}));  // [774x770]
  {c_type}* wff1 = ({c_type}*)aligned_alloc({alignment}, H_ff1 * I * sizeof({c_type}));
  {c_type}* wff2 = ({c_type}*)aligned_alloc({alignment}, I * H * sizeof({c_type}));

  {c_type}* bufq = ({c_type}*)aligned_alloc({alignment}, S * H_o_q * sizeof({c_type}));  // [135x772]
  {c_type}* bufk = ({c_type}*)aligned_alloc({alignment}, S * H * sizeof({c_type}));
  {c_type}* bufv = ({c_type}*)aligned_alloc({alignment}, S * H_o_v * sizeof({c_type}));  // [135x774]
  {c_type}* bufq_o = ({c_type}*)aligned_alloc({alignment}, S * H * sizeof({c_type}));
  {c_type}* bufv_o = ({c_type}*)aligned_alloc({alignment}, S * H * sizeof({c_type}));
  {c_type}* attn = ({c_type}*)aligned_alloc({alignment}, S * H * sizeof({c_type}));
  {c_type}* ff1 = ({c_type}*)aligned_alloc({alignment}, S * I * sizeof({c_type}));
  {c_type}* ff1gelu = ({c_type}*)aligned_alloc({alignment}, S * I * sizeof({c_type}));
  {c_type}* ff2_scalar = ({c_type}*)aligned_alloc({alignment}, S * 773 * sizeof({c_type}));  // Output dimension 773
  {c_type}* ff2_vector = ({c_type}*)aligned_alloc({alignment}, S * 773 * sizeof({c_type}));  // Output dimension 773

  if (!input || !wq || !wk || !wv || !wo_q || !wo_v || !wff1 || !wff2 ||
      !bufq || !bufk || !bufv || !attn || !ff1 || !ff1gelu ||
      !ff2_scalar || !ff2_vector) {{
    fprintf(stderr, "Allocation failed\\n");
    return 1;
  }}

  // Initialize with deterministic pseudo-random values.
  srand(0);
  for (int i = 0; i < S * H_input; ++i) {{
    input[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_q * H_o_q; ++i) {{
    wq[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_k * H; ++i) {{
    wk[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_v * H_o_v; ++i) {{
    wv[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_o_q * H; ++i) {{
    wo_q[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_o_v * H; ++i) {{
    wo_v[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < H_ff1 * I; ++i) {{
    wff1[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}
  for (int i = 0; i < I * H; ++i) {{
    wff2[i] = ({c_type})rand() / ({c_type})RAND_MAX;
  }}

  // Input memref descriptors for different K dimensions
  // All point to the same underlying buffer but with different size1 (K dimension)
  const int64_t input_q_offset = 0;
  const int64_t input_q_size0 = S;
  const int64_t input_q_size1 = H_q;  // 775, remainder 7
  const int64_t input_q_stride1 = 1;
  const int64_t input_q_stride0 = H_input;  // Stride based on full buffer size

  const int64_t input_k_offset = 0;
  const int64_t input_k_size0 = S;
  const int64_t input_k_size1 = H_k;  // 771, remainder 3
  const int64_t input_k_stride1 = 1;
  const int64_t input_k_stride0 = H_input;

  const int64_t input_v_offset = 0;
  const int64_t input_v_size0 = S;
  const int64_t input_v_size1 = H_v;  // 770, remainder 2
  const int64_t input_v_stride1 = 1;
  const int64_t input_v_stride0 = H_input;

  const int64_t input_ff1_offset = 0;
  const int64_t input_ff1_size0 = S;
  const int64_t input_ff1_size1 = H_ff1;  // 769, remainder 1
  const int64_t input_ff1_stride1 = 1;
  const int64_t input_ff1_stride0 = H_input;

  // Weight matrices with different K dimensions
  const int64_t wq_offset = 0;
  const int64_t wq_size0 = H_q;  // K dimension
  const int64_t wq_size1 = H_o_q;  // Output dimension 772
  const int64_t wq_stride1 = 1;
  const int64_t wq_stride0 = H_o_q;

  const int64_t wk_offset = 0;
  const int64_t wk_size0 = H_k;  // K dimension
  const int64_t wk_size1 = H;
  const int64_t wk_stride1 = 1;
  const int64_t wk_stride0 = H;

  const int64_t wv_offset = 0;
  const int64_t wv_size0 = H_v;  // K dimension
  const int64_t wv_size1 = H_o_v;  // Output dimension 774
  const int64_t wv_stride1 = 1;
  const int64_t wv_stride0 = H_o_v;

  const int64_t wo_q_offset = 0;
  const int64_t wo_q_size0 = H_o_q;  // K dimension (772)
  const int64_t wo_q_size1 = H;
  const int64_t wo_q_stride1 = 1;
  const int64_t wo_q_stride0 = H;

  const int64_t wo_v_offset = 0;
  const int64_t wo_v_size0 = H_o_v;  // K dimension (774)
  const int64_t wo_v_size1 = H;
  const int64_t wo_v_stride1 = 1;
  const int64_t wo_v_stride0 = H;

  const int64_t wff1_offset = 0;
  const int64_t wff1_size0 = H_ff1;  // K dimension (769, remainder 1)
  const int64_t wff1_size1 = I;
  const int64_t wff1_stride1 = 1;
  const int64_t wff1_stride0 = I;

  const int64_t wff2_offset = 0;
  const int64_t wff2_size0 = I;  // K dimension (3075)
  const int64_t wff2_size1 = 773;  // Output dimension 773 (remainder 5)
  const int64_t wff2_stride1 = 1;
  const int64_t wff2_stride0 = 773;

  const int64_t buf3_offset = 0;
  const int64_t buf3_size0 = S;
  const int64_t buf3_size1 = H;
  const int64_t buf3_stride1 = 1;
  const int64_t buf3_stride0 = H;

  const int64_t bufq_offset = 0;
  const int64_t bufq_size0 = S;
  const int64_t bufq_size1 = H_o_q;  // 772
  const int64_t bufq_stride1 = 1;
  const int64_t bufq_stride0 = H_o_q;

  const int64_t bufv_offset = 0;
  const int64_t bufv_size0 = S;
  const int64_t bufv_size1 = H_o_v;  // 774
  const int64_t bufv_stride1 = 1;
  const int64_t bufv_stride0 = H_o_v;

  const int64_t buf3072_offset = 0;
  const int64_t buf3072_size0 = S;
  const int64_t buf3072_size1 = I;
  const int64_t buf3072_stride1 = 1;
  const int64_t buf3072_stride0 = I;

  clock_t start = clock();
  for (int it = 0; it < iterations; ++it) {{
    transformer_block(
      input, input, input_q_offset,
      input_q_size0, input_q_size1,
      input_q_stride0, input_q_stride1,
      input, input, input_k_offset,
      input_k_size0, input_k_size1,
      input_k_stride0, input_k_stride1,
      input, input, input_v_offset,
      input_v_size0, input_v_size1,
      input_v_stride0, input_v_stride1,
      input, input, input_ff1_offset,
      input_ff1_size0, input_ff1_size1,
      input_ff1_stride0, input_ff1_stride1,
      wq, wq, wq_offset, wq_size0, wq_size1, wq_stride0, wq_stride1,
      wk, wk, wk_offset, wk_size0, wk_size1, wk_stride0, wk_stride1,
      wv, wv, wv_offset, wv_size0, wv_size1, wv_stride0, wv_stride1,
      wo_q, wo_q, wo_q_offset, wo_q_size0, wo_q_size1, wo_q_stride0, wo_q_stride1,
      wo_v, wo_v, wo_v_offset, wo_v_size0, wo_v_size1, wo_v_stride0, wo_v_stride1,
      wff1, wff1, wff1_offset, wff1_size0, wff1_size1, wff1_stride0, wff1_stride1,
      wff2, wff2, wff2_offset, wff2_size0, wff2_size1, wff2_stride0, wff2_stride1,
      bufq, bufq, bufq_offset, bufq_size0, bufq_size1,
      bufq_stride0, bufq_stride1,
      bufk, bufk, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      bufv, bufv, bufv_offset, bufv_size0, bufv_size1,
      bufv_stride0, bufv_stride1,
      bufq_o, bufq_o, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      bufv_o, bufv_o, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      attn, attn, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      ff1, ff1, buf3072_offset, buf3072_size0, buf3072_size1,
      buf3072_stride0, buf3072_stride1,
      ff1gelu, ff1gelu, buf3072_offset, buf3072_size0, buf3072_size1,
      buf3072_stride0, buf3072_stride1,
      ff2_scalar, ff2_scalar, buf3_offset, buf3_size0, 773,  // Output dimension 773
      buf3_stride0, 1);
  }}
  clock_t end = clock();
  double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;

  start = clock();
  for (int it = 0; it < iterations; ++it) {{
    vectorized_transformer_block(
      input, input, input_q_offset,
      input_q_size0, input_q_size1,
      input_q_stride0, input_q_stride1,
      input, input, input_k_offset,
      input_k_size0, input_k_size1,
      input_k_stride0, input_k_stride1,
      input, input, input_v_offset,
      input_v_size0, input_v_size1,
      input_v_stride0, input_v_stride1,
      input, input, input_ff1_offset,
      input_ff1_size0, input_ff1_size1,
      input_ff1_stride0, input_ff1_stride1,
      wq, wq, wq_offset, wq_size0, wq_size1, wq_stride0, wq_stride1,
      wk, wk, wk_offset, wk_size0, wk_size1, wk_stride0, wk_stride1,
      wv, wv, wv_offset, wv_size0, wv_size1, wv_stride0, wv_stride1,
      wo_q, wo_q, wo_q_offset, wo_q_size0, wo_q_size1, wo_q_stride0, wo_q_stride1,
      wo_v, wo_v, wo_v_offset, wo_v_size0, wo_v_size1, wo_v_stride0, wo_v_stride1,
      wff1, wff1, wff1_offset, wff1_size0, wff1_size1, wff1_stride0, wff1_stride1,
      wff2, wff2, wff2_offset, wff2_size0, wff2_size1, wff2_stride0, wff2_stride1,
      bufq, bufq, bufq_offset, bufq_size0, bufq_size1,
      bufq_stride0, bufq_stride1,
      bufk, bufk, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      bufv, bufv, bufv_offset, bufv_size0, bufv_size1,
      bufv_stride0, bufv_stride1,
      bufq_o, bufq_o, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      bufv_o, bufv_o, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      attn, attn, buf3_offset, buf3_size0, buf3_size1,
      buf3_stride0, buf3_stride1,
      ff1, ff1, buf3072_offset, buf3072_size0, buf3072_size1,
      buf3072_stride0, buf3072_stride1,
      ff1gelu, ff1gelu, buf3072_offset, buf3072_size0, buf3072_size1,
      buf3072_stride0, buf3072_stride1,
      ff2_vector, ff2_vector, buf3_offset, buf3_size0, 773,  // Output dimension 773
      buf3_stride0, 1);
  }}
  end = clock();
  double vector_time = (double)(end - start) / CLOCKS_PER_SEC;

  double speedup = scalar_time / vector_time;
  printf("Speedup: %.4fx\\n", speedup);

  free(input);
  free(wq);
  free(wk);
  free(wv);
  free(wo_q);
  free(wo_v);
  free(wff1);
  free(wff2);
  free(bufq);
  free(bufk);
  free(bufv);
  free(attn);
  free(ff1);
  free(ff1gelu);
  free(ff2_scalar);
  free(ff2_vector);

  return 0;
}}
"""
        )


def compile_and_benchmark_strategy(
    test_file: str,
    vector_isa: str,
    strategy_type: str,
    num_runs: int,
    llvm_path_override: Optional[str] = None,
) -> Tuple[Optional[float], list[str]]:
    """Compile scalar + vectorized transformer block for a given strategy and return (median_speedup, chosen_strategies_list)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llvm_path = llvm_path_override or os.environ.get(
        "LLVM_PROJECT_PATH",
        "/home/mazaya/Documents/cmu/interviews/vorticity/llvm-project-vorticity",
    )

    c_type = "float"
    alignment = 32

    temp_dir = tempfile.mkdtemp()
    try:
        scalar_o = compile_scalar_block(
            test_file, llvm_path, temp_dir, vector_isa)
        vector_o, chosen_strategy = compile_vectorized_block(
            test_file, llvm_path, temp_dir, vector_isa, strategy_type
        )
        if not vector_o:
            return None, chosen_strategy

        wrapper_c = os.path.join(temp_dir, "wrapper.c")
        write_wrapper_c(wrapper_c, c_type=c_type, alignment=alignment)

        # ISA-specific clang flags.
        if vector_isa == "avx512":
            clang_flags = "-mavx512f -mavx512vl"
        elif vector_isa == "avx2":
            clang_flags = "-mavx2 -mfma"
        elif vector_isa == "sve":
            clang_flags = "-march=armv8-a+sve"
        else:  # avx
            clang_flags = "-mavx -mfma"

        executable = os.path.join(temp_dir, "benchmark")
        clang_cmd = ["clang", "-O3"] + clang_flags.split() + [
            wrapper_c,
            scalar_o,
            vector_o,
            "-o",
            executable,
            "-lm",
        ]
        subprocess.run(clang_cmd, check=True, capture_output=True)

        speedups = []
        for _ in range(num_runs):
            result = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                output = result.stdout + result.stderr
                print(
                    f"    Error (exit code {result.returncode}) while running strategy {strategy_type}: "
                    f"{output[:400]}"
                )
                return None, chosen_strategy

            output = result.stdout + result.stderr
            match = re.search(r"Speedup:\s+([0-9]*\.[0-9]+)x", output)
            if not match:
                print(
                    f"    Could not parse speedup for {strategy_type}, output was:\n{output[:400]}"
                )
                return None, chosen_strategy
            speedups.append(float(match.group(1)))

        if not speedups:
            return None, chosen_strategy

        median_speedup = float(np.median(speedups))
        return median_speedup, chosen_strategy

    except subprocess.CalledProcessError as e:
        msg = ""
        if e.stderr:
            msg = e.stderr[:500]
        elif e.output:
            msg = e.output[:500]
        else:
            msg = str(e)
        print(f"    Compilation error for {strategy_type}: {msg}")
        return None, None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_machine_config():
    """Load machine configuration from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "machine_config.json")

    if not os.path.exists(config_file):
        default_config = {
            "machines": {
                "local": {
                    "type": "local",
                    "description": "Local machine",
                    "output_dir": "output/local"
                }
            },
            "default_machine": "local"
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(config_file, 'r') as f:
        return json.load(f)


def cross_compile_and_benchmark_remote(test_file: str, vector_isa: str, machine_config: dict) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Optional[list[str]]]]:
    """Cross-compile locally and run benchmarks on remote machine (ARM or AVX512)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llvm_path = os.environ.get(
        "LLVM_PROJECT_PATH",
        "/home/mazaya/Documents/cmu/interviews/vorticity/llvm-project-vorticity",
    )

    ssh_host = machine_config["ssh_host"]
    ssh_key = machine_config.get("ssh_key", "")
    remote_path = machine_config["remote_path"]

    # Set compilation flags based on ISA
    if vector_isa == "sve":
        llc_attrs = "-mattr=+sve"
        vector_width_flag = "--linalg-to-vector-vector-width=256"
        llc_march = "aarch64"
        clang_flags = "-march=armv8-a+sve"
        cross_compile = True
        target_triple = "aarch64-linux-gnu"
    elif vector_isa == "avx512":
        llc_attrs = "-mattr=+avx512f"
        vector_width_flag = "--linalg-to-vector-vector-width=512"
        llc_march = "x86-64"
        clang_flags = "-march=skylake-avx512"
        cross_compile = False  # x86-64 to x86-64, no cross-compilation needed
        target_triple = None
    else:
        # For other ISAs, use old method
        return run_benchmark_remote_old(test_file, vector_isa, machine_config)

    test_file_abs = os.path.abspath(test_file)
    test_file_name = os.path.basename(test_file)

    target_name = "ARM" if vector_isa in ["sve", "sme"] else "AVX512"
    print(f"  Cross-compiling {test_file_name} locally for {target_name}...")

    c_type = "float"
    alignment = 32

    temp_dir = tempfile.mkdtemp()
    strategies = ["scalar_remainder", "unrolled_remainder",
                  "masked_remainder", "heuristic", "model"]
    results: Dict[str, Dict[str, Optional[float]]] = {}
    chosen_debug: Dict[str, Optional[list[str]]] = {}

    try:
        # Preprocess test file
        preprocessed_file = os.path.join(temp_dir, "preprocessed.mlir")
        with open(test_file, "r") as f_in:
            content = f_in.read()
        if "func.return" not in content:
            content = re.sub(r"(?<!func\.)\breturn\b", "func.return", content)
        if "module {" not in content:
            content = f"module {{\n{content}\n}}\n"
        with open(preprocessed_file, "w") as f_out:
            f_out.write(content)

        # Compile each strategy
        for strategy in strategies:
            print(f"    Compiling {strategy}...")
            strategy_dir = os.path.join(temp_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)

            # Run vector-shape-opt
            vector_opt_cmd = [
                os.path.join(script_dir, "build", "tools",
                             "vector-shape-opt", "vector-shape-opt"),
                "--linalg-to-vector",
                vector_width_flag,
            ]

            if strategy == "unrolled_remainder":
                vector_opt_cmd.append("--linalg-to-vector-unroll-scalar-k")
            elif strategy == "masked_remainder":
                vector_opt_cmd.append(
                    "--linalg-to-vector-use-masked-remainder")
            elif strategy == "heuristic":
                vector_opt_cmd.append("--linalg-to-vector-use-heuristic")
                vector_opt_cmd.append("--linalg-to-vector-debug-strategy")
            elif strategy == "model":
                vector_opt_cmd.append("--linalg-to-vector-use-model")
                vector_opt_cmd.append("--linalg-to-vector-debug-strategy")

            result = subprocess.run(
                vector_opt_cmd + [preprocessed_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print(
                    f"      Error compiling {strategy}: {result.stderr[:500]}")
                continue

            # Extract strategy choices
            chosen_strategies = []
            if strategy in ("heuristic", "model") and result.stderr:
                for line in result.stderr.splitlines():
                    if "[Strategy Debug] Selected:" in line or "[Model Strategy] Selected:" in line or "[Strategy]" in line:
                        line_upper = line.upper()
                        if "NO_MASKING" in line_upper:
                            chosen_strategies.append("NO_MASKING")
                        elif "UNROLL_REMAINDER" in line_upper:
                            chosen_strategies.append("UNROLL_REMAINDER")
                        elif "MASK_REMAINDER" in line_upper:
                            chosen_strategies.append("MASK_REMAINDER")
                        elif "MASK_BODY" in line_upper:
                            chosen_strategies.append("MASK_BODY")
            if chosen_strategies:
                chosen_debug[strategy] = chosen_strategies

            vectorized_mlir = os.path.join(strategy_dir, "vectorized.mlir")
            with open(vectorized_mlir, "w") as f:
                f.write(result.stdout)

            # Lower to LLVM IR
            vectorized_lowered = os.path.join(
                strategy_dir, "vectorized_lowered.mlir")
            vectorized_ll = os.path.join(strategy_dir, "vectorized.ll")

            subprocess.run(
                [
                    f"{llvm_path}/build/bin/mlir-opt",
                    vectorized_mlir,
                    "--expand-strided-metadata",
                    "--memref-expand",
                    "--lower-vector-multi-reduction",
                    "--convert-vector-to-scf",
                    "--convert-vector-to-llvm",
                    "--convert-scf-to-cf",
                    "--convert-cf-to-llvm",
                    "--convert-func-to-llvm",
                    "--convert-arith-to-llvm",
                    "--finalize-memref-to-llvm",
                    "--convert-ub-to-llvm",
                    "--reconcile-unrealized-casts",
                    "-o",
                    vectorized_lowered,
                ],
                check=True,
                capture_output=True,
            )

            # Rename functions
            with open(vectorized_lowered, "r") as f:
                content = f.read()
            content = content.replace(
                "@transformer_block", "@vectorized_transformer_block")
            content = content.replace("@approx_gelu", "@approx_gelu_vec")
            with open(vectorized_lowered, "w") as f:
                f.write(content)

            subprocess.run(
                [
                    f"{llvm_path}/build/bin/mlir-translate",
                    "--mlir-to-llvmir",
                    vectorized_lowered,
                    "-o",
                    vectorized_ll,
                ],
                check=True,
                capture_output=True,
            )

            # Compile to object file
            vectorized_o = os.path.join(strategy_dir, "vectorized.o")
            subprocess.run(
                [
                    f"{llvm_path}/build/bin/llc",
                    f"-march={llc_march}",
                    llc_attrs,
                    "-O3",
                    "-filetype=obj",
                    vectorized_ll,
                    "-o",
                    vectorized_o,
                ],
                check=True,
                capture_output=True,
            )

            # Compile scalar version
            scalar_mlir = os.path.join(strategy_dir, "scalar.mlir")
            scalar_ll = os.path.join(strategy_dir, "scalar.ll")
            scalar_o = os.path.join(strategy_dir, "scalar.o")

            subprocess.run(
                [
                    f"{llvm_path}/build/bin/mlir-opt",
                    preprocessed_file,
                    "--linalg-generalize-named-ops",
                    "--expand-strided-metadata",
                    "--convert-linalg-to-loops",
                    "--convert-scf-to-cf",
                    "--convert-cf-to-llvm",
                    "--convert-func-to-llvm",
                    "--memref-expand",
                    "--finalize-memref-to-llvm",
                    "--convert-arith-to-llvm",
                    "--convert-ub-to-llvm",
                    "--reconcile-unrealized-casts",
                    "-o",
                    scalar_mlir,
                ],
                check=True,
                capture_output=True,
            )

            subprocess.run(
                [
                    f"{llvm_path}/build/bin/mlir-translate",
                    "--mlir-to-llvmir",
                    scalar_mlir,
                    "-o",
                    scalar_ll,
                ],
                check=True,
                capture_output=True,
            )

            llc_cmd = [
                f"{llvm_path}/build/bin/llc",
                f"-march={llc_march}",
                "-O3",
                "-filetype=obj",
                scalar_ll,
                "-o",
                scalar_o,
            ]
            # Add attrs for AVX512 (x86-64), but not for ARM cross-compilation
            if not cross_compile:
                llc_cmd.insert(2, llc_attrs)
            subprocess.run(
                llc_cmd,
                check=True,
                capture_output=True,
            )

            # Create wrapper
            wrapper_c = os.path.join(strategy_dir, "wrapper.c")
            write_wrapper_c(wrapper_c, c_type=c_type, alignment=alignment)

            # Compile wrapper
            wrapper_o = os.path.join(strategy_dir, "wrapper.o")
            if cross_compile:
                # Cross-compile for ARM
                try:
                    subprocess.run(
                        ["aarch64-linux-gnu-gcc", "-O3",
                            "-c", wrapper_c, "-o", wrapper_o],
                        check=True,
                        capture_output=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    subprocess.run(
                        ["clang", f"--target={target_triple}",
                            "-O3", "-c", wrapper_c, "-o", wrapper_o],
                        check=True,
                        capture_output=True,
                    )
            else:
                # Compile for x86-64 (same architecture)
                subprocess.run(
                    ["clang", "-O3", clang_flags,
                        "-c", wrapper_c, "-o", wrapper_o],
                    check=True,
                    capture_output=True,
                )

        # Copy binaries to remote machine
        print(f"  Copying binaries to {target_name} machine...")
        ssh_cmd_prep = ["ssh"]
        if ssh_key:
            ssh_cmd_prep.extend(["-i", ssh_key])
        ssh_cmd_prep.extend(
            [ssh_host, f"mkdir -p {remote_path}/benchmark_binaries"])
        subprocess.run(ssh_cmd_prep, capture_output=True, check=True)

        for strategy in strategies:
            strategy_dir = os.path.join(temp_dir, strategy)
            if os.path.exists(strategy_dir):
                scp_cmd_strategy = ["scp", "-r"]
                if ssh_key:
                    scp_cmd_strategy.extend(["-i", ssh_key])
                scp_cmd_strategy.extend(
                    [strategy_dir, f"{ssh_host}:{remote_path}/benchmark_binaries/"])
                subprocess.run(scp_cmd_strategy,
                               capture_output=True, check=True)

        # Run benchmarks on remote machine
        print(f"  Running benchmarks on {target_name} machine...")
        ssh_cmd = ["ssh"]
        if ssh_key:
            ssh_cmd.extend(["-i", ssh_key])
        ssh_cmd.append(ssh_host)

        remote_script = f"""
cd {remote_path}/benchmark_binaries
for strategy in scalar_remainder unrolled_remainder masked_remainder heuristic model; do
    echo "Benchmarking strategy: $strategy ..."
    if [ -d "$strategy" ]; then
        cd $strategy
        if [ ! -f scalar.o ] || [ ! -f vectorized.o ] || [ ! -f wrapper.o ]; then
            echo "  FAILED"
        elif gcc -O3 scalar.o vectorized.o wrapper.o -o benchmark -lm 2>&1 || clang -O3 scalar.o vectorized.o wrapper.o -o benchmark -lm 2>&1; then
            if [ -f benchmark ]; then
                ./benchmark 2>&1
            else
                echo "  FAILED"
            fi
        else
            echo "  FAILED"
        fi
        cd ..
    else
        echo "  FAILED"
    fi
done
"""

        result = subprocess.run(
            ssh_cmd + [remote_script],
            capture_output=True,
            text=True,
            timeout=3600,
        )

        # Parse results
        output = result.stdout + result.stderr
        current_strategy = None

        for line in output.split("\n"):
            for strategy in strategies:
                if f"Benchmarking strategy: {strategy} ..." in line:
                    current_strategy = strategy
                    break

            if current_strategy and "Speedup:" in line:
                match = re.search(r"Speedup:\s+([0-9]*\.[0-9]+)x", line)
                if match:
                    speedup = float(match.group(1))
                    results[current_strategy] = {"speedup": speedup}
                    print(f"    {current_strategy}: {speedup:.3f}x")
                current_strategy = None

            if current_strategy and ("FAILED" in line or "Error" in line):
                print(f"    {current_strategy}: FAILED")
                results[current_strategy] = {"speedup": None}
                current_strategy = None

        return results, chosen_debug

    except Exception as e:
        print(f"    Error: {str(e)}")
        return results, chosen_debug
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_benchmark_remote_old(test_file: str, vector_isa: str, machine_config: dict) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Optional[list[str]]]]:
    """Run benchmark on remote machine via SSH (for non-ARM targets)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script_name = "benchmark_transformer_block.py"
    local_benchmark_script = os.path.join(script_dir, benchmark_script_name)

    ssh_host = machine_config["ssh_host"]
    ssh_key = machine_config.get("ssh_key", "")
    remote_path = machine_config["remote_path"]
    llvm_path = machine_config.get("llvm_project_path", "")

    test_file_abs = os.path.abspath(test_file)
    test_file_name = os.path.basename(test_file)

    print(f"  Copying scripts and {test_file_name} to remote machine...")
    scp_cmd_script = ["scp"]
    scp_cmd_file = ["scp"]
    if ssh_key:
        scp_cmd_script.extend(["-i", ssh_key])
        scp_cmd_file.extend(["-i", ssh_key])

    scp_cmd_script.extend(
        [local_benchmark_script, f"{ssh_host}:{remote_path}/"])
    subprocess.run(scp_cmd_script, capture_output=True, check=True)

    # Ensure tests directory exists on remote
    ssh_cmd_prep = ["ssh"]
    if ssh_key:
        ssh_cmd_prep.extend(["-i", ssh_key])
    ssh_cmd_prep.extend([ssh_host, f"mkdir -p {remote_path}/tests"])
    subprocess.run(ssh_cmd_prep, capture_output=True, check=True)

    scp_cmd_file.extend([test_file_abs, f"{ssh_host}:{remote_path}/tests/"])
    subprocess.run(scp_cmd_file, capture_output=True, check=True)

    remote_test_file = f"tests/{test_file_name}"

    ssh_cmd = ["ssh"]
    if ssh_key:
        ssh_cmd.extend(["-i", ssh_key])
    ssh_cmd.append(ssh_host)

    remote_cmd = f"""
cd {remote_path} && \
export LLVM_PROJECT_PATH="{llvm_path}" && \
export PATH="$LLVM_PROJECT_PATH/build/bin:$PATH" && \
chmod +x {benchmark_script_name} && \
python3 {benchmark_script_name} --{vector_isa} {remote_test_file} 2>&1
"""

    print(f"  Running benchmark on remote machine...")
    try:
        result = subprocess.run(
            ssh_cmd + [remote_cmd],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout (compilation can be slow)
        )
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Remote benchmark timed out after 2 hours")
        print(f"    This may indicate the benchmark is hanging or compilation is very slow")
        print(f"    Try running with a shorter timeout or check the remote machine")
        return {}, {}

    output = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"    Remote execution error (return code {result.returncode}):")
        print(f"    STDOUT: {result.stdout[:2000]}")
        print(f"    STDERR: {result.stderr[:2000]}")
        return {}, {}

    # Parse results from output
    results: Dict[str, Dict[str, Optional[float]]] = {}
    chosen_debug: Dict[str, Optional[str]] = {}
    strategies = [
        "scalar_remainder",
        "unrolled_remainder",
        "masked_remainder",
        "heuristic",
        "model",
    ]

    lines = output.split('\n')
    current_strategy = None

    for i, line in enumerate(lines):
        for strategy in strategies:
            if f'Benchmarking strategy: {strategy} ...' in line:
                current_strategy = strategy
                break

        if current_strategy and 'Speedup:' in line:
            match = re.search(r'Speedup:\s+([0-9]*\.[0-9]+)x', line)
            if match:
                speedup = float(match.group(1))
                results[current_strategy] = {"speedup": speedup}
                print(f"    {current_strategy}: {speedup:.3f}x")
            current_strategy = None

        if current_strategy and 'Chosen internal strategy:' in line:
            match = re.search(r'Chosen internal strategy:\s+(\w+)', line)
            if match:
                chosen_debug[current_strategy] = [match.group(1)]
            elif 'Chosen internal strategies' in line:
                # Parse multiple strategies
                match = re.search(
                    r'Chosen internal strategies.*?:\s+(.+)', line)
                if match:
                    strategies_str = match.group(1)
                    chosen_debug[current_strategy] = [s.strip()
                                                      for s in strategies_str.split(',')]

        if current_strategy and ('FAILED' in line or 'Error' in line or 'Exception' in line):
            print(f"    {current_strategy}: FAILED")
            results[current_strategy] = {"speedup": None}
            current_strategy = None

    return results, chosen_debug


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_test = get_default_test_file()

    # Load machine config
    config = load_machine_config()
    machines = config["machines"]
    default_machine = config.get("default_machine", "local")

    # CLI parsing: [--machine <name>] [--avx|--avx2|--avx512|--sve] [path_to_block.mlir]
    vector_isa = "avx"
    machine_name = default_machine
    args = sys.argv[1:]

    if "--machine" in args:
        idx = args.index("--machine")
        if idx + 1 < len(args):
            machine_name = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
        else:
            print("Error: --machine requires a machine name")
            return 1

    if "--avx512" in args:
        vector_isa = "avx512"
        args.remove("--avx512")
    elif "--avx2" in args:
        vector_isa = "avx2"
        args.remove("--avx2")
    elif "--sve" in args:
        vector_isa = "sve"
        args.remove("--sve")
    elif "--avx" in args:
        vector_isa = "avx"
        args.remove("--avx")

    if machine_name not in machines:
        print(f"Error: Unknown machine '{machine_name}'")
        return 1

    machine_config = machines[machine_name]
    machine_type = machine_config.get("type", "local")

    if args:
        test_file = os.path.abspath(args[0])
    else:
        test_file = default_test

    if not os.path.isfile(test_file):
        print(f"Error: test file '{test_file}' not found")
        return 1

    print(f"Using machine: {machine_name}")
    print(f"Using test file: {test_file}")
    print(f"Vector ISA: {vector_isa.upper()}")
    print()

    strategies = [
        "scalar_remainder",
        "unrolled_remainder",
        "masked_remainder",
        "heuristic",
        "model",
    ]

    # Per-strategy results, keyed by strategy name.
    #   {
    #     "speedup": <scalar_vs_vector_speedup>
    #   }
    results: Dict[str, Dict[str, Optional[float]]] = {}
    chosen_debug: Dict[str, Optional[str]] = {}

    # Run benchmarks locally or remotely
    if machine_config["type"] == "remote":
        # Use cross-compilation for ARM targets (SVE/SME) and AVX512
        if vector_isa in ["sve", "sme", "avx512"]:
            results, chosen_debug = cross_compile_and_benchmark_remote(
                test_file, vector_isa, machine_config)
        else:
            results, chosen_debug = run_benchmark_remote_old(
                test_file, vector_isa, machine_config)
    else:
        num_runs = 10
        llvm_path = machine_config.get("llvm_project_path", None)

        for strategy in strategies:
            print(f"Benchmarking strategy: {strategy} ...")
            speedup, chosen_strategies = compile_and_benchmark_strategy(
                test_file, vector_isa, strategy, num_runs=num_runs, llvm_path_override=llvm_path
            )
            chosen_debug[strategy] = chosen_strategies
            if speedup is not None:
                print(f"  Speedup: {speedup:.3f}x")
                results[strategy] = {"speedup": speedup}
            else:
                print(f"  FAILED")
                results[strategy] = {"speedup": None}
            if chosen_strategies:
                if len(chosen_strategies) == 1:
                    print(
                        f"  Chosen internal strategy: {chosen_strategies[0]}")
                else:
                    print(
                        f"  Chosen internal strategies ({len(chosen_strategies)} matmuls): {', '.join(chosen_strategies)}")
            elif strategy in ("heuristic", "model"):
                # If we expected a strategy but didn't get one, show a hint
                print(
                    f"  Note: Could not extract internal strategy choice (check if --linalg-to-vector-debug-strategy is enabled)")
            print()

    # Print summary table.
    print("=" * 80)
    print("Transformer Block Benchmark Summary")
    print("=" * 80)

    header = (
        f"{'Strategy':<18} "
        f"{'Speedup vs Scalar':<20} "
        f"{'Internal Choices (N matmuls)':<30}"
    )
    print(header)
    print("-" * 100)

    for strategy in strategies:
        s = results.get(strategy, {})
        spd = s.get("speedup")
        spd_str = f"{spd:.3f}x" if spd is not None else "N/A"
        internal_list = chosen_debug.get(strategy)
        if isinstance(internal_list, list):
            if len(internal_list) == 0:
                internal_str = "-"
            elif len(internal_list) == 1:
                internal_str = internal_list[0]
            else:
                # Show count and unique strategies, or all if few
                unique_strategies = list(set(internal_list))
                if len(unique_strategies) <= 3 and len(internal_list) <= 7:
                    internal_str = f"{len(internal_list)}: {', '.join(internal_list)}"
                else:
                    # Summarize: show count and unique set
                    counts = {}
                    for st in internal_list:
                        counts[st] = counts.get(st, 0) + 1
                    summary_parts = [f"{k}({v})" for k, v in counts.items()]
                    internal_str = f"{len(internal_list)}: {', '.join(summary_parts)}"
        else:
            internal_str = str(internal_list) if internal_list else "-"
        row = (
            f"{strategy:<18} "
            f"{spd_str:<20} "
            f"{internal_str:<30}"
        )
        print(row)
    print("-" * 100)

    # Save JSON next to script. Use ISA-specific filename so multiple runs
    # for different ISAs don't overwrite each other.
    output_json = os.path.join(
        script_dir, f"transformer_block_results_{vector_isa}.json")
    with open(output_json, "w") as f:
        json.dump(
            {
                "machine": machine_name,
                "test_file": test_file,
                "vector_isa": vector_isa,
                "strategies": strategies,
                "results": results,
                "internal_choices": chosen_debug,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
