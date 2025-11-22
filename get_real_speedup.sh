#!/bin/bash

# Get actual runtime speedup by creating and running executables
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_PROJECT_PATH="/home/mazaya/Documents/cmu/interviews/vorticity/llvm-project-vorticity"
TEST_FILE="${1:-tests/test.mlir}"
ITERATIONS=100000

echo "=================================================================================="
echo "MEASURING ACTUAL RUNTIME SPEEDUP"
echo "=================================================================================="
echo ""

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Extract matrix dimensions from MLIR file
# Pattern: %A: memref<MxK>, %B: memref<KxN>, %C: memref<MxN>
DIMS=$(grep -oP 'memref<\K\d+x\d+' "$TEST_FILE" | head -3)
M=$(echo "$DIMS" | head -1 | cut -d'x' -f1)
K=$(echo "$DIMS" | head -1 | cut -d'x' -f2)
N=$(echo "$DIMS" | tail -1 | cut -d'x' -f2)

# Fallback to square if extraction fails
if [ -z "$M" ] || [ -z "$K" ] || [ -z "$N" ]; then
    SIZE=$(grep -oP 'memref<\K\d+' "$TEST_FILE" | head -1 || echo "4")
    M=$SIZE
    K=$SIZE
    N=$SIZE
fi

echo "Matrix dimensions: M=$M, K=$K, N=$N"


# Extract function name from MLIR file
FUNC_NAME=$(grep -oP 'func\.func @\K\w+' "$TEST_FILE" | head -1 || echo "foo")
echo "Found function name: $FUNC_NAME"

# Compile scalar version
echo "1. Compiling SCALAR version..."
cp "$TEST_FILE" "$TEMP_DIR/scalar.mlir"

"${LLVM_PROJECT_PATH}/build/bin/mlir-opt" "$TEMP_DIR/scalar.mlir" \
    --linalg-generalize-named-ops \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --memref-expand \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    > "$TEMP_DIR/scalar_lowered.mlir" 2>&1

# Rename function to scalar_matmul
sed -i "s/@${FUNC_NAME}/@scalar_matmul/g" "$TEMP_DIR/scalar_lowered.mlir"

"${LLVM_PROJECT_PATH}/build/bin/mlir-translate" --mlir-to-llvmir "$TEMP_DIR/scalar_lowered.mlir" > "$TEMP_DIR/scalar.ll" 2>&1
"${LLVM_PROJECT_PATH}/build/bin/llc" -march=x86-64 -O3 -filetype=obj "$TEMP_DIR/scalar.ll" -o "$TEMP_DIR/scalar.o" 2>&1

echo "  ✓ Scalar compiled"

# Compile AVX version
echo "2. Compiling AVX version..."
"${SCRIPT_DIR}/build/tools/vector-shape-opt/vector-shape-opt" --linalg-to-vector "$TEST_FILE" > "$TEMP_DIR/avx.mlir" 2>&1

"${LLVM_PROJECT_PATH}/build/bin/mlir-opt" "$TEMP_DIR/avx.mlir" \
    --memref-expand \
    --finalize-memref-to-llvm \
    --convert-vector-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    > "$TEMP_DIR/avx_lowered.mlir" 2>&1

# Rename function to avx_matmul
sed -i "s/@${FUNC_NAME}/@avx_matmul/g" "$TEMP_DIR/avx_lowered.mlir"

"${LLVM_PROJECT_PATH}/build/bin/mlir-translate" --mlir-to-llvmir "$TEMP_DIR/avx_lowered.mlir" > "$TEMP_DIR/avx.ll" 2>&1
"${LLVM_PROJECT_PATH}/build/bin/llc" -march=x86-64 -mattr=+avx,+fma -O3 -filetype=obj "$TEMP_DIR/avx.ll" -o "$TEMP_DIR/avx.o" 2>&1

echo "  ✓ AVX compiled"


# Check function signatures
echo ""
echo "Checking function signatures..."
echo "Scalar function:"
grep "^define" "$TEMP_DIR/scalar.ll" | head -1
echo "AVX function:"
grep "^define" "$TEMP_DIR/avx.ll" | head -1

echo ""
echo "Note: The function signatures are complex (memref descriptors)."
echo "To get actual runtime, we need to match these signatures exactly."
echo ""
echo "Attempting to link and run..."

# Find clang
CLANG=""
for candidate in "${LLVM_PROJECT_PATH}/build/bin/clang" \
    "${LLVM_PROJECT_PATH}/bin/clang" \
    "clang" \
    "/usr/bin/clang"; do
    if [ -f "$candidate" ] || command -v "$candidate" &> /dev/null; then
        CLANG="$candidate"
        break
    fi
done

if [ -z "$CLANG" ]; then
    echo "Error: clang not found"
    exit 1
fi

# Create wrapper that matches the exact signature (21 parameters)
cat > "$TEMP_DIR/wrapper.c" << 'EOFWRAP'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

// Exact function signature from LLVM IR
typedef void (*matmul_func_t)(void*, void*, long, long, long, long, long,
                              void*, void*, long, long, long, long, long,
                              void*, void*, long, long, long, long, long);

extern void scalar_matmul(void*, void*, long, long, long, long, long,
                          void*, void*, long, long, long, long, long,
                          void*, void*, long, long, long, long, long);

extern void avx_matmul(void*, void*, long, long, long, long, long,
                       void*, void*, long, long, long, long, long,
                       void*, void*, long, long, long, long, long);

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void call_matmul(matmul_func_t func, float* A, float* B, float* C, int M, int K, int N) {
    // Memref descriptor layout:
    // ptr allocated, ptr aligned, i64 offset, i64 sizes[0], i64 sizes[1], i64 strides[0], i64 strides[1]
    // A is MxK, B is KxN, C is MxN
    void* A_alloc = A;
    void* A_aligned = A;
    long A_offset = 0;
    long A_sizes[2] = {M, K};
    long A_strides[2] = {K, 1};
    
    void* B_alloc = B;
    void* B_aligned = B;
    long B_offset = 0;
    long B_sizes[2] = {K, N};
    long B_strides[2] = {N, 1};
    
    void* C_alloc = C;
    void* C_aligned = C;
    long C_offset = 0;
    long C_sizes[2] = {M, N};
    long C_strides[2] = {N, 1};
    
    func(A_alloc, A_aligned, A_offset, A_sizes[0], A_sizes[1], A_strides[0], A_strides[1],
         B_alloc, B_aligned, B_offset, B_sizes[0], B_sizes[1], B_strides[0], B_strides[1],
         C_alloc, C_aligned, C_offset, C_sizes[0], C_sizes[1], C_strides[0], C_strides[1]);
}

int main(int argc, char** argv) {
    int M = 4, K = 4, N = 4;
    int iterations = 100000;
    int warmup = 10000;
    
    // Parse dimensions from command line: M K N iterations
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) N = atoi(argv[3]);
    if (argc > 4) iterations = atoi(argv[4]);
    
    // Allocate matrices: A is MxK, B is KxN, C is MxN
    float* A = (float*)aligned_alloc(32, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(32, K * N * sizeof(float));
    float* C = (float*)aligned_alloc(32, M * N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    printf("Benchmarking %dx%d * %dx%d = %dx%d matrix multiplication\n", M, K, K, N, M, N);
    printf("Iterations: %d (warmup: %d)\n\n", iterations, warmup);
    
    // Warmup scalar
    for (int i = 0; i < warmup; i++) {
        call_matmul(scalar_matmul, A, B, C, M, K, N);
    }
    memset(C, 0, M * N * sizeof(float));
    
    // Time scalar
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        call_matmul(scalar_matmul, A, B, C, M, K, N);
    }
    double scalar_time = get_time() - start;
    
    memset(C, 0, M * N * sizeof(float));
    
    // Warmup AVX
    for (int i = 0; i < warmup; i++) {
        call_matmul(avx_matmul, A, B, C, M, K, N);
    }
    memset(C, 0, M * N * sizeof(float));
    
    // Time AVX
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        call_matmul(avx_matmul, A, B, C, M, K, N);
    }
    double avx_time = get_time() - start;
    
    printf("Results:\n");
    printf("  Scalar time: %.6f seconds (%.3f microseconds/iteration)\n", 
           scalar_time, scalar_time / iterations * 1e6);
    printf("  AVX time:    %.6f seconds (%.3f microseconds/iteration)\n", 
           avx_time, avx_time / iterations * 1e6);
    
    if (avx_time > 0) {
        double speedup = scalar_time / avx_time;
        printf("  Speedup:     %.2fx\n", speedup);
        printf("  Improvement: %.1f%%\n", (1 - avx_time/scalar_time) * 100);
    }
    
    free(A);
    free(B);
    free(C);
    return 0;
}
EOFWRAP

# Try to link AVX version
"$CLANG" -O3 -mavx -mfma "$TEMP_DIR/wrapper.c" "$TEMP_DIR/scalar.o" "$TEMP_DIR/avx.o" -o "$TEMP_DIR/benchmark_avx" -lm 2>&1 || {
    echo "Linking failed - function signatures don't match C interface"
    echo ""
    echo "The actual function signatures are:"
    echo "Scalar:"
    grep "^define" "$TEMP_DIR/scalar.ll" | head -1
    echo ""
    echo "AVX:"
    grep "^define" "$TEMP_DIR/avx.ll" | head -1
    echo ""
    echo "These need to be wrapped properly to call from C."
    exit 1
}

echo "✓ Linked successfully!"
echo ""

# Now compile and link AVX unrolled version separately
echo "=================================================================================="
echo "Compiling AVX unrolled version separately..."
echo "=================================================================================="
echo ""

# Compile AVX unrolled version
echo "Compiling AVX unrolled version..."
"${SCRIPT_DIR}/build/tools/vector-shape-opt/vector-shape-opt" --linalg-to-vector --linalg-to-vector-unroll-scalar-k "$TEST_FILE" > "$TEMP_DIR/avx_unrolled.mlir" 2>&1

"${LLVM_PROJECT_PATH}/build/bin/mlir-opt" "$TEMP_DIR/avx_unrolled.mlir" \
    --memref-expand \
    --finalize-memref-to-llvm \
    --convert-vector-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    > "$TEMP_DIR/avx_unrolled_lowered.mlir" 2>&1

# Rename function to avx_matmul (same name for consistency in wrapper)
sed -i "s/@${FUNC_NAME}/@avx_matmul/g" "$TEMP_DIR/avx_unrolled_lowered.mlir"

"${LLVM_PROJECT_PATH}/build/bin/mlir-translate" --mlir-to-llvmir "$TEMP_DIR/avx_unrolled_lowered.mlir" > "$TEMP_DIR/avx_unrolled.ll" 2>&1
"${LLVM_PROJECT_PATH}/build/bin/llc" -march=x86-64 -mattr=+avx,+fma -O3 -filetype=obj "$TEMP_DIR/avx_unrolled.ll" -o "$TEMP_DIR/avx_unrolled.o" 2>&1

echo "  ✓ AVX unrolled compiled"

# Link AVX unrolled version
"$CLANG" -O3 -mavx -mfma "$TEMP_DIR/wrapper.c" "$TEMP_DIR/scalar.o" "$TEMP_DIR/avx_unrolled.o" -o "$TEMP_DIR/benchmark_avx_unrolled" -lm 2>&1 || {
    echo "Linking failed for unrolled version"
    exit 1
}

echo "✓ Linked successfully!"
echo ""

# Run both benchmarks separately
echo "=================================================================================="
echo "Running benchmark: Scalar vs AVX (with scalar remainder)"
echo "=================================================================================="
"$TEMP_DIR/benchmark_avx" "$M" "$K" "$N" "$ITERATIONS"

echo ""
echo "=================================================================================="
echo "Running benchmark: Scalar vs AVX (with unrolled remainder)"
echo "=================================================================================="
"$TEMP_DIR/benchmark_avx_unrolled" "$M" "$K" "$N" "$ITERATIONS"

