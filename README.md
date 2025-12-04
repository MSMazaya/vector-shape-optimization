# Vector Shape Optimization in MLIR

This project implements an MLIR compiler pass that systematically explores vector shapes and remainder-handling strategies for matrix multiplication operations, guided by analytical and neural-network-based cost models.

## Prerequisites

### Required Dependencies

1. **LLVM/MLIR (v17.0.6)**: 
   - Download from https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6
   - Build with MLIR enabled: `cmake -DLLVM_ENABLE_PROJECTS="mlir" ...`

2. **Python 3.8+** with packages:
   ```bash
   pip install torch numpy pandas scikit-learn
   ```

3. **Optional (for transformer block generation)**:
   - `torch-mlir`: https://github.com/llvm/torch-mlir

### Hardware Requirements

- **x86-64** with AVX, AVX2, or AVX-512 support (for Intel benchmarks)
- **ARM64** with SVE support (for ARM benchmarks)
- AWS EC2 instances tested:
  - `c7i.xlarge` (Intel Sapphire Rapids, AVX-512)
  - ARM instances with SVE support

## Building

### 1. Build LLVM/MLIR

```bash
# Clone and build LLVM with MLIR
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-17.0.6
mkdir build && cd build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Set `LLVM_PATH` environment variable:
```bash
export LLVM_PATH=/path/to/llvm-project
```

### 2. Build This Project

```bash
cd vector-shape-opt
mkdir build && cd build
cmake .. -DMLIR_DIR=$LLVM_PATH/build/lib/cmake/mlir
make -j$(nproc)
```

The build produces:
- `build/tools/vector-shape-opt/vector-shape-opt`: MLIR opt tool with our passes

## Usage

### Basic Vectorization

Apply the vectorization pass to an MLIR file:

```bash
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-vector-width=512 \
  input.mlir
```

### Strategy Selection

#### Analytical Heuristic

```bash
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-use-heuristic \
  --linalg-to-vector-vector-width=512 \
  input.mlir
```

#### Neural Network Model

```bash
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-use-model \
  --linalg-to-vector-vector-width=512 \
  input.mlir
```

#### Explicit Strategies

```bash
# Masked remainder
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-use-masked-remainder \
  --linalg-to-vector-vector-width=512 \
  input.mlir

# Unrolled remainder
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-unroll-scalar-k \
  --linalg-to-vector-vector-width=512 \
  input.mlir
```

### Debug Output

Enable debug output for strategy selection:

```bash
./build/tools/vector-shape-opt/vector-shape-opt \
  --linalg-to-vector \
  --linalg-to-vector-use-heuristic \
  --linalg-to-vector-debug-strategy \
  --linalg-to-vector-vector-width=512 \
  input.mlir
```

## Running Benchmarks

### Transformer Block Benchmark

Benchmarks a BERT-like transformer encoder block:

```bash
python3 benchmark_transformer_block.py \
  --vector-isa avx512 \
  --test-file tests/transformer_block_bert_like.mlir
```

Outputs JSON results and summary table comparing:
- Scalar (baseline)
- Masked remainder
- Unrolled remainder
- Heuristic selection
- Neural network model selection

### HEIR MLP Benchmark

Benchmarks the HEIR MLP (MNIST) workload:

```bash
python3 benchmark_heir_mlp.py \
  --vector-isa avx512 \
  --test-file tests/mlp_heir_mnist.mlir
```

### Synthetic Matrix Benchmarks

Benchmarks various matrix sizes (2x2 to 16x16):

```bash
python3 benchmark_performance.py \
  --vector-isa avx512 \
  --max-dim 16
```

### Parameter Sweep

Sweep (a, b) parameters for the analytical cost model:

```bash
python3 sweep_heuristic_params.py \
  --avx512 \
  --max-tests 20 \
  --a-values 0.25 0.5 1.0 \
  --b-values 0.1 0.25 0.5
```
