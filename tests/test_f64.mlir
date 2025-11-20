// Test case: Double precision (f64) matrix multiplication
module {
  func.func @matmul_f64(%A: memref<4x4xf64>, %B: memref<4x4xf64>, %C: memref<4x4xf64>) {
    linalg.matmul ins(%A, %B : memref<4x4xf64>, memref<4x4xf64>)
                 outs(%C : memref<4x4xf64>)
    return
  }
}

