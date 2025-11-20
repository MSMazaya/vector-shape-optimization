// Test case: Large 16x16 matrix multiplication
module {
  func.func @matmul_16x16(%A: memref<16x16xf32>, %B: memref<16x16xf32>, %C: memref<16x16xf32>) {
    linalg.matmul ins(%A, %B : memref<16x16xf32>, memref<16x16xf32>)
                 outs(%C : memref<16x16xf32>)
    return
  }
}

