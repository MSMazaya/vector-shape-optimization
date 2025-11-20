// Test case: Odd-sized 5x5 matrix multiplication (tests non-power-of-2)
module {
  func.func @matmul_5x5(%A: memref<5x5xf32>, %B: memref<5x5xf32>, %C: memref<5x5xf32>) {
    linalg.matmul ins(%A, %B : memref<5x5xf32>, memref<5x5xf32>)
                 outs(%C : memref<5x5xf32>)
    return
  }
}

