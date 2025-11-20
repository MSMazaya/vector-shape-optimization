// Test case: Rectangular matrix multiplication (MxK * KxN)
module {
  func.func @matmul_rectangular(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) {
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x6xf32>)
                 outs(%C : memref<4x6xf32>)
    return
  }
}

