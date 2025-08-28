// RUN: %orchestra-opt --lower-rock-to-amdgpu %s | FileCheck %s

func.func @test_gemm(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = rock.gemm arch = "cdna", matrix_a = %arg0, matrix_b = %arg1, matrix_c = %arg2 : tensor<32x16xf32>, tensor<16x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_gemm
// CHECK: bufferization.to_memref
// CHECK: vector.transfer_read
// CHECK: amdgpu.mfma
// CHECK: vector.transfer_write
// CHECK: bufferization.to_tensor
// CHECK-NOT: rock.gemm
