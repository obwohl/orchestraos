// Test the full pipeline from linalg -> rock -> amdgpu
// RUN: %orchestra-opt %s --lower-linalg-to-rock --lower-rock-to-amdgpu="arch=amdgcn-amd-amdhsa" | FileCheck %s --check-prefix=CHECK-PIPELINE

// Test just the rock -> amdgpu lowering
// RUN: %orchestra-opt %s --lower-rock-to-amdgpu | FileCheck %s --check-prefix=CHECK-UNIT

// CHECK-PIPELINE: gpu.module @orchestra_gpu_module {
// CHECK-PIPELINE:   gpu.func @gemm_kernel_0
// CHECK-PIPELINE:     amdgpu.mfma
// CHECK-PIPELINE:   }
// CHECK-PIPELINE: }
// CHECK-PIPELINE-LABEL: func.func @main
// CHECK-PIPELINE-NOT: linalg.matmul
// CHECK-PIPELINE-NOT: rock.gemm
// CHECK-PIPELINE: gpu.launch_func
func.func @main() -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %A = tensor.empty() : tensor<128x128xf32>
  %B = tensor.empty() : tensor<128x128xf32>
  %C = tensor.empty() : tensor<128x128xf32>
  %D = linalg.fill ins(%c0 : f32) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %E = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%D : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %E : tensor<128x128xf32>
}

// -----

// CHECK-UNIT-LABEL: func.func @test_gemm
// CHECK-UNIT-NOT: rock.gemm
// CHECK-UNIT: amdgpu.mfma
func.func @test_gemm(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = rock.gemm arch = "cdna", matrix_a = %arg0, matrix_b = %arg1, matrix_c = %arg2 : tensor<32x16xf32>, tensor<16x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}