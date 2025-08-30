// RUN: %orchestra-opt --lower-rock-to-amdgpu %s | FileCheck %s

func.func @test_gemm(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "rock.gemm"(%arg0, %arg1) {arch = "cdna3"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_gemm
// CHECK:         %cst = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK:         return %cst
// CHECK-NOT:     rock.gemm
