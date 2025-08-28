// RUN: %orchestra-opt %s --lower-orchestra-to-gpu="gpu-arch=rocdl" | FileCheck %s

// CHECK-LABEL: gpu.func @rock_gemm_test
// CHECK:         vector.load
// CHECK:         vector.load
// CHECK:         vector.fma
// CHECK:         vector.store
// CHECK-NOT:     rock.gemm
// CHECK:         gpu.return

gpu.module @test_rock_lowering {
  gpu.func @rock_gemm_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = "rock.gemm"(%arg0, %arg1) {arch = ""} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    gpu.return %result : tensor<4x4xf32>
  }
}
