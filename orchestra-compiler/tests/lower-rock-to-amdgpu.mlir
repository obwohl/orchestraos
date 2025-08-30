// RUN: %orchestra-opt --lower-rock-to-amdgpu %s | FileCheck %s

// CHECK-LABEL: func.func @test_gemm(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<32x32xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x32xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<32x32xf32>
func.func @test_gemm(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "rock.gemm"(%arg0, %arg1, %arg2) {arch = "cdna3"} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK:      scf.for %{{.*}} =
// CHECK:        scf.for %{{.*}} =
// CHECK:          scf.for %{{.*}} =
// CHECK:            amdgpu.mfma
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK-NOT:  rock.gemm
