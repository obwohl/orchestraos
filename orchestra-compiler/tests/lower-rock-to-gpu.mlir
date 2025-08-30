// RUN: %orchestra-opt %s -lower-rock-to-gpu | %FileCheck %s

// This test verifies that the `lower-rock-to-gpu` pass correctly
// converts a `rock.gemm` operation into a three-level nested `scf.for` loop.

// CHECK-LABEL: func @test_gemm
// CHECK-SAME: (%[[ARG0:.+]]: tensor<64x128xf32>, %[[ARG1:.+]]: tensor<128x256xf32>)
func.func @test_gemm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<64x256xf32> {
  // CHECK-NOT: rock.gemm
  // CHECK:       scf.for
  // CHECK:         scf.for
  // CHECK:           %[[ACC:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK:           scf.for {{.*}} iter_args({{.*}} = %[[ACC]])
  // CHECK-DAG:         tensor.extract
  // CHECK-DAG:         tensor.extract
  // CHECK-DAG:         amdgpu.mfma
  // CHECK:           }
  // CHECK:         }
  // CHECK:       }
  %0 = "rock.gemm"(%arg0, %arg1) {arch = "cdna3"} : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}
