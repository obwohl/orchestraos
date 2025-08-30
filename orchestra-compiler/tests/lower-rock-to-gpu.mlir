// RUN: %orchestra-opt %s -lower-rock-to-gpu | %FileCheck %s

// This test verifies that the `lower-rock-to-gpu` pass correctly
// converts a `rock.gemm` operation into a three-level nested `scf.for` loop.

// CHECK-LABEL: func @test_gemm
// CHECK-SAME: (%[[ARG0:.+]]: tensor<64x128xf32>, %[[ARG1:.+]]: tensor<128x256xf32>)
func.func @test_gemm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<64x256xf32> {
  // CHECK-NOT: rock.gemm
  // CHECK-NOT: amdgpu.mfma
  //
  // CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
  // CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty
  // CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
  // CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
  // CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
  //
  // CHECK:       scf.for %[[IV0:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%[[ITER0:.+]] = %[[EMPTY]]) -> (tensor<64x256xf32>) {
  // CHECK:         scf.for %[[IV1:.+]] = %[[C0]] to %[[C256]] step {{.+}} iter_args(%[[ITER1:.+]] = %[[ITER0]]) -> (tensor<64x256xf32>) {
  //
  // CHECK:           %[[ACC_INIT:.+]] = vector.transfer_read %[[ITER1]][%[[IV0]], %[[IV1]]]
  //
  // CHECK:           %[[REDUCTION_LOOP:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C128]] step %[[C2]] iter_args(%[[ACC_ITER:.+]] = %[[ACC_INIT]]) -> (vector<32x32xf32>) {
  //
  // CHECK:             %[[A_TILE:.+]] = vector.transfer_read %{{.*}}[%[[IV0]], %[[IV2]]]
  // CHECK:             %[[B_TILE:.+]] = vector.transfer_read %{{.*}}[%[[IV2]], %[[IV1]]]
  //
  // CHECK:             %[[CONTRACT:.+]] = vector.contract {{.*}} %[[A_TILE]], %[[B_TILE]], %[[ACC_ITER]]
  // CHECK:             scf.yield %[[CONTRACT]]
  // CHECK:           }
  //
  // CHECK:           %[[UPDATED_C:.+]] = vector.transfer_write %[[REDUCTION_LOOP]], %[[ITER1]][%[[IV0]], %[[IV1]]]
  // CHECK:           scf.yield %[[UPDATED_C]]
  // CHECK:         }
  // CHECK:       }
  %0 = "rock.gemm"(%arg0, %arg1) {arch = "cdna3"} : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}
