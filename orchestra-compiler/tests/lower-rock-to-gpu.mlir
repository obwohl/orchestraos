// RUN: %orchestra-opt %s -lower-rock-to-gpu | FileCheck %s

// This test verifies that the `lower-rock-to-gpu` pass correctly
// converts a `rock.gemm` operation into a three-level nested `scf.for` loop.

// CHECK-LABEL: func @test_gemm
func.func @test_gemm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<64x256xf32> {
  // CHECK-NOT: rock.gemm
  // CHECK:      %[[EMPTY:.+]] = tensor.empty()
  // CHECK:      %[[C0:.+]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C64:.+]] = arith.constant 64 : index
  // CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
  // CHECK-NEXT: %[[C128:.+]] = arith.constant 128 : index
  // CHECK-NEXT: %[[C32_M:.+]] = arith.constant 32 : index
  // CHECK-NEXT: %[[C32_N:.+]] = arith.constant 32 : index
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : index
  // CHECK:      scf.for %{{.*}} = %[[C0]] to %[[C64]] step %[[C32_M]] iter_args(%{{.*}} = %[[EMPTY]])
  // CHECK:        scf.for %{{.*}} = %[[C0]] to %[[C256]] step %[[C32_N]]
  // CHECK:          scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C2]]
  // CHECK:            yield
  // CHECK:          }
  // CHECK:          yield
  // CHECK:        }
  // CHECK:        yield
  // CHECK:      }
  %0 = "rock.gemm"(%arg0, %arg1) {arch = "cdna3"} : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}
