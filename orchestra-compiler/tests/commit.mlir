// RUN: %orchestra-opt %s | %orchestra-opt | FileCheck %s
// CHECK-LABEL: func @commit_test
func.func @commit_test(%arg0: memref<10xf32>) -> memref<10xf32> {
  // CHECK: %0 = orchestra.commit %arg0 : memref<10xf32>
  %0 = orchestra.commit %arg0 : memref<10xf32>
  return %0 : memref<10xf32>
}
