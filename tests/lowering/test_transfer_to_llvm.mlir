// RUN: orchestra-opt %s --lower-orchestra-to-llvm | FileCheck %s

module {
  func.func @test_transfer(%arg0: memref<10x20xf32>) {
    %0 = "orchestra.transfer"(%arg0) {from = @DRAM, to = @HBM} : (memref<10x20xf32>) -> memref<10x20xf32>
    return
  }
}

// CHECK-LABEL: func @test_transfer
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<10x20xf32>
// CHECK:         call @memcpy
// CHECK:         return
