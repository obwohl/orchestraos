// RUN: %orchestra-opt %s --lower-orchestra-to-gpu | %FileCheck %s

// CHECK-LABEL: func @lower_transfer
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<10x20xf32>
//       CHECK:   gpu.memcpy %[[ALLOC]], %arg0 : memref<10x20xf32>, memref<10x20xf32>
//       CHECK:   return %[[ALLOC]]
func.func @lower_transfer(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
  %0 = orchestra.transfer %arg0 from @host to @gpu0 : memref<10x20xf32>
  return %0 : memref<10x20xf32>
}
