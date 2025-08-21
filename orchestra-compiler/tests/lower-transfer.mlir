// RUN: %orchestra-opt %s --lower-orchestra-to-gpu | %FileCheck %s

// CHECK-LABEL: gpu.func @lower_transfer
//  CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[TOKEN:.*]] = nvgpu.device_async_copy %arg0[%[[C0]]], %[[ALLOC]][%[[C0]]], 4
//  CHECK-NEXT:   nvgpu.device_async_wait %[[TOKEN]]
//  CHECK-NEXT:   memref.dealloc %[[ALLOC]]
//  CHECK-NEXT:   gpu.return
gpu.module @test {
  gpu.func @lower_transfer(%arg0: memref<4xf32>) {
    %0 = orchestra.transfer %arg0 from @host to @gpu0 : memref<4xf32>
    memref.dealloc %0 : memref<4xf32>
    gpu.return
  }
}
