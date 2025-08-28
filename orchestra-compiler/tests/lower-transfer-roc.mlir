// RUN: %orchestra-opt %s --lower-orchestra-to-gpu="gpu-arch=rocdl" | %FileCheck %s

// CHECK-LABEL: gpu.func @lower_transfer
// Use CHECK-DAG for the initial operations which may be reordered.
// CHECK-DAG:     %[[alloc:.*]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[src_ptr_int:.*]] = memref.extract_aligned_pointer_as_index %arg0
// CHECK-DAG:     %[[dst_ptr_int:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]]
// CHECK-DAG:     %[[src_i64:.*]] = arith.index_cast %[[src_ptr_int]] : index to i64
// CHECK-DAG:     %[[dst_i64:.*]] = arith.index_cast %[[dst_ptr_int]] : index to i64
//
// The following operations should be in order.
// CHECK:         %[[src_ptr:.*]] = llvm.inttoptr %[[src_i64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[dst_ptr:.*]] = llvm.inttoptr %[[dst_i64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[src_addrspace:.*]] = llvm.addrspacecast %[[src_ptr]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[dst_addrspace:.*]] = llvm.addrspacecast %[[dst_ptr]] : !llvm.ptr to !llvm.ptr<3>
// CHECK-NEXT:    %[[c16:.*]] = arith.constant 16 : i32
// CHECK-NEXT:    %[[c0_0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[c0_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    rocdl.global.load.lds %[[src_addrspace]], %[[dst_addrspace]], %[[c16]], %[[c0_0]], %[[c0_1]]
// CHECK-NEXT:    gpu.return

gpu.module @test attributes { rocdl.target = "gfx908" } {
  gpu.func @lower_transfer(%arg0: memref<4xf32>) {
    // The transfer op will be replaced by the ROCDL equivalent.
    %0 = orchestra.transfer %arg0 from @host to @gpu0 : memref<4xf32>
    memref.dealloc %0 : memref<4xf32>
    gpu.return
  }
}
