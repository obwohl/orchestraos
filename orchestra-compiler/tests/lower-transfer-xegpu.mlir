// RUN: orchestra-opt %s --lower-orchestra-to-gpu=gpu-arch=xegpu --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_basic_f32
// CHECK-SAME: (%[[SRC:.*]]: memref<256x256xf32>, %[[DST:.*]]: memref<256x256xf32>)
gpu.module @test_module_f32 {
  gpu.func @test_basic_f32(%src: memref<256x256xf32>, %dst: memref<256x256xf32>) {
    orchestra.transfer %src to %dst : memref<256x256xf32>
    gpu.return
  }
}
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32>
// CHECK:        xegpu.load_nd %{{.*}} : !xegpu.tensor_desc<32x32xf32> -> vector<32x32xf32>
// CHECK:      }
// CHECK:      xegpu.fence

// -----

// CHECK-LABEL: func.func @test_f16
// CHECK-SAME: (%[[SRC:.*]]: memref<128x128xf16>, %[[DST:.*]]: memref<128x128xf16>)
gpu.module @test_module_f16 {
  gpu.func @test_f16(%src: memref<128x128xf16>, %dst: memref<128x128xf16>) {
    orchestra.transfer %src to %dst : memref<128x128xf16>
    gpu.return
  }
}
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<128x128xf16> -> !xegpu.tensor_desc<32x32xf16>
// CHECK:        xegpu.load_nd %{{.*}} : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
// CHECK:      }
// CHECK:      xegpu.fence

// -----

// CHECK-LABEL: func.func @test_bf16
// CHECK-SAME: (%[[SRC:.*]]: memref<128x128xbf16>, %[[DST:.*]]: memref<128x128xbf16>)
gpu.module @test_module_bf16 {
  gpu.func @test_bf16(%src: memref<128x128xbf16>, %dst: memref<128x128xbf16>) {
    orchestra.transfer %src to %dst : memref<128x128xbf16>
    gpu.return
  }
}
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<128x128xbf16> -> !xegpu.tensor_desc<32x32xbf16>
// CHECK:        xegpu.load_nd %{{.*}} : !xegpu.tensor_desc<32x32xbf16> -> vector<32x32xbf16>
// CHECK:      }
// CHECK:      xegpu.fence

// -----

// CHECK-LABEL: func.func @test_strided_source
// CHECK-SAME: (%[[SRC:.*]]: memref<256x256xf32, strided<[512, 1]>>, %[[DST:.*]]: memref<256x256xf32>)
gpu.module @test_module_strided {
  gpu.func @test_strided_source(%src: memref<256x256xf32, strided<[512, 1]>>, %dst: memref<256x256xf32>) {
    orchestra.transfer %src to %dst : memref<256x256xf32, strided<[512, 1]>>
    gpu.return
  }
}
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<256x256xf32, strided<[512, 1]>> -> !xegpu.tensor_desc<32x32xf32>
// CHECK:      }
// CHECK:      xegpu.fence

// -----

// CHECK-LABEL: func.func @test_global_to_workgroup
// CHECK-SAME: (%[[SRC:.*]]: memref<128x128xf32, #gpu.address_space<global>>, %[[DST:.*]]: memref<128x128xf32, #gpu.address_space<workgroup>>)
gpu.module @test_module_memspace {
  gpu.func @test_global_to_workgroup(%src: memref<128x128xf32, #gpu.address_space<global>>, %dst: memref<128x128xf32, #gpu.address_space<workgroup>>) {
    orchestra.transfer %src to %dst : memref<128x128xf32, #gpu.address_space<global>>
    gpu.return
  }
}
// CHECK:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<128x128xf32, #gpu.address_space<global>> -> !xegpu.tensor_desc<32x32xf32, #gpu.address_space<global>>
// CHECK:        xegpu.create_nd_tdesc %{{.*}} : memref<128x128xf32, #gpu.address_space<workgroup>> -> !xegpu.tensor_desc<32x32xf32, #gpu.address_space<workgroup>>
// CHECK:      }
// CHECK:      xegpu.fence
