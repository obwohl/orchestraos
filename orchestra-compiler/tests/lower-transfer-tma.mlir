// RUN: %orchestra-opt %s -lower-orchestra-to-gpu="gpu-arch=nvgpu" | %FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @tma_test_module attributes {sm_arch = 100} {
    gpu.func @main() kernel {
      // CHECK: gpu.func @main
      // CHECK:   [[MBARRIER:%.+]] = nvgpu.mbarrier.create
      // CHECK:   nvgpu.mbarrier.init [[MBARRIER]]
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.0 : f32
      %0 = memref.alloc() : memref<16x16xf32>
      %1 = orchestra.transfer %0 from @DRAM to @L2 : memref<16x16xf32>
      // CHECK:   [[DESC:%.+]] = nvgpu.tma.create.descriptor
      // CHECK:   [[ALLOC:%.+]] = memref.alloc
      // CHECK:   nvgpu.tma.async.load {{.+}} to [[ALLOC]]
      // CHECK:   [[TOKEN:%.+]] = nvgpu.mbarrier.arrive
      %2 = memref.load %1[%c0, %c0] : memref<16x16xf32>
      // CHECK:   scf.while
      // CHECK:     nvgpu.mbarrier.test.wait
      // CHECK:   }
      // CHECK:   memref.load [[ALLOC]]
      gpu.return
    }
  }
}
