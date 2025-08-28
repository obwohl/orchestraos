// RUN: %orchestra-opt --lower-rock-to-amdgpu %s | FileCheck %s

gpu.module @test_module {
  gpu.func @test_gemm(%arg0: vector<128xf32>, %arg1: vector<128xf32>) -> vector<1024xf32> {
    %c0 = arith.constant 0.0 : f32
    %c = arith.constant dense<0.0> : vector<1024xf32>

    // CHECK-LABEL: @test_gemm
    // CHECK: amdgpu.mfma
    %0 = rock.gemm "gfx90a", %arg0, %arg1 : (vector<128xf32>, vector<128xf32>) -> vector<1024xf32>

    gpu.return %0 : vector<1024xf32>
  }
}
