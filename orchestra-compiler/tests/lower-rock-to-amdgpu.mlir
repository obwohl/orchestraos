// RUN: %orchestra-opt %s --lower-linalg-to-rock --lower-rock-to-amdgpu="arch=amdgcn-amd-amdhsa" | FileCheck %s

// CHECK: gpu.module @orchestra_gpu_module
// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.matmul
// CHECK-NOT: rock.gemm
// CHECK: gpu.launch_func

func.func @main() -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %A = tensor.empty() : tensor<128x128xf32>
  %B = tensor.empty() : tensor<128x128xf32>
  %C = tensor.empty() : tensor<128x128xf32>
  %D = linalg.fill ins(%c0 : f32) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %E = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%D : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %E : tensor<128x128xf32>
}
