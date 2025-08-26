// RUN: orchestra-opt --lower-orchestra-to-gpu --gpu-arch=rocdl %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x16xf32>, %input_conv: tensor<1x16x16x3xf32>, %filter_conv: tensor<3x3x3x8xf32>, %output_conv: tensor<1x14x14x8xf32>) {

    // CHECK-LABEL: gpu.module @orchestra_module
    orchestra.schedule {
      // CHECK: gpu.func @task_matmul
      // CHECK: rock.gemm
      %r0 = orchestra.task -> (tensor<4x16xf32>) target = {arch = "rocdl", device_id = 0} {
        %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<4x16xf32>) -> tensor<4x16xf32>
        orchestra.yield %0 : tensor<4x16xf32>
      }

      // CHECK: gpu.func @task_conv
      // CHECK: rock.conv
      %r1 = orchestra.task -> (tensor<1x14x14x8xf32>) target = {arch = "rocdl", device_id = 0} {
        %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
             ins(%input_conv, %filter_conv : tensor<1x16x16x3xf32>, tensor<3x3x3x8xf32>)
             outs(%output_conv : tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>
        orchestra.yield %1 : tensor<1x14x14x8xf32>
      }
    }
    return
  }
}
