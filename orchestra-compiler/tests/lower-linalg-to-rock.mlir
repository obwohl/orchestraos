// RUN: /app/orchestra-compiler/build/orchestra-opt/orchestra-opt %s -lower-linalg-to-rock | FileCheck %s

func.func @test_gemm(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_gemm
// CHECK: rock.gemm
// CHECK-NOT: linalg.matmul
