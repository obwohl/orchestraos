// RUN: ! %orchestra-opt -verify-diagnostics %s

func.func @test_invalid_commit_operand(%arg0: tensor<4xf32>) {
  // expected-error @+1 {{'orchestra.commit' op operand must be MemRef-type, but got 'tensor<4xf32>'}}
  %0 = orchestra.commit %arg0 : tensor<4xf32>
  return
}

func.func @test_invalid_commit_result(%arg0: memref<4xf32>) -> tensor<4xf32> {
    // expected-error @+1 {{'orchestra.commit' op result must be MemRef-type, but got 'tensor<4xf32>'}}
    %0 = orchestra.commit %arg0 : memref<4xf32>
    return %0
}
