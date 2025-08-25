// RUN: %orchestra-opt %s -lower-orchestra-to-standard | %FileCheck %s

func.func @test_commit(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK-LABEL: @test_commit
  // CHECK:      {{.*}} = arith.select %arg0, %arg1, %arg2 : f32
  %0 = orchestra.commit %arg0 true(%arg1) false(%arg2) : (i1, f32, f32) -> f32
  return %0 : f32
}
