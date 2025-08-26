// RUN: %orchestra-opt %s --divergence-to-speculation | FileCheck %s

// CHECK-LABEL: func @test_speculate_candidate
// CHECK-NOT: scf.if
// CHECK: %[[THEN_TASK:.*]] = "orchestra.task"() <{target = {arch = "unknown", device_id = 0 : i32}}>
// CHECK: %[[ELSE_TASK:.*]] = "orchestra.task"() <{target = {arch = "unknown", device_id = 0 : i32}}>
// CHECK: "orchestra.select"(%arg0, %[[THEN_TASK]], %[[ELSE_TASK]])
func.func @test_speculate_candidate(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  %0 = scf.if %arg0 -> (f32) {
    %1 = arith.addf %arg1, %arg1 : f32
    scf.yield %1 : f32
  } else {
    %2 = arith.mulf %arg2, %arg2 : f32
    scf.yield %2 : f32
  }
  return %0 : f32
}
