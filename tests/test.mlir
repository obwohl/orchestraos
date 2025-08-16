// RUN: orchestra-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_commit
func.func @test_commit(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK: "orchestra.schedule"
  %0 = "orchestra.schedule"() ({
    %true_val = "arith.constant"() {value = 1.0 : f32} : () -> f32
    %false_val = "arith.constant"() {value = 0.0 : f32} : () -> f32
    %res = "orchestra.commit"(%arg0, %true_val, %false_val) : (i1, f32, f32) -> f32
    "orchestra.yield"(%res) : (f32) -> ()
  }) : () -> f32
  return %0 : f32
}
