// RUN: %orchestra-opt %s | %orchestra-opt | FileCheck %s
// CHECK-LABEL: func @select_test
func.func @select_test(%cond: i1, %true_val: f32, %false_val: f32) -> f32 {
  // CHECK: %{{.*}} = "orchestra.select"(%{{.*}}, %{{.*}}, %{{.*}}) <{num_true = 0 : i32}> {num_true = 1 : i32} : (i1, f32, f32) -> f32
  %0 = "orchestra.select"(%cond, %true_val, %false_val) {num_true = 1 : i32} : (i1, f32, f32) -> f32
  return %0 : f32
}
