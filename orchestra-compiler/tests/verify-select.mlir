// RUN: ! %orchestra-opt -verify-diagnostics %s

// -----

func.func @test_invalid_mismatched_true_false_count(%cond: i1, %t1: f32, %f1: f32, %f2: f32) {
  // expected-error@+1 {{'orchestra.select' op has mismatched variadic operand sizes}}
  %0 = orchestra.select %cond, %t1, %f1, %f2 {num_true = 1} : (i1, f32, f32, f32) -> f32
  return
}

// -----

func.func @test_invalid_mismatched_true_false_types(%cond: i1, %t1: f32, %f1: i32) {
  // expected-error@+1 {{'orchestra.select' op requires 'true' and 'false' value types to match}}
  %0 = orchestra.select %cond, %t1, %f1 {num_true = 1} : (i1, f32, i32) -> f32
  return
}

// -----

func.func @test_invalid_mismatched_result_count(%cond: i1, %t1: f32, %f1: f32) {
  // expected-error@+1 {{'orchestra.select' op requires number of results to match number of values in each branch}}
  %0, %1 = orchestra.select %cond, %t1, %f1 {num_true = 1} : (i1, f32, f32) -> (f32, f32)
  return
}

// -----

func.func @test_invalid_mismatched_result_types(%cond: i1, %t1: f32, %f1: f32) {
  // expected-error@+1 {{'orchestra.select' op requires result types to match operand types}}
  %0 = orchestra.select %cond, %t1, %f1 {num_true = 1} : (i1, f32, f32) -> i32
  return
}
