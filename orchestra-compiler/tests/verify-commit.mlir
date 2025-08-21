// RUN: %orchestra-opt --verify-diagnostics --split-input-file %s

//===----------------------------------------------------------------------===//
// Valid Cases
//===----------------------------------------------------------------------===//

func.func @test_valid_single_value(%cond: i1, %true: f32, %false: f32) {
  %0 = orchestra.commit %cond, 1 of %true, %false : (i1, f32, f32) -> f32
  return
}

// -----

func.func @test_valid_multiple_values(%cond: i1, %t1: f32, %t2: tensor<4xf32>, %f1: f32, %f2: tensor<4xf32>) {
  %0, %1 = orchestra.commit %cond, 2 of %t1, %t2, %f1, %f2 : (i1, f32, tensor<4xf32>, f32, tensor<4xf32>) -> (f32, tensor<4xf32>)
  return
}

// -----

func.func @test_valid_zero_values(%cond: i1) {
  orchestra.commit %cond, 0 of : (i1) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid Cases
//===----------------------------------------------------------------------===//

// -----

func.func @test_invalid_mismatched_true_false_count(%cond: i1, %t1: f32, %f1: f32, %f2: f32) {
  // expected-error@+1 {{'orchestra.commit' op has mismatched variadic operand sizes}}
  %0 = orchestra.commit %cond, 1 of %t1, %f1, %f2 : (i1, f32, f32, f32) -> f32
  return
}

// -----

func.func @test_invalid_mismatched_true_false_types(%cond: i1, %t1: f32, %f1: i32) {
  // expected-error@+1 {{'orchestra.commit' op requires 'true' and 'false' value types to match}}
  %0 = orchestra.commit %cond, 1 of %t1, %f1 : (i1, f32, i32) -> f32
  return
}

// -----

func.func @test_invalid_mismatched_result_count(%cond: i1, %t1: f32, %f1: f32) {
  // expected-error@+1 {{'orchestra.commit' op requires number of results to match number of values in each branch}}
  %0, %1 = orchestra.commit %cond, 1 of %t1, %f1 : (i1, f32, f32) -> (f32, f32)
  return
}

// -----

func.func @test_invalid_mismatched_result_types(%cond: i1, %t1: f32, %f1: f32) {
  // expected-error@+1 {{'orchestra.commit' op requires result types to match operand types}}
  %0 = orchestra.commit %cond, 1 of %t1, %f1 : (i1, f32, f32) -> i32
  return
}
