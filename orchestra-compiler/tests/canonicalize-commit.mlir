// RUN: %orchestra-opt %s -canonicalize | %FileCheck %s

// CHECK-LABEL: func @test_fold_commit_true
// CHECK-SAME:    ([[ARG:%.+]]: i32)
// CHECK:         return [[ARG]] : i32
func.func @test_fold_commit_true(%arg0: i32) -> i32 {
  %true = arith.constant true
  %c42 = arith.constant 42 : i32
  %res = "orchestra.commit"(%true, %arg0, %c42) <{num_true = 1 : i32}> : (i1, i32, i32) -> i32
  return %res : i32
}

// CHECK-LABEL: func @test_fold_commit_false
// CHECK-SAME:    ([[ARG:%.+]]: i32)
// CHECK:         %[[CST:.*]] = arith.constant 42 : i32
// CHECK:         return %[[CST]] : i32
func.func @test_fold_commit_false(%arg0: i32) -> i32 {
  %false = arith.constant false
  %c42 = arith.constant 42 : i32
  %res = "orchestra.commit"(%false, %arg0, %c42) <{num_true = 1 : i32}> : (i1, i32, i32) -> i32
  return %res : i32
}
