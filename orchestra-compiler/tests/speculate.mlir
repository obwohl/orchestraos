// RUN: %orchestra-opt %s --divergence-to-speculation | %FileCheck %s

// CHECK-LABEL: func @test_speculate_candidate
// CHECK-SAME:      (%[[COND:.*]]: i1, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) -> f32 {
// CHECK: %[[THEN_TASK:.*]] = "orchestra.task"() <{target_arch = {arch = "unknown"}}> ({
// CHECK-NEXT:    ^bb0(%[[ARG1_IN_REGION:.*]]: f32):
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[ARG1_IN_REGION]], %[[ARG1_IN_REGION]]
// CHECK-NEXT:      orchestra.return
// CHECK-NEXT:    }) : () -> f32
// CHECK: %[[ELSE_TASK:.*]] = "orchestra.task"() <{target_arch = {arch = "unknown"}}> ({
// CHECK-NEXT:    ^bb0(%[[ARG2_IN_REGION:.*]]: f32):
// CHECK-NEXT:      %[[MUL:.*]] = arith.mulf %[[ARG2_IN_REGION]], %[[ARG2_IN_REGION]]
// CHECK-NEXT:      orchestra.return
// CHECK-NEXT:    }) : () -> f32
// CHECK: %[[COMMIT:.*]] = "orchestra.commit"(%[[COND]], %[[THEN_TASK]], %[[ELSE_TASK]]) <{num_true = 1 : i32}> : (i1, f32, f32) -> f32
// CHECK: return %[[COMMIT]]
// CHECK: }
func.func @test_speculate_candidate(%cond: i1, %arg1: f32, %arg2: f32) -> f32 {
  %result = scf.if %cond -> (f32) {
    %add = arith.addf %arg1, %arg1 : f32
    scf.yield %add : f32
  } else {
    %mul = arith.mulf %arg2, %arg2 : f32
    scf.yield %mul : f32
  }
  return %result : f32
}

// -----

// CHECK-LABEL: func @test_no_speculate_side_effects
// CHECK:         scf.if
func.func @test_no_speculate_side_effects(%cond: i1, %arg1: f32, %mem: memref<f32>) {
  scf.if %cond {
    memref.store %arg1, %mem[] : memref<f32>
    scf.yield
  } else {
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: func @test_no_speculate_no_else
// CHECK-NOT:     scf.if
func.func @test_no_speculate_no_else(%cond: i1) {
  scf.if %cond {
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: func @test_no_speculate_no_results
// CHECK-NOT:     scf.if
func.func @test_no_speculate_no_results(%cond: i1) {
  scf.if %cond {
    scf.yield
  } else {
    scf.yield
  }
  return
}
