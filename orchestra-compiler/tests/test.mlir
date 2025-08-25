// RUN: %orchestra-opt %s | %FileCheck %s

// CHECK-LABEL: "orchestra.schedule"
"orchestra.schedule"() ({
  // CHECK: orchestra.task ID("test") target_arch<{arch = "test"}>
  orchestra.task ID("test") target_arch<{arch = "test"}> region {
    // CHECK: %[[COND:.*]] = arith.constant true
    %cond = arith.constant true
    // CHECK: %[[TRUE:.*]] = arith.constant 1.000000e+00 : f32
    %true_val = arith.constant 1.0 : f32
    // CHECK: %[[FALSE:.*]] = arith.constant 0.000000e+00 : f32
    %false_val = arith.constant 0.0 : f32
    // CHECK: %{{.*}} = orchestra.commit %[[COND]], 1 of %[[TRUE]], %[[FALSE]]
    %res = orchestra.commit %cond, 1 of %true_val, %false_val : (i1, f32, f32) -> f32
    "orchestra.yield"() : () -> ()
  }
  // CHECK: "orchestra.yield"()
  "orchestra.yield"() : () -> ()
}) : () -> ()
