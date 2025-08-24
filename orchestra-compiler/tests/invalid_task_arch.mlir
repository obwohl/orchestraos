// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s
// CHECK: error: invalid properties {target_arch = 42 : i64} for op orchestra.task: expected string property to come from string attribute
"orchestra.schedule"() ({
  "orchestra.task"() <{target_arch = 42}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
