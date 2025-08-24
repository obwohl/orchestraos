// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// CHECK: error: 'orchestra.task' op requires a non-empty 'target_arch' property, but got ''
"orchestra.schedule"() ({
  "orchestra.task"() <{target = {device = "gpu"}}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
