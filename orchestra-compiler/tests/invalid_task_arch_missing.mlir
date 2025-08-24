// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// CHECK: error: 'orchestra.task' op requires 'target' attribute to have an 'arch' key
"orchestra.schedule"() ({
  "orchestra.task"() <{target = {device = "gpu"}}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
