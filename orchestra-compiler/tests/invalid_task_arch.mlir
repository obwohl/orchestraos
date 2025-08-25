// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

"orchestra.schedule"() ({
  // CHECK: error: 'orchestra.task' op requires attribute 'target_arch'
  "orchestra.task"() ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
