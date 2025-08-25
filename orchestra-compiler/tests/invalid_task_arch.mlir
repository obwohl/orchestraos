// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

"orchestra.schedule"() ({
  // CHECK: error: 'orchestra.task' op requires attribute 'arch'
  "orchestra.task"() ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()
