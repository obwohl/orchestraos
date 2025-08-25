// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// CHECK: error: 'orchestra.task' op requires a 'target' property
"orchestra.schedule"() ({
  "orchestra.task"() ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()
