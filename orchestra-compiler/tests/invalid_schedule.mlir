// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

"orchestra.schedule"() ({
  // CHECK: error: 'orchestra.task' op requires a non-empty 'arch' property
  "orchestra.task"() <{arch = ""}> ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()
