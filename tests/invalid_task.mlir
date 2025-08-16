// RUN: not orchestra-opt %s --verify-diagnostics 2>&1 | FileCheck %s
// CHECK: error: 'orchestra.task' op requires attribute 'target'
"orchestra.schedule"() ({
  "orchestra.task"() ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
