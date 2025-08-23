// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s
// CHECK: error: 'orchestra.task' op requires 'arch' key in 'target' attribute to be a StringAttr
"orchestra.schedule"() ({
  "orchestra.task"() <{target = {arch = 42}}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()
