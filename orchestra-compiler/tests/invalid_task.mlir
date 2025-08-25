// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

"orchestra.schedule"() ({
  // CHECK: error: 'orchestra.task' op requires a non-empty 'arch' in 'target_arch'
  "orchestra.task"() <{target_arch = {}}> ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()
