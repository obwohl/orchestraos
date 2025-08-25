// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s
// CHECK: error: 'orchestra.task' op requires a non-empty 'task_id' property
"orchestra.schedule"() ({
  orchestra.task ID("") target_arch<{arch = "cpu"}> region {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()
