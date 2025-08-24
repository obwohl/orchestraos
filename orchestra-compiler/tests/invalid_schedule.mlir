// RUN: not %orchestra-opt %s 2>&1 | %FileCheck %s

// CHECK: error: 'orchestra.task' op has a duplicate task_id 'task1'
"orchestra.schedule"() ({
  orchestra.task "task1" on "cpu" {} : () -> () {
    "orchestra.yield"() : () -> ()
  }
  orchestra.task "task1" on "cpu" {} : () -> () {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()
