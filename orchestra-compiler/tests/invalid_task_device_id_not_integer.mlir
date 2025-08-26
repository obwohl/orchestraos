// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

"orchestra.schedule"() ({
  // CHECK: error: 'orchestra.task' op 'device_id' key in 'target' dictionary must be an integer
  "orchestra.task"() <{target = {arch = "cpu", device_id = "not_an_integer"}}> ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()
