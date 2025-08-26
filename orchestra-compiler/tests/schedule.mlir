// RUN: %orchestra-opt %s -verify-diagnostics

// A valid schedule with two tasks.
"orchestra.schedule"() ({
  "orchestra.task"() <{target = {arch = "cpu", device_id = 0}}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.task"() <{target = {arch = "gpu", device_id = 1}}> ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  "orchestra.yield"() : () -> ()
}) : () -> ()

// -----

"orchestra.schedule"() ({
  // expected-error@+1 {{only 'orchestra.task' and 'orchestra.yield' operations are allowed inside a 'orchestra.schedule'}}
  %c0 = "arith.constant"() {value = 0 : i32} : () -> i32
  "orchestra.yield"() : () -> ()
}) : () -> ()
