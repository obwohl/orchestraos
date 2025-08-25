// RUN: %orchestra-opt %s -verify-diagnostics

// A valid schedule with two tasks.
"orchestra.schedule"() ({
  "orchestra.task"() <{target_arch = {arch = "cpu"}}> ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.task"() <{target_arch = {arch = "gpu"}}> ({
    "orchestra.return"() : () -> ()
  }) : () -> ()
  "orchestra.return"() : () -> ()
}) : () -> ()

// -----

"orchestra.schedule"() ({
  // expected-error@+1 {{only 'orchestra.task' operations are allowed inside a 'orchestra.schedule'}}
  %c0 = "arith.constant"() {value = 0 : i32} : () -> i32
  "orchestra.return"() : () -> ()
}) : () -> ()
