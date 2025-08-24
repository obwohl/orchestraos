// RUN: %orchestra-opt %s -verify-diagnostics

// A valid schedule with two tasks.
"orchestra.schedule"() ({
  orchestra.task "task1" on "cpu" {} : () -> () {
    "orchestra.yield"() : () -> ()
  }
  orchestra.task "task2" on "gpu" {} : () -> () {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()

// -----

"orchestra.schedule"() ({
  // expected-error@+1 {{only 'orchestra.task' operations are allowed inside a 'orchestra.schedule'}}
  %c0 = "arith.constant"() {value = 0 : i32} : () -> i32
  "orchestra.yield"() : () -> ()
}) : () -> ()
