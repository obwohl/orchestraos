// RUN: %orchestra-opt %s -verify-diagnostics

// A valid schedule with two tasks.
"orchestra.schedule"() ({
  orchestra.task ID("task1") target_arch<{arch = "cpu"}> region {
    "orchestra.yield"() : () -> ()
  }
  orchestra.task ID("task2") target_arch<{arch = "gpu"}> region {
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
