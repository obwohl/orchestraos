// RUN: %orchestra-opt %s -verify-diagnostics

// A valid schedule with two tasks.
orchestra.schedule {
  orchestra.task target = #orchestra.target<arch = "cpu", device_id = 0> {
    orchestra.return
  }
  orchestra.task target = #orchestra.target<arch = "gpu", device_id = 1> {
    orchestra.return
  }
  orchestra.return
}

// -----

"orchestra.schedule"() ({
  // expected-error@+1 {{only 'orchestra.task' operations are allowed inside a 'orchestra.schedule'}}
  %c0 = "arith.constant"() {value = 0 : i32} : () -> i32
  "orchestra.return"() : () -> ()
}) : () -> ()
