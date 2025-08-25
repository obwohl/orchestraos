// RUN: %orchestra-opt %s -verify-diagnostics

"orchestra.schedule"() ({
  // expected-error@+1 {{'target_arch' dictionary attribute must contain an 'arch' key}}
  orchestra.task ID("test1") target_arch<{device = "cpu"}> region {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()

// -----

"orchestra.schedule"() ({
  // expected-error@+1 {{the 'arch' key in 'target_arch' must be a string attribute}}
  orchestra.task ID("test2") target_arch<{arch = 42}> region {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()
