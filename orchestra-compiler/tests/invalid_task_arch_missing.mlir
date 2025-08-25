// RUN: %orchestra-opt %s -verify-diagnostics

"orchestra.schedule"() ({
  // expected-error@+1 {{expected 'target_arch'}}
  orchestra.task ID("test") region {
    "orchestra.yield"() : () -> ()
  }
  "orchestra.yield"() : () -> ()
}) : () -> ()
