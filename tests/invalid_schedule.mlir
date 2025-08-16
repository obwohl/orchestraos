// RUN: not orchestra-opt %s --verify-diagnostics 2>&1 | FileCheck %s
// CHECK: error: 'orchestra.schedule' op must be a top-level operation
func.func @test_invalid_schedule() {
  "orchestra.schedule"() ({
    "orchestra.yield"() : () -> ()
  }) : () -> ()
  return
}
