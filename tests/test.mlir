// RUN: %orchestra-opt %s | FileCheck %s
// CHECK: "orchestra.my_op"

module {
  func.func @test_orchestra_op() {
    "orchestra.my_op"() : () -> ()
    return
  }
}
