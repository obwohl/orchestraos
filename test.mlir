module {
  func.func @test_orchestra_op() {
    "orchestra.my_op"() : () -> ()
    return
  }
}
