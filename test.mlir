module {
  func.func @main() {
    "orchestra.dummy_op"() : () -> ()
    func.return
  }
}