// RUN: %orchestra-opt %s --split-input-file --verify-diagnostics

// Test that an empty 'from' attribute is caught by the verifier.
module {
  func.func @test_empty_from(%arg0: tensor<16xf32>) {
    // expected-error@+1 {{'orchestra.transfer' op requires a non-empty 'from' attribute}}
    %0 = orchestra.transfer %arg0 from @"" to @MEM : tensor<16xf32>
    return
  }
}

// -----

// Test that an empty 'to' attribute is caught by the verifier.
module {
  func.func @test_empty_to(%arg0: tensor<16xf32>) {
    // expected-error@+1 {{'orchestra.transfer' op requires a non-empty 'to' attribute}}
    %0 = orchestra.transfer %arg0 from @MEM to @"" : tensor<16xf32>
    return
  }
}
