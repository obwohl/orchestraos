// RUN: %orchestra-opt %s --canonicalize | %FileCheck %s

// -----
// Test Case 1: Basic fusion of two consecutive transfer ops.
// CHECK-LABEL: func @test_basic_fusion
func.func @test_basic_fusion(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: %[[VAL:.*]] = orchestra.transfer %arg0 from @MEM1 to @MEM3
  // CHECK-NOT: orchestra.transfer %{{.*}} from @MEM2
  // CHECK: return %[[VAL]]
  %0 = orchestra.transfer %arg0 from @MEM1 to @MEM2 : tensor<16xf32>
  %1 = orchestra.transfer %0 from @MEM2 to @MEM3 : tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----
// Test Case 2: Negative case - do not fuse if the intermediate transfer has multiple users.
// CHECK-LABEL: func @test_multiple_uses
func.func @test_multiple_uses(%arg0: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) {
  // CHECK: %[[INTERMEDIATE:.*]] = orchestra.transfer %arg0 from @MEM1 to @MEM2
  // CHECK: %[[FINAL:.*]] = orchestra.transfer %[[INTERMEDIATE]] from @MEM2 to @MEM3
  // CHECK: return %[[INTERMEDIATE]], %[[FINAL]]
  %0 = orchestra.transfer %arg0 from @MEM1 to @MEM2 : tensor<16xf32>
  %1 = orchestra.transfer %0 from @MEM2 to @MEM3 : tensor<16xf32>
  return %0, %1 : tensor<16xf32>, tensor<16xf32>
}

// -----
// Test Case 3: Attribute propagation - higher priority should be kept.
// CHECK-LABEL: func @test_attribute_propagation
func.func @test_attribute_propagation(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: orchestra.transfer %arg0 from @MEM1 to @MEM3 {priority = 10 : i32}
  %0 = orchestra.transfer %arg0 from @MEM1 to @MEM2 {priority = 10 : i32} : tensor<16xf32>
  %1 = orchestra.transfer %0 from @MEM2 to @MEM3 {priority = 5 : i32} : tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----
// Test Case 4: Attribute propagation - second op has higher priority.
// CHECK-LABEL: func @test_attribute_propagation_reverse
func.func @test_attribute_propagation_reverse(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: orchestra.transfer %arg0 from @MEM1 to @MEM3 {priority = 20 : i32}
  %0 = orchestra.transfer %arg0 from @MEM1 to @MEM2 {priority = 10 : i32} : tensor<16xf32>
  %1 = orchestra.transfer %0 from @MEM2 to @MEM3 {priority = 20 : i32} : tensor<16xf32>
  return %1 : tensor<16xf32>
}
