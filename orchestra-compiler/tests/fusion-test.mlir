// RUN: %orchestra-opt %s -transform-interpreter -canonicalize | %FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %producer = transform.structured.match ops{["linalg.generic"]} attributes{__producer__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match ops{["linalg.generic"]} attributes{__consumer__} in %arg0 : (!transform.any_op) -> !transform.any_op

    %tiled_consumer, %loop = transform.structured.tile_using_for %consumer tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_producer, %fused_loop = transform.structured.fuse_into_containing_op %producer into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }

  func.func @fuse_generics(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = linalg.generic {
      __producer__,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<256xf32>) outs(%arg1 : tensor<256xf32>) {
    ^bb0(%in: f32, %out_val: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
    } -> tensor<256xf32>

    %1 = linalg.generic {
      __consumer__,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%0 : tensor<256xf32>) outs(%arg1 : tensor<256xf32>) {
    ^bb0(%in: f32, %out_val: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}

// CHECK-LABEL: func.func @fuse_generics
// CHECK:   scf.for
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK-NOT: linalg.generic
