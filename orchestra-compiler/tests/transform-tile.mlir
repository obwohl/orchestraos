// RUN: %orchestra-opt %s -transform-interpreter -canonicalize | %FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">

    %generic = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.op<"func.func">) -> !transform.op<"linalg.generic">

    %tiled, %loops = transform.structured.tile_using_for %generic tile_sizes [4]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.op<"scf.for">)

    transform.yield
  }
}

// CHECK-LABEL: func @tile_generic
// CHECK: scf.for
// CHECK: linalg.generic
func.func @tile_generic(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<10xf32>) outs(%arg1 : tensor<10xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
