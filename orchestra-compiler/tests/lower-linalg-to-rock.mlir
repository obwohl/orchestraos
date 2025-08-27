// RUN: orchestra-opt %s -lower-linalg-to-rock | FileCheck %s

#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @test_gemm
func.func @test_gemm(%arg0: tensor<4x8xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // CHECK: %0 = rock.gemm(%arg0, %arg1, %arg2) {aTransposed = false, bTransposed = true, cTransposed = false, storeMethod = "set"} : (tensor<4x8xf32>, tensor<16x8xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  %0 = linalg.generic {
    indexing_maps = [#map_a, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x16xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %mul = arith.mulf %in1, %in2 : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
