func.func @mfma_test(%a: vector<64xf32>, %b: vector<64xf32>, %c: vector<32x32xf32>) -> vector<32x32xf32> {
  %d = "amdgpu.mfma"(%a, %b, %c) {m = 32 : i32, n = 32 : i32, k = 2 : i32, blocks = 1 : i32} : (vector<64xf32>, vector<64xf32>, vector<32x32xf32>) -> vector<32x32xf32>
  return %d : vector<32x32xf32>
}
