// Test that the barrier op is parsed correctly.
// RUN: %orchestra-opt %s | %FileCheck %s --check-prefix=CHECK-PARSING

// Test that the barrier op is erased by the lowering pass.
// RUN: %orchestra-opt %s --lower-orchestra-to-standard | %FileCheck %s --check-prefix=CHECK-LOWERING

// CHECK-PARSING-LABEL: func.func @test_barrier()
// CHECK-PARSING:         orchestra.barrier
// CHECK-PARSING:         return

// CHECK-LOWERING-LABEL: func.func @test_barrier()
// CHECK-LOWERING-NOT:     orchestra.barrier
// CHECK-LOWERING:         return

func.func @test_barrier() {
  orchestra.barrier
  return
}
