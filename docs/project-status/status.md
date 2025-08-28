# OrchestraOS Compiler: Development Status & Roadmap

**Last Updated:** August 28, 2025

## 1. High-Level Summary

The OrchestraOS compiler has a stable, buildable, and verifiable foundation. The core `OrchestraIR` dialect is functional, and the project's test suite passes against the required LLVM 20 environment. The compiler is now entering a new phase of development focused on implementing the multi-vendor hardware support strategy outlined in the architectural blueprint, with a clear roadmap for adding comprehensive support for AMD, Google, and AWS accelerators.

## 2. Current Feature Status

This section provides a verified, at-a-glance view of implemented features, aligned with the architectural components from the project blueprint.

*   **✅ Build & Test Environment**
    *   The project builds successfully using CMake/Ninja against LLVM 20.
    *   The full `lit` test suite (`check-orchestra`) passes (14/14 tests).

*   **✅ Core Dialect: `OrchestraIR`**
    *   ✅ **`orchestra.schedule`**: Implemented with a verifier that checks for valid child operations.
    *   ✅ **`orchestra.task`**: Modernized with a C++ helper class (`OrchestraTarget`) that provides a type-safe API for the `target` attribute. The verifier now uses this class to enforce the schema.
    *   ✅ **`orchestra.transfer`**: Implemented with canonicalization patterns and a verifier for its attributes.
    *   ✅ **`orchestra.commit`**: Implemented for conditional selection.
    *   ✅ **`orchestra.yield`**: The dialect's terminator is now consistently named `yield`.
    *   ✅ **`orchestra.barrier`**: A new barrier synchronization operation has been added.
    *   ❌ **MLIR Properties Migration**: The planned migration to the MLIR Property system for `orchestra.task` was blocked by a persistent TableGen bug in MLIR 20.1.8. The current C++ helper class is a robust workaround.

*   **✅ Transformation & Optimization Framework**
    *   ✅ **Speculative Execution**: The `--divergence-to-speculation` pass is enabled and correctly lowers `scf.if` to `orchestra.task` and `orchestra.commit`. It is **not** disabled.
    *   ✅ **Declarative Optimization**: The `-transform-interpreter` is integrated. A test for producer-consumer fusion (`fusion-test.mlir`) is working.

*   **🟡 Backend Lowering Paths**
    *   ✅ **NVIDIA GPU Lowering**: Supports asynchronous data movement for both Blackwell (TMA) and older architectures (Hopper).
    *   🟡 **AMD GPU Lowering**: Initial support via `--gpu-arch=rocdl` is present for data movement. Matrix acceleration lowering is not yet implemented.
    *   ❌ **Intel GPU Lowering**: The `xegpu` path is incomplete and its tests are disabled.
    *   ❌ **StableHLO Bridge**: The lowering path to StableHLO and the bridge mechanism for Google/AWS are not yet implemented.
    *   ❌ **Feedback-Driven Optimization (FDO)**: The entire FDO loop is not implemented.

## 3. Development Roadmap

This section serves as the new, verified to-do list for the project.

*   **Milestone 1: Foundational Architectural Enhancements**
    *   ✅ **Task 1.1: Modernize `orchestra.task` and Normalize Terminators.** The `orchestra.task` operation has been modernized with a type-safe C++ helper class for its `target` attribute, and `orchestra.return` has been renamed to `orchestra.yield` for consistency.
    *   ❌ **Task 1.2: Migrate to MLIR Properties System.** This task is now blocked pending a future upgrade of the MLIR dependency. The `Property` system in MLIR 20.1.8 has a code generation bug that prevents its use for `orchestra.task`. The C++ helper class serves as a robust workaround.

*   **Milestone 2: AMD GPU Support (rocMLIR Integration)**
    *   [x] **Task 2.0: Implement `rock` Dialect Scaffolding.** The initial `rock` dialect has been implemented and integrated into the build system. This includes the definition of a `rock.gemm` operation that uses the MLIR property system for its attributes. This foundational work unblocks further development on the AMD GPU lowering path.
    *   [x] **Task 2.1: Implement `linalg-to-rock` Lowering.** Create a new pass to lower `linalg.generic` operations to the `rock` dialect, guided by the `rocMLIR` kernel generator "contract".
    *   [x] **Task 2.2: Implement `rock-to-amdgpu` and `rocdl` Lowering.** A pass to lower the `rock` dialect has been implemented. It currently provides a placeholder lowering for `rock.gemm` to `vector.fma` to enable testing and verification of the pass structure. A full implementation targeting `amdgpu.mfma` intrinsics with tiling is a future task.
    *   [ ] **Task 2.3: Create AMD-specific Transform Script.** Author `amd_instinct_cdna3_strategy.mlir` to apply `linalg`-level optimizations co-designed with the `rocMLIR` lowering contract.

*   **Milestone 3: Google TPU & AWS Trainium Support (StableHLO Bridge)**
    *   [ ] **Task 3.1: Implement `linalg-to-stablehlo` Lowering Pass.** Create a robust pass to convert `linalg` dialect operations into their `stablehlo` equivalents.
    *   [ ] **Task 3.2: Implement the "StableHLO Bridge" Mechanism.** Integrate the external `stablehlo-translate` tool into the build and execution pipeline to serialize `stablehlo` IR into a portable bytecode artifact.
    *   [ ] **Task 3.3: Create Google/AWS-specific Transform Scripts.** Author `google_tpu_strategy.mlir` and `aws_trainium_strategy.mlir` to perform high-level `linalg` optimizations before handoff.

*   **Milestone 4: Advanced AWS Integration (Neuron Kernel Interface)**
    *   [ ] **Task 4.1: Extend `orchestra.task` for NKI.** Add an optional `nki_source` string attribute to the `orchestra.task` operation to hold custom Neuron Kernel Interface source code.
    *   [ ] **Task 4.2: Implement `orchestra-lower-nki-tasks` Pass.** This pass will identify tasks with `nki_source`, invoke the NKI compiler, and link the resulting custom kernel.
