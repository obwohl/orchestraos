# OrchestraOS Compiler: Development Status & Roadmap

**Last Updated:** August 25, 2025

## 1. High-Level Summary

The OrchestraOS compiler has a stable, buildable, and verifiable foundation. The core `OrchestraIR` dialect is functional, and the project's test suite passes against the required LLVM 20 environment. The compiler is now entering a new phase of development focused on implementing the multi-vendor hardware support strategy outlined in the architectural blueprint, with a clear roadmap for adding comprehensive support for AMD, Google, and AWS accelerators.

## 2. Current Feature Status

This section provides a verified, at-a-glance view of implemented features, aligned with the architectural components from the project blueprint.

*   **✅ Build & Test Environment**
    *   The project builds successfully using CMake/Ninja against LLVM 20.
    *   The full `lit` test suite (`check-orchestra`) passes (16/16 tests).

*   **✅ Core Dialect: `OrchestraIR`**
    *   ✅ **`orchestra.schedule`**: Implemented with a verifier for unique task IDs.
    *   ✅ **`orchestra.task`**: The `target` attribute schema is now finalized and enforced by a verifier.
    *   ✅ **`orchestra.transfer`**: Implemented with canonicalization patterns and a verifier for its attributes.
    *   ✅ **`orchestra.commit`**: Implemented as a `MemRef`-to-`MemRef` op representing data commitment, not as a simple token.
    *   ✅ **`orchestra.select`**: Implemented for conditional selection, replacing the old functionality of `orchestra.commit`.
    *   🟡 **MLIR Properties Migration**: In progress. The dialect-wide `usePropertiesForAttributes` flag has been enabled, automatically migrating simple inherent attributes (like on `orchestra.transfer`) to property-based storage. Complex attributes (like the `DictionaryAttr` on `orchestra.task`) require custom attribute classes for structured access and are being handled separately.

*   **✅ Transformation & Optimization Framework**
    *   ✅ **Speculative Execution**: The `--divergence-to-speculation` pass is enabled and correctly lowers `scf.if` to `orchestra.task` and `orchestra.select`. It is **not** disabled.
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
    *   ✅ **Task 1.1: Finalize `orchestra.task` Target Schema.** Define and enforce a formal schema for the `target` property on `orchestra.task`, including a mandatory `arch` key and optional, target-specific keys.
    *   🟡 **Task 1.2: Migrate to MLIR Properties System.**
        *   ✅ **Task 1.2.1: Enable Dialect-wide Property Storage.** The `usePropertiesForAttributes` flag is enabled, transparently converting storage for all inherent attributes.
        *   ✅ **Task 1.2.2: Add Verifier for `orchestra.transfer`**. A verifier was added to ensure the correctness of its attributes, which now use property storage.
        *   [ ] **Task 1.2.3: Implement Custom Attribute Class for `orchestra.task`**. The `target` `DictionaryAttr` requires a custom attribute class for type-safe, structured access.

*   **Milestone 2: AMD GPU Support (rocMLIR Integration)**
    *   [ ] **Task 2.1: Implement `linalg-to-rock` Lowering.** Create a new pass to lower `linalg.generic` operations to the `rock` dialect, guided by the `rocMLIR` kernel generator "contract".
    *   [ ] **Task 2.2: Implement `rock-to-amdgpu` and `rocdl` Lowering.** Develop the pipeline to lower the `rock` dialect to `amdgpu` and `rocdl`, including mappings for matrix acceleration primitives (`amdgpu.mfma`).
    *   [ ] **Task 2.3: Create AMD-specific Transform Script.** Author `amd_instinct_cdna3_strategy.mlir` to apply `linalg`-level optimizations co-designed with the `rocMLIR` lowering contract.

*   **Milestone 3: Google TPU & AWS Trainium Support (StableHLO Bridge)**
    *   [ ] **Task 3.1: Implement `linalg-to-stablehlo` Lowering Pass.** Create a robust pass to convert `linalg` dialect operations into their `stablehlo` equivalents.
    *   [ ] **Task 3.2: Implement the "StableHLO Bridge" Mechanism.** Integrate the external `stablehlo-translate` tool into the build and execution pipeline to serialize `stablehlo` IR into a portable bytecode artifact.
    *   [ ] **Task 3.3: Create Google/AWS-specific Transform Scripts.** Author `google_tpu_strategy.mlir` and `aws_trainium_strategy.mlir` to perform high-level `linalg` optimizations before handoff.

*   **Milestone 4: Advanced AWS Integration (Neuron Kernel Interface)**
    *   [ ] **Task 4.1: Extend `orchestra.task` for NKI.** Add an optional `nki_source` string attribute to the `orchestra.task` operation to hold custom Neuron Kernel Interface source code.
    *   [ ] **Task 4.2: Implement `orchestra-lower-nki-tasks` Pass.** This pass will identify tasks with `nki_source`, invoke the NKI compiler, and link the resulting custom kernel.
