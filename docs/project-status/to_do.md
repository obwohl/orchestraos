# OrchestraOS Compiler Task List

This document outlines the engineering tasks required to evolve the OrchestraOS compiler.

## Milestone 1: Core Dialect Refactoring and Cleanup

This milestone focuses on solidifying the semantics of the core Orchestra dialect and cleaning up the codebase to provide a stable foundation for future work.

-   [x] **Task 1.1: Refactor `orchestra.commit` Operation**
    -   Redefine `orchestra.commit` as a pure scheduling barrier and memory coherency token.
    -   Update the TableGen definition, C++ implementation, and verifier.
    -   Remove obsolete tests and lowering patterns related to the old `commit` op.

-   [x] **Task 1.2: Introduce `orchestra.select` for Conditional Selection**
    -   Define a new `orchestra.select` operation to handle polymorphic, variadic conditional selection.
    -   Implement the TableGen definition, C++ class, and verifier.
    -   Add comprehensive tests for the new operation.

-   [x] **Task 1.3: Restore Speculative Execution Pass**
    -   Refactor the `divergence-to-speculation` pass to use the new `orchestra.select` operation.
    -   Re-enable the pass and its corresponding test.

-   [x] **Task 1.4: Document Core Dialect Semantics**
    -   Create a new documentation file (`docs/architecture/dialect_semantics.md`) to provide a detailed rationale for the design of `orchestra.commit` and `orchestra.select`.
    -   Update README files to link to the new documentation.

## Milestone 2: Foundational Architectural Enhancements

These tasks modernize the compiler's core infrastructure to support multi-vendor targets and align with current MLIR best practices.

-   [ ] **Task 2.1: Finalize `orchestra.task` Target Schema.**
    -   Define and enforce a formal schema for the `target` property on `orchestra.task`.
    -   The schema (within the `DictionaryAttr` property) must include a mandatory `arch` key (e.g., "nvidia_blackwell", "amd_cdna3") and allow optional, target-specific keys.

-   [ ] **Task 2.2: Complete Migration to MLIR Properties System.**
    -   The `Orchestra` dialect should use the MLIR `Properties` system for improved performance and type safety.
    -   [x] **1.2.1:** Enable `usePropertiesForAttributes = 1;` for the `Orchestra` dialect in `OrchestraOps.td`.
    -   [x] **1.2.2:** Migrate the `target` attribute of `orchestra.task` to `Properties`.
    -   [ ] **1.2.3:** Investigate migrating attributes of other core operations (e.g., `orchestra.transfer`, `orchestra.schedule`) to the `Properties` system where appropriate.

## Milestone 3: AMD GPU Support (rocMLIR Integration)

This milestone focuses on adding end-to-end support for AMD Instinct-series GPUs.

-   [ ] **Task 3.1: Implement `linalg-to-rock` Lowering.**
    -   Create a new pass to lower `linalg.generic` operations to the `rock` dialect, which is the entry point for the `rocMLIR` kernel generator.
    -   This pass must be guided by the "contract" of the `rocMLIR` tool to produce optimal `linalg` patterns.

-   [ ] **Task 3.2: Implement `rock-to-amdgpu` and `rocdl` Lowering.**
    -   Develop the pipeline to lower the `rock` dialect to the `amdgpu` and `rocdl` dialects.
    -   Ensure correct mapping to hardware primitives like `amdgpu.mfma` for matrix operations.
    -   Handle synchronization with `gpu.barrier`.

-   [ ] **Task 3.3: Create AMD-specific Transform Script.**
    -   Author `amd_instinct_cdna3_strategy.mlir` for the `transform` dialect interpreter.
    -   This script will apply tiling, packing, and fusion patterns at the `linalg` level, co-designed with the `rocMLIR` lowering contract to generate high-performance kernels.

## Milestone 4: Google TPU & AWS Trainium Support (StableHLO Bridge)

This milestone focuses on integrating with Google and AWS compilers via the StableHLO standard.

-   [ ] **Task 4.1: Implement `linalg-to-stablehlo` Lowering Pass.**
    -   Create a robust pass to convert `linalg` dialect operations into their `stablehlo` equivalents.
    -   This pass will be the common backend for both Google and AWS targets.

-   [ ] **Task 4.2: Implement the "StableHLO Bridge" Mechanism.**
    -   Integrate the external `stablehlo-translate` tool into the build and execution pipeline.
    -   The compiler driver will invoke this tool as a subprocess to serialize the textual `stablehlo` IR into a versioned, portable bytecode artifact.
    -   This artifact is the handoff point to the vendor compilers (`xla_compile`, `neuronx-cc`).

-   [ ] **Task 4.3: Create Google/AWS-specific Transform Scripts.**
    -   Author `google_tpu_strategy.mlir` and `aws_trainium_strategy.mlir`.
    -   These scripts will perform high-level `linalg` optimizations and canonicalizations to simplify the graph before lowering to `stablehlo`.

## Milestone 5: Advanced AWS Integration (Neuron Kernel Interface)

This milestone adds expert-level performance tuning capabilities for AWS Trainium.

-   [ ] **Task 5.1: Extend `orchestra.task` for NKI.**
    -   Add an optional `nki_source` string attribute to the `orchestra.task` operation to hold custom Neuron Kernel Interface source code.

-   [ ] **Task 5.2: Implement `orchestra-lower-nki-tasks` Pass.**
    -   This pass will identify tasks with `nki_source`, invoke the NKI compiler offline, and link the resulting custom kernel into the final executable.
