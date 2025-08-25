# OrchestraOS Compiler Task List

This document outlines the engineering tasks required to evolve the OrchestraOS compiler into a multi-vendor, heterogeneous orchestration platform. The tasks are derived from the strategic blueprints in `docs/architecture`.

## Milestone 1: Foundational Architectural Enhancements

These tasks modernize the compiler's core infrastructure to support multi-vendor targets.

-   [ ] **Task 1.1: Enhance `orchestra.task` Schema.**
    -   Modify the `orchestra.task` operation to use a `DictionaryAttr` for its `target` attribute.
    -   Enforce a schema with a mandatory `arch` key (e.g., "nvidia\_blackwell", "amd\_cdna3") to enable target-specific dispatch.
    -   Allow for optional, architecture-specific keys to provide fine-grained control (e.g., `mfma=true`).

-   [x] **Task 1.2: Refactor Core Operations to Use MLIR Properties.**
    -   Migrate the `target` attribute of `orchestra.task` and the `num_true` attribute of `orchestra.commit` from generic attributes to the MLIR `Properties` system for improved performance and type safety.
    -   Enable `usePropertiesForAttributes = 1;` for the `Orchestra` dialect in `OrchestraOps.td`.

## Milestone 2: AMD GPU Support (rocMLIR Integration)

This milestone focuses on adding end-to-end support for AMD Instinct-series GPUs.

-   [ ] **Task 2.1: Implement `linalg-to-rock` Lowering.**
    -   Create a new pass to lower `linalg.generic` operations to the `rock` dialect, which is the entry point for the `rocMLIR` kernel generator.
    -   This pass must be guided by the "contract" of the `rocMLIR` tool to produce optimal `linalg` patterns.

-   [ ] **Task 2.2: Implement `rock-to-amdgpu` and `rocdl` Lowering.**
    -   Develop the pipeline to lower the `rock` dialect to the `amdgpu` and `rocdl` dialects.
    -   Ensure correct mapping to hardware primitives like `amdgpu.mfma` for matrix operations.
    -   Handle synchronization with `gpu.barrier`.

-   [ ] **Task 2.3: Create AMD-specific Transform Script.**
    -   Author `amd_instinct_cdna3_strategy.mlir` for the `transform` dialect interpreter.
    -   This script will apply tiling, packing, and fusion patterns at the `linalg` level, co-designed with the `rocMLIR` lowering contract to generate high-performance kernels.

## Milestone 3: Google TPU & AWS Trainium Support (StableHLO Bridge)

This milestone focuses on integrating with Google and AWS compilers via the StableHLO standard.

-   [ ] **Task 3.1: Implement `linalg-to-stablehlo` Lowering Pass.**
    -   Create a robust pass to convert `linalg` dialect operations into their `stablehlo` equivalents.
    -   This pass will be the common backend for both Google and AWS targets.

-   [ ] **Task 3.2: Implement the "StableHLO Bridge" Mechanism.**
    -   Integrate the external `stablehlo-translate` tool into the build and execution pipeline.
    -   The compiler driver will invoke this tool as a subprocess to serialize the textual `stablehlo` IR into a versioned, portable bytecode artifact.
    -   This artifact is the handoff point to the vendor compilers (`xla_compile`, `neuronx-cc`).

-   [ ] **Task 3.3: Create Google/AWS-specific Transform Scripts.**
    -   Author `google_tpu_strategy.mlir` and `aws_trainium_strategy.mlir`.
    -   These scripts will perform high-level `linalg` optimizations and canonicalizations to simplify the graph before lowering to `stablehlo`.

## Milestone 4: Advanced AWS Integration (Neuron Kernel Interface)

This milestone adds expert-level performance tuning capabilities for AWS Trainium.

-   [ ] **Task 4.1: Extend `orchestra.task` for NKI.**
    -   Add an optional `nki_source` string attribute to the `orchestra.task` operation to hold custom Neuron Kernel Interface source code.

-   [ ] **Task 4.2: Implement `orchestra-lower-nki-tasks` Pass.**
    -   This pass will identify tasks with `nki_source`, invoke the NKI compiler offline, and link the resulting custom kernel into the final executable.
