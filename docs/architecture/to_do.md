Of course. Based on the provided `orchestra - tech - MLIR 20 Blueprint` and the `Blueprint addendum`, here is a revised and updated upgrade plan, structured as a concise to-do list that reflects the new multi-vendor strategy and removes the now-obsolete Intel backend work.

***

## Orchestra Compiler: A Prioritized Modernization and Expansion Plan

### Overview

This document outlines a high-level, phased to-do list for evolving the Orchestra compiler. It synthesizes the foundational goals of the original MLIR 20 Blueprint with the critical multi-vendor expansion strategy detailed in the addendum. The priority is to modernize the core infrastructure before implementing the new, state-of-the-art backends for strategic hardware targets.

### Phase 1: Core IR and Framework Modernization

This phase is the highest priority and a prerequisite for all subsequent work. Its goal is to refactor the existing codebase to use modern MLIR infrastructure, which is essential for building a scalable and maintainable multi-target compiler.

*   **Task 1.1: Modernize Core Dialect with the `Properties` System.**
    *   **What:** Refactor the `OrchestraIR` TableGen definitions (`OrchestraOps.td`). Migrate key attributes, such as `target` on `orchestra.task`, to use the `Properties` system instead of generic dictionary attributes.
    *   **Why:** To improve compile-time performance, add C++ type safety, and align the core dialect with current MLIR best practices, creating a robust foundation for all future development.

*   **Task 1.2: Enhance `orchestra.task` Target Schema.**
    *   **What:** Formalize the `target` attribute schema on `orchestra.task` to include a mandatory `arch` key (e.g., "amd_cdna3", "google_tpu_v5e").
    *   **Why:** This is a critical IR enhancement required to programmatically dispatch to the correct target-specific lowering and optimization pipelines, forming the basis of the multi-vendor strategy.

*   **Task 1.3: Refactor Existing Patterns with PDL.**
    *   **What:** Rewrite existing imperative C++ patterns, such as the `SpeculateIfOpPattern`, to use the declarative Pattern Description Language (PDL).
    *   **Why:** To improve the readability and maintainability of core transformations by separating the matching logic from the rewrite logic, aligning with modern MLIR development philosophy.

### Phase 2: Declarative Optimization and Multi-Vendor Backend Implementation

With a modernized core, this phase focuses on implementing the primary strategic goal: performant, multi-vendor support for all Tier-1 and Tier-2 hardware targets.

*   **Task 2.1: Implement the Declarative Optimization Framework.**
    *   **What:** Replace the hardcoded C++ optimization passes with the `transform` dialect-based framework. Integrate the standard `-transform-interpreter` pass into the main compiler pipeline.
    *   **Why:** To empower performance engineers to rapidly iterate on complex, hardware-specific optimization strategies (e.g., tiling, fusion) by editing simple MLIR scripts, which is the key to unlocking performance across multiple, diverse hardware backends.

*   **Task 2.2: Implement "Gray-Box" Lowering for Google TPU.**
    *   **What:** Build a new compiler pass pipeline (`orchestra-lower-to-stablehlo`) that lowers `linalg` dialect operations to the `StableHLO` dialect.
    *   **Why:** To support Google TPUs by handing off a standardized IR to Google's highly optimized, proprietary XLA compiler, leveraging their toolchain investment as a backend.

*   **Task 2.3: Implement "Black-Box" Lowering for AWS Trainium.**
    *   **What:** Utilize the `orchestra-lower-to-stablehlo` pipeline to target the AWS Neuron SDK, treating the Neuron Compiler as a black-box backend.
    *   **Why:** To support AWS Trainium accelerators by integrating with the official, MLIR-based Neuron toolchain via the common `StableHLO` entry point.
    *   **Sub-task:** Enhance `OrchestraIR` to support the Neuron Kernel Interface (NKI) "escape hatch" for expert-tuned, performance-critical kernels.

*   **Task 2.4: Implement "White-Box" Lowering for AMD Instinct.**
    *   **What:** Build a complete, end-to-end lowering path that converts `linalg` to the `rock` dialect and subsequently to the `amdgpu` and `rocdl` dialects.
    *   **Why:** To achieve deep integration and co-designed optimization for AMD GPUs by leveraging the open-source `rocMLIR` kernel generator and its transparent, multi-level dialect stack.

*   **Task 2.5: Enhance NVIDIA Backend for Blackwell Architecture.**
    *   **What:** Add an architecture-aware code path to the existing NVIDIA lowering pass. When targeting Blackwell (sm_100+), generate `nvgpu.tma.*` operations for data transfer and use the `nvgpu.mbarrier` family for synchronization.
    *   **Why:** To ensure state-of-the-art performance on the latest incumbent hardware by leveraging its most powerful and efficient primitives for data movement and synchronization.

### Phase 3: Advanced Features and Finalization

This final phase focuses on implementing advanced, system-level capabilities and ensuring the long-term health of the project.

*   **Task 3.1: Implement the Feedback-Driven Optimization (FDO) Loop.**
    *   **What:** Build the end-to-end FDO system, including the branch profiling pass, the `DivergenceProfile` data contract, and the JIT recompilation service integrated via a custom MLIR `Action`.
    *   **Why:** To create a dynamic, "learning" compiler that can adapt to data-dependent workloads at runtime, unlocking performance in scenarios where static analysis is insufficient.

*   **Task 3.2: Update All Documentation.**
    *   **What:** Thoroughly review and update all project documentation, including the architectural blueprints, to reflect the new, multi-vendor architecture and modernized infrastructure.
    *   **Why:** To ensure the project remains accessible, maintainable, and aligned with its state-of-the-art vision for all current and future developers.