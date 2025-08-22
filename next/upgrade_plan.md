# Orchestra Compiler: A Prioritized Modernization Plan

## 1. Executive Summary

This document outlines a high-level, prioritized plan for modernizing the Orchestra compiler to align with its state-of-the-art (SOTA) architectural blueprint. The current compiler is a stable, functional, first-generation tool with a passing test suite. This plan provides a practical, phased roadmap for its evolution.

The guiding principle of this plan is to prioritize foundational changes first, ensuring that each phase builds upon a stable and modernized core. The plan is broken into three phases, with tasks ordered by dependency and impact.

### 1.1 PDL-First Development Mandate

Following the successful modernization of core components in Phase 1, all new pattern-based development within the Orchestra compiler must adhere to a PDL-first approach. This means that any new rewrite patterns, transformations, or optimizations should be implemented using MLIR's Pattern Description Language (PDL) or PDLL, leveraging the declarative paradigm for improved readability, maintainability, and extensibility. This mandate applies to all subsequent phases and tasks where pattern-based logic is involved.

## 2. Phase 1: Foundational Modernization

This phase is the highest priority and is a prerequisite for all subsequent work. Its goal is to update the existing, functional codebase to use modern MLIR infrastructure, improving its performance, maintainability, and robustness.

*   **Task 1.1: Modernize Core Dialect with the `Properties` System.**
    *   **What:** Refactor the TableGen definitions in `OrchestraOps.td`. The `target` attribute of `orchestra.task` and the `num_true` attribute of `orchestra.commit` will be migrated from being standard arguments to using the `Properties` system.
    *   **Why:** This is the most fundamental modernization. It provides significant compile-time performance benefits, adds C++ type safety for these critical attributes, and aligns the core dialect with current MLIR best practices. This provides a solid foundation for all future dialect and pass development.

*   **Task 1.2: Modernize Speculative Execution Pass with PDL.**
    *   **What:** Rewrite the `SpeculateIfOpPattern` C++ pattern. The imperative C++ matching logic will be replaced with a declarative PDL (`.pdll`) pattern. The existing C++ rewrite logic will be refactored into a focused rewriter function called by the PDL engine.
    *   **Why:** This modernizes the implementation of an existing, core feature. It decouples the "what" of the pattern from the "how" of the rewrite, making the pass more readable, maintainable, and easier to extend. This should be done after the `Properties` migration so the new C++ rewriter can benefit from the modern API.

## 3. Phase 2: Implementation of SOTA Features

With a modernized core and the establishment of the PDL-first development mandate, the project can begin implementing the major new features from the blueprint. All new pattern-based implementations within this phase must adhere to the PDL-first approach. This phase is ordered to deliver the highest-impact, most self-contained features first.

*   **Task 2.1: Implement the Declarative Optimization Framework.**
    *   **What:** Build the hardware-aware optimization framework using the `transform` dialect. This involves creating a library of target-specific transform scripts (e.g., for fusing operations on NVIDIA vs. Intel GPUs) and integrating the standard `-transform-interpreter` pass into the main compiler pipeline. Any new rewrite patterns or declarative sequences defined as part of this framework should utilize PDL.
    *   **Why:** This is arguably the most impactful new feature for performance engineers. It makes complex optimization strategies (fusion, tiling, layout) scriptable and modular, dramatically accelerating the tuning cycle. It is a large but self-contained feature that delivers immense value.

*   **Task 2.2: Implement the SOTA Intel GPU Backend (`XeVM`).**
    *   **What:** Architect and build a new lowering path from Orchestra/Linalg to the `XeVM` dialect. This will involve deprecating and removing the current, non-functional `xegpu` pass. The new pass must target key `XeVM` primitives like `xevm.blockload2d` for data transfer and `xevm.mma` for matrix math. Any pattern-based lowering rules should be implemented using PDL.
    *   **Why:** The compiler currently lacks a functional Intel GPU backend. Delivering a performant, SOTA backend is a critical feature for supporting a major hardware vendor. This should be prioritized over enhancing the already-functional NVIDIA backend.

*   **Task 2.3: Enhance the NVIDIA GPU Backend for Blackwell.**
    *   **What:** Add an architecture-aware code path to the existing `LowerOrchestraToNVGPUPass`. When targeting NVIDIA Blackwell (sm_100+), this path should lower `orchestra.transfer` to use the Tensor Memory Accelerator (TMA) via `nvgpu.tma.*` operations, with `nvgpu.mbarrier` for synchronization. Any new rewrite patterns for this lowering should be implemented using PDL.
    *   **Why:** This is an incremental but vital enhancement to the existing, working NVIDIA backend. It ensures SOTA performance on the latest hardware but is a lower priority than building the missing Intel backend from scratch.

*   **Task 2.4: Implement the Feedback-Driven Optimization (FDO) Loop.**
    *   **What:** Build the end-to-end FDO system. This includes the `OrchestraBranchProfiler` pass, the `DivergenceProfile` data contract, and the JIT recompilation service integrated via a custom MLIR `Action`. Any pattern-based components should adhere to the PDL-first mandate.
    *   **Why:** This is the most complex feature, as it involves interactions with an external runtime system. It should be implemented last, as it can then leverage the full power of the modernized compiler, including the declarative optimization framework and the SOTA backends, to perform its dynamic recompilations.

## 4. Phase 3: Documentation and Finalization

*   **Task 3.1: Update All Documentation.**
    *   **What:** Thoroughly review and update all project documentation, including the main `README.md`, guides, and the blueprint itself, to reflect the new, modernized architecture.
    *   **Why:** Ensures that the project remains accessible and maintainable for current and future developers.