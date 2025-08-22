# Orchestra Compiler - Project Status

**Last Updated: 2025-08-21**

## 1. Summary

The Orchestra compiler is a functional, first-generation compiler for heterogeneous systems. The project is in a **stable, buildable, and verifiable state**, with a full test suite that passes against LLVM 20. The compiler provides a solid foundation for research and development in compiler-driven hardware orchestration.

A comprehensive architectural plan, the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**, exists and details the roadmap for evolving the compiler into a state-of-the-art system. However, the implementation of this blueprint has not yet begun. The project's current state is therefore that of the initial, foundational implementation.

## 2. Verified Current State

The following features have been verified through code analysis and a successful run of the project's `lit` test suite.

*   **Build System:** The project is built with CMake and Ninja. The test suite is correctly integrated using a `check-orchestra` target. The build environment requires `FileCheck` and `llvm-lit` paths to be configured correctly.
*   **Core Dialect (`OrchestraIR`):** The core dialect is implemented and functional. It has been modernized to use the MLIR v20 `Properties` system for its core operation attributes, providing improved type safety and performance.
*   **Speculative Execution:** The `--divergence-to-speculation` pass successfully converts `scf.if` operations into speculative `orchestra.task` operations. This feature is tested and functional. The implementation uses a standard C++ rewrite pattern.
*   **GPU Lowering (NVIDIA Hopper):** The `--lower-orchestra-to-gpu` pass provides a lowering path to the `nvgpu` dialect. It correctly generates asynchronous data transfers (`nvgpu.device_async_copy`), making it suitable for NVIDIA Hopper-class GPUs.
*   **Standard Lowering:** The `--lower-orchestra-to-standard` pass correctly lowers `orchestra.commit` to `arith.select`.

## 3. Known Limitations & Deviations from SOTA Blueprint

The current implementation does not include the state-of-the-art (SOTA) features outlined in the MLIR 2.0 blueprint. Key deviations include:

*   **No Declarative Patterns:** Passes are implemented using imperative C++ patterns, not the declarative, more maintainable PDL (Pattern Description Language) framework.
*   **No Hardware-Aware Optimization Framework:** The compiler lacks a flexible, scriptable framework for hardware-aware optimizations. The SOTA approach using the `transform` dialect is not implemented.
*   **No Feedback-Driven Optimization (FDO):** The compiler is purely Ahead-of-Time (AOT). The entire FDO loop (instrumentation, runtime profiling, JIT recompilation) is not implemented.
*   **Limited SOTA GPU Support:**
    *   **NVIDIA:** The backend does not support SOTA features of the Blackwell architecture (Tensor Memory Accelerator, `mbarrier`).
    *   **Intel:** The backend targets an incomplete, experimental `xegpu` lowering, not the SOTA `XeVM` dialect required for maximum performance on modern Intel GPUs. The `xegpu` tests are disabled.

## 4. Next Steps

The immediate next step for the project is to begin the implementation of the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**. A detailed, prioritized roadmap for this modernization effort has been created and can be found at **[next/upgrade_plan.md](../../next/upgrade_plan.md)**.
