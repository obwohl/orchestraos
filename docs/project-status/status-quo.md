# Orchestra Compiler - Project Status

**Last Updated: 2025-08-21**

## 1. Summary

The Orchestra compiler is a functional, first-generation compiler for heterogeneous systems. The project is in a **stable, buildable, and verifiable state**, with a full test suite that passes against LLVM 20. The compiler provides a solid foundation for research and development in compiler-driven hardware orchestration.

A comprehensive architectural plan, the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**, exists and details the roadmap for evolving the compiler into a state-of-the-art system. However, the implementation of this blueprint has not yet begun. The project's current state is therefore that of the initial, foundational implementation.

## 2. Verified Current State

The following features have been verified through code analysis and a successful run of the project's `lit` test suite.

*   **Build System:** The project is built with CMake and Ninja. The test suite is correctly integrated using a `check-orchestra` target. The build environment requires `FileCheck` and `llvm-lit` paths to be configured correctly.
*   **Core Dialect (`OrchestraIR`):** The core dialect is implemented and functional.
    *   **Partial Migration to Properties:** The `orchestra.commit` operation has been successfully migrated to use the `Properties` system, providing improved type safety and performance. The `orchestra.task` operation still needs to be migrated.
    *   The `target` attribute on `orchestra.task` has been enhanced to require a mandatory `arch` key, which is enforced by a custom verifier.
*   **Speculative Execution:** The `--divergence-to-speculation` pass successfully converts `scf.if` operations into speculative `orchestra.task` operations. This feature is tested and functional. The implementation has been migrated from C++ to the declarative Pattern Description Language (PDL).
*   **GPU Lowering (NVIDIA):** The `--lower-orchestra-to-gpu` pass provides a lowering path to the `nvgpu` dialect. It includes an architecture-aware code path:
    *   For NVIDIA Blackwell GPUs (`sm_100` and newer), it generates SOTA asynchronous data transfers using the Tensor Memory Accelerator (`nvgpu.tma.async.load`) and `mbarrier` synchronization.
    *   For older NVIDIA GPUs (e.g., Hopper), it generates standard asynchronous copies (`nvgpu.device_async_copy`).
*   **Standard Lowering:** The `--lower-orchestra-to-standard` pass correctly lowers `orchestra.commit` to `arith.select`.
*   **Declarative Optimization Framework:** The compiler now includes a declarative optimization framework based on the MLIR `transform` dialect. The `-transform-interpreter` pass is integrated and can be used to apply hardware-specific optimization scripts. A library of such scripts can be found in `orchestra-compiler/transforms`.

## 3. Known Limitations & Deviations from SOTA Blueprint

The current implementation does not include all of the state-of-the-art (SOTA) features outlined in the MLIR 2.0 blueprint. Key deviations include:

*   **No Declarative Patterns:** Most passes are implemented using imperative C++ patterns, not the declarative, more maintainable PDL (Pattern Description Language) framework. The exception is the `SpeculateIfOpPattern`, which has been migrated to PDL.
*   **No Feedback-Driven Optimization (FDO):** The compiler is purely Ahead-of-Time (AOT). The entire FDO loop (instrumentation, runtime profiling, JIT recompilation) is not implemented.
*   **Limited SOTA GPU Support:**
    *   **Intel:** The backend targets an incomplete, experimental `xegpu` lowering, not the SOTA `XeVM` dialect required for maximum performance on modern Intel GPUs. The `xegpu` tests are disabled.

## 4. Next Steps

The immediate next step for the project is to begin the implementation of the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**. A detailed, prioritized roadmap for this modernization effort has been created and can be found at **[next/upgrade_plan.md](../../next/upgrade_plan.md)**.
