# Orchestra Compiler - Project Status

**Last Updated: 2025-08-24**

## 1. Summary

The Orchestra compiler is a functional, first-generation compiler for heterogeneous systems. The project is in a **stable, buildable, and verifiable state**, with a full test suite that passes against LLVM 20. The compiler provides a solid foundation for research and development in compiler-driven hardware orchestration.

A comprehensive architectural plan, the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**, exists and details the roadmap for evolving the compiler into a state-of-the-art system. However, the implementation of this blueprint has not yet begun. The project's current state is therefore that of the initial, foundational implementation.

## 2. Verified Current State

The following features have been verified through code analysis and a successful run of the project's `lit` test suite.

*   **Build System:** The project is built with CMake and Ninja. The test suite is correctly integrated using a `check-orchestra` target. The build environment requires `FileCheck` and `llvm-lit` paths to be configured correctly.
*   **Core Dialect (`OrchestraIR`):** The core dialect is implemented and functional.
*   **`orchestra.task` Improvements:** The `orchestra.task` operation has been significantly improved for better usability and robustness:
    *   **Dictionary-based Target:** The `target_arch` attribute is now a `DictionaryAttr`. This provides a structured way to specify target architecture properties.
    *   **Custom Verifier:** A C++ verifier ensures that the `target_arch` dictionary contains a non-empty string attribute named `arch`.
*   **Speculative Execution:** The `--divergence-to-speculation` pass successfully converts `scf.if` operations into speculative `orchestra.task` operations. This feature is tested and functional. The implementation uses a PDL pattern that calls a C++ rewriter. The pattern's C++ constraints have been refactored for improved clarity and modularity.
*   **GPU Lowering (NVIDIA):** The `--lower-orchestra-to-gpu` pass provides a lowering path to the `nvgpu` dialect. It includes an architecture-aware code path:
    *   For NVIDIA Blackwell GPUs (`sm_100` and newer), it generates SOTA asynchronous data transfers using the Tensor Memory Accelerator (`nvgpu.tma.async.load`) and `mbarrier` synchronization.
    *   For older NVIDIA GPUs (e.g., Hopper), it generates standard asynchronous copies (`nvgpu.device_async_copy`).
*   **GPU Lowering (AMD):** Initial support for AMD GPUs has been added via the `--gpu-arch=rocdl` flag. This pass lowers `orchestra.transfer` operations to the `rocdl` dialect, using `rocdl.global.load.lds` for asynchronous data copies.
*   **Standard Lowering:** The `--lower-orchestra-to-standard` pass correctly lowers `orchestra.commit` to `arith.select`.
*   **Declarative Optimization Framework:** The compiler now includes a declarative optimization framework based on the MLIR `transform` dialect. The `-transform-interpreter` pass is integrated and can be used to apply hardware-specific optimization scripts. A library of such scripts can be found in `orchestra-compiler/transforms`. A working example of producer-consumer fusion using this framework can be found in `orchestra-compiler/tests/fusion-test.mlir`.
*   **Build System and Stability:** The project was initially failing to build against the specified LLVM 20 environment. This has been fixed. The canonicalization pattern for `orchestra.transfer` was updated to correctly handle optional attributes. A failing test related to the `orchestra.task` verifier was fixed, making the codebase stable and the test suite fully passing.
*   **Code Formatting:** A `.clang-format` file based on the LLVM style has been added to the repository root to ensure a consistent code style. A Python script at `scripts/run-clang-format.py` has been added to automatically format all C++ source and header files in the `orchestra-compiler` directory.
*   **`orchestra.schedule` Verifier:** A verifier has been added to the `orchestra.schedule` operation to ensure that all child `orchestra.task` operations have a unique `task_id`. This is a critical feature for enabling reliable scheduling and analysis passes. The `task_id` property was also added to the `orchestra.task` operation as part of this work.

## 3. Known Limitations & Deviations from SOTA Blueprint

The current implementation does not include all of the state-of-the-art (SOTA) features outlined in the MLIR 2.0 blueprint. Key deviations include:

*   **No Declarative Patterns:** Most passes are implemented using imperative C++ patterns. The `SpeculateIfOpPattern` has been migrated to PDL for its matching logic, but still uses a C++ rewriter for the complex rewrite logic. This is a common and acceptable practice in MLIR for complex patterns.
*   **No Feedback-Driven Optimization (FDO):** The compiler is purely Ahead-of-Time (AOT). The entire FDO loop (instrumentation, runtime profiling, JIT recompilation) is not implemented.
*   **Limited SOTA GPU Support:**
    *   **Intel:** The backend targets an incomplete, experimental `xegpu` lowering, not the SOTA `XeVM` dialect required for maximum performance on modern Intel GPUs. The `xegpu` tests are disabled.
    *   **AMD:** The newly added ROCDL backend provides initial data movement capabilities but does not yet implement lowering for matrix acceleration primitives (e.g., to `rocdl.mfma`).
*   **PDL Lowering for `orchestra.commit`:** An attempt was made to refactor the C++ `CommitOpLowering` pattern to PDL. This was unsuccessful because the lowering occurs in a `DialectConversion` pass, which requires a `ConversionPattern` with access to adapted operands. The PDL tooling generates a `RewritePattern` which does not have this capability, making it unsuitable for this specific conversion. The C++ implementation remains the correct approach.

## 4. Next Steps

The immediate next step for the project is to begin the implementation of the **[Orchestra MLIR 2.0 Blueprint](../architecture/orchestra%20-%20tech%20-%20MLIR%2020%20Blueprint.md)**. A detailed, prioritized roadmap for this modernization effort has been created and can be found at **[next/upgrade_plan.md](../../next/upgrade_plan.md)**.
