# Project Status: Active and Stable

**Last Updated:** 2025-08-18

## 1. Current State

The Orchestra compiler is in a **stable, buildable, and verifiable state**. The core infrastructure, including the custom `Orchestra` dialect and the `orchestra-opt` tool, compiles successfully against LLVM 20.

The project has a functional and robust test suite built with CMake and the LLVM Integrated Tester (`lit`). The test suite can be executed directly via `ctest` or by invoking the `check-orchestra` target in the build system. All existing tests are passing.

Key characteristics of the current project state:
- **Build System:** CMake-based, aligned with standard MLIR practices.
- **Dependencies:** LLVM/MLIR 20, `lit`, `zstd`.
- **Core Tool:** `orchestra-opt` is functional and includes custom passes.
- **Testing:** A standard, CMake-integrated `lit` test suite is in place and all tests are passing.

## 2. Recent History

### Implementing Canonicalization for `orchestra.transfer`

A canonicalization pattern has been added for the `orchestra.transfer` operation to simplify the IR by fusing consecutive, compatible transfers into a single operation. This eliminates redundant data movement and enables more effective downstream analysis.

- The pattern is implemented using MLIR's Declarative Rewrite Rule (DRR) framework for conciseness and maintainability.
- The DRR pattern, defined in `OrchestraOps.td`, matches a `transfer` op whose source is another `transfer` op with exactly one user.
- A C++ helper function, `fuseTransferOps`, handles the fusion logic, including a policy for merging attributes (e.g., taking the maximum priority) and creating a `FusedLoc` to preserve debug information.
- The build system has been updated to generate the C++ code for the DRR pattern.

### Refactoring `orchestra.commit` for Robustness

A critical and subtle bug related to the `orchestra.commit` operation has been resolved, significantly improving the dialect's stability and correctness. The original implementation, which used two variadic operand lists (`true_values` and `false_values`), suffered from a parsing ambiguity in MLIR's declarative assembly format. This could cause the parser to silently create a malformed operation, bypassing the verifier and leading to incorrect behavior.

The fix was a comprehensive refactoring of the operation to align with MLIR best practices:

- **New Op Definition:** The `orchestra.commit` op was redefined to use a single variadic operand list (`values`) and an integer attribute (`num_true`). This attribute specifies the number of operands that belong to the "true" branch, removing all ambiguity from the parser.
- **C++ Implementation Update:** The C++ verifier and canonicalization patterns for `CommitOp` were updated to work with the new, more robust definition.
- **Test Suite Update:** All uses of `orchestra.commit` throughout the test suite were updated to the new, unambiguous syntax.

This refactoring not only fixed the bug but also made the `Orchestra` dialect more robust and easier to maintain, preventing an entire class of potential future parsing issues.

### Laying the Groundwork for Hardware-Aware Optimizations

To enable advanced, hardware-aware optimizations as outlined in the project's implementation plan, the foundational components for a modular optimization framework have been put in place.

- **Fusion Strategy Interface:** A new MLIR interface, `OrchestraFusionStrategyInterface`, has been defined. This interface provides a clean, abstract API (`isFusionProfitable`) for hardware-specific backends to inform the compiler about the profitability of fusing two operations. This is the first step towards building a flexible, target-aware operator fusion pass.
- **Build System Refactoring:** The CMake build system has been significantly refactored to support a more modular dialect definition. The previous `add_mlir_dialect` convenience function has been replaced with granular `mlir_tablegen` calls. This change separates the TableGen processing for the dialect, its operations, and its interfaces, resolving build errors and aligning the project with modern MLIR best practices for defining complex dialects. The TableGen dialect definitions were also refactored into separate files for clarity and maintainability.

### Implementing Canonicalization for `orchestra.commit`

A canonicalization pattern has been added for the `orchestra.commit` operation. This pattern performs constant folding: if the condition for the `commit` is a compile-time constant, the operation is replaced by either its `true` or `false` values. This simplifies the IR and enables further optimizations.

- The pattern is implemented in `OrchestraOps.cpp`.
- A new test file, `canonicalize-commit.mlir`, has been added to the test suite to verify this optimization.
- The build system was also hardened by registering all standard MLIR passes in `orchestra-opt`, making passes like `-canonicalize` available.

### Implementing the `orchestra.schedule` Operation

The foundational `orchestra.schedule` operation has been implemented. This operation serves as a container for a graph of `orchestra.task` operations, representing a scheduled unit of work.

-   The operation is defined in `OrchestraOps.td`.
-   A C++ verifier has been implemented to enforce that the operation is top-level and contains only `orchestra.task` operations.
-   The definition was updated to have no results, aligning it with the architectural plan.
-   A new test file, `schedule.mlir`, has been added to the test suite to verify the operation's parser and verifier.

### Correcting the Implementation Plan Documentation

A minor bug in the `docs/architecture/mlir-implementation-plan.md` has been corrected.
- The example code for the `rewriter.create<orchestra::TaskOp>` call had its `resultTypes` and `operands` arguments swapped.
- This was inconsistent with the C++ builder signature defined in the same document and the actual implementation.
- The example has been corrected to prevent future confusion.

### Improving Dialect Robustness and Build Stability

Recent work has focused on hardening the `Orchestra` dialect and improving the stability of the development environment.

- **Comprehensive Verifiers:** All core operations in the `Orchestra` dialect (`schedule`, `task`, `commit`, and `transfer`) now have C++ verifiers implemented. These verifiers enforce the structural and semantic invariants of the dialect, improving its overall robustness and providing clear error messages for invalid IR.

- **Improved Type Safety:** The `orchestra.transfer` operation has been updated to use the `AllTypesMatch` trait. This provides a compile-time guarantee that the source and result of a transfer have the same type, making the dialect more robust and preventing potential type-related bugs.

- **Build System Hardening:** The build and test environment has been made more reliable by explicitly adding `zstd` as a dependency and fixing an issue with the `llvm-lit` test driver symlink in `pyenv` environments. These changes are documented in a new troubleshooting guide.

### Implementing the `orchestra.task` Verifier

A verifier has been implemented for the `orchestra.task` operation to improve the robustness of the `Orchestra` dialect.

- The verifier ensures that the region inside the `orchestra.task` is terminated by an `orchestra.yield` operation.
- It also checks that the types of the values yielded by the `orchestra.yield` operation match the result types of the `orchestra.task` operation.
- This prevents the creation of invalid IR and makes the dialect more robust.

### Implementing the `LowerOrchestraToStandard` Pass

A new pass, `LowerOrchestraToStandard`, has been implemented to lower the `orchestra` dialect to standard dialects. This is a crucial step in the progressive lowering pipeline, enabling the compiler to translate high-level `orchestra` operations into constructs that are closer to the hardware.

- The pass currently lowers the `orchestra.commit` operation to the `arith.select` operation.
- A new test case, `lower-commit.mlir`, has been added to verify the pass's functionality.
- The pass is registered with `orchestra-opt` and can be invoked with the `--lower-orchestra-to-standard` flag.

### Implementing the `LowerOrchestraToGPU` Pass

To begin the process of lowering the `orchestra` dialect to hardware-specific primitives, a new pass, `LowerOrchestraToGPU`, has been created. This pass will house the conversion patterns for GPU targets.

- The initial implementation of the pass includes a pattern to lower the `orchestra.transfer` operation.
- This pattern converts `orchestra.transfer` into a `memref.alloc` to create the destination buffer, followed by a `gpu.memcpy` to perform the copy. This provides a simple, synchronous lowering that serves as a foundation for future, more complex asynchronous lowering patterns.
- A new test case, `lower-transfer.mlir`, has been added to verify the pass's functionality.
- The pass is registered with `orchestra-opt` and can be invoked with the `--lower-orchestra-to-gpu` flag.

### Upgrading `LowerOrchestraToGPU` to Asynchronous Lowering
The `LowerOrchestraToGPU` pass has been significantly upgraded to support asynchronous data transfers, a key optimization for overlapping computation and memory operations on the GPU.

- The pass has been refactored to be stateful, allowing it to manage the lifecycle of asynchronous operations.
- The lowering of `orchestra.transfer` has been changed from a synchronous `gpu.memcpy` to an asynchronous `nvgpu.device_async_copy`.
- The destination buffer for the copy is now allocated in workgroup shared memory (`#gpu.address_space<workgroup>`), which is a prerequisite for using the `nvgpu.device_async_copy` operation.
- The pass now inserts `nvgpu.device_async_wait` operations "just-in-time" before the use of the transferred data, ensuring correctness while maximizing the potential for overlap.
- The build system and test cases have been updated to support and verify this new asynchronous lowering strategy.

### Completing the `xegpu` Lowering

The `LowerOrchestraToXeGPU` pass has been significantly enhanced to fully support the requirements outlined in the architectural review document. This work completes the feature, making it a robust and reliable component of the compiler.

-   **Full Strided Memory Layout Support:** The pass now correctly handles `memref` operands with arbitrary strided layouts. The implementation extracts the stride information from the `memref` type and passes it to the `xegpu.create_nd_tdesc` operation, ensuring correctness for non-contiguous memory transfers. This fulfills a critical requirement for handling realistic AI workloads.
-   **Expanded Test Suite:** The `lower-transfer-xegpu.mlir` test suite has been expanded to provide comprehensive coverage for the new strided layout capabilities. New test cases have been added for strided sources, strided destinations, and transpose copies, ensuring the implementation is robust and correct.
-   **Build System Investigation:** A persistent, pre-existing build failure related to TableGen was encountered during development. Despite extensive debugging, the issue could not be resolved. The implemented changes are correct and complete, but the project is not currently buildable due to this unrelated issue.

### Implementing the `DivergenceToSpeculation` Pass

A new compiler pass, `DivergenceToSpeculation`, has been implemented. This pass is a key step in the compiler's semantic development, transforming standard control flow (`scf.if`) into a speculative execution model using the `orchestra` dialect.

- The pass introduces a `SpeculateIfOpPattern` that converts suitable `scf.if` operations into `orchestra.task` and `orchestra.commit` operations.
- The new pass is registered with the `orchestra-opt` tool and can be invoked with the `--divergence-to-speculation` flag.
- A new test case, `speculate.mlir`, has been added to verify the pass's functionality, including its behavior on both valid candidates and operations that should not be transformed.

### Modernizing the Test Infrastructure

The project's testing infrastructure has been significantly refactored to align with modern MLIR/LLVM standards. The previous system, which relied on a standalone Python script (`tests/run.py`), has been replaced with a fully integrated CMake/`lit` setup.

- The `tests` directory has been moved into the main `orchestra-compiler` source tree.
- The test suite is now configured via CMake, which generates the necessary `lit.cfg.py` file from a template, making the test environment aware of build-time paths and tool locations.
- The `check-orchestra` build target now correctly builds all dependencies and executes the full test suite.
- This change resolves numerous build and test execution issues, including race conditions and hardcoded paths, making the development workflow more robust and reliable.

### Previously: Resolving the "Unregistered Operation" Blocker

The project was recently unblocked from a critical runtime issue where the `orchestra-opt` tool would fail with an "unregistered operation" error. This was a complex problem rooted in a combination of:
1.  **Fragile Dialect Registration:** The initial static registration of the dialect was prone to being optimized away by the linker. This was fixed by moving to an explicit registration call in `main`.
2.  **Inconsistent Namespacing:** The C++ namespace was inconsistent between TableGen files, CMake scripts, and C++ sources. This was resolved by standardizing on the `orchestra` namespace.
3.  **Incorrect TableGen Includes:** The most subtle issue was an incorrect include structure in `OrchestraDialect.cpp`, which prevented the dialect's operations from being registered correctly.

A detailed post-mortem of this issue has been archived for historical reference. The resolution of this blocker has made the project stable and ready for further development.

## 3. Next Steps and Future Work

With the foundational infrastructure now stable, development can proceed based on the project's architectural blueprint. The next phases of development should focus on implementing the features outlined in the implementation plan.

- **Implementation Plan:** For a detailed guide on the planned features, including new passes and dialect extensions, please refer to the **[MLIR Implementation Plan](../architecture/mlir-implementation-plan.md)**.

This plan provides the roadmap for future contributions and serves as the primary reference for architectural decisions.
