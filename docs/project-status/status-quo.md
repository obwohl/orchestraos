# Project Status: Active and Stable

**Last Updated:** 2025-08-16

## 1. Current State

The Orchestra compiler is in a **stable, buildable, and verifiable state**. The core infrastructure, including the custom `Orchestra` dialect and the `orchestra-opt` tool, compiles successfully against LLVM 20.

The project has a functional and robust test suite built with CMake and the LLVM Integrated Tester (`lit`). The test suite can be executed directly via `ctest` or by invoking the `check-orchestra` target in the build system. All existing tests are passing.

Key characteristics of the current project state:
- **Build System:** CMake-based, aligned with standard MLIR practices.
- **Dependencies:** LLVM/MLIR 20, `lit`.
- **Core Tool:** `orchestra-opt` is functional and includes custom passes.
- **Testing:** A standard, CMake-integrated `lit` test suite is in place and all tests are passing.

## 2. Recent History

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
