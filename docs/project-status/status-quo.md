# Current Project Status: Build Fixed, Runtime Issue Identified

Last updated: 2025-08-11

## 1. Overview

This document provides a summary of the current status of the `orchestra-compiler` project.

**The project is now buildable.** A comprehensive effort has been undertaken to refactor and correct the project's build system and source files to align with canonical MLIR out-of-tree dialect practices.

However, the project is **not yet functional**. While the `orchestra-opt` tool compiles and links successfully, it fails at runtime with an "unregistered operation" error. This document details the fixes applied and the analysis of the new blocking issue.

## 2. Build System and Source Code Fixes

The initial "unbuildable" state of the project has been resolved. The following actions were taken:

*   **CMake Refactoring**: All `CMakeLists.txt` files (root, `include/`, `lib/`, `orchestra-opt/`) were overwritten with correct, canonical versions based on official MLIR examples and best practices. This resolved all CMake configuration and dependency issues.
*   **C++ and TableGen Correction**: The C++ source files (`.h`, `.cpp`) and the TableGen definition file (`.td`) for the `Orchestra` dialect were corrected to ensure proper dialect and operation definition and registration.
*   **MLIR 18.1 Compatibility**: A known issue with MLIR 18.1's new "Properties" system was proactively addressed by setting `usePropertiesForAttributes = 0;` in the dialect's TableGen definition, which resolved the initial C++ compilation errors.
*   **Linker Debugging**: Several linker issues were resolved by adding the necessary MLIR libraries (`MLIRMlirOptMain`) to the executable's link list.

## 3. Current Blocking Issue: "Unregistered Operation" Runtime Error

Despite the successful build, running the `orchestra-opt` tool on a test file containing a custom dialect operation fails with the following error:

```
error: unregistered operation 'orchestra.my_op' found in dialect ('orchestra') that does not allow unknown operations
```

### 3.1. Diagnostic Analysis

This error indicates that while the `Orchestra` dialect itself is being loaded, its custom operations (like `my_op`) are not being registered with the MLIR context. An extensive investigation has traced this to a subtle and persistent linker issue.

Using the `nm` utility, we have definitively confirmed that the C++ symbols for the operation classes (e.g., `orchestra::MyOp`) are being stripped from the final `orchestra-opt` executable during the link stage. This happens because the linker's static library optimization (elision) incorrectly determines that the operation's code is "unused," as it is only referenced via static initializers.

The standard solution for this problem, linking the dialect library with the `--whole-archive` flag, has been implemented but **did not solve the issue**. The symbols are still being stripped.

## 4. Next Steps

The project is currently blocked on this advanced linker issue. The immediate next step is to conduct deep research to understand why the GNU linker (`ld`) is ignoring the `--whole-archive` flag in this specific context. The research should investigate:
*   Potential negative interactions between CMake, LLVM/MLIR's build scripts, and the GNU linker.
*   Quirks in the specific version of the toolchain being used.
*   Alternative methods or linker flags to more forcefully prevent static library symbol stripping.

Once this linker issue is understood and resolved, the project should become fully functional.
