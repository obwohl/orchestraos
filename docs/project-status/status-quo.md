# Project Status

Last updated: 2025-08-11

## High-Level Summary

The `orchestra-compiler` project is currently **unbuildable**. The primary blocker is a series of cascading C++ compilation errors in the `Orchestra` dialect library (`lib/Orchestra`). These errors stem from a non-standard and problematic interaction between the custom CMake build scripts and the MLIR TableGen code generation process.

The core issue is that the generated C++ header files for the dialect's operations do not correctly separate declarations from definitions, leading to circular dependencies and "type not found" errors that are extremely difficult to resolve by modifying the C++ source code alone.

## Detailed Build Failure Analysis

The process of attempting to build the project revealed the following sequence of issues:

1.  **TableGen Syntax Errors:** The initial build failed due to several syntax errors in `include/Orchestra/OrchestraOps.td`. These were corrected.

2.  **Missing `BytecodeOpInterface` Header:** After fixing the TableGen errors, the build failed with errors indicating that `mlir::BytecodeOpInterface` was not defined. This was resolved by adding `#include "mlir/Bytecode/BytecodeOpInterface.h"` to `lib/Orchestra/OrchestraDialect.cpp`.

3.  **Persistent "Type Not Found" Errors:** The final and most persistent error is `‘CommitOpGenericAdaptorBase’ does not name a type`. This error, and many others like it, occurs during the compilation of `lib/Orchestra/OrchestraDialect.cpp`.

## Investigation and Failed Attempts

Extensive efforts were made to resolve the compilation errors by restructuring `lib/Orchestra/OrchestraDialect.cpp`. The following standard MLIR patterns were attempted, and all of them failed:

*   **Standard Include Order:** Arranging the file to first include dialect and op headers, then the generated `...Dialect.cpp.inc` file, and finally the op definitions via `#define GET_OP_CLASSES` and `#include "...Ops.cpp.inc"`.
*   **Manual `initialize()` function:** Several variations of a manually implemented `OrchestraDialect::initialize()` function were tried, using different combinations of `GET_OP_LIST` and includes of `.h.inc` and `.cpp.inc` files.
*   **Pre-emptive Declaration:** Using `#define GET_OP_CLASSES` before including `OrchestraOps.h` to force the inclusion of the class declarations.

All of these attempts failed, which strongly indicates that the problem is not in the C++ source file's structure, but in the generated files themselves.

## Root Cause Analysis

The root cause is the custom CMake function `add_mlir_dialect` in `MyAddMLIR.cmake`. It generates the TableGen files in a way that is inconsistent with modern MLIR standards. Specifically:

```cmake
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.h -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  ...
endfunction()
```

The line `mlir_tablegen(${dialect}.h -gen-op-decls)` is intended to create a header file with only declarations. However, the build process seems to be producing a `.h` file that also contains implementations, guarded by a `#define`. This creates an unresolvable dependency cycle.

## Next Steps

**The primary focus for the next developer must be to fix the TableGen file generation process.**

1.  **Correct `add_mlir_dialect`:** The `add_mlir_dialect` function in `MyAddMLIR.cmake` must be corrected to generate clean header files (`.h` and `.h.inc`) with only declarations, and a single source file (`.cpp.inc`) with only definitions. The current implementation is flawed. A direct comparison with the `add_mlir_dialect` function in a recent version of MLIR or the `mlir-standalone-template` is the recommended approach.

2.  **Standardize `OrchestraDialect.cpp`:** Once the file generation is corrected, the `lib/Orchestra/OrchestraDialect.cpp` file should be updated to use the standard, simplified MLIR pattern for dialect implementation.

Fixing the build is the absolute prerequisite for any other work on this project.