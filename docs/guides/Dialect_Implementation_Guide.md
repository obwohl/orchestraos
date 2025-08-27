# A Developer's Guide to Implementing a New Dialect

## 1. Overview

This guide provides the canonical, step-by-step process for adding a new dialect to the Orchestra Compiler. It is the result of an extensive debugging process and contains the definitive, working patterns for this specific project. Following this guide will prevent common build system and C++ compilation errors.

## 2. Core Architectural Principles

*   **Use the Properties System:** All new operations that have attributes should use MLIR's modern Properties system for type safety and API clarity. This is a mandatory architectural requirement.
*   **Isolate Dialects:** Each new dialect should be implemented in its own isolated library (e.g., `OrchestraMyDialect`) to ensure clean separation and avoid build system conflicts.

## 3. The Correct Implementation Pattern

This section details the proven, working patterns for the build system and C++ source files.

### 3.1. CMake Configuration (`CMakeLists.txt`)

The project uses a legacy CMake pattern for TableGen. You **must** use the `mlir_tablegen` command, not the modern `add_mlir_tablegen`.

**Example:** The `CMakeLists.txt` for a new dialect library (e.g., in `lib/Orchestra/Dialects/MyDialect/`) should mimic the structure of the one in `include/Orchestra/`. It involves:
1.  Using `set(LLVM_TARGET_DEFINITIONS ...)` to specify the input `.td` file.
2.  Calling `mlir_tablegen(...)` with the legacy generators (`-gen-op-defs`, `-gen-dialect-defs`, etc.).
3.  Manually collecting the generated file paths into lists (`set(MY_DIALECT_INCGEN_HDRS ...)`).
4.  Creating a final dependency target with `add_custom_target(MyDialectIncGen ...)`.
5.  Defining the final library with `add_library(OrchestraMyDialect ...)`.

### 3.2. TableGen Syntax (`.td` files)

To use properties with the project's legacy TableGen generator (`-gen-op-defs`), you **must** use the "in-situ" syntax.

*   **Correct Syntax:** Define properties directly inside the `arguments` block.
    ```tablegen
    def MyOp : Op<MyDialect, "my_op"> {
      let arguments = (ins
        // This works:
        AttrProperty<"my_prop", StrAttr, "...">:$properties,
        F32Tensor:$input
      );
      // ...
    }
    ```
*   **Incorrect Syntax:** The `let properties = [...]` block is **not supported** by the generator and will cause build failures.

### 3.3. C++ Implementation (`.h`, `.cpp`)

To avoid "incomplete type" C++ compilation errors, a specific file structure and include order is required:

1.  **`MyDialect.h`:** Standard dialect header.
2.  **`MyOps.h`:** Header for the dialect's operations. It must include `mlir/IR/OpDefinition.h` and the generated `...h.inc` file inside a `#define GET_OP_CLASSES` block.
3.  **`MyDialect.cpp`:** The dialect's main source file. It **must** include its own operations header (e.g., `#include "MyDialect/MyOps.h"`) before the `addOperations<...>()` call.
4.  **`MyOps.cpp`:** A small "shell" source file whose only purpose is to include and materialize the generated C++ definitions from the `...cpp.inc` file.

## 4. Summary Checklist

When adding a new operation or dialect:

- [ ] Create an isolated library for the dialect.
- [ ] Use the `mlir_tablegen` command in your `CMakeLists.txt`.
- [ ] Use "in-situ" syntax for any properties in your `.td` files.
- [ ] Create both `MyDialect.cpp` and `MyOps.cpp` files.
- [ ] Ensure `MyDialect.cpp` includes `MyOps.h`.
- [ ] Add the new library to the parent `CMakeLists.txt` with `add_subdirectory()`.
- [ ] Link the new library against the `orchestra-opt` executable.
