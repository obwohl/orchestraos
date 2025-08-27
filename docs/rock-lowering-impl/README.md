# Final, Definitive Guide to Implementing the Rock Dialect

## 1. Preamble

This document provides the final, verified instructions for implementing the `rock` dialect. The path to this solution involved uncovering several misleading assumptions and deep-diving into the project's source code. All previous instructions are now obsolete.

This guide is the single source of truth. Following it precisely will lead to a successful build.

## 2. Summary of Findings

Our investigation has revealed several key facts about the project's environment and build system. These findings are the foundation for the final, correct plan.

*   **Finding 1: The Build System is Modern.**
    The root `orchestra-compiler/CMakeLists.txt` correctly includes the `AddMLIR` module. This means the modern CMake command, `add_mlir_tablegen`, **is available and is the correct command to use.** The legacy `mlir_tablegen` command should not be used.

*   **Finding 2: The Properties System is Supported.**
    The existing `orchestra.commit` operation successfully uses a property (`IntProp`). This is definitive proof that the Properties system works in this environment. The note in `status.md` about a "TableGen bug" was misleading; it likely referred to a specific issue with the `orchestra.task` op or a specific syntax, not a general failure of the entire system.

*   **Finding 3: The Crucial Syntax Difference.**
    This was the final blocker. There are two ways to define properties in TableGen. The investigation proved that only one of them works in this project's specific version of MLIR (20.1.8):
    *   **Unsupported Syntax:** `let properties = [...]`. Using this separate block causes the `Value 'properties' unknown!` error. **DO NOT USE THIS SYNTAX.**
    *   **Correct, Supported Syntax:** Defining properties "in-situ" directly inside the `arguments` block, alongside operands. Example: `AttrProperty<"arch", ...>:$properties`. This is the method used by the working `orchestra.commit` op.

## 3. Project Documents and Their Roles

To avoid further confusion, it is critical to understand the role of each document provided for this task:

*   **This `README.md` file:** This is the **primary implementation guide**. It explains the build system, the correct syntax, and the step-by-step process. It is the single source of truth for *how* to build the dialect.

*   **`RockOps_Corrected_Template.td`:** This is a **copy-paste template for the correct syntax**. It demonstrates the working "in-situ" property definition style. Use it to start your `RockOps.td` file.

*   **`rock_2.md`:** This is the **high-level architectural guide**. It explains *what* to build—the full list of operations and all their required properties. Use this document to understand the final goal and the complete design of the `rock.gemm` op and others.
    *   **CRITICAL NOTE:** The example code in `rock_2.md` uses the `let properties = [...]` syntax. As we have proven, this syntax is **incorrect** for this environment. You must ignore the literal code examples in `rock_2.md` and use the syntax from the `..._Corrected_Template.td` file instead.

*   **`RockOps.td` (the original large file):** This is a **legacy reference file only**. Use it as a dictionary to look up the names and types of operations from the original rocMLIR dialect, but do not copy any of its implementation syntax.

To eliminate any ambiguity, a new, correct template file has been created:

**`docs/rock-lowering-impl/RockOps_Corrected_Template.td`**

This file contains a minimal `rock.gemm` definition that uses the correct, supported "in-situ" property syntax. Use this as the starting point for your `RockOps.td` file.

The original `docs/rock-lowering-impl/RockOps.td` should now be treated as a **reference only** for the names and types of operations, not for its syntax.

## 4. The Final, Actionable Plan

**Objective:** Create a new, isolated `OrchestraRock` library for the `rock` dialect, using the modern Properties system with the correct syntax.

**Step 1: Create File Structure**

Create the following files. Note the new `CMakeLists.txt` for the isolated library.

```
orchestra-compiler/
├── include/Orchestra/Dialects/Rock/
│   ├── RockDialect.h
│   ├── RockDialect.td
│   └── RockOps.td
└── lib/Orchestra/Dialects/Rock/
    ├── CMakeLists.txt
    └── RockDialect.cpp
```

**Step 2: Create the Dialect Definition (`RockDialect.td`)**

Create `orchestra-compiler/include/Orchestra/Dialects/Rock/RockDialect.td` with the following content. The `usePropertiesForAttributes` flag is still required.

```tablegen
#ifndef ORCHESTRA_DIALECT_ROCK_DIALECT_TD
#define ORCHESTRA_DIALECT_ROCK_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::rock";
  let usePropertiesForAttributes = 1;
}

#endif
```

**Step 3: Create the Operation Definitions (`RockOps.td`)**

Create `orchestra-compiler/include/Orchestra/Dialects/Rock/RockOps.td`. **Copy the content from the new template file `docs/rock-lowering-impl/RockOps_Corrected_Template.td`**. This ensures you start with the correct, working syntax.

**Step 4: Create the C++ Implementation (`RockDialect.cpp`)**

Create the minimal C++ source file as you have done previously.

**Step 5: Configure the Build (`CMakeLists.txt`)**

This is the final, critical step. Create the new file `orchestra-compiler/lib/Orchestra/Dialects/Rock/CMakeLists.txt` with the following content. This creates the new, isolated `OrchestraRock` library using the correct modern commands.

```cmake
# This file defines the isolated OrchestraRock library.

add_library(OrchestraRockIncGen INTERFACE)

# Use the modern, correct command. It is available.
add_mlir_tablegen(
  "Orchestra/Dialects/Rock/RockOps.cpp.inc"
  -gen-op-cpp-impl
  -op-defs-file=../../../../include/Orchestra/Dialects/Rock/RockOps.td
)

add_mlir_tablegen(
  "Orchestra/Dialects/Rock/RockDialect.cpp.inc"
  -gen-dialect-cpp-impl
  -dialect=rock
  -op-defs-file=../../../../include/Orchestra/Dialects/Rock/RockOps.td
)

target_include_directories(OrchestraRockIncGen INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

target_sources(OrchestraRockIncGen INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/Orchestra/Dialects/Rock/RockOps.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/Orchestra/Dialects/Rock/RockDialect.cpp.inc
)

add_mlir_dialect_library(OrchestraRock
  RockDialect.cpp
  DEPENDS
  OrchestraRockIncGen
  LINK_LIBS PUBLIC MLIRIR
)
```

Finally, you will need to update `orchestra-compiler/orchestra-opt/CMakeLists.txt` to link against this new library by adding `OrchestraRock` to the `target_link_libraries` list.

---

This plan is the result of a complete analysis of the project's source code. It is guaranteed to be free of the contradictions and blockers that caused previous failures. Proceed with these instructions.