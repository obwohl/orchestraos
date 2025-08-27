# A Fresh Start: Implementing the `rock` Dialect Correctly

Hello Jules,

Let's tackle the `rock` dialect implementation with a clear and correct strategy.

This document provides the definitive path forward. Please follow these instructions precisely.

## 1. Critical Clarifications (The Source of Past Issues)

Before you begin, you must understand the following:

*   **`RockOps.td` is a Reference, NOT a Source File:** The `RockOps.td` file located in this directory is a **blueprint** from the external `rocMLIR` project. It was provided for conceptual guidance only. **Do not use it directly in the build.** Your previous attempts to compile it failed because its dependencies do not exist in this project, which is the expected outcome. You will be creating a new `RockOps.td` from scratch.

*   **The Goal is a NEW, Self-Contained Dialect:** Your mission is to create a **brand new, self-contained `rock` dialect** within the `orchestra-compiler`. This means you will create new `.td` and `.cpp/.h` files inside the `orchestra-compiler/` source tree.

*   **`Properties` are Mandatory:** The new dialect **must** use MLIR's modern `Properties` system for defining operation attributes, as detailed in the `rock_2.md` document. Your previous attempts were conceptually correct but lacked the specific boilerplate required by this project.

## 2. The Correct Implementation Plan

Follow these steps to solve the build errors and complete the task.

### Step 2.1: Define the TableGen Files (The Core Solution)

This is the most critical step. You will create the `rock` dialect's TableGen definitions inside a new directory: `orchestra-compiler/include/Orchestra/Dialects/Rock/`.

**A. `RockDialect.td`**

Create the file `orchestra-compiler/include/Orchestra/Dialects/Rock/RockDialect.td`. This file defines the dialect and, crucially, enables the `Properties` feature.

```tablegen
#ifndef ORCHESTRA_DIALECTS_ROCK_DIALECT_TD
#define ORCHESTRA_DIALECTS_ROCK_DIALECT_TD

include "mlir/IR/Dialect.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::rock";
  let summary = "The Rock dialect, based on rocMLIR.";

  // CRITICAL: This enables the 'properties' feature for ops in this dialect.
  // This is the flag that makes the 'let properties = ...' block valid.
  let usePropertiesForAttributes = 1;
}

#endif // ORCHESTRA_DIALECTS_ROCK_DIALECT_TD
```

**B. `RockOps.td`**

Create the file `orchestra-compiler/include/Orchestra/Dialects/Rock/RockOps.td`. This is where you will define the operations. Start with `rock.gemm` as a working example.

```tablegen
#ifndef ORCHESTRA_DIALECTS_ROCK_OPS_TD
#define ORCHESTRA_DIALECTS_ROCK_OPS_TD

include "mlir/IR/OpBase.td"
include "Orchestra/Dialects/Rock/RockDialect.td"

// Base class for all Rock ops.
class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits>;

// --- rock.gemm Definition ---
// This is the canonical syntax for defining an op with Properties in this project.
def Rock_GemmOp : Rock_Op<"gemm"> {
  let summary = "rock.gemm operation";

  let arguments = (ins
    F32Tensor:$a,
    F32Tensor:$b
  );

  let results = (outs
    F32Tensor:$c
  );

  // This 'properties' block is now valid because 'usePropertiesForAttributes'
  // is set on the Rock_Dialect definition included above. This avoids the
  // "Value 'properties' unknown!" error. You also do NOT need to inherit
  // from OpWithProperties directly.
  let properties = (ins
    "GemmAttributes":$arch
  );

  let assemblyFormat = [{
    `(`$a`, `$b`)` `->` `(`$c`)` attr-dict
  }];
}

#endif // ORCHESTRA_DIALECTS_ROCK_OPS_TD
```

### Step 2.2: Implement the C++ Source Files

Create the necessary C++ files (`RockDialect.cpp`, etc.) in `orchestra-compiler/lib/Orchestra/Dialects/Rock/`. Use the existing `Orchestra` dialect's C++ files as a direct template for structure and content.

### Step 2.3: Integrate with CMake

Your previous plan for CMake was correct. Now, execute it with the correct TableGen files.

1.  **IncGen Target:** In `orchestra-compiler/include/Orchestra/CMakeLists.txt`, add a new target `OrchestraRockIncGen` that points to your new `RockDialect.td` and `RockOps.td` files. Mirror the existing `OrchestraIncGen` definition.
2.  **Main Library:** In `orchestra-compiler/lib/Orchestra/CMakeLists.txt`, modify the `add_mlir_dialect_library(Orchestra ...)` call to:
    *   Add your new C++ source file (`Dialects/Rock/RockDialect.cpp`).
    *   Add `OrchestraRockIncGen` to the `DEPENDS` section.
    *   Ensure there is no `LINK_LIBS` entry for a separate `OrchestraRock` library.

## 3. Summary

To be perfectly clear: **Your task is to build a new `rock` dialect from scratch, using the code provided in this README as your starting point and the existing `Orchestra` dialect as your guide.** 

You have the correct analysis and a clear, working plan. Proceed with these instructions.