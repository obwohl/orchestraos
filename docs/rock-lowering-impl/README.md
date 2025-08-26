# Implementation Plan: Lowering to Rock Dialect

## 1. Objective

This document outlines the plan to implement a new `Rock` MLIR dialect within the `orchestra-compiler`. This dialect will serve as a lowering target for AMD GPUs, starting with a `rock.gemm` operation.

**The implementation must follow the modern MLIR approach using `Properties` as detailed in the accompanying `rock_2.md` file.**

## 2. Key Resources

*   **`rock_2.md`**: The strategic guide for this task. It mandates the use of `Properties`.
*   **`rocops.td`**: The original TableGen file project. This is the **technical specification** for the operation attributes.

https://github.com/ROCm/rocMLIR

## 3. Actionable Implementation Steps

### Step 1: Create Directory Structure

Create the following directory structure inside the `orchestra-compiler/` directory:

```
orchestra-compiler/
├── include/
│   └── Orchestra/
│       └── Dialects/
│           └── Rock/
└── lib/
    └── Orchestra/
        └── Dialects/
            └── Rock/
```

### Step 2: Create and Implement `RockOps.td`

1.  Create a new file: `orchestra-compiler/include/Orchestra/Dialects/Rock/RockOps.td`.
2.  Define the `Rock` dialect in this file.
3.  Define the `Rock_GemmOp` using the `OpWithProperties` base class.
4.  The properties of this operation should be derived directly from the attributes in the reference `rocops.td` file. Pay close attention to the data types.

**Example Structure for `RockOps.td`:**

```tablegen
#ifndef ORCHESTRA_DIALECT_ROCK_OPS_TD
#define ORCHESTRA_DIALECT_ROCK_OPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::orchestra::rock";
  let summary = "A dialect for lowering to AMD ROCm primitives.";
}

// Define a reusable Property for the architecture
def Rock_Architecture : StringAttr<"Architecture string (e.g., 'gxf908')">;

class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits>;

def Rock_GemmOp : Rock_Op<"gemm", [NoSideEffect]> {
  let summary = "A generalized matrix multiplication operation for Rock devices.";

  let arguments = (
    ins
      Matrix_t:$matrixA,
      Matrix_t:$matrixB,
      Matrix_t:$matrixC
  );

  let results = (
    outs
      Matrix_t:$result
  );

  let properties = (
    ins
      // Infer the properties and types from rocops.td
      // Example:
      Rock_Architecture:$architecture,
      I64Attr:$numCU,
      StringAttr:$filterLayout,
      StringAttr:$inputLayout,
      StringAttr:$outputLayout
      // ... add all other relevant attributes from rocops.td as properties
  );

  let assemblyFormat = [{
    `(`$matrixA`, `$matrixB`, `$matrixC`)`
    `attr_dict`
    `:` functional_type($matrixA, $matrixB, $matrixC, $result)
  }];
}

#endif // ORCHESTRA_DIALECT_ROCK_OPS_TD
```

### Step 3: Implement C++ Dialect Files

1.  Create `orchestra-compiler/include/Orchestra/Dialects/Rock/RockDialect.h`.
2.  Create `orchestra-compiler/lib/Orchestra/Dialects/Rock/RockDialect.cpp`.
3.  Implement the basic dialect class structure. The TableGen files will generate most of the operation-specific code for you.

### Step 4: Integrate with Build System (CMake)

Update the `CMakeLists.txt` files to include the new dialect.

1.  **In `orchestra-compiler/include/Orchestra/Dialects/CMakeLists.txt`** (or equivalent), add:
    ```cmake
    add_subdirectory(Rock)
    ```
2.  **In `orchestra-compiler/lib/Orchestra/Dialects/CMakeLists.txt`** (or equivalent), add:
    ```cmake
    add_subdirectory(Rock)
    ```
3.  **Create `orchestra-compiler/lib/Orchestra/Dialects/Rock/CMakeLists.txt`** and add the following to build the C++ files:
    ```cmake
    add_orchestra_dialect_library(OrchestraRockDialect
      RockDialect.cpp
      DEPENDS
      OrchestraRockDialectIncGen
    )
    ```
4.  **Create `orchestra-compiler/include/Orchestra/Dialects/Rock/CMakeLists.txt`** and add the following to process the TableGen file:
    ```cmake
    include(MLIRTableGen)

    add_mlir_tablegen(RockOps.h.inc -gen-op-decls)
    add_mlir_tablegen(RockOps.cpp.inc -gen-op-defs)

    add_public_tablegen_target(OrchestraRockDialectIncGen)
    ```

*(Note: The exact CMake function names (`add_orchestra_dialect_library`, etc.) might need to be adapted to your project's specific CMake macros.)*

### Step 5: Implement the Lowering Pass

1.  Create a new pass file, e.g., `orchestra-compiler/lib/Orchestra/Transforms/LowerToRock.cpp`.
2.  Implement the logic to convert a higher-level `gemm` operation (e.g., `linalg.gemm`) into your new `rock.gemm` op.
3.  Register the new pass so it can be used by `orchestra-opt`.

### Step 6: Final Verification

Ensure the entire `orchestra-compiler` project compiles successfully after making these changes.
