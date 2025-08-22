# Research Summary: MLIR PDL and PDLL

This document summarizes the findings from researching the MLIR Pattern Description Language (PDL) and its frontend, PDLL. This research was conducted to support **Task 1.2** of the modernization plan: rewriting the C++ `SpeculateIfOpPattern` using PDL.

## Key Concepts

### PDLL: The Language for Patterns

PDLL is a declarative language for defining MLIR rewrite patterns in `.pdll` files. It is designed to be more intuitive and structured than the older TableGen-based DRR (Declarative Rewrite Rules).

- **`Pattern` Definition**: Patterns are defined with the `Pattern` keyword. They can be named and assigned a `benefit` to guide the pattern application strategy.
  ```pdll
  Pattern ReshapeReshapeOptPattern with benefit(10) {
    ...
  }
  ```

- **Matcher and Rewriter Sections**: A pattern body is logically separated into two parts:
    1.  **Matcher**: Describes the input IR graph to be matched. This consists of all statements before the final rewrite operation.
    2.  **Rewriter**: Describes the transformation to be applied. This is initiated by the last statement in the pattern, which must be an `erase`, `replace`, or `rewrite` statement.

### Core Constructs

- **`let` variables**: Used to declare variables for MLIR entities like `Op`, `Value`, `Type`, and `Attr`.
- **`op<...>` expression**: Used to match existing operations or create new ones. The syntax is similar to the generic MLIR textual format.
- **`.td` Includes**: PDLL files can `#include` TableGen (`.td`) files. This is crucial as it imports Operation Definition Specification (ODS) information, allowing PDLL to understand an operation's arguments, results, and named attributes, leading to more powerful and readable patterns.

### Integrating with C++

The most critical feature for Task 1.2 is the ability to mix declarative PDLL matching with imperative C++ rewriting.

- **Native Rewriters**: A C++ function can be exposed to PDLL as a "native rewriter." This allows the complex rewrite logic to remain in C++, while the pattern matching is handled declaratively by PDLL.

- **Declaration in PDLL**: The C++ function is declared in the `.pdll` file with its signature.

  ```pdll
  // Declares a C++ function `MyRewriteFunction` that takes a Value
  // and returns a new Operation.
  Rewrite MyRewriteFunction(value: Value) -> Op;
  ```

- **Definition with C++ Implementation**: The C++ implementation can be provided directly within the `.pdll` file inside a `[{ ... }]` block. This is the mechanism we will use.

  ```pdll
  Rewrite MyCppRewrite(value: Value) -> Op [{
    // Native C++ code goes here.
    // The 'rewriter' object is available implicitly.
    // The arguments (e.g., 'value') are available by name.
    return rewriter.create<MyNewOp>(value.getLoc(), value);
  }];
  ```

- **Calling from a Pattern**: Once defined, the native rewriter can be called from the rewrite section of a pattern.

  ```pdll
  Pattern MyPattern {
    // Matcher section
    let root = op<my_dialect.old_op>(input: Value);

    // Rewriter section
    replace root with MyCppRewrite(input);
  }
  ```

## Plan for Task 1.2

Based on this research, the plan is as follows:
1.  **Isolate**: Identify the existing C++ `SpeculateIfOpPattern`.
2.  **Extract**: Refactor the rewrite logic of the pattern into a standalone C++ function.
3.  **Declare**: Create a new `.pdll` file.
4.  **Match**: In the `.pdll` file, write a declarative `Pattern` that matches the same IR structure as the original C++ pattern.
5.  **Rewrite**: In the same `.pdll` file, define a native `Rewrite` that wraps the extracted C++ function.
6.  **Connect**: Call the native `Rewrite` from the `Pattern`.
7.  **Integrate**: Update the build system (CMake) to process the new `.pdll` file.
8.  **Replace**: Remove the old C++ `OpRewritePattern` from the pass.
9.  **Verify**: Build the compiler and run the test suite to ensure the behavior is identical.
