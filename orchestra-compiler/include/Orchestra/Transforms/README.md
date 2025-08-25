# Orchestra Transformation Pass Declarations

This directory contains the header files that declare the transformation passes for the Orchestra compiler.

## Purpose

The primary role of the files in this directory is to define the public C++ interface for the compiler's custom passes. This typically involves declaring the factory functions that are used to create instances of the passes.

## Key Files

*   **`Passes.h`**: This is the main header file for the transformation passes. It declares the `create...Pass()` factory functions for all passes that are part of the Orchestra compiler. These functions are then used to register the passes with the `PassManager` and make them available to tools like `orchestra-opt`.

## Implementation vs. Declaration

It is important to note that this directory only contains the *declarations* of the passes. The actual C++ *implementations* are located in the corresponding directory under `orchestra-compiler/lib/Orchestra/Transforms/`. This separation of interface and implementation is a standard software engineering practice and is followed throughout the MLIR codebase.
