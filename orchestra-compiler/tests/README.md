# Orchestra Compiler Test Suite

This directory contains the regression and unit test suite for the Orchestra compiler. The tests are built on top of the LLVM/MLIR testing infrastructure, specifically using the **LLVM Integrated Tester (`lit`)**.

## Testing Philosophy

The primary goal of this test suite is to ensure the correctness and stability of the Orchestra dialect, its operations, and its transformation passes. All new features, bug fixes, or refactors must be accompanied by corresponding tests.

## How Tests Work

Tests are written as `.mlir` files that contain one or more MLIR operations. The tests use a special `RUN` command, which is a comment that `lit` parses and executes as a shell command.

The `RUN` line typically invokes `orchestra-opt` on the test file itself and pipes the output to `FileCheck`. `FileCheck` is a utility that verifies that the output of the command matches a set of expected patterns, which are specified in the test file using `CHECK:` prefixes.

### Example Test (`speculate.mlir`)

```mlir
// RUN: orchestra-opt %s -divergence-to-speculation | FileCheck %s

// CHECK-LABEL: func @speculate_if(
// CHECK-NOT:   scf.if
// CHECK:       orchestra.task
// CHECK:       orchestra.task
// CHECK:       orchestra.commit
func.func @speculate_if(%arg0: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %res = scf.if %arg0 -> i32 {
    scf.yield %c0 : i32
  } else {
    scf.yield %c1 : i32
  }
  return %res : i32
}
```

In this example:
1.  The `RUN` line executes `orchestra-opt` on this file (`%s`), runs the `-divergence-to-speculation` pass, and pipes the result to `FileCheck`.
2.  `FileCheck` then verifies that the output contains the patterns specified by the `CHECK:` comments. It ensures that the `scf.if` operation has been replaced by two `orchestra.task` operations and one `orchestra.commit` operation.

## Running the Tests

The tests are executed as part of the build process using the `check-orchestra` target.

```bash
# From the repository root
cmake --build orchestra-compiler/build --target check-orchestra
```

This command will automatically discover all the test files in this directory, run them through `lit`, and report any failures.

## Adding a New Test

To add a new test:

1.  Create a new `.mlir` file in this directory.
2.  Write the MLIR IR that you want to test.
3.  Add a `RUN` line that invokes `orchestra-opt` with the appropriate passes.
4.  Add `CHECK:` lines to verify the output of the transformation.
5.  Re-run the test suite using the `check-orchestra` target to ensure your new test passes and that you haven't introduced any regressions.
