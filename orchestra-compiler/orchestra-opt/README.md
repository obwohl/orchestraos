# Orchestra Optimizer (`orchestra-opt`)

This directory contains the source code for `orchestra-opt`, the core command-line tool for the Orchestra compiler.

## Purpose

`orchestra-opt` is a customized version of MLIR's standard `mlir-opt` tool. It serves two primary purposes:

1.  **Registers the Orchestra Dialect:** It registers the `Orchestra` dialect and all its associated operations, types, and interfaces with the MLIR context. This allows the tool to parse, process, and print MLIR files containing Orchestra IR.
2.  **Registers Custom Passes:** It registers the custom transformation passes defined for the Orchestra compiler, making them available to be run via the `--pass-pipeline` option.

## Key Files

*   **`main.cpp`**: The main entry point for the `orchestra-opt` executable. Its primary responsibility is to register the necessary dialects and passes and then delegate to the standard MLIR command-line driver.

## Usage

The `orchestra-opt` tool is built in the `orchestra-compiler/build/orchestra-opt/` directory. For detailed usage instructions and common options, please refer to the `README.md` in the parent `orchestra-compiler` directory.
