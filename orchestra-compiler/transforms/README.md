# Declarative Optimization Framework

This directory contains a library of target-specific transform scripts for the Orchestra compiler. These scripts are written in the MLIR `transform` dialect and are used to control the hardware-aware optimization process.

## Overview

The Orchestra compiler uses a declarative approach to hardware-aware optimization. Instead of hardcoding optimization strategies in C++ passes, we use scripts written in the `transform` dialect to orchestrate the application of transformations. This approach provides several advantages:

*   **Agility:** Performance engineers can easily write and modify optimization scripts without recompiling the compiler.
*   **Composability:** Complex optimization pipelines can be built by composing simpler transformations.
*   **Clarity:** The optimization strategy is expressed in a clear, declarative way.

## Usage

The `-transform-interpreter` pass is used to apply a transform script to a payload IR. The pass takes the path to the transform script as an argument.

For example, to apply the `fuse_generic_ops.mlir` script to an input file, you would run:

```
orchestra-opt input.mlir -transform-interpreter=path/to/fuse_generic_ops.mlir
```

## Available Scripts

*   `fuse_generic_ops.mlir`: Fuses a sequence of `linalg.generic` operations. (Note: this script is currently a placeholder and does not perform fusion correctly. It is being used to test the integration of the transform dialect.)
