# Declarative Optimization Framework

This directory is intended to contain a library of target-specific transform scripts for the Orchestra compiler. These scripts are written in the MLIR `transform` dialect and are used to control the hardware-aware optimization process.

## Overview

The Orchestra compiler uses a declarative approach to hardware-aware optimization. Instead of hardcoding optimization strategies in C++ passes, we use scripts written in the `transform` dialect to orchestrate the application of transformations. This approach provides several advantages:

*   **Agility:** Performance engineers can easily write and modify optimization scripts without recompiling the compiler.
*   **Composability:** Complex optimization pipelines can be built by composing simpler transformations.
*   **Clarity:** The optimization strategy is expressed in a clear, declarative way.

## Usage

The `-transform-interpreter` pass is used to apply a transform script to a payload IR. The pass takes the path to the transform script as an argument.

For example, to apply a transform script to an input file, you would run:

```
orchestra-opt input.mlir -transform-interpreter=path/to/your_transform_script.mlir
```

## Available Scripts

This directory is currently a placeholder for where target-specific transform scripts will be stored. As we develop optimization strategies for different hardware backends (e.g., NVIDIA, AMD), this directory will be populated with the corresponding `.mlir` transform files.
