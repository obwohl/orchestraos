# Current Build Status and Troubleshooting Guide

Last updated: 2025-08-11

## 1. Overview

This document provides a summary of the current build status of the `orchestra-compiler` project and a guide for troubleshooting the issues that have been encountered.

**The project is currently unbuildable.** The build fails with a series of C++ compilation errors that stem from a non-standard project structure and incorrect handling of MLIR's TableGen-generated files.

This guide will walk through the debugging process that has been performed so far and provide a clear path to resolving the build issues.

## 2. The Core Problem: Project Structure

The fundamental issue is that this project does not follow the canonical MLIR out-of-tree project structure. The `cmake-build-guide.md` in this directory provides a detailed explanation of the correct structure, which is to separate the project into `include/` and `lib/` directories. This project has a flat structure, which leads to a number of problems with CMake's dependency management.

## 3. Debugging Log and Resolved Issues

The following is a log of the issues that have been encountered and resolved so far.

### 3.1. TableGen Syntax Errors

The initial build failed due to syntax errors in `include/Orchestra/OrchestraOps.td`. These were corrected.

### 3.2. Missing `BytecodeOpInterface` Header

After fixing the TableGen errors, the build failed with errors indicating that `mlir::BytecodeOpInterface` was not defined. This was resolved by adding `#include "mlir/Bytecode/BytecodeOpInterface.h"` to `lib/Orchestra/OrchestraDialect.cpp`.

### 3.3. Persistent "Type Not Found" Errors

The final and most persistent error is `‘CommitOpGenericAdaptorBase’ does not name a type`. This error, and many others like it, occurs during the compilation of `lib/Orchestra/OrchestraDialect.cpp`. This error is a direct result of the incorrect project structure.

## 4. Path to Resolution

The path to fixing the build is to restructure the project to follow the guidelines in `cmake-build-guide.md`. This involves the following steps:

1.  **Restructure the project:** Create `include/` and `lib/` directories and move the files as described in the guide.
2.  **Update the CMakeLists.txt files:** The root `CMakeLists.txt` and the new `CMakeLists.txt` files in the `include/` and `lib/` directories must be updated to match the guide.
3.  **Standardize `OrchestraDialect.cpp`:** The `lib/Orchestra/OrchestraDialect.cpp` file should be updated to use the standard, simplified MLIR pattern for dialect implementation, as described in the guide.

By following these steps, the build should be successfully restored.
