## 1. Project Overview

The OrchestraOS compiler is an ambitious project to build a "meta-OS" that elevates scheduling and data movement to first-class citizens of the compiler's IR. The goal is to enable global, hardware-aware optimization of complex workloads, particularly in the domain of AI and machine learning.

The core of the project is the `orchestra-compiler`, which implements a custom MLIR dialect (`OrchestraIR`) and a set of tools for processing it.


### 2. Build and Test Instructions

The project uses a standard CMake-based workflow.

```bash
# 1. Configure the build using CMake.
#    This command should be run from the root of the repository.
#    It tells CMake where the source code is (-S) and where to put the build artifacts (-B).
cmake -S orchestra-compiler -B orchestra-compiler/build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DLLVM_TOOLS_DIR=/usr/lib/llvm-20/bin

# 2. Build the compiler and its tools.
cmake --build orchestra-compiler/build

# 3. Run the test suite.
#    This will build and run all tests, including the new ones for any new features.
cmake --build orchestra-compiler/build --target check-orchestra
```

This will produce the `orchestra-opt` executable in the `orchestra-compiler/build/orchestra-opt/` directory and run the full test suite to verify correctness.

---

## 3. Agent Workflow Protocol

This section outlines the mandatory protocol for working on this repository.

### Core Principles

*   **Work as autonomously as possible.** Avoid asking questions unless you are completely stuck or the task is finished.
*   Do not ask for intermediate confirmation (e.g., "Does this seem like the correct approach?"). If you feel the need to ask such a question, instead proceed with your proposed plan. My confirmation is granted by default.

### The Development Cycle

1.  **Sanity Check:** Before starting any new work, test the project as it is to ensure the baseline is stable. If this fails, your first task is to fix it.

2.  **Understand Context:** Thoroughly review the repository structure and all available documentation, especially the files linked in the "Documentation" section below.

3.  **Decide and Implement:** Check the repository (especially status-quo.md) against to_do.md and decide on the next task. Frame and formulate a small, testable, and achievable step. Do not solve a large problem in one chunk.
    *   The blueprints in docs/architecture are very important to understand in-depth.
    *   Use the documentation in the 'docs/guides' directory
    *   research leveraging your Google search tool.

4.  **Verify Continuously:** Use self-verification loops (running tests, checking logs) to confirm your changes are successful and correct.

### Problem-Solving Strategy

*   **DO NOT GIVE UP.**
*   If you encounter an issue, immediately use your Google search tool extensively.
*   If you are still stuck after searching, reframe the problem. Try to miniaturize it and solve a smaller, testable part.
*   Repeat this loop at least twice, ensuring each attempt is a new approach. Do not repeat the same errors.
*   Only after at least 20 failed attempts may you ask for help by writing a comprehensive deep-research question for a separate agent. This question must be specific and include all details, such as package versions and a full description of what you have already tried and why it failed.

### Finalizing Work and Committing

1.  **Ensure All Tests Pass:** First, ensure that **ALL** tests pass successfully. Do not proceed otherwise.

2.  **Prepare the Commit (Documentation First):** This step is a single, atomic action. You are not authorized to create a commit unless it includes the following documentation updates.
    *   **Update `docs/status-quo.md`:** Reflect the new state of the project.
    *   **Update local READMEs:** If you changed a specific component, update its README with details.
    *   **Commit:** Once, and only once, the documentation is updated and staged alongside the code changes, create a single, comprehensive commit. A commit that does not include documentation updates is a violation of this protocol.


**VERY IMPORTANT: If you find that ANY section of this whole AGENTS.md file is misleading or wrong, or you need to debug before anything real works, please feel free to change this file**