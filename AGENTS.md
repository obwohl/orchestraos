## 1. Project Overview

The OrchestraOS compiler is an ambitious project to build a "meta-OS" that elevates scheduling and data movement to first-class citizens of the compiler's IR. The goal is to enable global, hardware-aware optimization of complex workloads, particularly in the domain of AI and machine learning. The blueprint.md in docs/architecture is the ground-truth for our whole project. *Read and understand it carefully.*

The core of the project is the `orchestra-compiler`, which implements a custom MLIR dialect (`OrchestraIR`) and a set of tools for processing it.

---

## 2. Agent Workflow Protocol

This section outlines the mandatory protocol for working on this repository.

### Core Principles

*   **Work as autonomously as possible.** Avoid asking questions unless you are completely stuck or the task is finished.
*   Do not ask for intermediate confirmation (e.g., "Does this seem like the correct approach?"). If you feel the need to ask such a question, instead proceed with your proposed plan. My confirmation is granted by default.

### The Development Cycle

1.  **Sanity Check:** Before starting any new work, test the project as it is to ensure the baseline is stable. If this fails, your first task is to fix it.

2.  **Understand Context:** Thoroughly review the repository structure and all available documentation, especially the files linked in the "Documentation" section below.

3.  **Decide and Implement:** Check the docs/project-status/status.md and decide on the next task. Frame and formulate a small, testable, and achievable step. Do not solve a large problem in one chunk.
    *   The blueprint.md in docs/architecture is the ground-truth for our whole project. *Read and understand it carefully.*
    *   Use the documentation in the 'docs/guides' directory
    *   research leveraging your Google search tool.

4.  **Verify Continuously:** Use self-verification loops (running tests, checking logs) to confirm your changes are successful and correct.

### Problem-Solving Strategy

*   **DO NOT GIVE UP.**
*   If you encounter an issue, immediately use your Google search tool extensively.
*   If you are still stuck after searching, reframe the problem. Try to miniaturize it and solve a smaller, testable part.
*   Repeat this loop at least twice, ensuring each attempt is a new approach. Do not repeat the same errors.
*   Only after at least 10 consecutive failed attempts (without any progress) you may ask for help by writing a comprehensive deep-research question for a separate agent. This question must be specific and include all details, such as package versions and a full description of what you have already tried and why it failed.
*   When facing apparent difficulties with the file-system, first try to find out, where you are (pwd), make some basic sanity checks (like navigating), make yourself an overview about the whole environment via ls -R and then try to run the commands in smaller chunks, not all in one large chunk. That prevents time-outs.


### Finalizing Work and Committing

1.  **Ensure All Tests Pass:** First, ensure that **ALL** tests pass successfully. Do not proceed otherwise.

2.  **Prepare the Commit (Documentation First):** This step is a single, atomic action. You are not authorized to create a commit unless it includes the following documentation updates.
    *   **Update `docs/project-status/status.md`:** Reflect the new state of the project.
    *   **Update local READMEs:** If you changed a specific component, update its README with details.
    *   **Commit:** Once, and only once, the documentation is updated and staged alongside the code changes, create a single, comprehensive commit. A commit that does not include documentation updates is a violation of this protocol.

---

## 3. Build and Test Instructions

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

# 4. Contributing to OrchestraOS
Given the complexity and rapid evolution of this project, clear and consistent documentation is paramount.

## Development Workflow

This project uses a standard CMake-based build system.

A key part of the development process is testing. All new features must be accompanied by tests, and all tests must pass before a change is submitted. The test suite is built on top of LLVM's `lit` and can be run using the `check-orchestra` target in CMake.

## Documentation Principles

Our documentation strategy is designed to support developers working on the core compiler, especially during the scaffolding and iterative development phases. It aims to capture critical insights, common pitfalls, and design rationale that might not be immediately obvious from the code alone.

1.  **Developer-Centric:** Documentation is primarily for fellow and future developers. It should be technical, precise, and focused on enabling efficient understanding and modification of the codebase.
2.  **Contextual & Discoverable:** Documentation should live as close as possible to the code it describes. Information should be easy to find when navigating the project structure.
3.  **Actionable & Concise:** Guidelines should be clear, brief, and provide concrete examples where applicable. Avoid unnecessary prose.
4.  **Living Document:** Documentation is not a one-time effort. It must evolve with the codebase. If code changes, relevant documentation must be updated. If a new insight or workaround is discovered, it should be documented.
5.  **Focus on "Why":** While code explains "what" is being done, documentation should explain "why" it's being done, especially for non-obvious design choices, workarounds, or complex interactions.

## Documentation Guidelines

### README Guidelines

Every significant directory or component should contain a `README.md` file. These READMEs serve as entry points for understanding that specific part of the codebase.

*   **Purpose:** Clearly state the purpose and role of the directory/component within the larger OrchestraOS project.
*   **Key Files:** List and briefly describe the most important files within the directory.
*   **Build/Test Instructions:** If applicable, provide concise instructions for building and testing the component in isolation or as part of the larger system. 
*   **Common Pitfalls & Hot Tips:** Include a dedicated section for common issues, workarounds, and hard-won knowledge specific to this component. This is where subtle MLIR interactions, unexpected build behaviors, or non-obvious API usages should be documented.

### Code Comment Guidelines

Code comments should be used sparingly but effectively. They are not a substitute for clear, self-documenting code.

*   **Focus on "Why":** The primary purpose of a comment is to explain *why* a particular piece of code exists, *why* a specific design choice was made, or *why* a non-obvious workaround was implemented. Avoid comments that merely restate what the code is doing.
*   **Complex Logic:** Use comments to clarify complex algorithms, intricate data structures, or subtle interactions between components.
*   **Workarounds & Hacks:** Document any workarounds, hacks, or temporary solutions, explaining the underlying problem and what a proper fix would entail.
*   **External References:** If a solution or design was inspired by an external resource (e.g., an MLIR forum post, a research paper, a specific LLVM commit), include a link to that resource in the comment.
*   **Avoid Redundancy:** Do not comment on obvious code. If the code is hard to understand, refactor it first.

### Architectural Decision Record (ADR) Guidelines

For significant architectural or design decisions, an Architectural Decision Record (ADR) should be created. ADRs provide a historical log of decisions and their rationale.

*   **Location:** ADRs should reside in a dedicated `docs/adrs/` directory.
*   **Format:** Each ADR should be a separate Markdown file, named systematically (e.g., `ADR-001-DecisionName.md`).
*   **Content:** Each ADR should typically include:
    *   **Title:** A concise, descriptive title.
    *   **Status:** (e.g., Proposed, Accepted, Superseded, Deprecated).
    *   **Context:** The problem or situation that led to the decision.
    *   **Decision:** The chosen solution or approach.
    *   **Alternatives:** Other options considered and why they were rejected.
    *   **Consequences:** The positive and negative impacts of the decision.

### Troubleshooting & Hot Tips Guidelines

Hard-won knowledge and solutions to common or particularly difficult problems should be captured to prevent future developers from repeating the same debugging efforts.

*   **Location:** These tips can be integrated into relevant `README.md` files (for component-specific issues) or collected in a central `docs/TROUBLESHOOTING.md` for broader, cross-cutting issues.
*   **Content:**
    *   **Problem/Symptom:** Clearly describe the error message, unexpected behavior, or challenge encountered.
    *   **Root Cause:** Explain the underlying reason for the problem.
    *   **Solution:** Provide the steps or code changes required to resolve the issue.
    *   **Context/Caveats:** Add any important context, limitations, or alternative solutions.
    *   **Example:** For MLIR-related issues, include the exact error message, the relevant code snippet, and the fix.
