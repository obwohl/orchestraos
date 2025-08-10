# Contributing to OrchestraOS

Welcome to the OrchestraOS project! We appreciate your interest in contributing.

This document outlines the guidelines and best practices for contributing to the OrchestraOS codebase, with a particular focus on documentation. Given the complexity and rapid evolution of this project, clear and consistent documentation is paramount.

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
*   **Build/Test Instructions:** If applicable, provide concise instructions for building and testing the component in isolation or as part of the larger system. Link to more detailed build documentation (e.g., `docs/guides/cmake-build-guide.md`).
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