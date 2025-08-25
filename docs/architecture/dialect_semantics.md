# Orchestra Dialect Semantics

This document provides a detailed rationale for the core operations in the Orchestra dialect. It is intended to be a reference for compiler developers working on the OrchestraOS project, to ensure a clear and consistent understanding of the dialect's design principles.

## `orchestra.select`

The `orchestra.select` operation is a conditional selection mechanism that is polymorphic and variadic. It takes a condition, two sets of SSA values, and a `num_true` attribute, and returns one of the sets of values based on the condition.

### Why not `arith.select` or `scf.if`?

A core design principle of MLIR is to reuse operations from standard dialects whenever possible. The introduction of `orchestra.select` is justified by the fact that it fills a semantic gap not covered by existing standard operations.

*   **`arith.select`:** This operation is defined to work only on scalars and vectors (`IntegerLike` or `Vector` types). The `orchestra.select` operation is designed to be type-polymorphic, allowing it to select between any number of values of any type, including `memref` and `tensor`. This is a requirement for the `--divergence-to-speculation` pass, which must handle arbitrary SSA values yielded from `scf.if` regions.

*   **`scf.if`:** The purpose of the `--divergence-to-speculation` pass is to convert region-based control flow (`scf.if`) into a dataflow representation that can be optimized by the Orchestra scheduler. Lowering back to an `scf.if` would defeat the purpose of this pass. `orchestra.select` serves as the crucial merge point in this dataflow graph.

In summary, `orchestra.select` provides the essential vocabulary needed to express dataflow-based conditional selection on arbitrary SSA values, a concept not covered by existing standard operations.

## `orchestra.commit`

The `orchestra.commit` operation is a scheduling barrier and memory coherency token. It is a high-level scheduling primitive that provides guarantees to the Orchestra scheduler about the state of a `memref`.

### Semantic Contract

*   **What it is:** `orchestra.commit` signals that the contents of its `memref` operand are in a consistent, defined state, and all side effects from its producing operation are complete. It is a guarantee that this version of the buffer is "final" and ready for consumption by other tasks, potentially on other devices.

*   **What it is NOT:** It is **not** a copy or a DMA operation. It performs no data movement itself.

### Lifecycle and Lowering

The `orchestra.commit` op exists only in the `OrchestraIR` dialect. It is a critical input to the bufferization and scheduling passes. For example, a scheduler can only schedule an `orchestra.transfer` from memory space A to B *after* the corresponding `orchestra.commit` in memory space A has been reached.

After the scheduler has used the `commit` op to correctly order and lower other operations, the `commit` op has served its purpose and is **erased from the IR**. It does not lower to any other operation.

### Why an Operation, Not an Attribute?

Modeling this concept as an attribute on the `memref` would be insufficient. An attribute is passive metadata. An **operation** is an active participant in the IR graph, establishing a concrete point in the SSA dataflow chain. This allows the scheduler and other transformation passes to reason about it, move it, and use it to enforce ordering dependencies. This level of manipulation is only possible when the concept is modeled as a first-class operation.
