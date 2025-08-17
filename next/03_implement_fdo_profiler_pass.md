# Task: Implement the `OrchestraBranchProfiler` Instrumentation Pass

## Part A: Comprehensive Task Plan

### 1. Task Description

This task is to implement the `OrchestraBranchProfiler` pass, which is the first and most critical component of the feedback-driven optimization (FDO) framework outlined in the project's implementation plan. This pass will perform Ahead-Of-Time (AOT) instrumentation of GPU kernels to collect data on branch divergence.

The pass will operate on `gpu.func` operations. It will walk the operations within each function and, for every `scf.if` operation it finds, it will insert lightweight profiling code. This code will consist of:
1.  A shared memory buffer, allocated once per function, to store profiling counters.
2.  Hardware-accelerated atomic operations (`arith.atomic_rmw`) placed inside the `then` and `else` branches of the `scf.if`. These atomics will increment specific counters in the shared buffer, effectively counting how many threads in a workgroup execute each path.

### 2. Rationale

-   **Enables Feedback-Driven Optimization:** This pass is the cornerstone of the entire FDO system described in Section 3 of the `mlir-implementation-plan.md`. Without this instrumentation, the compiler has no way to gather the runtime data needed for more advanced optimizations like profile-guided data re-layout or thread remapping.
-   **Key Strategic Feature:** The implementation plan identifies the FDO loop as a key strategic differentiator for the Orchestra compiler, turning it into a dynamic, learning system. This task is the first concrete step towards realizing that vision.
-   **Self-Contained Module:** The profiler pass can be implemented and tested independently of the other FDO components (runtime monitor, JIT recompiler), making it an ideal first task.

### 3. Testing Strategy

The correctness of the pass will be verified with a new `lit` test file.

1.  **Create the test file:** `orchestra-compiler/tests/fdo-instrumentation.mlir`.
2.  **Structure the test:** The file will contain a `gpu.module` with a `gpu.func`. This function will include various `scf.if` operations to test different scenarios:
    -   An `scf.if` with both a `then` and an `else` block.
    -   An `scf.if` with only a `then` block.
    -   Nested `scf.if` operations.
3.  **Invoke the pass:** The `RUN` command will execute `orchestra-opt --orchestra-branch-profiler`.
4.  **Verify with `FileCheck`:** The test will use `FileCheck` to assert that the pass transformed the IR correctly. It will check for:
    -   `CHECK: memref.alloca.*#gpu.address_space<workgroup>`: Verifies that the shared counter buffer is allocated.
    -   `CHECK-DAG: arith.atomic_rmw`: Verifies that atomic operations are inserted.
    -   The test will have specific checks to ensure the atomics are inserted into the correct blocks (e.g., one in the `then` region, one in the `else` region) and are updating the correct indices in the counter buffer.
    -   The test will also verify that the pass does *not* modify operations that are not `scf.if`.

### 4. Documentation and SOTA Research

-   **Primary Documentation:** Section 3.1 of `docs/architecture/mlir-implementation-plan.md` provides the high-level architecture. It specifies that the pass should target `gpu.func`, find `scf.if`, and use `arith.atomic_rmw` to increment counters in a shared memory buffer.
-   **SOTA Research (Google Search):** My research will focus on the practical implementation details of such a pass.
    1.  **Pass Structure:** I will search for "mlir instrumentation pass example" and "mlir GPU pass structure" to decide between a `RewritePattern`-based approach and a direct walk of the IR. For instrumentation that adds operations rather than replacing them, a direct walk using `getOperation()->walk(...)` is often more straightforward.
    2.  **GPU Atomics in MLIR:** I will research the C++ builder APIs for `arith.atomic_rmw` and `memref.alloca` with GPU memory spaces. Search terms: "mlir C++ builder arith.atomic_rmw", "mlir gpu workgroup memory".
    3.  **Instrumentation in other compilers:** I will briefly research how Clang's `-p` or `-pg` profiling (gprof) or other FDO instrumentation passes are implemented to understand the general principles of assigning unique IDs to instrumentation sites and managing counter buffers.

## Part B: Deep-Research Question

### Question for Deep-Research Agent

**Question:**

I am implementing an MLIR instrumentation pass, `OrchestraBranchProfiler`, on **LLVM/MLIR 20**, to collect branch divergence statistics from GPU kernels. The pass needs to traverse `gpu.func` operations, identify all `scf.if` branches, and inject atomic counter increments into a shared memory buffer. What is the most robust, efficient, and extensible way to design this pass, particularly regarding buffer management, pass structure, and interaction with the GPU execution model?

**Onboarding Context:**

*   **Project:** OrchestraOS Compiler, an MLIR-based compiler for heterogeneous systems.
*   **Framework:** C++ with LLVM/MLIR version 20.
*   **Goal of the Pass:** To be the AOT instrumentation component of a larger Feedback-Driven Optimization (FDO) system.
*   **Core Logic:**
    1.  For each `gpu.func`, allocate a counter buffer in workgroup shared memory.
    2.  Assign a unique ID to each `scf.if` branch.
    3.  Insert `arith.atomic_rmw` operations into the `then` and `else` blocks to increment the counters corresponding to that branch's ID.

**My Questions:**

1.  **Buffer and Counter Management:** What is the canonical method for managing the instrumentation buffer and counter indices? Should the pass do a first walk to count all `scf.if` sites to determine the buffer size, then do a second walk to insert the `memref.alloca` and the atomics? Or is there a more efficient single-pass approach? How should I map each `scf.if` to a unique and stable index in the buffer, even as other passes might modify the IR? Should I add a custom attribute to the `scf.if` to store its ID?
2.  **Pass Structure: Rewrite vs. Direct Walk:** Since this pass adds new operations rather than rewriting existing ones, a direct IR walk (`getOperation()->walk(...)`) seems more natural than using `RewritePattern`s. Is this assumption correct? What are the trade-offs? Does a direct walk make it harder to manage IR modifications safely compared to using a `PatternRewriter`?
3.  **GPU Execution Model Interactions:** What are the subtle interactions with the GPU execution model I need to be aware of?
    -   Is `arith.atomic_rmw` on workgroup memory always the right choice for counters, or are there cases where it could cause performance issues (e.g., bank conflicts, serialization)?
    -   Does inserting these atomics inside `scf.if` regions pose any risk to the compiler's ability to perform other optimizations, like reconvergence analysis or predication?
    -   Should a `gpu.barrier` be inserted anywhere, for instance, after the instrumentation logic, to ensure counter visibility before the function returns?
4.  **Extensibility and Configuration:** The pass currently targets `scf.if`. In the future, we might want to profile `scf.for` loops or other operations. What is the best way to design the pass for extensibility? Is creating a custom `ProfilableOpInterface` that operations can implement the right approach? Or should the pass be configured with a list of operation names to instrument? I am looking for a design that is both powerful and adheres to MLIR's philosophy of modularity and extensibility.

I am seeking an answer that provides architectural guidance for building a production-quality instrumentation pass, drawing on best practices from mature compiler projects.
