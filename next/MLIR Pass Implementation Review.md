

# **Architectural Review and Implementation Guide: OrchestraBranchProfiler Pass**

## **Executive Summary**

This report presents an expert architectural review of the proposed OrchestraBranchProfiler pass for the OrchestraOS Compiler. The developer's initial plan provides a solid foundation, correctly identifying the core transformation required to instrument GPU branch operations for future feedback-driven optimizations (FDO). The high-level goal is well-defined and represents a valuable step toward enabling profile-guided compilation.

However, a deep analysis reveals several areas where the plan can be significantly improved to align with modern MLIR best practices, enhance robustness, and ensure long-term maintainability. The most critical findings and recommendations are as follows:

* **Strategic Shift to Idiomatic MLIR:** The proposed imperative "walk-and-modify" implementation should be replaced with a more idiomatic, stateful RewritePattern-based approach. This strategic shift enhances modularity by encapsulating the transformation logic, improves maintainability, and aligns the pass with the broader MLIR compiler ecosystem's design patterns.  
* **API Correctness and Modernization:** The implementation plan contains several API calls that are either outdated or non-idiomatic for the specified MLIR 20 version. The revised plan provides corrected, modern C++ API usage to prevent compilation issues and improve code clarity.  
* **Robustness and Edge Case Handling:** The initial specification is under-defined for common IR structures, such as scf.if operations without else blocks or those nested within other control flow constructs. The requirements have been expanded to mandate specific, predictable behavior for these cases, preventing potential compiler crashes or incorrect transformations.  
* **Testing Sufficiency:** The proposed testing strategy, while a good start, is insufficient to guarantee correctness across the identified edge cases. A more comprehensive test suite is specified to validate the pass's behavior against a variety of IR inputs.

Adopting these recommendations will result in a pass that is not only functionally correct but also robust, maintainable, and architecturally sound within the MLIR framework, providing a reliable foundation for the OrchestraOS FDO system.

## **Revised Requirements Specification**

### **2.1 Feature Title**

Implement the OrchestraBranchProfiler Instrumentation Pass

### **2.2 Goal / User Story**

As a compiler developer, I want to instrument gpu.func operations to collect runtime data on divergent branches (scf.if with both then and else regions). This data will be stored in workgroup-local memory and is the essential first step for building a future feedback-driven optimization (FDO) system that can use real-world execution profiles to guide compilation and improve performance.

### **2.3 Enhanced Acceptance Criteria**

1. A new MLIR pass, OrchestraBranchProfiler, is created and registered with the orchestra-opt tool under the command-line flag \--orchestra-branch-profiler.  
2. The pass must be an OperationPass that operates on gpu.func operations.  
3. **Conditionality:** For each gpu.func processed, the pass must perform an analysis step. *If and only if* the function contains one or more scf.if operations with a non-empty else region, the pass will proceed with instrumentation. Otherwise, the function must remain unmodified.  
4. **Buffer Allocation:** A single memref.alloca operation must be inserted at the beginning of the function's entry block.  
   * The allocated buffer must have a memref type with an element type of i32.  
   * The memory space must be workgroup memory, specified as \#gpu.address\_space\<workgroup\>.  
   * The size of the buffer must be exactly $N \* 2$, where $N$ is the total count of qualifying scf.if operations within the function.  
5. **Branch Instrumentation:** For each qualifying scf.if operation, the pass must assign a unique, stable base index $i$ (where $0 \\le i \< N$).  
   * A memref.atomic\_rmw operation of kind addi must be inserted at the beginning of the then block, incrementing the counter at index $i \* 2$.  
   * A memref.atomic\_rmw operation of kind addi must be inserted at the beginning of the else block, incrementing the counter at index $i \* 2 \+ 1$.  
6. **Synchronization:** A single gpu.barrier operation must be inserted at the end of the gpu.func, immediately before the function's terminator (gpu.return), to ensure all atomic updates are complete and visible across the workgroup. This barrier should only be inserted if instrumentation occurred.  
7. **Test Verification:** A new Lit test file must be added to the test suite to verify the pass's correctness against all specified behaviors, including the edge cases outlined below.

### **2.4 Robustness and Edge Case Mandates**

The initial plan lacks specificity for common variations in the MLIR IR. A robust compiler pass must have well-defined behavior for all possible valid inputs to prevent crashes or incorrect transformations. This is a critical consideration when designing passes that operate on a flexible, multi-level IR like MLIR.1 The requirement is to instrument

scf.if operations with both then and else blocks. This implies that other forms of control flow must be handled gracefully. For instance, scf.if operations can exist without a meaningful else block, and some functions may not contain any branches at all.3 The pass must not fail or produce incorrect code in these scenarios. Furthermore, the indexing logic for counters must be stable and predictable, even in the presence of complex, nested control flow. These are not mere implementation details; they are fundamental requirements for correctness.

* **No Qualifying Branches:** If a gpu.func contains zero scf.if operations with non-empty else regions, the pass must make no modifications to the function. No memref.alloca or gpu.barrier should be inserted.  
* **scf.if without else:** scf.if operations that do not have an else region, or have an empty else region, must be ignored and not contribute to the counter $N$.  
* **Nested Control Flow:** The pass must correctly identify and instrument qualifying scf.if operations regardless of their nesting depth within other scf or standard dialect operations (e.g., inside an scf.for). The enumeration of branches must be based on a stable walk order, such as a pre-order traversal of the operation graph.

### **2.5 Expanded Testing Requirements**

The proposed single test case covers the ideal "happy path" but is insufficient to guarantee the robustness mandated above. A comprehensive test suite is the only way to ensure the pass is correct across a variety of inputs. MLIR's testing infrastructure, particularly the LLVM Integrated Tester (lit) and FileCheck, is designed for precisely this kind of fine-grained, declarative verification.4 To validate the new robustness mandates, specific test cases are required. A test for a function with no branches ensures the no-op case works correctly. A test with mixed

scf.if operations (some with an else block, some without) verifies the filtering logic. A test with nested branches verifies the stability of the walking and indexing logic. Each of these tests targets a specific requirement, creating a direct and verifiable link between specification and implementation.

**Table 1: Mandatory Test Cases for OrchestraBranchProfiler**

| Test File Name | Purpose | Key FileCheck Assertions |
| :---- | :---- | :---- |
| instrument-basic.mlir | Verify correct instrumentation of a simple gpu.func with two qualifying scf.ifs at the same nesting level. | CHECK: memref.alloca()\[4 x i32\], CHECK: atomic\_rmw {{.\*}}\[%c0\], CHECK: atomic\_rmw {{.\*}}\[%c1\], CHECK: atomic\_rmw {{.\*}}\[%c2\], CHECK: atomic\_rmw {{.\*}}\[%c3\], CHECK: gpu.barrier |
| instrument-no-op.mlir | Verify the pass does nothing to a gpu.func with no scf.if operations. | CHECK-NOT: memref.alloca, CHECK-NOT: gpu.barrier, CHECK-NOT: atomic\_rmw |
| instrument-no-else.mlir | Verify the pass ignores scf.if operations that lack a non-empty else region, while still instrumenting a qualifying one. | CHECK: memref.alloca()\[2 x i32\], CHECK-NOT: inside the no-else if region. |
| instrument-nested.mlir | Verify correct and stable indexing for nested scf.if operations inside an scf.for loop. | CHECK: memref.alloca()\[4 x i32\], CHECK: correct index ordering based on pre-order walk. |

## **Revised & Enhanced Implementation Plan**

### **3.1 Strategic Recommendation: Adopting a Stateful RewritePattern Approach**

The proposed imperative "walk-and-modify" approach is functional but deviates from idiomatic MLIR design. The MLIR framework is architected around the concept of Directed Acyclic Graph (DAG) to DAG rewriting, which is most effectively implemented using patterns.5 However, this specific instrumentation task introduces a global, stateful component—the single, shared counter buffer—that a simple, stateless

RewritePattern cannot handle in isolation. A stateless OpRewritePattern\<scf::IfOp\> would match each scf.if operation independently and would incorrectly attempt to allocate a new buffer for each one.

The optimal solution is a hybrid architecture that combines the strengths of both an OperationPass and a RewritePattern. This design separates the global analysis and setup from the local transformation logic. The OperationPass acts as the state manager: it analyzes the entire gpu.func, creates the single shared memref.alloca buffer, and determines the unique index for each branch. It then configures and invokes a stateful RewritePattern, passing this shared state into the pattern's constructor. The pattern's sole responsibility is the local transformation of a single scf.if operation using the provided state.

This refined architecture offers several advantages:

* **Modularity:** The logic for matching and instrumenting an scf.if is cleanly encapsulated within its own RewritePattern class, separating it from the pass driver logic.  
* **Composability:** This pattern can be added to a RewritePatternSet and applied alongside other patterns by a greedy driver, leveraging MLIR's powerful and efficient pattern application engine.6  
* **Maintainability:** The clear separation of concerns (global analysis vs. local rewrite) makes the code easier to understand, debug, and extend in the future.

### **3.2 Revised Step-by-Step Implementation Guide**

1. **Define Pass and Pattern Structure:**  
   * In orchestra-compiler/lib/Orchestra/Transforms/OrchestraBranchProfiler.cpp, define the main pass class OrchestraBranchProfiler, which will inherit from mlir::OperationPass\<mlir::gpu::GPUFuncOp\>.  
   * Within the same file, define a private helper class IfInstrumentationPattern that inherits from mlir::OpRewritePattern\<mlir::scf::IfOp\>.  
   * The constructor for IfInstrumentationPattern will be stateful, accepting the mlir::Value of the counter buffer and a reference to a map (e.g., llvm::DenseMap\<mlir::Operation\*, unsigned\>) that stores the pre-calculated base index for each scf.if operation.  
2. **Implement IfInstrumentationPattern::matchAndRewrite:**  
   * This method will be invoked by the pattern driver for each scf::IfOp in the function. It receives the target scf::IfOp and a mlir::PatternRewriter.  
   * The method will use the state passed during its construction. It will look up the current ifOp in the index map to retrieve its unique base index, $id$.  
   * Using the PatternRewriter, it will create two memref.atomic\_rmw operations. The first will be inserted at the beginning of the then block to increment the counter at index $id \* 2$. The second will be inserted at the beginning of the else block for index $id \* 2 \+ 1$.  
   * This pattern does not need to replace the scf.if op. Since PatternRewriter is used to insert the atomic ops directly into the regions of the existing ifOp, the transformation can be done in-place. The method should return success() to indicate that the IR was modified.  
3. **Implement OrchestraBranchProfiler::runOnOperation:**  
   * **Phase 1: Analysis (Collect):**  
     * Get the gpu::GPUFuncOp using getOperation().  
     * Perform a read-only walk of the function to collect all scf::IfOps that satisfy the condition \!ifOp.getElseRegion().empty() into an llvm::SmallVector.  
     * If this vector is empty, return immediately, as no instrumentation is needed.  
   * **Phase 2: Global Setup (Modify):**  
     * Create an mlir::OpBuilder with its insertion point set to the beginning of the gpu.func's entry block.  
     * Create the memref.alloca operation for the counter buffer. The size will be collected\_if\_ops.size() \* 2\.  
     * Create and populate an llvm::DenseMap\<mlir::Operation\*, unsigned\> by iterating over the collected if ops with an index. This map associates each scf.if operation with its unique base index.  
   * **Phase 3: Pattern Application:**  
     * Obtain the MLIRContext from the operation.  
     * Create an mlir::RewritePatternSet.  
     * Add an instance of IfInstrumentationPattern to the set, passing the newly allocated buffer Value and the index map to its constructor.  
     * Use a greedy pattern driver, such as mlir::applyPatternsAndFoldGreedily, to apply the pattern set to the gpu.func. This will execute the matchAndRewrite logic on all qualifying scf.if operations.  
   * **Phase 4: Finalization (Modify):**  
     * Create a new mlir::OpBuilder positioned immediately before the function's terminator (gpu.return).  
     * Create a single gpu.barrier operation to ensure all workgroup threads have completed their atomic updates before the kernel finishes.  
4. **Register Pass and Add Tests:**  
   * Register the OrchestraBranchProfiler pass in Passes.cpp as originally planned.  
   * Implement the comprehensive test suite described in the revised requirements, creating separate .mlir files for each edge case.

### **3.3 API and Syntax Validation (MLIR 20\)**

The developer's plan correctly identified several key APIs but also raised questions about their precise usage. The following table provides authoritative corrections and clarifies best practices for the MLIR 20 C++ API, ensuring the implementation is correct and idiomatic.

**Table 2: API Usage Corrections and Best Practices (MLIR 20\)**

| Area | Original Proposal / Question | Corrected & Idiomatic Approach (MLIR 20\) | Rationale & References |
| :---- | :---- | :---- | :---- |
| **Pass Inheritance** | mlir::Pass\<OrchestraBranchProfiler, mlir::gpu::GPUFuncOp\> | struct OrchestraBranchProfiler : public mlir::OperationPass\<mlir::gpu::GPUFuncOp\> | The mlir::FunctionPass base class has been deprecated in favor of the more generic mlir::OperationPass. This modern base class is templated on the specific operation type it targets, making the pass structure more explicit and type-safe. 7 |
| **Checking else Block** | ifOp.hasElse() | \!ifOp.getElseRegion().empty() | The hasElse() method only checks for the syntactic presence of the else region construct. A more robust check is \!getElseRegion().empty(), which correctly handles cases where an else region exists but is empty (e.g., else {}), ensuring that only truly divergent branches are instrumented. 3 |
| **MemRefType Creation** | MemRefType::get({numCounters}, i32Type, {}, gpu::AddressSpaceAttr::get(...)); | mlir::MemRefType::get({numCounters}, i32Type, {}, mlir::gpu::AddressSpaceAttr::get(context, mlir::gpu::AddressSpace::Workgroup)); | The static get method is a valid approach. The key correction is the proper construction of the AddressSpaceAttr using the gpu dialect's C++ enum mlir::gpu::AddressSpace::Workgroup, which provides type safety over raw integer values. The Builder pattern is an alternative idiomatic approach. 10 |
| **Atomic Operation** | builder.create\<memref::AtomicRMWOp\>(loc, arith::AtomicRMWKind::addi,...); | Correct as proposed. | The plan correctly identifies memref.atomic\_rmw as the appropriate operation for atomic modifications of memref buffers, distinguishing it from arith.atomic\_rmw, which operates on tensors. 12 |
| **Insertion Before Terminator** | builder.setInsertionPoint(gpuFunc.front().getTerminator()); | mlir::OpBuilder builder(gpuFunc.front().getTerminator()); | While setting the insertion point to the terminator operation works, constructing the OpBuilder directly with the terminator operation as its insertion point is a more concise and common idiom. This places new operations immediately before the terminator. 15 |

### **3.4 Alternative Strategy Analysis: Declarative Rewrite Rules (DRR)**

A comprehensive architectural review should also consider viable alternatives. For many pattern-matching tasks in MLIR, Declarative Rewrite Rules (DRR), defined using TableGen, offer a powerful and concise alternative to imperative C++ patterns.17

* **What is DRR?** DRR allows developers to define rewrite patterns in a declarative format within .td (TableGen definition) files. The syntax specifies a source IR pattern to match and a target IR pattern to generate. The MLIR build system then uses TableGen to auto-generate the corresponding C++ pattern-matching and rewriting logic.  
* **Pros of DRR:**  
  * **Conciseness:** A simple rewrite can often be expressed in a few lines of TableGen, which is significantly more compact than the equivalent C++ class definition.  
  * **Separation of Concerns:** It cleanly separates the definition of the transformation logic from the C++ source code, placing it alongside other dialect definitions in TableGen files.  
* **Cons of DRR (for this specific task):**  
  * **State Management:** DRR is fundamentally designed for local, stateless rewrites. The core requirement of this pass—to reference a single buffer allocated once per function and to assign a unique, sequential index to each match—is inherently stateful. Injecting this per-pass state (the buffer Value and the unique index) into a declarative pattern is not directly supported and would require complex, non-standard workarounds using NativeCodeCall that would negate the benefits of conciseness and clarity.  
* **Conclusion:** The stateful nature of the branch profiling task makes it a poor fit for the stateless design of Declarative Rewrite Rules. For transformations that require context or state accumulated across an entire function or module, the recommended hybrid approach—using a C++ OperationPass to manage state and a C++ RewritePattern to perform the local transformation—provides the necessary flexibility and clarity, making it the superior architectural choice for this implementation.

#### **Referenzen**

1. MLIR (software) \- Wikipedia, Zugriff am August 18, 2025, [https://en.wikipedia.org/wiki/MLIR\_(software)](https://en.wikipedia.org/wiki/MLIR_\(software\))  
2. Introduction \- tt-mlir documentation, Zugriff am August 18, 2025, [https://docs.tenstorrent.com/tt-mlir/](https://docs.tenstorrent.com/tt-mlir/)  
3. lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp Source File \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/doxygen/SCFToControlFlow\_8cpp\_source.html](https://mlir.llvm.org/doxygen/SCFToControlFlow_8cpp_source.html)  
4. MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/](https://mlir.llvm.org/)  
5. mlir/docs/PatternRewriter.md · doe \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md)  
6. Pattern Rewriting : Generic DAG-to-DAG Rewriting \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/PatternRewriter/](https://mlir.llvm.org/docs/PatternRewriter/)  
7. Create your Own MLIR Pass — NVIDIA CUDA Quantum documentation \- GitHub Pages, Zugriff am August 18, 2025, [https://nvidia.github.io/cuda-quantum/0.3.0/using/advanced/mlir\_pass.html](https://nvidia.github.io/cuda-quantum/0.3.0/using/advanced/mlir_pass.html)  
8. D117182 \[mlir\]\[Pass\] Deprecate FunctionPass in favor of OperationPass  
9. D154149 \[mlir\]\[gpu\] Add the \`gpu-module-to-binary\` pass. \- LLVM Phabricator archive, Zugriff am August 18, 2025, [https://reviews.llvm.org/D154149](https://reviews.llvm.org/D154149)  
10. mlir::MemRefType::Builder Class Reference \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1MemRefType\_1\_1Builder.html](https://mlir.llvm.org/doxygen/classmlir_1_1MemRefType_1_1Builder.html)  
11. mlir/lib/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/d6b73da15211d2286c6b0750b68d139104d463b9/mlir/lib/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/d6b73da15211d2286c6b0750b68d139104d463b9/mlir/lib/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp)  
12. lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp Source File \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/doxygen/MemRefToLLVM\_8cpp\_source.html](https://mlir.llvm.org/doxygen/MemRefToLLVM_8cpp_source.html)  
13. mlir/test/Dialect/MemRef/ops.mlir · llvmorg-14.0.4 ... \- Sign in · GitLab, Zugriff am August 18, 2025, [https://gitlab.ispras.ru/mvg/mvg-oss/llvm-project/-/blob/llvmorg-14.0.4/mlir/test/Dialect/MemRef/ops.mlir](https://gitlab.ispras.ru/mvg/mvg-oss/llvm-project/-/blob/llvmorg-14.0.4/mlir/test/Dialect/MemRef/ops.mlir)  
14. Sanitizing MLIR Programs with Runtime Operation Verification \- LLVM, Zugriff am August 18, 2025, [https://llvm.org/devmtg/2025-06/slides/technical-talk/springer-sanitizing.pdf](https://llvm.org/devmtg/2025-06/slides/technical-talk/springer-sanitizing.pdf)  
15. mlir::OpBuilder Class Reference \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1OpBuilder.html](https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html)  
16. llvm-project/mlir/include/mlir/IR/Builders.h at main \- GitHub, Zugriff am August 18, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Builders.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Builders.h)  
17. MLIR — Canonicalizers and Declarative Rewrite Patterns \- Math ∩ Programming, Zugriff am August 18, 2025, [https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/](https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/)  
18. Chapter 3: High-level Language-Specific Analysis and ..., Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)