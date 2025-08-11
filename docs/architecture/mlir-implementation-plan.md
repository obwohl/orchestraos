

> **🛑 Implementation Status Notice**
>
> This document outlines the **intended architecture and a future implementation plan** for the Orchestra compiler. It describes features and components that are the *goal* of this project.
>
> The current implementation is **incomplete and not in a buildable state**. Significant work is required to realize the design described here. For a detailed analysis of the current blockers and the actual state of the codebase, please refer to the project's primary status document:
>
> **[Current Project Status](../project-status/status-quo.md)**

# **A Comprehensive Implementation Blueprint for the OrchestraOS MLIR Compiler Core**

## **Introduction**

### **Purpose and Scope**

This document provides the single, authoritative engineering blueprint for the core components of the OrchestraOS compiler. It is intended to supersede all previous design documents, technical notes, and architectural proposals by presenting a unified, corrected, and exhaustive plan for implementation. The primary objective is to resolve inconsistencies, update outdated syntax, and provide a complete technical specification for the proprietary OrchestraIR dialect and its associated optimization passes. The scope of this blueprint encompasses the full lifecycle of OrchestraIR, from its formal definition and the compiler passes that generate it, through a suite of advanced, hardware-aware optimizations, to its final lowering into hardware-specific primitives for execution on heterogeneous compute platforms.

### **Architectural Vision**

The foundational philosophy of OrchestraOS is the establishment of a compiler-centric orchestration layer designed to master the complexity of modern, heterogeneous AI hardware.1 The central architectural principle is to elevate data movement, task scheduling, and execution strategy to first-class citizens of the Intermediate Representation (IR). Unlike traditional runtime systems that treat workloads as opaque entities, the OrchestraOS compiler possesses a deep semantic understanding of the computational graph, enabling it to make globally optimal decisions about how and where to execute work.1

The strategic selection of MLIR as the foundational infrastructure is a cornerstone of this vision. MLIR's extensible, multi-level dialect system provides the ideal framework for representing a program at numerous levels of abstraction simultaneously—from high-level framework graphs down to low-level, hardware-specific instructions. This modularity, combined with its rapid adoption as the de-facto industry standard, makes MLIR the superior choice for building a future-proof and technologically defensible compiler platform.1

### **Document Structure**

This blueprint is structured to guide a logical, phased implementation of the compiler core. The report begins by establishing the foundational language of the system, the OrchestraIR dialect. Subsequent sections detail the compiler transformations that operate on and produce this dialect, progressing from high-level, architecture-agnostic optimizations to low-level, hardware-specific code generation.

* **Section 1** provides a formal, modernized specification for the OrchestraIR dialect, defining its core operations and their semantics.  
* **Section 2** details the implementation of the divergence-to-speculation pass, a key optimization for mitigating control-flow bottlenecks on SIMT architectures.  
* **Section 3** presents a state-of-the-art framework for feedback-driven optimization, enabling the compiler to adapt dynamically to runtime workload behavior.  
* **Section 4** outlines the critical lowering pathways from the abstract OrchestraIR dialect to concrete, hardware-specific primitives for NVIDIA and Intel GPUs.  
* **Section 5** describes a modular framework for advanced, hardware-aware optimizations, including operator fusion and memory layout transformation, operating on the linalg dialect.  
* **Section 6** concludes by synthesizing these components into a coherent compiler pipeline, providing recommendations for pass ordering and future-looking architectural enhancements.

## **Section 1: The OrchestraIR Dialect: A Formal and Modernized Specification**

### **1.1. Rationale and Role in the Progressive Lowering Pipeline**

The OrchestraIR dialect is the heart of the OrchestraOS compiler's proprietary technology. It is a mid-level dialect designed to serve as the crucial bridge between a purely logical, framework-agnostic representation of a computation (e.g., in the torch or tf dialects) and the concrete, hardware-specific representations required for final code generation (e.g., the nvgpu, amdgpu, xegpu, and llvm dialects). The primary purpose of OrchestraIR is to explicitly materialize the decisions of a high-level, topology-aware scheduling pass directly into the IR.1

This design establishes a clean and powerful separation of concerns. High-level framework dialects describe *what* to compute, while OrchestraIR describes *how and where* to compute it. The complex, potentially heuristic- or learning-based logic of the scheduler is encapsulated in a single pass that produces OrchestraIR. This IR then serves as a stable, well-defined input to a series of more deterministic lowering and optimization passes. This approach moves scheduling and data placement decisions from an opaque, external runtime system into the compiler's domain, where they can be analyzed and optimized with a global view of the program's structure and the hardware's topology.1 This transformation of the compiler's role into that of a "meta-OS" or orchestrator is the core strategic value of the architecture, and the

OrchestraIR dialect is the language this system uses to express its decisions.

### **1.2. Operation Definition Specification (ODS) in TableGen**

The formal definition of the dialect is specified declaratively using MLIR's TableGen format (.td files). This serves as the single source of truth from which the C++ classes and methods for the dialect are automatically generated, minimizing boilerplate and ensuring consistency.2 The following are the canonical TableGen definitions for the core

OrchestraIR operations.

**orchestra.schedule:** This operation acts as a container for a physically scheduled subgraph. It contains a single region holding a Directed Acyclic Graph (DAG) of orchestra.task operations, where the edges of the DAG are represented by the SSA use-def chain.1

Code-Snippet

// In OrchestraOps.td  
def Orchestra\_ScheduleOp : Orchestra\_Op\<"schedule"\> {  
  let summary \= "Container for a physically scheduled DAG of tasks.";  
  let description \=;

  let results \= (outs Variadic\<AnyType\>:$results);  
  let regions \= (region SizedRegion:$body);  
  let hasCanonicalizer \= 1;  
}

orchestra.task**:** This operation encapsulates an atomic unit of computation assigned to a specific hardware resource. Its target attribute is defined as a DictionaryAttr, providing a flexible and extensible mechanism for specifying fine-grained placement constraints, an enhancement over the simple string attribute initially proposed.1

Code-Snippet

// In OrchestraOps.td  
def Orchestra\_TaskOp : Orchestra\_Op\<"task"\> {  
  let summary \= "An asynchronous unit of computation assigned to a resource.";  
  let description \=;

  let arguments \= (ins Variadic\<AnyType\>:$operands);  
  let results \= (outs Variadic\<AnyType\>:$results);

  let attributes \= (ins DictionaryAttr:$target);  
  let regions \= (region SizedRegion:$body);

  // Define a custom builder for convenience.  
  let builders \=;  
}

**orchestra.transfer:** This operation makes data movement between different logical memory spaces an explicit, first-class citizen of the IR. The from and to locations are represented by SymbolRefAttr, which will be resolved to physical memory spaces during a later lowering stage.1

Code-Snippet

// In OrchestraOps.td  
def Orchestra\_TransferOp : Orchestra\_Op\<"transfer"\> {  
  let summary \= "Explicitly represents data movement between memory spaces.";  
  let description \=;

  let arguments \= (ins AnyShaped:$source);  
  let results \= (outs SameAs\<"$source"\>:$result);

  let attributes \= (ins SymbolRefAttr:$from, SymbolRefAttr:$to);  
}

**orchestra.commit:** This operation is essential for the speculative execution pattern but was missing from formal specifications in the source material.1 The following definition synthesizes its intended semantics. It selects one of two sets of SSA values based on a boolean condition, materializing the result of the speculation.

Code-Snippet

// In OrchestraOps.td  
def Orchestra\_CommitOp : Orchestra\_Op\<"commit"\> {  
  let summary \= "Selects one of two SSA values based on a boolean condition.";  
  let description \=;

  let arguments \= (ins I1:$condition,  
                       Variadic\<AnyType\>:$true\_values,  
                       Variadic\<AnyType\>:$false\_values);  
  let results \= (outs Variadic\<AnyType\>:$results);

  let hasVerifier \= 1;  
}

**orchestra.yield:** This is a standard terminator operation for regions within the OrchestraIR dialect, analogous to func.return or scf.yield.

Code-Snippet

// In OrchestraOps.td  
def Orchestra\_YieldOp : Orchestra\_Op\<"yield"\> {  
  let summary \= "Terminator for regions in OrchestraIR operations.";  
  let description \= \[{  
    A terminator operation for regions within \`orchestra.schedule\` and  
    \`orchestra.task\`. It yields values from the region to the parent op.  
  }\];

  let arguments \= (ins Variadic\<AnyType\>:$operands);  
  let traits \=;  
}

### **1.3. C++ API and Builder Implementation Guidelines**

While TableGen generates the core C++ class definitions, certain logic must be hand-written in the corresponding .cpp file (e.g., OrchestraDialect.cpp).2 This includes the implementation of verifiers and custom builders.

**Verifiers:** Each operation should have a verifier to enforce semantic invariants that cannot be captured by the type system alone. For example, the orchestra.commit verifier must ensure that the types and number of values from the true\_values and false\_values operands match each other and the operation's result types.

C++

// In OrchestraOps.cpp  
mlir::LogicalResult orchestra::CommitOp::verify() {  
  if (getTrueValues().getTypes()\!= getFalseValues().getTypes()) {  
    return emitOpError("requires 'true' and 'false' value types to match");  
  }  
  if (getTrueValues().getTypes()\!= getResultTypes()) {  
    return emitOpError("requires result types to match operand types");  
  }  
  return mlir::success();  
}

**Custom Builders:** Convenience builders simplify the programmatic creation of operations within compiler passes. The orchestra.task builder, for example, should automatically create the entry block of its region with the correct number and types of block arguments to match its operands.

C++

// In OrchestraOps.cpp  
void orchestra::TaskOp::build(mlir::OpBuilder \&builder,  
                              mlir::OperationState \&state,  
                              mlir::ValueRange operands,  
                              mlir::TypeRange resultTypes,  
                              mlir::DictionaryAttr target) {  
  state.addOperands(operands);  
  state.addTypes(resultTypes);  
  state.addAttribute("target", target);

  // Create the region and add a block with arguments  
  // corresponding to the task's operands.  
  mlir::Region \*bodyRegion \= state.addRegion();  
  mlir::Block \*bodyBlock \= new mlir::Block();  
  bodyRegion-\>push\_back(bodyBlock);  
  bodyBlock-\>addArguments(operands.getTypes(),  
                          SmallVector\<mlir::Location\>(operands.size(), state.location));  
}

| Operation | Syntax Example | Operands | Attributes | Core Semantics & Rationale |
| :---- | :---- | :---- | :---- | :---- |
| orchestra.schedule | orchestra.schedule {... } | None | None | Contains a region with a DAG of orchestra.task operations, representing the full, physically scheduled execution plan. |
| orchestra.task | %res \= orchestra.task (%in) target={...} {... } | Variadic inputs | target: DictionaryAttr specifying hardware resource. | Encapsulates a unit of work. The target attribute constrains its placement. The SSA use-def chain defines its dependencies. |
| orchestra.transfer | %gpu\_data \= orchestra.transfer %host\_data from @host to @gpu0 | The data to be transferred. | from, to: SymbolRefAttr pointing to resource handles. | Explicitly represents the movement of data between two distinct memory locations, making communication a first-class optimizable operation. |
| orchestra.commit | %res \= orchestra.commit %cond, %true\_val, %false\_val | i1 condition, variadic true values, variadic false values. | None | Selects one of two sets of SSA values based on a boolean condition. Core op for materializing the result of speculative execution. |

## **Section 2: Implementation of the Divergence-to-Speculation Pass**

### **2.1. The SpeculateIfOpPattern: A Pattern-Based Rewrite for scf.if**

The primary strategy for mitigating the "Penalty of Divergence" on SIMT architectures like GPUs is to transform conditional control dependencies into parallel data dependencies.1 This is achieved through a compiler pass that identifies suitable conditional constructs and rewrites them into a speculative execution pattern using the

OrchestraIR dialect.

The implementation will leverage MLIR's declarative pattern rewriting infrastructure. This is the idiomatic and most robust approach for such transformations. The core logic will be encapsulated in a C++ class, SpeculateIfOpPattern, that inherits from mlir::OpRewritePattern\<scf::IfOp\>.1 The MLIR pass manager will invoke this pattern on every

scf.if operation it encounters, allowing the pattern to match and rewrite candidates.

### **2.2. Step-by-Step Rewriter Logic with Corrected MLIR C++ APIs**

The transformation logic is executed within the matchAndRewrite method of the pattern. All modifications to the IR must be performed through the mlir::PatternRewriter instance provided to this method to ensure IR integrity and proper tracking of changes.6 The following steps outline a correct and robust implementation.

Step 1: Matching Candidate Operations  
The first step is to filter for suitable scf.if operations. The speculative execution pattern is only applicable to conditionals that have both a then and an else branch and that produce one or more results. An if without an else or without results does not fit the model where two paths are executed and a final result is selected.1

C++

// In SpeculateIfOpPattern.cpp  
mlir::LogicalResult SpeculateIfOpPattern::matchAndRewrite(  
      mlir::scf::IfOp ifOp, mlir::PatternRewriter \&rewriter) const {  
  // 1\. Check for suitability. The pattern requires an 'else' block  
  // and must produce results.  
  if (\!ifOp.getElseRegion().hasOneBlock() |

| ifOp.getNumResults() \== 0) {  
    return rewriter.notifyMatchFailure(ifOp, "not a candidate for speculation");  
  }  
  //... further steps  
}

Step 2: Identifying External SSA Value Dependencies  
A critical and subtle aspect of this transformation is handling SSA values that are defined outside the scf.if but used within its regions. While an scf.if region can implicitly "capture" any dominating SSA value, an orchestra.task is an isolated unit and must receive all external data as explicit operands.1 Therefore, the pattern must programmatically identify these captured values. An SSA  
Value is considered external to a Region if its defining operation is not located within that region, or if it is not a block argument of that region's entry block.9 This requires a manual walk of the operations within each region.

C++

// Helper function to find all SSA Values used in a region but defined outside.  
static llvm::SetVector\<mlir::Value\> getUsedExternalValues(mlir::Region \&region) {  
  llvm::SetVector\<mlir::Value\> externalValues;  
  region.walk(\[&\](mlir::Operation \*op) {  
    for (mlir::Value operand : op-\>getOperands()) {  
      // An operand is external if its defining op is not within the region,  
      // or it is a block argument of a different region.  
      if (operand.getParentRegion()\!= \&region) {  
        externalValues.insert(operand);  
      }  
    }  
  });  
  return externalValues;  
}

// Inside matchAndRewrite:  
auto \&thenRegion \= ifOp.getThenRegion();  
auto \&elseRegion \= ifOp.getElseRegion();  
auto thenExternalValues \= getUsedExternalValues(thenRegion);  
auto elseExternalValues \= getUsedExternalValues(elseRegion);

Step 3: Cloning and Remapping Regions  
The bodies of the original scf.if regions must be moved into the newly created orchestra.tasks. The mlir::IRMapping class is essential for this process. It tracks the mapping from old values (the external dependencies) to new ones (the block arguments of the task's region), and the rewriter.clone method uses this mapping to automatically update the operands of the cloned operations.1

C++

// Helper function to clone a region and remap its arguments.  
static void cloneAndRemapRegion(mlir::Region \&sourceRegion,  
                                mlir::Region \&destRegion,  
                                llvm::ArrayRef\<mlir::Value\> externalValues,  
                                mlir::PatternRewriter \&rewriter) {  
  mlir::IRMapping mapper;  
  auto destArgs \= destRegion.getArguments();  
  assert(externalValues.size() \== destArgs.size());  
  mapper.map(externalValues, destArgs);

  // Clone the operations from the source region into the destination region.  
  for (auto \&op : sourceRegion.front().without\_terminator()) {  
    rewriter.clone(op, mapper);  
  }

  // Find the original \`scf.yield\` and use its operands, remapped,  
  // to create the new \`orchestra.yield\`.  
  auto sourceYield \= cast\<mlir::scf::YieldOp\>(sourceRegion.front().getTerminator());  
  llvm::SmallVector\<mlir::Value\> yieldOperands;  
  for (mlir::Value operand : sourceYield.getOperands()) {  
    yieldOperands.push\_back(mapper.lookupOrDefault(operand));  
  }  
  rewriter.create\<orchestra::YieldOp\>(sourceYield.getLoc(), yieldOperands);  
}

Step 4: Constructing the DAG  
With the external dependencies identified and the cloning logic defined, the rewriter can now construct the new OrchestraIR DAG. It creates two orchestra.task operations—one for the then path and one for the else path—followed by the orchestra.commit operation that selects between their results. The key is that %true\_result and %false\_result have no SSA dependency on each other, which explicitly signals to the scheduler that they can be executed in parallel.1

C++

// Inside matchAndRewrite:  
mlir::Location loc \= ifOp.getLoc();  
mlir::TypeRange resultTypes \= ifOp.getResultTypes();  
auto targetAttr \= rewriter.getDictionaryAttr({}); // Placeholder target

// Create the 'then' task and populate its body.  
auto thenTask \= rewriter.create\<orchestra::TaskOp\>(  
    loc, resultTypes, thenExternalValues.getArrayRef(), targetAttr);  
cloneAndRemapRegion(thenRegion, thenTask.getBodyRegion(),  
                    thenExternalValues.getArrayRef(), rewriter);

// Create the 'else' task and populate its body.  
auto elseTask \= rewriter.create\<orchestra::TaskOp\>(  
    loc, resultTypes, elseExternalValues.getArrayRef(), targetAttr);  
cloneAndRemapRegion(elseRegion, elseTask.getBodyRegion(),  
                    elseExternalValues.getArrayRef(), rewriter);

// Create the commit operation to select the final result.  
auto commitOp \= rewriter.create\<orchestra::CommitOp\>(  
    loc, resultTypes, ifOp.getCondition(), thenTask.getResults(),  
    elseTask.getResults());

Step 5: Finalizing the Rewrite  
The final action is to replace the original scf.if operation with the results of the newly created orchestra.commit operation. This completes the rewrite pattern.1

C++

// Inside matchAndRewrite:  
rewriter.replaceOp(ifOp, commitOp.getResults());  
return mlir::success();

### **2.3. Critical Safety Precondition: Verifying Side-Effect-Free Regions**

A critical consideration in the application of this pass is the fundamental constraint imposed by its speculative nature. While the transformation is most impactful on the complex, data-dependent control flow characteristic of agentic AI, this same complexity often involves stateful operations that introduce side effects.1 The speculative execution of an irreversible action, such as a memory write (

memref.store), would create a race condition and lead to catastrophic correctness failures. Therefore, the applicability of this powerful optimization is strictly limited to subgraphs that are demonstrably pure and functional.1

To ensure program correctness, the pass must include a non-negotiable safety check. Before applying the rewrite, the pattern's match logic must verify that the then and else regions are free of side effects. MLIR provides the MemoryEffectOpInterface for this purpose.10 Operations that read from or write to memory implement this interface, allowing their effects to be queried programmatically. The

match logic must be augmented to walk all nested operations within the scf.if regions and check their memory effects. If any operation reports a Write effect, the pattern must fail to match, preventing the transformation.

C++

// In SpeculateIfOpPattern::matchAndRewrite, before creating tasks:  
auto hasSideEffects \=  
    \[&\](mlir::Region \&region) {  
      auto walkResult \= region.walk(\[&\](mlir::Operation \*op) {  
        auto memInterface \= dyn\_cast\<mlir::MemoryEffectOpInterface\>(op);  
        if (\!memInterface) return mlir::WalkResult::advance();

        llvm::SmallVector\<mlir::MemoryEffects::EffectInstance\> effects;  
        memInterface.getEffects(effects);  
        for (const auto \&effect : effects) {  
          if (isa\<mlir::MemoryEffects::Write\>(effect.getEffect())) {  
            return mlir::WalkResult::interrupt(); // Found a write side effect.  
          }  
        }  
        return mlir::WalkResult::advance();  
      });  
      return walkResult.wasInterrupted(); // Interrupted means a side effect was found.  
    };

if (hasSideEffects(ifOp.getThenRegion()) |

| hasSideEffects(ifOp.getElseRegion())) {  
  return rewriter.notifyMatchFailure(  
      ifOp, "cannot speculate regions with memory write side effects");  
}

This constraint elevates the importance of upstream compiler passes that can perform function pure-ification or hoist memory operations out of conditional regions, as their success directly enables this powerful divergence mitigation strategy downstream.

### **2.4. Strategy for cf.cond\_br: Leveraging the Upstream \-lift-cf-to-scf Pass**

While it is possible to write a rewrite pattern that directly targets the lower-level cf.cond\_br operation, this approach is significantly more complex and brittle. An scf.if conveniently packages its logic into self-contained Regions, making them easy to extract. In contrast, a cf.cond\_br is merely a terminator branching to separate Blocks, requiring complex, manual control-flow analysis to identify the divergent paths and their re-convergence point.1

A far more robust and idiomatic MLIR strategy is to leverage existing infrastructure to normalize the IR before the speculation pass runs. The standard MLIR pass pipeline includes the \-lift-cf-to-scf pass, which is specifically designed to analyze Control Flow Graph (CFG) structures and replace patterns of conditional branches and re-convergences with their equivalent scf.if operations.12

Therefore, the recommended practice for the OrchestraOS compiler is to ensure that the \-lift-cf-to-scf pass is executed as a prerequisite to the divergence-to-speculation pass. This design choice dramatically simplifies the implementation of the proprietary pass, improves its robustness by relying on well-tested community-maintained logic, and adheres to the MLIR philosophy of composing small, focused passes to achieve a larger transformation.1

## **Section 3: A Framework for Feedback-Driven Divergence Mitigation**

### **3.1. System Architecture: Instrumentation, Runtime Profiling, and JIT Recompilation**

To address divergence patterns that are highly data-dependent and unpredictable at compile time, a static AOT approach is insufficient. A more powerful, adaptive solution is a feedback-driven optimization (FDO) loop that uses runtime performance data to guide compiler transformations.1 This requires a robust, closed-loop system architecture composed of three interconnected components.

1. **AOT Instrumentation:** During the initial compilation, a dedicated MLIR pass, OrchestraBranchProfiler, traverses gpu.func operations. It identifies potentially divergent branches (e.g., scf.if) and inserts lightweight profiling code. This instrumentation consists of simple, hardware-accelerated atomic operations (e.g., arith.atomic\_rmw) that increment counters in a pre-allocated shared memory buffer, tracking metrics like branch outcomes with minimal performance impact.1  
2. **Runtime Monitoring and Profiling:** A lightweight runtime service or daemon process periodically and asynchronously analyzes this shared buffer. It aggregates the raw statistics to build a structured "divergence profile" for the executing kernel, identifying which branches are the most frequent sources of divergence and whether stable patterns exist in their behavior.1  
3. **JIT Recompilation and Hot-Swapping:** When the runtime monitor detects a stable but suboptimal divergence pattern (e.g., a branch consistently causing a 50/50 split within warps), it triggers an adaptive response. It invokes a JIT compilation service via an RPC call, passing the original MLIR representation of the kernel along with the detailed divergence profile. The JIT service applies a specialized feedback-driven-remap pass, lowers the optimized MLIR to PTX (for NVIDIA), and uses the NVIDIA Runtime Compilation (NVRTC) library to compile the PTX into a binary CUBIN on-the-fly. This new binary is then dynamically loaded into the running application using CUDA runtime APIs, and the runtime's dispatch table is updated to route subsequent calls to the newly optimized kernel.1

This FDO system transforms the compiler from a static tool into a dynamic, learning system. The collected divergence profiles, paired with the heuristics that govern when and how to recompile, become a valuable and proprietary dataset. As the system is exposed to more workloads, its ability to predict and mitigate divergence improves, creating a compounding performance advantage and a deep, defensible intellectual property moat.1

### **3.2. The DivergenceProfile: A Formal Data Contract**

The DivergenceProfile is the critical data structure that serves as the interface between the runtime profiler and the JIT compiler. Its design must be robust and extensible to ensure the components can evolve independently. A versioned schema is essential for this purpose.1

| Field | Type | Description |
| :---- | :---- | :---- |
| profile\_version | string | Version of the profile schema for forward/backward compatibility. |
| kernel\_name | string | The mangled name of the gpu.func containing the branch. |
| branch\_id | uint32\_t | A unique, compiler-generated ID for the specific scf.if or cf.cond\_br. |
| invocation\_count | uint64\_t | Total times this branch was encountered by any thread. |
| divergence\_rate | float | Percentage of warps that experienced divergence at this branch. |
| path\_outcome\_counts | map\<string, uint64\_t\> | Counts for each path, e.g., {"then": 5.2M, "else": 4.8M}. |
| data\_correlation\_hint | optional | (Advanced) A serialized hint, e.g., a small histogram correlating input data ranges to branch outcomes. |

### **3.3. Implementation Pattern A: Profile-Guided Inter-Kernel Data Re-layout**

This pattern is an *inter-kernel* optimization that aims to re-sort a kernel's input data buffer so that elements known to cause threads to follow the same execution path are grouped together. When threads in a warp access contiguous memory, they will naturally load data that leads to convergent branching.1

* **Trigger Heuristic:** This pattern is ideal when the DivergenceProfile indicates a strong, monotonic correlation between a specific value in an input tensor and the resulting branch path.  
* **MLIR Implementation:** The transformation is applied not to the gpu.func itself, but to the host-side gpu.launch\_func operation that invokes it. A rewrite pattern matches the launch operation and inserts a high-level orchestra.transfer operation *before* the kernel is called. This orchestra.transfer is annotated with attributes derived from the profile (e.g., specifying the sort key), instructing it to perform the re-sorting. The launch operation's operand is then updated to point to the new, sorted buffer. The OrchestraOS scheduler ensures the sort completes before the kernel launch.

C++

// Conceptual C++ MLIR Rewrite Pattern for Data Re-layout  
struct RemapDataLayoutPattern : public mlir::OpRewritePattern\<mlir::gpu::LaunchFuncOp\> {  
  //... constructor taking DivergenceProfile...  
  mlir::LogicalResult matchAndRewrite(mlir::gpu::LaunchFuncOp launchOp,  
                                      mlir::PatternRewriter \&rewriter) const override {  
    // 1\. Analyze profile to confirm applicability.  
    if (\!profile.isApplicableForDataRelayout(launchOp.getKernelName()))  
      return mlir::failure();

    // 2\. Identify the input memref to be re-sorted from the profile.  
    unsigned divergentInputIdx \= profile.getDivergentInputIndex();  
    mlir::Value divergentInput \= launchOp.getKernelOperand(divergentInputIdx);  
    auto memrefType \= cast\<mlir::MemRefType\>(divergentInput.getType());

    // 3\. Create a new buffer for the sorted data.  
    mlir::Value sortedBuffer \= rewriter.create\<mlir::memref::AllocOp\>(launchOp.getLoc(), memrefType);

    // 4\. Insert an orchestra.transfer op to perform the sort.  
    rewriter.create\<orchestra::TransferOp\>(  
        launchOp.getLoc(), divergentInput, sortedBuffer,  
        profile.getSortKeyAsAttr(), // Attribute specifying how to sort  
        rewriter.getSymbolRefAttr("host\_dram"),  
        rewriter.getSymbolRefAttr("host\_dram"));

    // 5\. Replace the original operand of the launch op with the new sorted buffer.  
    rewriter.updateRootInPlace(launchOp, \[&\]() {  
      launchOp.getKernelOperandMutable(divergentInputIdx).assign(sortedBuffer);  
    });  
    return mlir::success();  
  }  
private:  
  const DivergenceProfile \&profile;  
};

### **3.4. Implementation Pattern B: Profile-Guided Intra-Kernel Thread-to-Data Remapping**

This more aggressive *intra-kernel* pattern modifies the kernel's internal logic. It dynamically re-assigns data items to threads within a single workgroup immediately before a divergent branch, forming new, convergent warps on-the-fly. This approach is inspired by academic research on on-GPU remapping, such as the WAGNERR algorithm.1

* **Trigger Heuristic:** This pattern is used when the divergence profile shows a stable but non-sortable pattern (e.g., a 50/50 split) or when divergence is caused by intermediate values computed within the kernel, making pre-sorting impossible.  
* **MLIR Implementation:** This pattern targets the divergent scf.if operation directly within the gpu.func. It injects a complex sequence of operations before the branch:  
  1. **Allocate Shared Memory:** Use memref.alloc with the \#gpu.address\_space\<workgroup\> attribute to create shared memory arrays for indices (true\_indices, false\_indices) and atomic counters (true\_counter, false\_counter).14  
  2. **Populate Index Arrays:** Each thread evaluates the branch condition. Based on the outcome, it uses arith.atomic\_rmw with kind addi to atomically increment the appropriate counter and get a unique slot in the corresponding index array. It then stores its original gpu.thread\_id into that slot.  
  3. **Synchronize:** A gpu.barrier is inserted to ensure all threads in the workgroup have populated the shared memory arrays before any thread proceeds.14  
  4. **Rewrite Control Flow:** The original scf.if is replaced. The new logic remaps threads based on their position in the new compact groups. For example, threads with a new ID less than the final value of true\_counter will execute the then logic. Inside this path, the thread uses its new ID to load the *original* thread ID from the true\_indices array. This original ID is then used for all subsequent data indexing, preserving correctness while ensuring the warps executing the then and else bodies are fully convergent.

This intra-kernel remapping introduces significant overhead from shared memory accesses, atomics, and synchronization. Its application must be governed by a cost model that weighs the profiled serialization penalty against the estimated cost of the remapping logic to ensure a net performance gain.1

## **Section 4: Lowering OrchestraIR to Hardware-Specific Primitives**

### **4.1. The Dialect Conversion Framework: A Primer**

The process of translating the high-level, abstract OrchestraIR dialect into low-level, hardware-specific operations is managed by MLIR's DialectConversion framework.1 This framework provides a robust and systematic way to perform such transformations through pattern-based rewriting. The conversion is governed by three key components 15:

1. **ConversionTarget:** This defines the "legality" of the final IR. For this lowering, the target will mark all operations in the OrchestraIR dialect as Illegal, signaling that they must be converted. Concurrently, operations in the target dialects (e.g., nvgpu, xegpu, arith, memref) will be marked as Legal. The conversion driver, applyFullConversion, will apply patterns until no illegal operations remain.15  
2. **RewritePatternSet:** This is a collection of ConversionPattern classes. Each pattern is responsible for matching a specific illegal operation (e.g., orchestra.transfer) and rewriting it into a sequence of legal ones.  
3. **TypeConverter:** This optional but crucial component manages changes in the type system during conversion. It is essential for handling the translation between OrchestraOS's logical memory spaces and the physical address spaces of the target hardware.17

### **4.2. A Custom TypeConverter for OrchestraOS Memory Spaces**

A critical challenge in lowering OrchestraIR is the management of memory spaces. The orchestra.transfer operation uses symbolic resource handles (e.g., @host\_dram, @gpu0\_hbm), represented as SymbolRefAttr. However, the target GPU dialects operate on memref types that use integer-based address spaces, such as \#gpu.address\_space\<global\> or \#gpu.address\_space\<workgroup\>.1

A standard LLVMTypeConverter is insufficient for this task. The solution is to implement a custom OrchestraTypeConverter that inherits from mlir::TypeConverter.18 This custom converter bridges the abstraction gap between the logical and physical memory models. Its implementation will center on the

addConversion method, which defines a conversion rule for mlir::MemRefType. This rule's callback function will inspect the memory space attribute of the input memref. If the attribute is a symbolic one from OrchestraIR, the converter will perform a lookup based on the target architecture and the symbolic name, constructing and returning a new MemRefType with the correct integer gpu.address\_space attribute.1

### **4.3. Lowering orchestra.commit to arith.select**

The lowering of the orchestra.commit operation is a straightforward one-to-one mapping. The arith.select operation has precisely the semantics required: it takes a boolean (i1) condition and two values, and selects one based on the condition.19 This operation is a standard primitive that LLVM and other code-generation backends can easily map to a low-cost conditional move (

cmov) instruction or an equivalent predicated instruction on the target hardware, directly fulfilling the design goal.1

The implementation is a simple ConversionPattern\<orchestra::CommitOp\> that uses the ConversionPatternRewriter to create an mlir::arith::SelectOp and replace the original commit operation with its result.1

C++

// In LowerOrchestraToStandard.cpp  
class CommitOpLowering : public mlir::ConversionPattern\<orchestra::CommitOp\> {  
public:  
  explicit CommitOpLowering(mlir::MLIRContext \*context)  
      : mlir::ConversionPattern\<orchestra::CommitOp\>(context) {}

  mlir::LogicalResult  
  matchAndRewrite(orchestra::CommitOp commitOp, OpAdaptor adaptor,  
                  mlir::ConversionPatternRewriter \&rewriter) const override {  
    // The OpAdaptor provides the operands, which may have already been  
    // type-converted.  
    rewriter.replaceOpWithNewOp\<mlir::arith::SelectOp\>(  
        commitOp, adaptor.getCondition(), adaptor.getTrueValues(),  
        adaptor.getFalseValues());  
    // Note: This simplified example assumes a single result. A full  
    // implementation would loop over all results and create multiple selects.  
    return mlir::success();  
  }  
};

### **4.4. A Stateful Pass for Lowering orchestra.transfer to nvgpu Asynchronous DMA**

Lowering orchestra.transfer to NVIDIA's token-based asynchronous copy model presents a unique challenge. A simple, stateless ConversionPattern that replaces orchestra.transfer with nvgpu.device\_async\_copy followed immediately by nvgpu.device\_async\_wait would render the copy synchronous, defeating the goal of overlapping data movement with computation.1

The correct implementation requires a **stateful pass**. The pass must operate in two conceptual phases:

1. **Rewrite Phase:** A ConversionPattern\<orchestra::TransferOp\> matches each transfer operation. It performs a local rewrite, creating the nvgpu.device\_async\_copy op.20 It then captures the  
   \!nvgpu.device\_async\_token returned by the copy op and stores it in a data structure managed by the pass instance, associating the token with the destination memref buffer. The original transfer op's result (the destination buffer handle) is replaced by the destination memref operand.  
2. **Synchronization Phase:** After the rewrite patterns have been applied across the entire function, the pass performs a second walk of the IR. When it encounters a use of a buffer that was the destination of an async copy, it inserts the necessary nvgpu.device\_async\_create\_group and nvgpu.device\_async\_wait operations immediately before the consuming operation. This "just-in-time" insertion of the wait barrier ensures it happens as late as possible, maximizing the potential overlap window.1 While MLIR passes are typically stateless for thread-safety, this state is confined to the pass's execution on a single function and is a necessary design pattern for managing this type of hardware asynchrony.21

### **4.5. Decomposing orchestra.transfer into xegpu Load/Store Sequences**

The lowering of orchestra.transfer to the Intel GPU dialect (xegpu) requires a fundamentally different approach. The xegpu dialect is designed around a tile-based programming model and does not have a single operation for a direct memory-to-memory transfer. The copy must be decomposed into a sequence of load and store operations.1

The implementation pattern is a ConversionPattern\<orchestra::TransferOp\> that replaces the single abstract operation with a multi-operation sequence, typically inside a loop (scf.for) that iterates over tiles of the data to be copied:

1. **Create Descriptors:** Inside the loop, xegpu.create\_nd\_tdesc is used to create "tensor descriptors" for the current source and destination tiles. A tensor descriptor is an opaque handle to a sub-region of memory that will be the target of a load or store.23  
2. **Load to Registers:** xegpu.load\_nd is used to load a 2D tile from the source memory (specified by its descriptor) into a vector type, which represents the data being held in physical registers.23  
3. **Store from Registers:** xegpu.store\_nd is used to store the contents of the vector from registers to the destination memory (specified by its descriptor).23  
4. **Synchronization:** Correctness hinges on the proper use of xegpu.fence. A fence with the appropriate scope (e.g., "workgroup") must be inserted after a sequence of stores to ensure that the written data is visible to all threads in the workgroup before any subsequent operation attempts to read it.23

| Feature | NVIDIA Target (nvgpu) | Intel Target (xegpu) |
| :---- | :---- | :---- |
| **Primary MLIR Op(s)** | nvgpu.device\_async\_copy 20 | xegpu.load\_nd, xegpu.store\_nd 23 |
| **Synchronization Primitive** | nvgpu.device\_async\_wait (on a \!nvgpu.device\_async\_token) 20 | xegpu.fence 23 |
| **Programming Model** | Explicit token-based asynchrony ("fire-and-forget" with wait) | Descriptor-based, tile-oriented load/store with memory barriers |
| **Key Memory Abstraction** | memref with gpu.address\_space attribute 14 | \!xegpu.tensor\_desc created from a memref 23 |

## **Section 5: Advanced Hardware-Aware Optimizations on the Linalg Dialect**

### **5.1. A Modular Framework for Operator Fusion**

Operator fusion is a critical optimization that combines multiple operations into a single, larger kernel, reducing kernel launch latency and intermediate data movement to and from global memory.1 The

linalg dialect is the ideal level of abstraction for this transformation, as its operations provide a structured, hardware-agnostic representation of tensor computations.24

A naive approach of writing fusion patterns for every possible pair of named linalg ops would lead to a combinatorial explosion of code. A more robust strategy is to first run the \-linalg-generalize-named-ops pass, which converts all named ops (linalg.matmul, linalg.add, etc.) into their equivalent linalg.generic form. This simplifies the fusion logic to a single case: merging two linalg.generic operations.1

To manage hardware heterogeneity, the fusion pass should be architected as a generic HardwareAwareFusionPass that delegates profitability decisions. The generic pass contains the core mechanics of merging linalg.generic operations. However, before performing a fusion, it queries a target-specific delegate to determine if the transformation will actually improve performance on the current hardware. This delegate is formalized as an MLIR OpInterface, OrchestraFusionStrategyInterface, which is attached to the top-level module for each backend. This design cleanly separates the generic *mechanism* of fusion from the hardware-specific *policy* of fusion, creating a powerful, modular, and extensible system.1

### **5.2. Target-Specific Fusion and Lowering Paths**

The implementation of the OrchestraFusionStrategyInterface will vary for each hardware target, encoding the unique constraints and capabilities of their respective matrix math accelerators.

* **NVIDIA Tensor Cores:** The isFusionProfitable method will return true for patterns like matmul \-\> elementwise where the matrix dimensions are compatible with nvgpu.mma.sync instructions (e.g., multiples of 16x8x16 for FP16 on Ampere). The lowering path involves tiling the fused linalg.generic, promoting data to shared memory, and finally converting the tiled computation (often via vector.contract) to nvgpu.mma.sync and nvgpu.ldmatrix.20  
* **AMD Matrix Accelerators:** The profitability function will check for alignment with the shape constraints of amdgpu.mfma (for CDNA) or amdgpu.wmma (for RDNA) intrinsics, which are typically 32x32 or 16x16.1 The lowering path is conceptually similar to the NVIDIA target but targets these AMD-specific operations.  
* **Intel XMX Engines:** The decision is strongly influenced by the requirements of the xegpu.dpas (Dot Product Accumulate Systolic) instruction. This instruction has specific shape and, critically, data layout requirements that must be considered as part of the fusion decision.23

| Hardware | Dialect::Op | Operand A (Type/Layout) | Operand B (Type/Layout) | Accumulator (Type/Layout) | Key Shape Constraints (M, N, K) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **NVIDIA (Ampere+)** | nvgpu.mma.sync | vector\<...xf16\> (Row Major) | vector\<...xf16\> (Col Major) | vector\<...xf32\> (Row Major) | Multiples of 16, 8, 16 |
| **AMD (CDNA)** | amdgpu.mfma | vector\<...xf32\> | vector\<...xf32\> | vector\<...xf32\> | 32, 32, 1 (f32) or 16, 16, 4 (f16) |
| **Intel (Xe-HPG+)** | xegpu.dpas | vector\<...xf16\> | vector\<...xf16\> (VNNI Packed) | vector\<...xf32\> | 8, 16, 16 (for d=8, sd=2) |

### **5.3. Integrated Memory Layout Transformation**

The physical layout of data in memory has a profound impact on performance, particularly on GPUs where coalesced memory accesses are key to achieving high bandwidth.1 Many frameworks default to NCHW (Number, Channels, Height, Width), but hardware accelerators like NVIDIA's Tensor Cores often perform better with NHWC, where all channel data for a single spatial location is contiguous.1

The compiler must be able to transform these layouts. This can be implemented as a rewrite pattern that matches a layout-sensitive operation like linalg.conv\_2d\_nchw\_fchw, inserts linalg.transpose operations around it to change the layout of its inputs and outputs, and then replaces the original op with its NHWC-based equivalent, linalg.conv\_2d\_nhwc\_hwcf. A sophisticated pass will propagate this layout change through an entire subgraph of operations to minimize the number of expensive transpose operations.1

Crucially, fusion and layout transformation decisions are not independent and must be considered holistically. A fusion may only become profitable if a layout transformation is also performed. For example, fusing a matmul and add for an Intel XMX target is highly desirable because the result can be lowered to a single xegpu.dpas instruction. However, this is only possible if the RHS operand is transformed into the VNNI packed layout. This transformation is best performed by the xegpu.load\_nd operation itself, which can perform the packing in hardware during the memory load.23 Therefore, the

isFusionProfitable method in the OrchestraFusionStrategyInterface must implement an integrated cost model that evaluates the combined cost of the fused operation *and* any required layout transformations to make a globally optimal decision.

## **Section 6: Compiler Pipeline Integration and Strategic Recommendations**

### **6.1. Recommended Pass Ordering and Dependencies**

The effectiveness and correctness of the OrchestraOS compiler depend critically on the precise ordering of its passes. The overall flow follows the progressive lowering philosophy of MLIR, gradually transforming the IR from high-level, abstract representations to low-level, hardware-specific forms. The following table outlines the recommended pipeline structure, synthesizing the flow described across the source documents.1

| Stage | Input Dialect(s) | Key Transformation Passes | Output Dialect(s) |  |
| :---- | :---- | :---- | :---- | :---- |
| 1\. Ingestion | torch, tf, onnx | Framework-specific normalization, functionalization, shape inference 1 | Normalized func, torch |  |
| 2\. Scheduling | func | Proprietary Topology-Aware Scheduling Pass | OrchestraIR |  |
| 3\. High-Level Opt. | OrchestraIR, scf, cf | \-lift-cf-to-scf, divergence-to-speculation | OrchestraIR, scf |  |
| 4\. Structured Lowering | OrchestraIR | orchestra-to-linalg, orchestra-transfer-to-dma | linalg, scf, memref, nvgpu |  |
| 5\. Code Generation | linalg, scf, memref | Tiling, Fusion, linalg-to-nvgpu, \-one-shot-bufferize 1 | scf, memref, nvgpu, amdgpu, xegpu |  |
| 6\. Executable Lowering | scf, memref, hardware dialects | scf-to-cf, \-gpu-lower-to-nvvm-pipeline 1, | gpu-module-to-binary 29 | llvm, gpu.binary |

Key dependencies must be strictly enforced. The \-lift-cf-to-scf pass must run before the divergence-to-speculation pass to normalize its input.1 The hardware-aware fusion and layout transformation passes operate on the

linalg and tensor dialects and must therefore run before bufferization (-one-shot-bufferize), which converts value-semantic tensors into memory-based memrefs.1

### **6.2. Leveraging the transform Dialect for Declarative Optimization Control**

Instead of hardcoding a single, monolithic optimization strategy into the C++ passes, a more flexible and powerful approach is to use the MLIR transform dialect.30 This dialect allows compiler transformations to be controlled by a script written in MLIR itself. The C++ implementation of fusion or tiling would expose its core rewrite patterns as

transform dialect operations (e.g., orchestra.tile\_linalg\_op, orchestra.fuse\_producer). A separate transform IR script can then be used to precisely define the optimization strategy for a given model or hardware target.1

This architecture cleanly separates the *mechanism* of a transformation (the C++ patterns) from the *policy* of its application (the transform script). This provides maximum flexibility, allowing performance engineers to script and tune complex optimization sequences without needing to recompile the compiler. It is the state-of-the-art approach for building adaptable, high-performance compilers and is strongly recommended for the evolution of the OrchestraOS optimization pipeline.

### **6.3. Summary of Best Practices and Critical Implementation Considerations**

The successful implementation of the OrchestraOS compiler core hinges on adherence to several cross-cutting principles:

* **Correctness:** The highest priority must be placed on correctness. This includes the non-negotiable safety checks for side effects in speculative execution, the careful management of synchronization primitives (gpu.barrier, nvgpu.device\_async\_wait, xegpu.fence) in all lowering passes, and the use of verifiers to enforce dialect invariants.  
* **Modularity:** The proposed architecture emphasizes modularity and extensibility. The use of the OpInterface-based delegate pattern for hardware-specific logic is a key example. This design contains the complexity of supporting new hardware to a well-defined set of new classes, ensuring the compiler remains robust and scalable as it evolves.  
* **Performance:** Achieving optimal performance requires a holistic view. The compiler's cost models must be sophisticated enough to reason about the complex trade-offs between computation, data movement, and layout transformation. Decisions should not be made in isolation but with an understanding of their impact on the entire pipeline.  
* **Strategy:** The architectural vision of the compiler as a "meta-OS" should guide all development. Features that enhance the compiler's ability to reason about the system globally, such as the feedback-driven optimization loop, are not just features but key strategic differentiators. The data generated by these systems is a valuable asset that creates a compounding performance advantage over time.

#### **Referenzen**

1. orchestra \- mlir 3 \- MLIR Hardware-Aware Compiler Optimization  
2. MLIR Dialects in Catalyst \- PennyLane Documentation, Zugriff am August 8, 2025, [https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html)  
3. Defining Dialects \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/DefiningDialects/](https://mlir.llvm.org/docs/DefiningDialects/)  
4. MLIR Dialects in Catalyst \- PennyLane Documentation, Zugriff am August 8, 2025, [https://docs.pennylane.ai/projects/catalyst/en/latest/dev/dialects.html](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/dialects.html)  
5. mlir::OpRewritePattern\< SourceOp \> Struct Template Reference, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/structmlir\_1\_1OpRewritePattern.html](https://mlir.llvm.org/doxygen/structmlir_1_1OpRewritePattern.html)  
6. mlir::PatternRewriter Class Reference, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1PatternRewriter.html](https://mlir.llvm.org/doxygen/classmlir_1_1PatternRewriter.html)  
7. mlir/docs/PatternRewriter.md · doe \- GitLab, Zugriff am August 8, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md)  
8. mlir::Region Class Reference \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1Region.html](https://mlir.llvm.org/doxygen/classmlir_1_1Region.html)  
9. mlir::Value Class Reference \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1Value.html](https://mlir.llvm.org/doxygen/classmlir_1_1Value.html)  
10. llvm-project/mlir/include/mlir/Interfaces/SideEffectInterfaces.td at main \- GitHub, Zugriff am August 8, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td)  
11. MLIR Side Effect Modelling \- LLVM, Zugriff am August 8, 2025, [https://llvm.org/devmtg/2023-10/slides/quicktalks/Niu-MLIRSideEffectModeling.pdf](https://llvm.org/devmtg/2023-10/slides/quicktalks/Niu-MLIRSideEffectModeling.pdf)  
12. Passes \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Passes/](https://mlir.llvm.org/docs/Passes/)  
13. 39 On-GPU Thread-Data Remapping for Branch Divergence Reduction \- HKU, Zugriff am August 8, 2025, [https://i.cs.hku.hk/\~clwang/papers/2019-Billy-TACO-On-GPU-TDR.pdf](https://i.cs.hku.hk/~clwang/papers/2019-Billy-TACO-On-GPU-TDR.pdf)  
14. 'gpu' Dialect \- MLIR, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/GPU/](https://mlir.llvm.org/docs/Dialects/GPU/)  
15. Dialect Conversion \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/DialectConversion/](https://mlir.llvm.org/docs/DialectConversion/)  
16. mlir/docs/DialectConversion.md · main · Kevin Sala / llvm-project \- GitLab, Zugriff am August 8, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/DialectConversion.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/DialectConversion.md)  
17. MLIR — Dialect Conversion \- Math ∩ Programming, Zugriff am August 8, 2025, [https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/](https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/)  
18. mlir::TypeConverter Class Reference \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1TypeConverter.html](https://mlir.llvm.org/doxygen/classmlir_1_1TypeConverter.html)  
19. 'arith' Dialect \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/ArithOps/](https://mlir.llvm.org/docs/Dialects/ArithOps/)  
20. 'nvgpu' Dialect \- MLIR, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/NVGPU/](https://mlir.llvm.org/docs/Dialects/NVGPU/)  
21. Developer Guide \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/getting\_started/DeveloperGuide/](https://mlir.llvm.org/getting_started/DeveloperGuide/)  
22. Pass Infrastructure \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/PassManagement/](https://mlir.llvm.org/docs/PassManagement/)  
23. 'xegpu' Dialect \- MLIR, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/XeGPU/](https://mlir.llvm.org/docs/Dialects/XeGPU/)  
24. MLIR Part 4 \- Linear Algebra in MLIR \- Stephen Diehl, Zugriff am August 8, 2025, [https://www.stephendiehl.com/posts/mlir\_linear\_algebra/](https://www.stephendiehl.com/posts/mlir_linear_algebra/)  
25. 'linalg' Dialect \- MLIR, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/Linalg/](https://mlir.llvm.org/docs/Dialects/Linalg/)  
26. mlir::OpInterface\< ConcreteType, Traits \> Class Template Reference \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1OpInterface.html](https://mlir.llvm.org/doxygen/classmlir_1_1OpInterface.html)  
27. Interfaces \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Interfaces/](https://mlir.llvm.org/docs/Interfaces/)  
28. 'amdgpu' Dialect \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Dialects/AMDGPU/](https://mlir.llvm.org/docs/Dialects/AMDGPU/)  
29. llvm-project/mlir/docs/Dialects/GPU.md at main \- GitHub, Zugriff am August 8, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md)  
30. The MLIR Transform Dialect \- arXiv, Zugriff am August 8, 2025, [https://arxiv.org/html/2409.03864v2](https://arxiv.org/html/2409.03864v2)  
31. \[2404.19350\] Transform Dialect Tutorial \- arXiv, Zugriff am August 8, 2025, [https://arxiv.org/abs/2404.19350](https://arxiv.org/abs/2404.19350)