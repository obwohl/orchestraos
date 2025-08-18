

# **Architecting a Robust and Extensible GPU Profiling Pass in MLIR**

## **A Principled Approach to Instrumentation Buffer Management**

The foundation of any robust instrumentation pass lies in its management of memory and state. For the OrchestraBranchProfiler, this translates to the meticulous allocation and indexing of the shared memory buffer that will store branch divergence counters. A naive, single-pass implementation that interleaves analysis with IR mutation is fraught with peril, risking iterator invalidation and incorrect behavior. The canonical and most robust methodology, mirroring advanced frameworks within MLIR, is a two-phase design pattern that strictly separates analysis from transformation. This approach, combined with a stable mechanism for identifying instrumentation sites, forms the bedrock of a correct, efficient, and maintainable pass.

### **The Two-Phase Design Pattern: Analysis and Transformation**

The core logic of the OrchestraBranchProfiler pass should be structured within its runOnOperation method as two distinct phases. This separation is paramount for safely managing IR modifications, particularly when the necessary transformations, such as determining the total buffer size, depend on global properties of the gpu.func being processed.

The first phase is a comprehensive, read-only **analysis walk**. This walk traverses the entire body of the gpu.func operation, typically using gpuFuncOp.walk(...), to identify and collect all scf.if operations targeted for instrumentation. The collected operations should be stored in a container, such as a SmallVector\<scf::IfOp, 16\>, while a counter tallies the total number of instrumentation sites. This read-only traversal provides a stable, unmodified snapshot of the IR. This stability is critical for correctness, as subsequent modifications will not invalidate iterators or alter the structure of the code being analyzed. This design philosophy is not novel; it is a cornerstone of sophisticated MLIR passes like One-Shot Bufferize, which performs a whole-function analysis to make optimal bufferization decisions before committing to any IR rewrites.1

The second phase is the **transformation phase**, which leverages the information gathered during analysis to perform all necessary IR mutations. This phase is further subdivided into two steps. First, a single memref.alloca operation is inserted at the beginning of the gpu.func's entry block. The size of this allocation is determined precisely by the total count of scf.if sites gathered in the analysis phase. Second, the pass iterates over the SmallVector of scf.if operations collected previously. For each operation, it assigns a unique, stable identifier and injects the corresponding arith.atomic\_rmw operations into its then and else blocks.

By separating analysis from transformation, this design systematically avoids common pitfalls like iterator invalidation and ensures that operations are inserted in a logically sound order—specifically, the buffer allocation must dominate all its subsequent uses. This structured approach is not merely a suggestion but a practical necessity for complying with the MLIR pass manager's strict rules, which forbid a pass from modifying parent or sibling operations while traversing a nested operation.3

### **Workgroup Memory Allocation in gpu.func**

The allocation of the counter buffer must be performed at the entry point of the GPU kernel to ensure it is available throughout the kernel's execution. An OpBuilder provides the necessary API to create and insert new operations into the IR. The insertion point should be explicitly set to the beginning of the entry block of the gpu.func.

The choice of allocation operation is memref.alloca, which is the generic, target-independent operation for requesting stack-like memory allocation within a function's scope.7 The crucial step is to annotate this allocation with the correct memory space. The

gpu dialect defines an attribute for this purpose: gpu::AddressSpaceAttr. By specifying gpu::AddressSpace::Workgroup, we instruct the compiler that this memref should reside in the workgroup's shared memory, a resource accessible by all threads within a thread block.9 This high-level, abstract representation is then lowered by subsequent passes in the GPU compilation pipeline (e.g.,

\-convert-gpu-to-nvvm or \-convert-gpu-to-rocdl) to the target-specific instructions and memory qualifiers. Using the more generic memref.alloca rather than a dialect-specific gpu.alloc promotes better layering and modularity, as the core instrumentation logic remains agnostic to the specifics of the GPU dialect, depending on it only for the address space enumeration.

The following C++ code demonstrates the creation of the workgroup memory buffer:

C++

// Inside runOnOperation(), after the analysis walk has determined 'totalIfSites'  
gpu::GPUFuncOp gpuFunc \= getOperation();  
OpBuilder builder(gpuFunc.getBody());  
auto loc \= gpuFunc.getLoc();

// Each 'scf.if' requires two counters: one for the 'then' branch and one for the 'else'.  
int64\_t numCounters \= totalIfSites \* 2;  
if (numCounters \== 0) return; // No instrumentation needed.

auto counterType \= builder.getI32Type();  
auto memrefType \= MemRefType::get({numCounters}, counterType, {},   
    gpu::AddressSpaceAttr::get(builder.getContext(), gpu::AddressSpace::Workgroup));

// Set the insertion point to the beginning of the function's entry block.  
builder.setInsertionPointToStart(\&gpuFunc.front());  
Value counterBuffer \= builder.create\<memref::AllocaOp\>(loc, memrefType);

### **Achieving Stable Instrumentation via Operation Attributes**

A robust Feedback-Driven Optimization (FDO) system requires a stable mapping between a source-level construct and its corresponding profile data. This mapping must persist across multiple compilations, even in the face of unrelated code changes or compiler optimizations that might reorder or duplicate operations. Relying on transient properties like memory addresses or the relative order of operations within the IR is inherently fragile. While SSA value names (%0, %1, etc.) provide identification in the textual format, they are not persistent properties of the in-memory IR and are regenerated during printing.14

The canonical solution in MLIR is to attach a custom attribute to each instrumented operation. This attribute stores a unique and stable identifier, durably linking the IR construct to its counter slot. According to the MLIR language reference, attributes intended as external metadata rather than as an inherent part of an operation's semantics should be "discardable" and namespaced with a dialect prefix to prevent collisions.14 For this pass, an attribute named

"orchestra.prof\_id" would be appropriate.

The implementation is straightforward: during the transformation phase, as the pass iterates through the collected scf.if operations, it assigns each a unique, monotonically increasing integer ID. This ID is then attached to the operation as an IntegerAttr.

C++

// During the transformation loop  
int64\_t id \= it.index(); // 'it' is from llvm::enumerate  
scf::IfOp ifOp \= it.value();  
ifOp-\>setAttr("orchestra.prof\_id", builder.getI64IntegerAttr(id));

The counter indices for the then and else blocks can then be deterministically calculated from this stable ID, for example, as $2 \\times id$ and $2 \\times id \+ 1$, respectively. This practice of attaching unique ID attributes for tracking and debugging purposes is a well-established and effective technique within the MLIR community.15 This approach ensures that the profiling data can be reliably mapped back to the original source code, a fundamental requirement for any FDO system.

## **Pass Structure: Balancing Simplicity and Safety**

The architectural choice of how to structure an MLIR pass—specifically, whether to use the declarative RewritePattern framework or a more direct, imperative IR walk—has significant implications for simplicity, performance, and safety. While the RewritePattern engine is a powerful tool for complex, iterative transformations, an insertion-only instrumentation pass like OrchestraBranchProfiler is better served by a direct Operation::walk. This approach is simpler, more performant, and, when combined with a disciplined mutation protocol, equally safe.

### **The Case for a Direct Operation::walk over RewritePattern**

The RewritePattern infrastructure, driven by mechanisms like applyPatternsGreedily, is fundamentally designed for complex, graph-like, DAG-to-DAG transformations. Its strength lies in managing a worklist of operations and applying a set of rewrite rules until a fixed point is reached, where one rewrite may create new opportunities for other patterns to match.16 This is the ideal tool for tasks like canonicalization, dialect lowering, or complex algebraic simplification.

However, the OrchestraBranchProfiler pass does not fit this model. Its task is not to rewrite or replace existing operations in a way that requires convergence. Instead, it performs a single, targeted transformation: inserting new atomic operations into the regions of existing scf.if ops. For this linear, non-iterative task, the overhead of the pattern driver's worklist management and fixed-point logic is unnecessary. A direct traversal using getOperation()-\>walk(...) is a more natural, simpler, and efficient solution. This aligns with contemporary best-practice guidance from MLIR experts, who explicitly recommend using Operation::walk unless fixed-point application (e.g., a rewrite creates an operation that must also be rewritten) or dialect conversion is required. For other cases, a direct walk is "faster, simpler and more predictable".18

A comparative analysis highlights the trade-offs:

* **RewritePattern Approach:**  
  * **Pros:** IR modifications are managed through a PatternRewriter, which provides a safe, transactional API and automatically handles worklist updates for the driver.16 Patterns are inherently composable, allowing them to be mixed and matched in different pass pipelines.  
  * **Cons:** Incurs the overhead of the pattern driver. The boilerplate required to define a pattern and a pass to run it is more verbose than a direct walk. The declarative nature can make the exact order of transformations less predictable.  
* **Direct walk Approach:**  
  * **Pros:** Offers maximum simplicity and performance for straightforward traversals. The developer has direct, imperative control over the traversal and the exact sequence of IR mutations.18  
  * **Cons:** The developer bears the full responsibility for ensuring the safety of IR mutations. The primary risk is iterator invalidation if the IR is modified during the walk itself.

For the OrchestraBranchProfiler, the benefits of the direct walk's simplicity and performance far outweigh the managed safety of the RewritePattern framework, provided a safe mutation protocol is strictly followed.

### **A Protocol for Safe IR Mutation within a Direct Walk**

The primary hazard when modifying the IR during a direct traversal is iterator invalidation. The walk function traverses the operation graph, and if an operation is modified or erased while the walker is active, the internal iterators can be invalidated, leading to undefined behavior or crashes.

The solution is a "Collect-and-Modify" protocol, which is a direct implementation of the two-phase design pattern discussed previously. This protocol combines the efficiency of a direct walk with the safety of a transactional approach.

1. **Collect:** The pass first performs a read-only walk over the gpu.func. The sole purpose of this walk is to identify all scf.if operations that require instrumentation. Pointers to these operations are collected and stored in a separate data structure, such as a SmallVector.  
2. **Modify:** Only after the walk has completed and the walker has been destroyed does the pass proceed to modify the IR. It iterates over the SmallVector of collected operations and is now free to insert new operations into their regions using an OpBuilder. Because this modification loop is decoupled from the IR traversal, the risk of iterator invalidation is completely eliminated.

The C++ structure for this protocol is clean and robust:

C++

\#**include** "mlir/IR/Builders.h"  
\#**include** "mlir/Pass/Pass.h"  
\#**include** "mlir/Dialect/GPU/IR/GPUDialect.h"  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
\#**include** "llvm/ADT/SmallVector.h"

//... Pass definition...

void OrchestraBranchProfiler::runOnOperation() {  
    gpu::GPUFuncOp gpuFunc \= getOperation();  
    MLIRContext \*context \= \&getContext();

    // Phase 1: Collect instrumentation targets in a read-only walk.  
    llvm::SmallVector\<scf::IfOp, 16\> ifOpsToInstrument;  
    gpuFunc.walk(\[&\](scf::IfOp ifOp) {  
        // In a real implementation, add logic to decide if this 'if' should be instrumented.  
        ifOpsToInstrument.push\_back(ifOp);  
    });

    // If no targets were found, preserve all analyses and exit early.  
    // This is a critical performance optimization.  
    if (ifOpsToInstrument.empty()) {  
        markAllAnalysesPreserved();  
        return;  
    }

    // Phase 2, Part A: Allocate the shared memory buffer.  
    OpBuilder builder(context);  
    builder.setInsertionPointToStart(\&gpuFunc.front());  
    //... (Buffer allocation logic as described in Section 1)...  
    Value counterBuffer \= /\*... create memref.alloca... \*/;

    // Phase 2, Part B: Instrument the collected sites.  
    // This loop iterates over a separate collection, not the live IR, ensuring safety.  
    for (auto it : llvm::enumerate(ifOpsToInstrument)) {  
        scf::IfOp ifOp \= it.value();  
        int64\_t id \= it.index();

        // Attach stable ID attribute.  
        ifOp-\>setAttr("orchestra.prof\_id", builder.getI64IntegerAttr(id));  
          
        // Delegate actual IR insertion to a helper function.  
        instrumentIfOp(ifOp, counterBuffer, id, builder);  
    }  
}

This pattern represents a standard and robust methodology for handling IR mutations in MLIR passes that do not use the pattern rewriter framework.20 It also provides a natural point to check if any work needs to be done. If the collection phase yields an empty list, the pass can call

markAllAnalysesPreserved() and return, signaling to the pass manager that no analyses were invalidated, thereby avoiding potentially expensive re-computations by subsequent passes.3 This level of explicit control over interaction with the pass manager is a key advantage of the imperative approach.

## **Mastering the GPU Execution Model**

Injecting instrumentation code into a GPU kernel requires a deep understanding of the parallel execution model. The goal is not merely correctness but also minimizing performance perturbation. The introduction of atomic operations into shared memory, while functionally correct, can create significant performance bottlenecks if not managed carefully. This section examines the performance characteristics of atomics, proposes strategies to mitigate common hazards like contention and bank conflicts, and discusses the interaction of instrumentation with compiler optimizations and memory visibility semantics.

### **Performance Deep Dive: The Cost of Atomics in Shared Memory**

The core of the instrumentation logic is an atomic read-modify-write operation on a counter in shared memory. The correct MLIR operation for this is memref.atomic\_rmw, which operates on a memref at a given set of indices.7 This is distinct from

arith.atomic\_rmw, which is a more abstract operation on scalar values.23 The

memref.atomic\_rmw will be lowered by the GPU pipeline into the appropriate target-specific atomic instruction.

The performance of atomic operations on GPUs is dominated by two factors:

1. **Atomic Contention:** The most significant cost arises from contention, which occurs when multiple threads attempt to perform an atomic operation on the *same memory address* simultaneously. Modern GPUs execute threads in groups called warps (typically 32 threads) in a SIMT (Single Instruction, Multiple Threads) fashion. If all threads in a warp execute an atomic operation on the same address, the hardware must serialize these requests, effectively turning parallel execution into sequential execution for that instruction. In the context of the OrchestraBranchProfiler, all threads that take the then branch will contend on a single counter, and all threads that take the else branch will contend on another. This serialization is the primary source of performance overhead from the instrumentation.24 The performance degradation is therefore directly related to the degree of divergence, which is precisely the phenomenon being measured.  
2. **Bank Conflicts:** A more subtle but still significant performance hazard is shared memory bank conflicts. To provide high bandwidth, on-chip shared memory is divided into a number of banks (e.g., 32 banks on NVIDIA architectures) that can service requests in parallel. A bank conflict occurs when multiple threads within the same warp access different memory addresses that map to the same memory bank.28 These conflicting requests are serialized. If the counter buffer is laid out as a simple, contiguous array of 32-bit integers, then  
   counter\[i\] and counter\[i+32\] will map to the same bank. If threads in a warp happen to be executing code that instruments two different scf.if sites whose counter indices are 32 apart, a bank conflict will occur, serializing the atomic updates even though they are to different counters.32

### **Proactive Contention Mitigation Strategies**

While contention on a single counter due to divergence is unavoidable with this profiling strategy, inter-counter interference from bank conflicts can and should be mitigated. The most practical and effective technique is to introduce padding into the counter buffer to alter its memory layout. By ensuring that adjacent counters or counters with a high probability of being accessed concurrently by the same warp map to different banks, serialization can be significantly reduced.

A simple and effective padding strategy is to change the stride of the counter array. Instead of allocating two 32-bit integers for each scf.if site, we can allocate three. The third element serves as padding, skewing the memory addresses to break the regular access patterns that lead to conflicts.

The following table compares a naive, contiguous layout with a padded layout, demonstrating how padding mitigates bank conflicts for a typical GPU with 32 memory banks.

| Feature | Naive Contiguous Layout | Padded Layout |
| :---- | :---- | :---- |
| **Description** | Counters are stored sequentially: \[if0\_t, if0\_e, if1\_t, if1\_e,...\] | A padding element is inserted after each counter pair: \[if0\_t, if0\_e, pad0, if1\_t, if1\_e, pad1,...\] |
| **Index Calculation** | idx\_then \= 2 \* id idx\_else \= 2 \* id \+ 1 | idx\_then \= 3 \* id idx\_else \= 3 \* id \+ 1 |
| **Bank Mapping (32-bit words)** | bank(id) \= (2 \* id) % 32 bank(id+16) \= (2 \* (id+16)) % 32 \= (2\*id \+ 32\) % 32 \= bank(id) | bank(id) \= (3 \* id) % 32 bank(id+16) \= (3 \* (id+16)) % 32 \= (3\*id \+ 48\) % 32 \= (bank(id) \+ 16\) % 32 |
| **Conflict Potential** | **High.** Counters for if\_site\_N and if\_site\_N+16 will always conflict, as they map to the same memory bank. | **Low.** Counters for if\_site\_N and if\_site\_N+16 are guaranteed to be in different banks, eliminating this common conflict pattern. |

This padding strategy can be implemented by simply adjusting the size of the memref.alloca and the index calculations used when creating the memref.atomic\_rmw operations. While this uses slightly more shared memory, the performance benefit from avoiding serialization far outweighs the cost of a few extra bytes per branch site. Other strategies, such as using hash functions to randomize bank mapping, are possible but add significant complexity.33 Simple padding provides a robust and low-cost solution.34

### **Instrumentation and Compiler Optimizations**

The introduction of operations with memory side effects, such as atomics, can interact with and potentially inhibit downstream compiler optimizations.

* **Thread Reconvergence:** High-performance GPU execution relies on the ability of threads within a warp to reconverge after a divergent branch. The hardware executes each divergent path serially and then reconverges all threads at the immediate post-dominator of the branch.36 The presence of an atomic operation inside a divergent branch is a side-effecting instruction that the compiler must preserve. This can limit the compiler's ability to perform optimizations that rely on reordering or speculating code across the branch, such as code hoisting or certain forms of predication.38 To minimize this interference, the  
  OrchestraBranchProfiler pass should be scheduled to run as late as possible in the compilation pipeline, after most functional and performance optimizations have already been applied.  
* **SPIR-V Conversion:** If the final compilation target is SPIR-V, the instrumentation pass must execute *before* the conversion to the spirv dialect. The SPIR-V dialect models structured control flow with a very rigid set of rules, requiring explicit spirv.mlir.selection and spirv.mlir.loop operations that define merge and continue blocks.39 Modifying the IR after it has been lowered to this structured form is complex and highly likely to violate the dialect's invariants. Therefore, the pass must operate on the higher-level  
  scf.if representation.

### **Ensuring Memory Visibility with gpu.barrier**

An atomic operation guarantees the atomicity of its read-modify-write sequence, preventing race conditions on the update itself. However, it does not, by itself, guarantee that the result of the update is immediately visible to all other threads in the workgroup, particularly those executing in different warps. Caches and memory ordering models can lead to scenarios where one warp's updates are not yet visible to another.

To ensure that the final counts in the shared memory buffer are consistent and reflect the contributions of all threads in the workgroup, a gpu.barrier operation must be inserted at the end of the gpu.func, immediately before the gpu.return operation. The gpu.barrier serves two critical purposes:

1. **Execution Synchronization:** It acts as a rendezvous point. No thread in the workgroup can proceed past the barrier until all other threads in the workgroup have reached it.11  
2. **Memory Fence:** It enforces memory visibility. It guarantees that all memory writes performed by any thread before the barrier are made visible to all other threads in the workgroup after the barrier.10

Without this final barrier, there is no guarantee that the counter values in shared memory are final and consistent when the kernel terminates. While some atomic operations on some architectures may have acquire/release semantics that provide stronger ordering guarantees, relying on this is non-portable and architecturally unsound. The explicit gpu.barrier is the canonical and correct mechanism for ensuring workgroup-wide memory consistency.42 It establishes a clear semantic boundary, signaling to the compiler that all preceding memory side effects must be completed and made visible before the kernel exits.

## **A Framework for Future-Proof, Extensible Profiling**

A successful compiler infrastructure is not only powerful but also extensible, capable of evolving to meet new requirements without extensive refactoring. The OrchestraBranchProfiler should be designed from the outset with this principle in mind. While the initial target is scf.if, the FDO system will inevitably need to profile other constructs, such as loop trip counts (scf.for) or other custom control flow operations. A design based on a configurable list of operation names is brittle and unscalable. The idiomatic and architecturally superior solution in MLIR is to use a custom OpInterface.

### **The MLIR Way: Interfaces over Configuration**

MLIR's design philosophy heavily favors composition and extensibility through dialects and interfaces.45 An

OpInterface is a formal contract that an operation can implement to expose specific behaviors or properties to generic passes and analyses.49 This approach is vastly superior to a hardcoded list of operation names for several reasons:

* **Decoupling:** The profiler pass no longer needs to have compile-time knowledge of scf.if, scf.for, or any other concrete operation. It only needs to know about the abstract ProfilableOpInterface contract. This decouples the transformation pass from the dialects it operates on.  
* **Extensibility:** To enable profiling for a new operation, potentially from a completely new dialect, a developer simply needs to provide an implementation of the ProfilableOpInterface for that operation. No modifications are required in the core profiler pass itself. This adheres to the open/closed principle, making the system easy to extend.  
* **Polymorphism:** The pass can interact with any operation that implements the interface through a common set of methods. It can use dyn\_cast\<ProfilableOpInterface\> to treat a generic Operation\* polymorphically, calling the appropriate implementation of the interface methods for that specific operation type.49

This contract-based design inverts the dependency: instead of the central pass knowing about all possible clients, each client (operation) declares its capability to the system by implementing the interface. This is a hallmark of a well-designed, scalable compiler architecture.

### **Design of a ProfilableOpInterface**

A custom interface, ProfilableOpInterface, can be defined declaratively using MLIR's TableGen framework. This interface will define the contract that all profilable operations must adhere to.

**ODS Definition (OrchestraInterfaces.td):**

Code-Snippet

\#include "mlir/IR/OpBase.td"

def ProfilableOpInterface : OpInterface\<"ProfilableOpInterface"\> {  
  let description \=;

  let methods \=;  
}

This definition specifies two methods:

1. unsigned getNumProfilingCounters(): This method is called during the analysis phase of the profiler. Each operation implementing the interface will report how many counter slots it requires in the shared memory buffer. For an scf.if, this would return 2\.  
2. void instrument(OpBuilder \&builder, Value buffer, int64\_t baseCounterIndex): This method is called during the transformation phase. The operation is given an OpBuilder, a handle to the shared memory buffer, and the baseCounterIndex at which its assigned counter slots begin. The operation is then responsible for implementing its own instrumentation logic, inserting the necessary atomic operations into the correct locations within its regions.

To apply this interface to an operation from an upstream dialect like scf without modifying its source code, MLIR provides the powerful mechanism of **external models**.49 An external model allows a downstream project to define an interface implementation for an existing type.

**External Model Implementation for scf.if:**

C++

// In a suitable C++ source file, e.g., OrchestraSCFExtensions.cpp

\#**include** "Orchestra/OrchestraInterfaces.h.inc" // Generated from the.td file  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
\#**include** "mlir/Dialect/MemRef/IR/MemRef.h"  
\#**include** "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {  
namespace orchestra {

// External model definition to attach ProfilableOpInterface to scf::IfOp  
struct ScfIfProfilableInterface  
    : public ProfilableOpInterface::ExternalModel\<ScfIfProfilableInterface, scf::IfOp\> {  
    
  unsigned getNumProfilingCounters(Operation \*op) const {  
    // An scf.if has a 'then' and an optional 'else' branch, requiring two counters.  
    return 2;  
  }

  void instrument(Operation \*op, OpBuilder \&builder, Value buffer, int64\_t baseCounterIndex) const {  
    auto ifOp \= cast\<scf::IfOp\>(op);  
    auto loc \= op-\>getLoc();

    // Create a constant '1' for incrementing the counter.  
    Value one \= builder.create\<arith::ConstantOp\>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));

    // Instrument the 'then' block.  
    builder.setInsertionPointToStart(ifOp.getThenBlock());  
    Value thenIndex \= builder.create\<arith::ConstantIndexOp\>(loc, baseCounterIndex);  
    builder.create\<memref::AtomicRMWOp\>(loc, arith::AtomicRMWKind::addi, one, buffer, ValueRange{thenIndex});

    // Instrument the 'else' block if it exists.  
    if (ifOp.hasElse()) {  
        builder.setInsertionPointToStart(ifOp.getElseBlock());  
        Value elseIndex \= builder.create\<arith::ConstantIndexOp\>(loc, baseCounterIndex \+ 1);  
        builder.create\<memref::AtomicRMWOp\>(loc, arith::AtomicRMWKind::addi, one, buffer, ValueRange{elseIndex});  
    }  
  }  
};

void registerExternalInterfaces(DialectRegistry \&registry) {  
    registry.addExtension(+(MLIRContext \*ctx, scf::SCFDialect \*dialect) {  
        scf::IfOp::attachInterface\<ScfIfProfilableInterface\>(\*ctx);  
    });  
}

} // namespace orchestra  
} // namespace mlir

This external model must then be registered with the MLIRContext when the compiler is initialized.

### **Evolving the Pass to be Interface-Driven**

With the interface defined and implemented, the OrchestraBranchProfiler pass can be rewritten to be completely generic. It no longer references scf.if directly, but instead operates on any Operation that implements ProfilableOpInterface.

Refactored Analysis Phase:  
The analysis walk is modified to find operations that can be cast to the interface. It queries each one to determine the total number of counters needed.

C++

// Refactored Phase 1: Collect and count using the interface.  
llvm::SmallVector\<Operation\*, 16\> profilableOps;  
int64\_t totalCounters \= 0;

gpuFunc.walk(\[&\](ProfilableOpInterface profilableOp) {  
    profilableOps.push\_back(profilableOp.getOperation());  
    totalCounters \+= profilableOp.getNumProfilingCounters();  
});

Refactored Transformation Phase:  
The transformation loop iterates over the collected generic Operation\* pointers, casts each to the interface, and calls the instrument method, delegating the specifics of the instrumentation to the operation's own implementation.

C++

// Refactored Phase 2: Allocate and Instrument.  
//... (Allocate buffer with size 'totalCounters')...  
Value counterBuffer \= /\*... \*/;

int64\_t currentCounterIndex \= 0;  
for (Operation \*op : profilableOps) {  
    auto profilableOp \= cast\<ProfilableOpInterface\>(op);

    // Attach stable ID, using the base index as the ID.  
    op-\>setAttr("orchestra.prof\_id", builder.getI64IntegerAttr(currentCounterIndex));  
      
    // Polymorphically call the instrument method.  
    profilableOp.instrument(builder, counterBuffer, currentCounterIndex);  
      
    // Advance the index for the next operation.  
    currentCounterIndex \+= profilableOp.getNumProfilingCounters();  
}

This interface-driven design is the epitome of MLIR's philosophy. It creates a clean, decoupled, and highly extensible system that can easily grow to support new profiling targets without incurring technical debt in the core profiler pass.

## **Conclusion**

The design of the OrchestraBranchProfiler pass presents a microcosm of the architectural principles that underpin robust and scalable compiler development within the MLIR framework. The proposed architecture addresses the user's requirements not just with functional solutions, but with a principled approach that emphasizes safety, performance, and long-term extensibility. The key architectural recommendations are as follows:

1. **Adopt a Two-Phase (Analyze-then-Transform) Buffer Management Strategy:** This pattern is non-negotiable for correctness and robustness. A preliminary, read-only walk of the gpu.func must be used to gather all instrumentation sites and calculate the required buffer size. Only after this analysis is complete should a second phase modify the IR to insert the memref.alloca for the shared memory buffer and the atomic operations at the collected sites. This avoids iterator invalidation and ensures logical consistency.  
2. **Utilize a Direct Operation::walk with the "Collect-and-Modify" Protocol:** For an insertion-only pass, a direct IR walk is simpler and more performant than the RewritePattern engine. The "Collect-and-Modify" protocol—collecting target operations in a separate container during a read-only walk before iterating over that container to perform modifications—provides the safety of a transactional approach while retaining the efficiency of a direct walk.  
3. **Implement Stable Identification with Custom Attributes:** To ensure the durability of profiling data across compilations, each instrumented scf.if operation must be tagged with a unique, stable ID. A custom, dialect-namespaced attribute (e.g., "orchestra.prof\_id") is the canonical MLIR mechanism for this purpose.  
4. **Mitigate GPU Performance Hazards Proactively:** The performance of atomic operations in shared memory is dominated by contention and bank conflicts. While contention from divergent threads is inherent to the profiling goal, inter-counter interference can be mitigated. The counter buffer should be explicitly padded to ensure that counters for different branch sites are distributed across shared memory banks, minimizing serialization. Furthermore, a gpu.barrier must be inserted before the kernel's return to guarantee workgroup-wide visibility of the final counter values.  
5. **Build for Extensibility with a Custom OpInterface:** The most critical long-term design decision is to build the pass around a custom ProfilableOpInterface. This interface defines a contract for any operation wishing to be instrumented, decoupling the profiler pass from concrete operation types. Using MLIR's external model mechanism, this interface can be implemented for existing operations in upstream dialects like scf without modification. This design ensures that the OrchestraOS profiling framework can be easily extended to new types of control flow or other operations in the future, embodying the modular and extensible spirit of MLIR.

By adhering to these architectural principles, the resulting OrchestraBranchProfiler pass will be more than a functional tool; it will be a robust, efficient, and future-proof component of the OrchestraOS compiler ecosystem.

#### **Referenzen**

1. mlir/docs/Bufferization.md · develop · undefined \- GitLab, Zugriff am August 18, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/develop/mlir/docs/Bufferization.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/develop/mlir/docs/Bufferization.md)  
2. Bufferization \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Bufferization/](https://mlir.llvm.org/docs/Bufferization/)  
3. llvm-project-with-mlir/mlir/g3doc/WritingAPass.md at master \- GitHub, Zugriff am August 18, 2025, [https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/WritingAPass.md](https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/WritingAPass.md)  
4. mlir/docs/PassManagement.md · 07b1177eed7549d0badf72078388422ce73167a0 · llvm-doe / llvm-project · GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/07b1177eed7549d0badf72078388422ce73167a0/mlir/docs/PassManagement.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/07b1177eed7549d0badf72078388422ce73167a0/mlir/docs/PassManagement.md)  
5. Pass Infrastructure \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/PassManagement/](https://mlir.llvm.org/docs/PassManagement/)  
6. mlir/docs/PassManagement.md · 1a88755c4c2dba26a4f63da740a26cf5a7b346a8 · zanef2 / HPAC \- GitLab at Illinois, Zugriff am August 18, 2025, [https://gitlab-03.engr.illinois.edu/zanef21/hpac/-/blob/1a88755c4c2dba26a4f63da740a26cf5a7b346a8/mlir/docs/PassManagement.md](https://gitlab-03.engr.illinois.edu/zanef21/hpac/-/blob/1a88755c4c2dba26a4f63da740a26cf5a7b346a8/mlir/docs/PassManagement.md)  
7. 'memref' Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/MemRef/](https://mlir.llvm.org/docs/Dialects/MemRef/)  
8. mlir::memref Namespace Reference, Zugriff am August 18, 2025, [https://mlir.llvm.org/doxygen/namespacemlir\_1\_1memref.html](https://mlir.llvm.org/doxygen/namespacemlir_1_1memref.html)  
9. llvm-project/mlir/docs/Dialects/GPU.md at main \- GitHub, Zugriff am August 18, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md)  
10. 'gpu' Dialect \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/GPU/](https://mlir.llvm.org/docs/Dialects/GPU/)  
11. GPU dialect \- joker-eph/llvm-project-with-mlir \- GitHub, Zugriff am August 18, 2025, [https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/Dialects/GPU.md](https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/Dialects/GPU.md)  
12. \[RFC\]\[GPU\] Functions and Memory Modeling \- Google Groups, Zugriff am August 18, 2025, [https://groups.google.com/a/tensorflow.org/g/mlir/c/RfXNP7Hklsc/m/MBNN7KhjAgAJ](https://groups.google.com/a/tensorflow.org/g/mlir/c/RfXNP7Hklsc/m/MBNN7KhjAgAJ)  
13. mlir/docs/Dialects/GPU.md · 5082acce4fd3646d5760c02b2c21d9cd2a1d7130 · llvm-doe / llvm-project \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md)  
14. MLIR Language Reference, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/LangRef/](https://mlir.llvm.org/docs/LangRef/)  
15. \[RFC\] Better support for dialects that want to make SSA names "load bearing" \- Page 2 \- MLIR \- LLVM Discourse, Zugriff am August 18, 2025, [https://discourse.llvm.org/t/rfc-better-support-for-dialects-that-want-to-make-ssa-names-load-bearing/674?page=2](https://discourse.llvm.org/t/rfc-better-support-for-dialects-that-want-to-make-ssa-names-load-bearing/674?page=2)  
16. mlir/docs/PatternRewriter.md · doe \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md)  
17. Pattern Rewriting : Generic DAG-to-DAG Rewriting \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/PatternRewriter/](https://mlir.llvm.org/docs/PatternRewriter/)  
18. The State of Pattern-Based IR Rewriting in MLIR \- LLVM, Zugriff am August 18, 2025, [https://llvm.org/devmtg/2024-10/slides/techtalk/Springer-Pattern-Based-IR-Rewriting-in-MLIR.pdf](https://llvm.org/devmtg/2024-10/slides/techtalk/Springer-Pattern-Based-IR-Rewriting-in-MLIR.pdf)  
19. The State of Pattern-Based IR Rewriting in MLIR \- Matthias Springer, Zugriff am August 18, 2025, [https://m-sp.org/downloads/llvm\_dev\_2024.pdf](https://m-sp.org/downloads/llvm_dev_2024.pdf)  
20. Understanding MLIR Passes Through a Simple Dialect Transformation \- Medium, Zugriff am August 18, 2025, [https://medium.com/@60b36t/understanding-mlir-passes-through-a-simple-dialect-transformation-879ca47f504f](https://medium.com/@60b36t/understanding-mlir-passes-through-a-simple-dialect-transformation-879ca47f504f)  
21. MLIR — Writing Our First Pass \- Math ∩ Programming, Zugriff am August 18, 2025, [https://www.jeremykun.com/2023/08/10/mlir-writing-our-first-pass/](https://www.jeremykun.com/2023/08/10/mlir-writing-our-first-pass/)  
22. Understanding the IR Structure \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)  
23. 'arith' Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/ArithOps/](https://mlir.llvm.org/docs/Dialects/ArithOps/)  
24. Modeling Utilization to Identify Shared-memory Atomic Bottlenecks \- arXiv, Zugriff am August 18, 2025, [https://arxiv.org/html/2503.17893v1](https://arxiv.org/html/2503.17893v1)  
25. Atomics and contention \- Arm GPU Best Practices Developer Guide, Zugriff am August 18, 2025, [https://developer.arm.com/documentation/101897/latest/Shader-code/Atomics](https://developer.arm.com/documentation/101897/latest/Shader-code/Atomics)  
26. Performance of atomic operations on shared memory \- Stack Overflow, Zugriff am August 18, 2025, [https://stackoverflow.com/questions/19505404/performance-of-atomic-operations-on-shared-memory](https://stackoverflow.com/questions/19505404/performance-of-atomic-operations-on-shared-memory)  
27. CUDA atomic operation performance in different scenarios \- Stack Overflow, Zugriff am August 18, 2025, [https://stackoverflow.com/questions/22367238/cuda-atomic-operation-performance-in-different-scenarios](https://stackoverflow.com/questions/22367238/cuda-atomic-operation-performance-in-different-scenarios)  
28. What is a "memory bank"? (shouldn't be hard to answer, but all my searches return "memory bank conflict", not what a "memory bank" actually is) : r/CUDA \- Reddit, Zugriff am August 18, 2025, [https://www.reddit.com/r/CUDA/comments/5xhdcy/what\_is\_a\_memory\_bank\_shouldnt\_be\_hard\_to\_answer/](https://www.reddit.com/r/CUDA/comments/5xhdcy/what_is_a_memory_bank_shouldnt_be_hard_to_answer/)  
29. NVIDIA CUDA Tutorial 9: Bank Conflicts \- YouTube, Zugriff am August 18, 2025, [https://www.youtube.com/watch?v=CZgM3DEBplE](https://www.youtube.com/watch?v=CZgM3DEBplE)  
30. Requesting clarification for Shared Memory Bank Conflicts and Shared memory access? \- CUDA Programming and Performance \- NVIDIA Developer Forums, Zugriff am August 18, 2025, [https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-bank-conflicts-and-shared-memory-access/268574](https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-bank-conflicts-and-shared-memory-access/268574)  
31. Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog, Zugriff am August 18, 2025, [https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)  
32. How can I diminish bank conflicts in this code? \- Stack Overflow, Zugriff am August 18, 2025, [https://stackoverflow.com/questions/14142934/how-can-i-diminish-bank-conflicts-in-this-code](https://stackoverflow.com/questions/14142934/how-can-i-diminish-bank-conflicts-in-this-code)  
33. Improving GPU performance : reducing memory conflicts and latency \- TUE Research portal \- Eindhoven University of Technology, Zugriff am August 18, 2025, [https://research.tue.nl/files/8808771/20151125\_Braak.pdf](https://research.tue.nl/files/8808771/20151125_Braak.pdf)  
34. Performance Modeling of Atomic Additions on GPU Scratchpad Memory \- ResearchGate, Zugriff am August 18, 2025, [https://www.researchgate.net/publication/258939369\_Performance\_Modeling\_of\_Atomic\_Additions\_on\_GPU\_Scratchpad\_Memory](https://www.researchgate.net/publication/258939369_Performance_Modeling_of_Atomic_Additions_on_GPU_Scratchpad_Memory)  
35. When is padding for shared memory really required? \- Stack Overflow, Zugriff am August 18, 2025, [https://stackoverflow.com/questions/15056842/when-is-padding-for-shared-memory-really-required](https://stackoverflow.com/questions/15056842/when-is-padding-for-shared-memory-really-required)  
36. On the Correctness of the SIMT Execution Model of GPUs, Zugriff am August 18, 2025, [https://d-nb.info/1253014310/34](https://d-nb.info/1253014310/34)  
37. Control Flow Management in Modern GPUs \- arXiv, Zugriff am August 18, 2025, [https://arxiv.org/pdf/2407.02944](https://arxiv.org/pdf/2407.02944)  
38. DARM: Control-Flow Melding for SIMT Thread Divergence Reduction \- LLVM, Zugriff am August 18, 2025, [https://llvm.org/devmtg/2022-11/slides/StudentTechTalk1-MergingSimilarControl-FlowRegions.pptx](https://llvm.org/devmtg/2022-11/slides/StudentTechTalk1-MergingSimilarControl-FlowRegions.pptx)  
39. SPIR-V Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/SPIR-V/](https://mlir.llvm.org/docs/Dialects/SPIR-V/)  
40. mlir/docs/Dialects/SPIR-V.md · 89bb0cae46f85bdfb04075b24f75064864708e78 \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/89bb0cae46f85bdfb04075b24f75064864708e78/mlir/docs/Dialects/SPIR-V.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/89bb0cae46f85bdfb04075b24f75064864708e78/mlir/docs/Dialects/SPIR-V.md)  
41. Part 2 – CUDA Advances, Zugriff am August 18, 2025, [https://pages.mini.pw.edu.pl/\~kaczmarskik/gpca/resources/Part2-advanced-cuda-handout-a4.pdf](https://pages.mini.pw.edu.pl/~kaczmarskik/gpca/resources/Part2-advanced-cuda-handout-a4.pdf)  
42. I don't understand the point of atomic operations : r/vulkan \- Reddit, Zugriff am August 18, 2025, [https://www.reddit.com/r/vulkan/comments/1g8k9i8/i\_dont\_understand\_the\_point\_of\_atomic\_operations/](https://www.reddit.com/r/vulkan/comments/1g8k9i8/i_dont_understand_the_point_of_atomic_operations/)  
43. Yet another article discussing atomics without discussing the memory model or me... | Hacker News, Zugriff am August 18, 2025, [https://news.ycombinator.com/item?id=20564921](https://news.ycombinator.com/item?id=20564921)  
44. Atomic Operations in CUDA \- NVIDIA Developer Forums, Zugriff am August 18, 2025, [https://forums.developer.nvidia.com/t/atomic-operations-in-cuda/9202](https://forums.developer.nvidia.com/t/atomic-operations-in-cuda/9202)  
45. MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/](https://mlir.llvm.org/)  
46. What about the MLIR compiler infrastructure? (Democratizing AI Compute, Part 8\) \- Modular, Zugriff am August 18, 2025, [https://www.modular.com/blog/democratizing-ai-compute-part-8-what-about-the-mlir-compiler-infrastructure](https://www.modular.com/blog/democratizing-ai-compute-part-8-what-about-the-mlir-compiler-infrastructure)  
47. MLIR: Scaling Compiler Infrastructure for Domain Specific Computation \- Reliable Computer Systems \- University of Waterloo, Zugriff am August 18, 2025, [https://rcs.uwaterloo.ca/\~ali/cs842-s23/papers/mlir.pdf](https://rcs.uwaterloo.ca/~ali/cs842-s23/papers/mlir.pdf)  
48. MLIR (software) \- Wikipedia, Zugriff am August 18, 2025, [https://en.wikipedia.org/wiki/MLIR\_(software)](https://en.wikipedia.org/wiki/MLIR_\(software\))  
49. Interfaces \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Interfaces/](https://mlir.llvm.org/docs/Interfaces/)  
50. Side Effects & Speculation \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)  
51. mlir/docs/Interfaces.md · 508c4efe1e9d95661b322818ae4d6a05b1913504 \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/508c4efe1e9d95661b322818ae4d6a05b1913504/mlir/docs/Interfaces.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/508c4efe1e9d95661b322818ae4d6a05b1913504/mlir/docs/Interfaces.md)  
52. Chapter 4: Enabling Generic Transformation with Interfaces \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)  
53. Deep Dive on MLIR Internals OpInterface Implementation \- LLVM, Zugriff am August 18, 2025, [https://llvm.org/devmtg/2024-04/slides/TechnicalTalks/Amini-DeepDiveOnMLIRInternals.pdf](https://llvm.org/devmtg/2024-04/slides/TechnicalTalks/Amini-DeepDiveOnMLIRInternals.pdf)  
54. FAQ \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/getting\_started/Faq/](https://mlir.llvm.org/getting_started/Faq/)