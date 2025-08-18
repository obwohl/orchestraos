

# **Idiomatic Region Transplantation and Terminator Replacement in MLIR Rewrite Patterns**

## **Introduction**

This report provides an exhaustive analysis of the correct methodology for transplanting the body of an operation's region into a new operation within an mlir::OpRewritePattern. This complex structural transformation is a common task in compiler development, particularly when lowering from a structured dialect like scf to a custom, domain-specific dialect. The primary challenge, and the impetus for this analysis, is a frequent verifier error: 'custom.yield' op must be the last operation in the parent block. This error signals an incomplete transformation that violates one of MLIR's core structural invariants.

We will address this verifier error by detailing the precise, sequential API calls required to correctly manipulate terminators during region transplantation. The focus will be on the robust, idiomatic patterns that leverage the full power of the PatternRewriter API, moving beyond simple operation cloning to perform complex structural IR surgery. The solution lies not in manual iteration over operations, but in a principled understanding and application of the high-level, transactional APIs designed for this exact purpose. This report will serve as an expert-level guide, establishing the foundational concepts of MLIR's structural IR, the design philosophy of the PatternRewriter, and a canonical implementation for this transformation, thereby enabling the development of correct and maintainable compiler passes.

## **The Architectural Roles of Regions, Blocks, and Terminators**

The verifier error at the heart of the user's query is not a superficial bug but a violation of foundational principles in MLIR's design. A comprehensive understanding of the structural hierarchy of MLIR is paramount to resolving this issue and preventing similar errors in future transformations.

### **The MLIR Structural Hierarchy**

MLIR's multi-level nature is physically represented in its nested, tree-like Intermediate Representation (IR) structure.1 This hierarchy is defined by a strict ownership model:

* An mlir::Operation can own one or more mlir::Regions.  
* A mlir::Region is a list of mlir::Blocks.2  
* A mlir::Block is an ordered list of mlir::Operations.3

This strict nesting allows MLIR to represent diverse computational structures, from the flat control-flow graphs (CFGs) of traditional compilers to the structured, single-block bodies of scf.for or scf.if operations, and even graph-based regions for dataflow programming models.1 The semantics of the operations within a region are defined by the parent operation that owns the region. For example, the

scf.if operation defines that its regions represent the "then" and "else" branches of a conditional statement.5

### **The Inviolable Single Terminator Invariant**

In the most common type of region, the SSACFG (Static Single Assignment Control Flow Graph) region, every block must adhere to a critical structural rule: it must end with exactly one "terminator" operation.2 A terminator is a special class of operation, marked with the

Terminator trait, whose purpose is to transfer control flow. Examples include cf.br (unconditional branch), cf.cond\_br (conditional branch), and scf.yield. The scf.yield operation, for instance, terminates the block within an scf operation's region and "yields" its operands as the results of the parent operation.5

The user's verifier error is a direct violation of this invariant. The transformation process described involves creating a new terminator (orchestra.yield) but fails to remove the original one (scf.yield). For a brief moment, before the pattern rewrite is fully committed, the block contains two operations marked with the Terminator trait. The MLIR verifier, which is designed to uphold the integrity of the IR, correctly identifies this invalid state and reports the failure. This invariant is not arbitrary; it is the bedrock upon which all control-flow analyses, dominator tree computations, and subsequent transformations are built. A malformed CFG with multiple terminators per block would render these algorithms useless.

The error, therefore, should be viewed as a feature—a safeguard that forces developers to reason about transformations not as a series of disconnected mutations, but as a sequence that must result in a valid IR state before the pattern successfully completes. The logical sequence of events leading to the error is as follows:

1. The rewrite pattern creates a new terminator, orchestra.yield.  
2. At this point, the block contains both the new orchestra.yield and the original scf.yield.  
3. The pattern's matchAndRewrite function returns success(), signaling to the driver that a change was made.  
4. The pattern driver, upon successful application, may invoke the verifier to ensure the IR remains valid.  
5. The verifier inspects the modified block, finds two terminators, and halts compilation with an error.

The resolution must therefore involve a sequence of API calls *within the pattern* that guarantees the block has precisely one terminator at the conclusion of the rewrite.

### **The mlir::Block API: A Framework's View of Terminators**

The design of the mlir::Block class API further underscores the special status of terminators within the MLIR framework. The class provides two specific methods for interacting with the end of a block: getTerminator() and without\_terminator().3

* getTerminator(): This method provides direct, privileged access to the block's terminator operation. It is the designated way to inspect, modify, or retrieve information (like yielded operands) from the operation that defines the block's control flow exit.  
* without\_terminator(): This method returns an iterator range over all operations in the block *except* for the final terminator. It provides a view of the block's "body"—the sequence of computations that execute before control is transferred.

This functional separation is a deliberate design choice. It allows algorithms to easily distinguish between the computational body of a block and its control-flow instruction. The task of transplanting a region's body and replacing its terminator maps perfectly onto this API design: the transformation must move the operations covered by without\_terminator() and replace the single operation returned by getTerminator().

## **The PatternRewriter as a Transactional State Machine**

To correctly implement complex structural transformations, it is essential to reframe the mlir::PatternRewriter not as a simple collection of builder methods, but as a mandatory, state-aware interface to the pattern application engine. Every mutation must be mediated by this class to ensure the integrity of the transformation process.

### **The Mandate for Rewriter-Only Mutations**

The MLIR documentation is unequivocal: "All IR mutations, including creation, must be performed by the given PatternRewriter".7 This rule is not merely a convention; it is a strict requirement for correctness. The pattern driver, which orchestrates the application of many

RewritePattern instances, may be performing speculative rewrites, managing complex undo-stacks for failed patterns, or tracking fine-grained IR changes for worklist-driven algorithms.

If a pattern were to perform a direct mutation—for example, by calling op-\>erase()—it would bypass the driver's tracking mechanisms. This could corrupt the driver's internal state, leading to unpredictable behavior, use-after-free errors, or silent miscompilations. The PatternRewriter 10 acts as a transactional API. It records a sequence of intended changes (creations, erasures, operand updates, replacements), and the driver commits these changes atomically only upon the successful completion of the entire pattern application process, such as a pass running

applyPartialConversion.11

### **Core API for Operation Manipulation**

The PatternRewriter provides a comprehensive suite of methods for operation-level changes. The most fundamental include:

* create\<OpTy\>(...): Creates a new operation of type OpTy.  
* eraseOp(op): Erases an existing operation.  
* replaceOp(op, newValues): Replaces all uses of the results of op with newValues and then erases op.  
* replaceOpWithNewOp\<OpTy\>(op,...): A convenient combination of create and replaceOp.

These methods ensure that all mutations are properly recorded and managed by the pattern driver.

### **The High-Level API for Structural Surgery**

For the task of region transplantation, the most critical methods are those designed for manipulating entire structural units like regions and blocks. The PatternRewriter inherits from RewriterBase, which provides a powerful, high-level API for this "structural surgery".10 Attempting to replicate this functionality manually is a common source of bugs. The existence of these APIs signals the intended level of abstraction for such transformations.

The design of these methods encapsulates the complex bookkeeping required for structural changes. Manually moving operations from one region to another would require painstakingly remapping all SSA uses of values defined outside the original region, handling block arguments correctly, and managing use-lists—a process that is incredibly error-prone. The high-level APIs, particularly those that accept an IRMapping object, automate this process, guaranteeing correctness. Therefore, these APIs are not just for convenience; they are the robust and idiomatic tools for performing these structural operations.

The following table summarizes the key methods for structural manipulation.

| Method Signature | Description | Idiomatic Use Case |
| :---- | :---- | :---- |
| void inlineRegionBefore(Region \&src, Region \&dest,...) | Moves all blocks from the src region to the dest region. | Transplanting a region's body into a new parent operation. |
| void cloneRegionBefore(Region \&src, Region \&dest,...) | Clones all blocks from the src region to the dest region. | Duplicating structured logic, for example, during loop unrolling. |
| void mergeBlocks(Block \*src, Block \*dest,...) | Moves operations from src to the end of dest and erases src. | Splicing two sequential basic blocks into one, common in CFG simplification. |
| Block \*splitBlock(Block \*block, iterator splitPoint) | Splits a block into two at splitPoint, creating a new block. | Creating a clean insertion point or "landing pad" between two sets of operations. |
| void moveBlockBefore(Block \*block, Block \*dest) | Moves a single block before another within the same region. | Reordering blocks within a region's Control Flow Graph. |

## **The Canonical Method for Region Body Transplantation and Terminator Replacement**

Integrating the architectural principles with the PatternRewriter's transactional API leads to a canonical, four-step algorithm for transforming an scf.if into a custom orchestra.task operation. This sequence guarantees that IR invariants are maintained and correctly produces the desired transformation.

### **1\. Preparation: Create the Destination Operation**

The first action within the matchAndRewrite function is to create the new orchestra.task operation. This new operation will serve as the container for the body of the original scf.if.

C++

// Inside matchAndRewrite(scf::IfOp ifOp, PatternRewriter \&rewriter) const  
auto loc \= ifOp.getLoc();

// Create the new 'orchestra.task' op. Its result types should match  
// the original 'scf.if' op. It is created with one empty region.  
auto newTaskOp \= rewriter.create\<orchestra::TaskOp\>(loc, ifOp.getResultTypes());

Crucially, the orchestra::TaskOp must be defined to have at least one region. The create call will instantiate it with an empty region, ready to receive the blocks from the scf.if operation.

### **2\. Transplantation: Move the Region Body Atomically**

This step leverages the high-level API to perform the core structural move. Instead of manually iterating and cloning operations, a single call to inlineRegionBefore transfers the entire body of the scf.if's "then" region.

C++

// Move all blocks from the 'then' region of the scf.if into the  
// new task's region. This is an atomic move of the entire block list.  
rewriter.inlineRegionBefore(ifOp.getThenRegion(), newTaskOp.getRegion(),  
                            newTaskOp.getRegion().end());

This call is the idiomatic solution for region transplantation. It correctly handles the transfer of ownership of the Block and all its Operations from the old parent to the new one. At this point, the newTaskOp contains a block that is identical to the original, including its scf.yield terminator. The IR is structurally valid but semantically incorrect for the orchestra dialect, which sets the stage for the next step.

### **3\. Terminator Surgery: The Critical Sequence**

This multi-part step directly addresses the verifier error by replacing the old terminator with the new one in a way that never violates the single-terminator invariant within the rewriter's transaction.

#### **Step 3a: Locate the Original Terminator**

After inlining, the block(s) inside newTaskOp now contain the original scf.yield. A handle to this operation is required to access its operands and to ultimately erase it.

C++

// The inlined region has a single block. Get a reference to it.  
Block \&block \= newTaskOp.getRegion().front();  
auto yieldOp \= cast\<scf::YieldOp\>(block.getTerminator());

#### **Step 3b: Capture Yielded Operands**

The operands of the scf.yield represent the data flow out of the original scf.if region. These values must be captured so they can be forwarded to the new orchestra.yield terminator.

C++

// Capture the values that were yielded by the original terminator.  
ValueRange yieldedOperands \= yieldOp.getOperands();

#### **Step 3c: Create the New Terminator**

With the necessary information captured, the new orchestra.yield can be created. The insertion point must be carefully managed. Setting it to be at the location of the old terminator ensures the new operation is inserted immediately before it.

C++

// Set the insertion point to be right before the old yield op.  
rewriter.setInsertionPoint(yieldOp);

// Create the new orchestra.yield, passing along the captured operands.  
rewriter.create\<orchestra::YieldOp\>(yieldOp.getLoc(), yieldedOperands);

#### **Step 3d: Erase the Original Terminator**

This is the final, critical part of the surgery and the step that resolves the verifier error. With the new terminator now in place, the old one is redundant and must be explicitly destroyed.

C++

// Erase the original scf.yield. The block now has a single, correct  
// terminator.  
rewriter.eraseOp(yieldOp);

After this call, the block within newTaskOp is terminated by orchestra.yield, satisfying both the structural single-terminator invariant and the semantic requirements of the orchestra dialect.

### **4\. Finalization: Replace the Original Operation**

The final step is to integrate the newly created and populated orchestra.task into the surrounding dataflow graph. The replaceOp method accomplishes this by redirecting all uses of the scf.if's results to the corresponding results of the newTaskOp.

C++

// Replace all uses of the original scf.if with the results of the new  
// orchestra.task. This also erases the now-empty scf.if op.  
rewriter.replaceOp(ifOp, newTaskOp.getResults());

return success();

This call completes the rewrite. It wires the new operation into the wider CFG and, as part of its contract, erases the original scf.if operation, which is now empty and fully replaced. The entire transformation is now complete and correct.

## **A Reference Implementation and In-Depth Analysis**

The canonical method described in the previous section can be encapsulated in a complete OpRewritePattern. Furthermore, examining existing patterns within the core MLIR codebase provides validation and deeper understanding of these structural transformation techniques.

### **Complete Reference Implementation**

The following C++ code provides a complete, commented implementation of the OpRewritePattern for converting scf.if to orchestra.task. This serves as a direct, actionable solution.

C++

\#**include** "mlir/IR/PatternMatch.h"  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
// Assume "Orchestra/OrchestraOps.h" defines orchestra::TaskOp and orchestra::YieldOp

namespace {

class ScfIfToOrchestraTask : public mlir::OpRewritePattern\<mlir::scf::IfOp\> {  
public:  
  using OpRewritePattern\<mlir::scf::IfOp\>::OpRewritePattern;

  mlir::LogicalResult  
  matchAndRewrite(mlir::scf::IfOp ifOp,  
                  mlir::PatternRewriter \&rewriter) const override {  
    // This pattern only handles 'scf.if' with a 'then' block and no 'else' block  
    // for simplicity. A robust implementation would handle the 'else' case.  
    if (\!ifOp.getElseRegion().empty()) {  
      return mlir::failure();  
    }

    auto loc \= ifOp.getLoc();

    // Step 1: Create the new 'orchestra.task' op.  
    // It is created with one empty region by default.  
    auto newTaskOp \= rewriter.create\<orchestra::TaskOp\>(loc, ifOp.getResultTypes());

    // Step 2: Transplant the region body atomically.  
    // Move all blocks from the 'then' region of the scf.if into the new task's region.  
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newTaskOp.getRegion(),  
                                newTaskOp.getRegion().end());

    // Step 3: Perform terminator surgery.  
    // The inlined region has a single block. Get a reference to it and its terminator.  
    mlir::Block \&block \= newTaskOp.getRegion().front();  
    auto yieldOp \= llvm::cast\<mlir::scf::YieldOp\>(block.getTerminator());

    // Capture the values that were yielded by the original terminator.  
    mlir::ValueRange yieldedOperands \= yieldOp.getOperands();

    // Set the insertion point to be right before the old yield op.  
    rewriter.setInsertionPoint(yieldOp);

    // Create the new orchestra.yield, passing along the captured operands.  
    rewriter.create\<orchestra::YieldOp\>(yieldOp.getLoc(), yieldedOperands);

    // Erase the original scf.yield. This is the crucial step to avoid the verifier error.  
    rewriter.eraseOp(yieldOp);

    // Step 4: Finalize by replacing the original operation.  
    // This replaces all uses of the scf.if's results and erases the ifOp.  
    rewriter.replaceOp(ifOp, newTaskOp.getResults());

    return mlir::success();  
  }  
};

} // namespace

### **Canonical Example: DetensorizeGenericOp from Linalg Transforms**

A powerful, real-world example of these techniques can be found in the MLIR codebase in the Linalg-to-Loops transformation pipeline. The DetensorizeGenericOp pattern is designed to "unwrap" a linalg.generic operation, inlining its region directly into the parent block.12 Its implementation of

matchAndRewrite demonstrates a sophisticated sequence of PatternRewriter calls.

1. Block \*newBlock \= rewriter.splitBlock(originalBlock, Block::iterator(op));  
   The first step is to split the block containing the linalg.generic op right before the op itself. This is a clever technique that creates a clean "landing pad." The linalg.generic op is now at the beginning of originalBlock, and newBlock contains all subsequent operations. This isolates the transformation and provides a clear insertion point.  
2. rewriter.inlineRegionBefore(op.getRegion(), newBlock);  
   Next, it inlines the linalg.generic op's region not into the original block, but into the parent region right at the start of the newBlock. This effectively places the body of the linalg.generic op where the op itself used to be.  
3. rewriter.replaceOp(op, yieldOp-\>getOperands());  
   Finally, it replaces the results of the original linalg.generic op with the operands of the linalg.yield op from the now-inlined region. This correctly wires the dataflow from the inlined operations to the operations that were originally users of the linalg.generic op.

This "split-then-inline" pattern is a powerful idiom for unwrapping region-based operations. While the user's case of creating a *new* parent operation does not require splitBlock, studying this example provides a deeper appreciation for how these high-level rewriter calls can be composed to perform complex and robust structural transformations.

### **Advanced Considerations**

A production-quality implementation would need to address more complex scenarios:

* **Handling scf.if with an else block:** The transformation must decide how to handle the else region. A common approach is to lower the structured scf.if to unstructured control flow. This would involve creating a continuation block and using cf.cond\_br to branch to the inlined "then" block or the inlined "else" block, both of which would then branch to the continuation block.13  
* **Multi-block Regions:** If the source region contains multiple blocks with internal control flow, inlineRegionBefore will correctly move all of them. The rewrite pattern is then responsible for ensuring the resulting CFG within the new parent operation is valid and that any terminators branching to blocks outside the original region are correctly handled.

## **Common Pitfalls and Anti-Patterns**

To fully appreciate the robustness of the canonical method, it is instructive to examine common but incorrect approaches. These anti-patterns often seem intuitive but fail to account for the complexities of SSA-based IR, leading to subtle bugs or outright invalid code.

### **The Manual Iteration Anti-Pattern**

A frequent mistake made by those new to MLIR's rewriting framework is to manually iterate through the operations of the source region's block and clone them one by one into the new region. This typically looks something like this:

C++

// ANTI-PATTERN: DO NOT USE  
for (mlir::Operation \&opToClone : ifOp.getThenRegion().front().without\_terminator()) {  
    rewriter.clone(opToClone); // Fails to handle SSA values correctly  
}

This approach is fundamentally flawed for several reasons:

1. **SSA Value Mismanagement:** This is the most critical failure. If an operation inside the loop uses an SSA value defined elsewhere within the scf.if's region (such as a block argument or the result of a preceding operation), the rewriter.clone() call will create a new operation with a dangling reference to the *old* value in the *old* region. The inlineRegionBefore and cloneRegionBefore methods use an internal IRMapping to automatically remap all such uses to the newly moved or cloned values, thus preserving the correctness of the dataflow graph.  
2. **Fragility and Verbosity:** The code is verbose and brittle. It requires explicit loops, conditional checks to skip the terminator, and manual creation of the new terminator. Any change to the structure of the source operation, such as the introduction of multiple blocks, would require significant changes to this fragile manual logic.  
3. **Inefficiency:** This per-operation approach is likely less efficient than the highly optimized internal implementation of inlineRegionBefore, which is designed to move entire blocks as a cohesive unit.

The following table provides a direct comparison of the idiomatic approach versus this anti-pattern.

| Feature | Idiomatic Approach (inlineRegionBefore) | Anti-Pattern (Manual Iteration & Cloning) |
| :---- | :---- | :---- |
| **SSA Value Handling** | **Automatic & Correct.** Uses IRMapping to remap all uses of values defined within the region. | **Manual & Error-Prone.** Fails to update uses of block arguments or values defined within the region, leading to broken IR. |
| **Block Arguments** | Handled correctly by the API. | Ignored. Cloned operations will have dangling references to the old block arguments. |
| **Code Verbosity** | **Concise.** A single, declarative function call. | **Verbose.** Requires explicit loops, conditional checks for terminators, and manual builder calls. |
| **Robustness** | **High.** Resilient to changes in the source region's structure. | **Low.** Brittle; any new type of operation in the source region may require changes to the cloning logic. |
| **Terminator Handling** | Moves the terminator along with the block, requiring explicit replacement. | Requires special logic to skip the original terminator and create a new one. |

### **Incorrect API Sequencing**

Even when using the correct APIs, the order of operations matters.

* **Erasing Before Creating:** Calling rewriter.eraseOp(yieldOp) *before* creating the new orchestra.yield. This is incorrect because it destroys the source of information (the yieldOp's operands) needed to create the new terminator. It also temporarily leaves the block without any terminator, which could fail verification in certain contexts.  
* **Forgetting to replaceOp:** Performing the entire region transplantation and terminator surgery correctly but forgetting the final rewriter.replaceOp(ifOp,...) call. This results in "dead" code. The new orchestra.task is created, but the old scf.if remains in the IR, and the new operation's results are unused. The transformation is effectively a no-op from a dataflow perspective.

## **Conclusion**

The task of transplanting a region's body and replacing its terminator is a microcosm of the challenges and design principles inherent in modern compiler infrastructure. The verifier error that motivates this analysis is not an obstacle but a guide, enforcing the structural integrity that makes MLIR a powerful and reliable framework. The resolution demonstrates that robust IR transformations are not achieved through low-level, manual manipulation of operations, but through a principled application of high-level, transactional APIs.

The key principles for successful region manipulation in MLIR can be summarized as follows:

1. **Respect Structural Invariants:** The single-terminator-per-block rule is inviolable in CFG regions. All transformations must conclude with the IR in a valid state that respects this and other invariants.  
2. **Use the PatternRewriter Exclusively:** All IR mutations must be performed through the PatternRewriter. This ensures that the transformation is correctly tracked, managed, and committed by the pattern application driver.  
3. **Leverage High-Level APIs for Structural Surgery:** For complex tasks like moving or cloning regions, methods such as inlineRegionBefore are not merely convenient but are the canonical tools for ensuring correctness. They abstract away the perilous complexity of SSA value remapping and use-list management.

The canonical four-step method—**Prepare, Transplant, Surgically Replace, and Finalize**—provides a robust and repeatable template for this class of transformation. By creating the new container, using inlineRegionBefore for the atomic move, performing a careful "get-create-erase" sequence on the terminator, and finally replacing the original operation, developers can build correct, idiomatic, and maintainable compiler passes that harness the full expressive power of MLIR.
