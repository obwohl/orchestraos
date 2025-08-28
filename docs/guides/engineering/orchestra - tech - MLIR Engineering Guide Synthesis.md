

# **An Authoritative Engineering Guide to Advanced Structural Transformations in MLIR**

## **Introduction**

### **Purpose and Audience**

This document serves as a definitive engineering guide for implementing complex, structural Intermediate Representation (IR) transformations within the Multi-Level Intermediate Representation (MLIR) framework. It is intended for compiler engineers and researchers who are actively developing within the MLIR ecosystem. The target audience is presumed to be familiar with the fundamental concepts of MLIR—such as dialects, operations, and attributes—but requires a deep, practical reference for mastering advanced rewriting techniques, including region transplantation, terminator replacement, and robust side-effect modeling. The content herein is meticulously updated to reflect the APIs and best practices of MLIR as of the LLVM 20 release cycle.

### **Core Philosophy**

The central philosophy of this guide is that robust and correct compiler transformations are not achieved through ad-hoc, low-level manipulation of the IR. Instead, they emerge from a disciplined adherence to MLIR's foundational structural invariants and a thorough understanding of the transactional nature of its pattern rewriting framework.1 The MLIR verifier, rather than being an obstacle, is a critical tool that guides developers toward safe and canonical implementation patterns. By internalizing the principles of invariant maintenance, transactional mutation, and idiomatic API usage, engineers can build sophisticated, correct, and maintainable compiler passes that harness the full power of MLIR's multi-level rewriting capabilities.

### **Guide Structure**

This guide is organized into four distinct parts, creating a pedagogical flow from foundational theory to practical implementation and advanced techniques.

* **Part I: Foundational Principles of MLIR's Structure and Semantics** establishes the architectural rules and invariants that govern all valid IR transformations.  
* **Part II: The Pattern Rewriting Engine: The Sole Arbiter of IR Mutation** details the design and contract of the PatternRewriter, the mandatory interface for all IR modifications within a pattern.  
* **Part III: A Practical Guide to Region Transplantation and Repair** provides detailed, step-by-step C++ implementation tutorials for the core tasks of moving region-based logic and repairing the surrounding IR.  
* **Part IV: Advanced Transformation Techniques and Best Practices** elevates the discussion to cover modern declarative rewriting with PDL/PDLL, sophisticated semantic modeling with interfaces, and effective interaction with the broader MLIR transformation ecosystem.

---

## **Part I: Foundational Principles of MLIR's Structure and Semantics**

Before attempting to modify the IR, a developer must possess a deep and intuitive understanding of its structure and the semantic contracts the MLIR verifier rigorously enforces. This section lays that essential groundwork.

### **Chapter 1: The Anatomy of MLIR IR**

#### **The Hierarchical Triad: Operations, Regions, and Blocks**

The MLIR IR is built upon a deeply nested, hierarchical data structure defined by a strict ownership model.1 This structure is composed of three primary components:

1. **Operation:** The fundamental unit of execution and the primary node in MLIR's graph-like data structure. Each operation has a unique name (e.g., func.func, arith.addi), a list of input operands (Values), a dictionary of compile-time constant attributes, and a list of results (Values) it produces.1  
2. **Region:** A container for a list of Blocks. An operation can own one or more regions, which represent a nested scope, often with distinct control-flow semantics. The body of a function (func.func), the body of a loop (scf.for), and the "then" and "else" clauses of a conditional (scf.if) are all modeled as regions.1  
3. **Block:** An ordered list of operations contained within a region. A block has a list of block arguments (Values defined upon entry) and a sequence of operations that execute in order. The final operation within a block must be a special kind of operation known as a terminator.1

This strict ownership model—Operations own Regions, Regions own Blocks, and Blocks own Operations—defines a non-negotiable tree-like structure for the IR. This physical representation is what enables MLIR's multi-level nature, allowing it to represent diverse computational structures, from flat control-flow graphs to the nested logic of structured control flow.1 The parent operation that owns a region defines the semantics of that region; for example, the

scf.if operation specifies that its two regions represent the "then" and "else" branches of a conditional statement.1

#### **Value Scoping and Implicit Capture**

MLIR's hierarchical structure gives rise to a sophisticated value scoping model based on Static Single Assignment (SSA) dominance.1 A

Value can be the result of an operation or an argument to a block. The fundamental rule of SSA is that any use of a Value must be dominated by its definition.

Crucially, this dominance relationship extends across region boundaries. An operation within a nested region is permitted to use a Value that is defined in a parent region or any ancestor region. This is a common and valid construct referred to as an "implicit capture".1 The value is effectively "captured" by the inner region from its lexical scope, much like a closure in a high-level programming language.

While implicit capture makes MLIR code concise and intuitive, it is also the direct source of the primary challenge in region transplantation. When a region is moved from its original parent operation to a new one, the lexical scope is broken. The original dominance relationships that made the implicit captures valid are severed. This transforms what might seem like a simple "move" operation into a complex semantic repair task. The transformation must fundamentally shift the IR's dependency model from one of implicit lexical scoping to one of explicit dataflow. This requires identifying all external values used by the region, adding them as explicit operands to the new parent operation, and remapping all internal uses to corresponding new block arguments. This reification of data dependencies is the central conceptual leap a developer must make to perform these transformations correctly.

### **Chapter 2: The Invariants of SSACFG Regions**

#### **The TerminatorOp Invariant**

MLIR supports two primary kinds of regions: Graph regions, which model dataflow graphs without explicit control flow, and SSACFG (Static Single Assignment Control Flow Graph) regions.1 The vast majority of compiler transformations operate on SSACFG regions, which impose a strict and inviolable contract on their constituent blocks.

The most critical rule is that **every block in an SSACFG region must end with exactly one operation that has the TerminatorOp trait**.1 A terminator operation is responsible for explicitly transferring control flow. This can be a transfer to another block within the same region (e.g.,

cf.br, cf.cond\_br), or it can be a transfer of control and data back to the parent operation that encloses the region (e.g., func.return, scf.yield).1

This invariant is not arbitrary; it is the bedrock upon which all robust control-flow analysis is built. Its existence allows for the construction of a well-defined Control Flow Graph (CFG), which is a prerequisite for essential compiler analyses like dominance, liveness, and dataflow analysis.1

#### **The Role of the Verifier**

The MLIR verifier acts as the guardian of these structural invariants. Any transformation that, even momentarily, leaves a block without a terminator will produce an invalid IR state that the verifier will detect and report as an error.1

This strictness should not be viewed as an obstacle but as a design guide. A common verifier error, such as 'custom.yield' op must be the last operation in the parent block, is not a bug to be worked around; it is a feature that enforces correct transformation sequencing.1 It signals that the single-terminator invariant has been violated, likely because a new terminator was added without removing the old one. This forces the developer to think transactionally: the IR must be in a valid state

*at the conclusion* of a pattern's application. The verifier's rigor thus guides the developer toward robust, safe implementation patterns, such as the "insert-then-erase" sequence for terminator replacement, which is detailed in Part III of this guide.

---

## **Part II: The Pattern Rewriting Engine: The Sole Arbiter of IR Mutation**

This section positions the PatternRewriter not as a mere collection of builder utilities, but as the mandatory, state-aware interface to the pattern application engine. All modifications to the IR within a pattern must be mediated by this class to ensure the integrity and correctness of the transformation process.

### **Chapter 3: The RewritePattern Contract**

When implementing a transformation using a C++ class derived from mlir::RewritePattern (or its typed variant mlir::OpRewritePattern), developers must abide by a strict contract to ensure correct interaction with the pattern application drivers.1

#### **The Mandate for Rewriter-Only Mutations**

The most critical rule is that **all IR mutations must be performed via the PatternRewriter instance** provided to the matchAndRewrite method.1 Direct calls to methods on IR objects, such as

op-\>erase(), are strictly forbidden. This is because the pattern drivers, such as the greedy rewrite driver, rely on notifications from the rewriter to manage their worklists, track changes, and potentially perform speculative rewrites.1 A direct mutation is invisible to the driver, which can corrupt its internal state and lead to incomplete transformations, use-after-free errors, or silent miscompilations.1

This reveals that a pattern application is not a direct manipulation of a global IR state but is better understood as a transaction. The PatternRewriter records a sequence of intended changes (creations, erasures, replacements), and the driver commits these changes atomically only upon the successful completion of the pattern. To help enforce this, MLIR provides an "expensive checks" build mode (-DMLIR\_ENABLE\_EXPENSIVE\_PATTERN\_API\_CHECKS=ON) which uses operation fingerprinting to catch violations of this rule.1

#### **The matchAndRewrite Success/Failure Contract**

The matchAndRewrite method must adhere to a strict semantic contract regarding its return value: it must return mlir::success() if and only if the IR was modified.1 Violating this contract has severe consequences:

* **Returning success() without changing the IR** can cause the greedy pattern driver to enter an infinite loop, as it will repeatedly apply the same pattern to the same operation without making progress.1  
* **Returning failure() after modifying the IR** can cause the driver to terminate prematurely, missing further optimization opportunities because it believes no change was made.1

#### **The "Analyze-Then-Mutate" Paradigm**

No mutations should occur within the matchAndRewrite method until the match is fully confirmed and all necessary analysis is complete.1 This "analyze-then-mutate" pattern ensures that the IR is not left in a partially transformed, potentially invalid state if the pattern ultimately decides not to apply the rewrite. All checks and data gathering should precede the first call to a mutating method on the

PatternRewriter.

### **Chapter 4: A Tour of Essential Rewriter APIs for Structural Surgery**

The PatternRewriter provides a rich set of APIs for performing both simple and complex structural changes to the IR. Mastering these APIs is essential for implementing correct and idiomatic transformations.

#### **Core Mutation APIs (Updated for MLIR v20)**

The following APIs are indispensable for most rewrite patterns:

* **Creation:** The standard method for creating a new operation. In modern MLIR, the builder-based create method has been deprecated in favor of a static create method on the operation class itself.5  
  * *Deprecated:* rewriter.create\<OpTy\>(loc,...);  
  * *Modern (v20+):* rewriter.create\<OpTy\>(loc,...); is still widely used in existing code, but the most modern approach is OpTy::create(rewriter, loc,...);. This guide will use the rewriter.create form for consistency with the source material's style, while noting the modern alternative. The rewriter.create form is still a valid and functional part of the API.  
* **Replacement:** The canonical way to finalize a pattern is with rewriter.replaceOp(Operation \*op, ValueRange newValues). This single call correctly patches all SSA use-def chains by replacing all uses of op's results with newValues, and then erases op.1 The variant  
  rewriter.replaceOpWithNewOp\<OpTy\>(...) combines creation and replacement into one call.  
* **Deletion:** rewriter.eraseOp(Operation \*op) is used to erase an operation that is known to have no uses. This is critical for cleaning up old terminators after they have been replaced.1

#### **High-Level Structural APIs: inlineRegionBefore vs. moveBlockBefore**

For complex structural changes like region transplantation, choosing the correct high-level API is paramount. The PatternRewriter offers two seemingly similar methods for moving blocks, but their semantics are profoundly different.1

* inlineRegionBefore(Region \&region, Region \&parent,...): This is the high-level, "smart" API designed specifically for region transplantation.1 It moves  
  *all* blocks from the source region into the parent region. More importantly, it is designed to facilitate the remapping of SSA values. When provided with new values corresponding to the source region's entry block arguments, it automatically replaces all uses of those old arguments with the new values, thus repairing a critical part of the SSA graph.1  
* moveBlockBefore(Block \*block,...): This is a low-level primitive that unlinks a single block and moves it. It performs no SSA value remapping whatsoever.1 Using this for region transplantation would require the developer to manually find and replace all uses of captured values and old block arguments—a complex and highly error-prone task. Attempting to replicate the functionality of  
  inlineRegionBefore with this primitive is a common and severe anti-pattern.

The choice between these two APIs is a critical decision point. The following table provides a direct comparison to guide this choice.

| Feature | inlineRegionBefore | moveBlockBefore |
| :---- | :---- | :---- |
| **Action** | Splices all blocks from a source region into a destination region. | Moves a single block. |
| **SSA Value Handling** | **Automatic & Correct.** Designed for moving logic into a new SSA context. Remaps entry block arguments to new SSA values. | **Manual & Error-Prone.** Requires manual replacement of all captured values and block arguments. |
| **Idiomatic Use Case** | Region transplantation, lowering region-bearing operations. | Low-level CFG reordering within the same region. |

---

## **Part III: A Practical Guide to Region Transplantation and Repair**

This part provides a detailed, practical walkthrough of the C++ implementation for transplanting a region and performing the necessary post-transformation repairs. The code and procedures are modernized for MLIR v20 and are heavily annotated to explain the role of each API call.

### **Chapter 5: Pre-Transformation Analysis: Identifying External Dependencies**

A robust rewrite pattern must separate its analysis phase from its mutation phase. Before any modifications are made, the pattern must gather all information necessary to ensure the transformation can be performed validly. For region transplantation, this means identifying the complete set of external values the region depends on—its implicit captures.

#### **The mlir::getUsedValuesDefinedAbove Utility**

MLIR provides an essential utility function for this analysis: mlir::getUsedValuesDefinedAbove, located in the RegionUtils.h header.1 Its most common signature is:

C++

void mlir::getUsedValuesDefinedAbove(Region \&region, SetVector\<Value\> \&values);

This function walks all operations within the specified region (and any of its nested descendant regions) and, for each operand, checks if the corresponding Value is defined outside the scope of that same region. If so, the Value is added to the output values container.1

The use of llvm::SetVector as the output container is a deliberate and critical design choice. It guarantees two properties:

1. **Uniqueness:** Each captured value is stored only once.  
2. **Deterministic Order:** The order in which values are inserted is preserved.

This stability is crucial, as the order of the captured values will directly dictate the order of operands in the new destination operation and the corresponding order of arguments in its region's entry block.1

#### **Building the Blueprint for the New Operation**

The SetVector populated by this function serves as a complete blueprint for the new destination operation. The array of captured values becomes the exact list of operands for the new operation, and their types determine the operand types in its signature.1 By performing this read-only analysis upfront, the pattern has all the information required for the rewrite before a single mutating call is made, perfectly aligning with the "analyze-then-mutate" paradigm.

### **Chapter 6: The Mechanics of Transplantation: A Step-by-Step C++ Implementation**

This chapter presents the complete, annotated C++ tutorial for transplanting a region, synthesized from the procedural knowledge in the source documents.1

#### **Step 1: Matching and Initial Validation**

The process begins by defining an OpRewritePattern for the source operation type and performing initial validation within the matchAndRewrite method.

C++

\#**include** "mlir/IR/PatternMatch.h"  
\#**include** "mlir/Transforms/RegionUtils.h"

// Assume my\_dialect::SourceOp and my\_dialect::DestOp are defined.  
struct TransplantPattern : public mlir::OpRewritePattern\<my\_dialect::SourceOp\> {  
  using OpRewritePattern\<my\_dialect::SourceOp\>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(  
      my\_dialect::SourceOp sourceOp,  
      mlir::PatternRewriter \&rewriter) const override {  
    // Ensure the source operation has one region with at least one block.  
    if (\!sourceOp-\>hasOneRegion() |

| sourceOp-\>getRegion(0).empty()) {  
      return mlir::failure();  
    }  
    //... proceed to Step 2

#### **Step 2: Identifying Captured Values**

As detailed previously, the first action is to perform the read-only analysis to find all external values used by the region.

C++

    //... continued from Step 1  
    mlir::Region \&sourceRegion \= sourceOp.getRegion();

    // Identify all values defined outside the region but used inside.  
    llvm::SetVector\<mlir::Value\> capturedValues;  
    mlir::getUsedValuesDefinedAbove(sourceRegion, capturedValues);

    //... proceed to Step 3

#### **Step 3: Creating the Destination Operation**

Using the gathered information, the new destination operation is created. The captured values become its operands. The insertion point is set to be immediately before the source operation.

C++

    //... continued from Step 2  
    // Set the insertion point for the new operation.  
    rewriter.setInsertionPoint(sourceOp);

    // The new operation will have the same result types as the old one.  
    mlir::TypeRange resultTypes \= sourceOp-\>getResultTypes();

    // Create the new destination operation, passing the captured values as operands.  
    auto destOp \= rewriter.create\<my\_dialect::DestOp\>(  
        sourceOp.getLoc(), resultTypes, capturedValues.getArrayRef());

    //... proceed to Step 4

At this point, a new my\_dialect::DestOp exists in the IR. The OpBuilder infrastructure has automatically created an empty region for it, along with an entry block whose arguments correspond one-to-one with the capturedValues passed to the create call.1

#### **Step 4: Moving and Remapping the Region Body**

This is the most critical step, where the logic from the source region is moved into the destination operation's new region. The inlineRegionBefore method performs the heavy lifting.

C++

    //... continued from Step 3  
    // Move the blocks from the source region to the destination region's end.  
    // This call also handles the remapping of captured values to the new  
    // block arguments of destOp's entry block.  
    rewriter.inlineRegionBefore(sourceRegion, destOp.getRegion(),  
                                destOp.getRegion().end());

    //... proceed to Step 5

This single API call accomplishes two distinct tasks: it moves all blocks from sourceRegion into destOp.getRegion(), and it implicitly performs a replaceAllUsesWith for each captured value. Every use of a value from capturedValues within the moved blocks is replaced with the corresponding new block argument from destOp's entry block.1 The deterministic ordering provided by

SetVector ensures this mapping is correct.

#### **Step 5: Finalizing the Rewrite**

With the region's logic transplanted and its SSA dependencies repaired, the final step is to replace the original source operation with the new destination operation.

C++

    //... continued from Step 4  
    // Replace all uses of the source op's results with the dest op's results.  
    // This also erases the source op.  
    rewriter.replaceOp(sourceOp, destOp.getResults());

    return mlir::success();  
  }  
};

The rewriter.replaceOp call finalizes the transformation by redirecting any downstream uses of sourceOp's results to the corresponding results from destOp, and then safely erases sourceOp from the IR.1

### **Chapter 7: Post-Transplantation Surgery: Safe Terminator Replacement**

The process detailed above successfully moves the computational logic but leaves one critical task unfinished: the terminators within the moved blocks are still the original ones, which are now likely invalid in the context of the new parent operation.1

#### **The Problem: Stale Terminators**

Every region-bearing operation defines a contract for how control and data should be returned from its region. An scf.for expects an scf.yield with no operands, while an scf.if expects an scf.yield whose operands match the scf.if's result types.1 When blocks are moved to a new

DestOp, they carry their old terminators with them. The DestOp will have its own rules for what terminator is required. Failure to replace the old, now-invalid terminators will result in a verifier failure.1

#### **The "Insert-Then-Erase" Safety Pattern**

The replacement must be done carefully to avoid violating the "every block must have a terminator" invariant. The correct procedure follows an "insert-then-erase" pattern: create the new terminator *before* erasing the old one.1 While this creates a transient state where the block has two terminators, this is a state the rewriting infrastructure is designed to handle within the atomic scope of a pattern application. The key is that the final state, when the pattern's transaction commits, must be valid.

The following C++ procedure should be applied after inlineRegionBefore but before the final replaceOp call.

C++

// This code snippet fits inside the matchAndRewrite function from Chapter 6,  
// after Step 4 (inlineRegionBefore) and before Step 5 (replaceOp).

// Iterate over each of the newly moved blocks in the destination region.  
for (mlir::Block \&block : destOp.getRegion()) {  
  // 1\. Get a handle to the old terminator.  
  mlir::Operation \*oldTerminator \= block.getTerminator();  
  assert(oldTerminator && "Block is expected to have a terminator");

  // 2\. Capture any operands from the old terminator that need to be yielded.  
  // This is necessary if the new terminator needs to forward these values.  
  mlir::ValueRange yieldedValues \= oldTerminator-\>getOperands();

  // 3\. Set the insertion point AT the old terminator.  
  // This ensures the new terminator is created right before the old one.  
  rewriter.setInsertionPoint(oldTerminator);

  // 4\. Create the new, correct terminator for the destination op.  
  // The specific type (e.g., my\_dialect::ReturnOp) depends on the  
  // requirements of DestOp.  
  rewriter.create\<my\_dialect::ReturnOp\>(oldTerminator-\>getLoc(), yieldedValues);

  // 5\. Now that a new terminator exists, erase the old one.  
  rewriter.eraseOp(oldTerminator);  
}

// Now, proceed with rewriter.replaceOp(sourceOp, destOp.getResults());

This sequence is a critical defensive programming pattern for manipulating IR structures with strict invariants. By creating the new terminator first, the block is never left in a state where block.getTerminator() would be null.1

---

## **Part IV: Advanced Transformation Techniques and Best Practices**

This final part elevates the guide to an expert-level reference by introducing modern declarative alternatives, addressing complex semantic modeling, and discussing the broader MLIR transformation ecosystem.

### **Chapter 8: Declarative Rewrites with PDL and PDLL**

While C++ patterns offer maximum power and flexibility, they can be verbose for purely structural matches. MLIR provides a modern, declarative alternative designed to overcome the limitations of previous approaches.

#### **Motivation: The Limits of C++ and TableGen-DRR**

The original declarative approach in MLIR, TableGen-based Declarative Rewrite Rules (DRR), was powerful but had significant limitations. TableGen's DAG-based structure struggled to intuitively represent core MLIR concepts like multi-result operations, variadic operands, and regions.8 This led to non-obvious syntax and made expressing complex structural patterns difficult.

#### **Introducing PDLL: A Language for Patterns**

To address these shortcomings, MLIR introduced the Pattern Descriptor Language (PDL) and its user-friendly frontend, PDLL.10 PDL is itself an MLIR dialect for representing rewrite patterns, and PDLL is a language that compiles down to PDL IR.10 This allows patterns to be expressed in a concise, declarative syntax that closely mirrors the structure of the MLIR they are matching.

For example, a simple pattern to replace a constant 10 with 11 can be expressed in PDLL as follows 10:

C++

\#**include** "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern ReplaceTenWithEleven {  
  let ten: attr\<"10 : i32"\>;  
  let constant \= op\<arith.constant\> {value \= ten};

  rewrite constant with {  
    let newConst \= op\<arith.constant\> {value \= attr\<"11 : i32"\>};  
    replace constant with newConst;  
  };  
}

The mlir-pdll tool parses this file and generates PDL dialect IR, which is then lowered into the C++ code that implements the pattern.10

#### **Best Practices and Trade-offs**

The existence of both C++ patterns and PDLL implies a trade-off. An expert developer does not use one exclusively but chooses the tool that best fits the complexity and nature of the pattern.

* **C++ OpRewritePattern** offers the full power of an imperative language. It is ideal for patterns that require complex algorithmic logic, sophisticated type computations, interaction with external data structures or libraries, or non-local analysis of the IR.  
* **PDLL** provides a concise, declarative syntax that excels at expressing complex structural graph matches. It is the superior tool for patterns that are primarily concerned with the shape and structure of the IR, especially those involving variadic operands, multiple results, or nested regions, which can be verbose and error-prone to express in C++.

The following table summarizes the trade-offs to guide the developer's decision.

| Aspect | C++ OpRewritePattern | Declarative PDLL |
| :---- | :---- | :---- |
| **Expressiveness** | Fully imperative; can perform arbitrary C++ computations and analysis. | Declarative; focused on graph matching and structural rewriting. |
| **Verbosity** | High for complex structural matches. | Low; concise syntax for structural patterns. |
| **Ideal Use Case** | Patterns with complex algorithmic logic, type computations, or non-local analysis. | Complex structural graph matches, especially with variadics, regions, or multi-result ops. |
| **Debugging** | Standard C++ debugger. | Debug via inspecting the generated PDL IR. |

### **Chapter 9: Modeling Side Effects to Prevent Erroneous Optimizations**

A common and subtle class of bugs in compilers arises from the incorrect modeling of operation side effects, leading to passes like Dead Code Elimination (DCE) erroneously removing essential code.1

#### **The Architectural Shift from Static Traits to Dynamic Interfaces**

Early iterations of MLIR used simple, static traits like HasNoSideEffect to model an operation's behavior.1 While simple, this binary approach was insufficient for sophisticated analyses. To address this, MLIR evolved to favor explicit, queryable interfaces, with the canonical approach for memory effects being the

MemoryEffectsOpInterface.1

The non-existence of a trait like RecursiveSideEffects is a deliberate architectural choice. The side effects of a container operation (one with a region) are an emergent property of the operations it contains at any given moment. This dynamic property cannot be captured by a static trait declared in a .td file. It requires a dynamic, query-based solution, which is precisely what an interface provides.1

#### **Implementing MemoryEffectsOpInterface: A Two-Part Tutorial**

The idiomatic pattern for modeling these dynamic side effects is a two-part solution involving both TableGen and C++.1

**Part 1: TableGen Declaration (.td)**

First, declare that the operation conforms to the interface in its TableGen definition. This acts as a contract, signaling to the MLIR framework that the C++ class will provide an implementation for the interface's methods.1

C++

// In OrchestraOps.td  
include "mlir/Interfaces/SideEffectInterfaces.td"

def Orchestra\_TaskOp : Orchestra\_Op\<"task", \[MemoryEffectsOpInterface\]\> {  
  //... other properties like summary, description...  
  let regions \= (region SizedRegion:$body);  
}

**Part 2: C++ Implementation (.cpp)**

Second, provide the C++ implementation for the getEffects method. This is where the "recursive" logic resides. The implementation traverses the operation's region, queries each nested operation for its own side effects, and accumulates them.1

C++

// In OrchestraOps.cpp  
\#**include** "mlir/Interfaces/SideEffectInterfaces.h"

void Orchestra\_TaskOp::getEffects(  
    SmallVectorImpl\<SideEffects::EffectInstance\<MemoryEffects::Effect\>\> \&effects) {  
  // Iterate over all operations within the task's body region.  
  for (Operation \&op : getBody().front().getOperations()) {  
    // Dynamically query if the nested op implements the interface.  
    if (auto memInterface \= dyn\_cast\<MemoryEffectsOpInterface\>(\&op)) {  
      // If so, delegate the query and accumulate its effects.  
      memInterface.getEffects(effects);  
      continue;  
    }

    // CONSERVATIVE FALLBACK: see below.  
    if (\!op.hasTrait\<OpTrait::HasNoSideEffect\>()) {  
      effects.emplace\_back(MemoryEffects::Read::get());  
      effects.emplace\_back(MemoryEffects::Write::get());  
    }  
  }  
}

#### **The Conservative Fallback: A Principle of Correctness**

The final if block in the C++ implementation embodies a critical principle of robust compiler design. If a nested operation's side effects are *unknown* (i.e., it does not implement MemoryEffectsOpInterface), the only safe assumption is that it has arbitrary side effects. The code conservatively assumes it can both read from and write to memory.1 This prevents incorrect transformations by DCE or other passes, prioritizing correctness above all else.

#### **Broader Impact: Becoming a "Good Citizen" in the MLIR Ecosystem**

Correctly implementing this interface is not just a localized bug fix. It makes the dialect a "good citizen" within the MLIR ecosystem. It enables a wide range of generic passes—such as Loop-Invariant Code Motion (LICM) and Common Subexpression Elimination (CSE)—to correctly reason about and optimize code containing the operation, unlocking the full power of MLIR's shared infrastructure.1

### **Chapter 10: Navigating the Broader Transformation Ecosystem**

#### **RewritePattern vs. ConversionPattern**

This guide has focused on mlir::RewritePattern, which is appropriate for transformations that occur within a single dialect or do not involve changes to the types of SSA values. However, for lowering processes that involve type conversions (e.g., from index to i64), a plain RewritePattern is insufficient. The correct tool is mlir::ConversionPattern, which is part of MLIR's powerful Dialect Conversion framework.1 A

ConversionPattern is used in conjunction with a TypeConverter and provides an enhanced rewriter that can automatically convert the types of operands and block arguments according to specified rules, greatly simplifying the implementation of type-changing lowerings.1

#### **Interacting with the Greedy Pattern Driver**

Most patterns are driven by the greedy pattern rewrite driver (applyPatternsAndFoldGreedily).1 Understanding its behavior is key to writing effective patterns:

* **Pattern Benefit:** Each pattern has a PatternBenefit, a numeric value indicating its priority. When multiple patterns match the same operation, the one with the highest benefit is applied first. For major structural transformations like region transplantation, it is advisable to assign a high benefit (greater than 1\) to ensure it runs before smaller, potentially interfering canonicalizations.1  
* **Bounded Recursion:** The driver is conservative about recursion and will halt if it detects that a pattern can be applied to its own output. If a pattern's recursion is intentional and known to terminate (e.g., peeling loop iterations), this must be explicitly signaled to the driver by calling setHasBoundedRewriteRecursion() in the pattern's constructor.1

#### **Effective Debugging Strategies**

Structural rewrites are complex and can easily introduce subtle bugs. A disciplined approach to debugging is essential 1:

* **Leverage the Verifier:** Use the \-verify-diagnostics pass to pinpoint where the IR became invalid. Sprinkling assert(succeeded(op-\>verify())) in C++ code can isolate the exact line that corrupts the IR.  
* **Use MLIR Pass Instrumentation:** The mlir-opt flags \--print-ir-before-all and \--print-ir-after-all are invaluable for inspecting the IR state before and after a pass runs.  
* **Enable Expensive Checks:** Compiling MLIR with \-DMLIR\_ENABLE\_EXPENSIVE\_PATTERN\_API\_CHECKS=ON enables internal diagnostics that catch common pattern implementation bugs, such as returning success() without changing the IR.1  
* **Use Debug Names and Filters:** Assigning debug names to patterns allows them to be selectively enabled or disabled with mlir-opt's \--debug-only and \-rewrite-pattern-filter options, making it easier to isolate the behavior of a single complex pattern.

## **Conclusion**

Region transplantation, terminator replacement, and side-effect modeling are powerful but intricate transformations within the MLIR ecosystem. Their successful implementation hinges on a disciplined adherence to the foundational principles of MLIR's IR structure and the strict contracts of the pattern rewriting framework.

The key principles for any developer undertaking these tasks are:

1. **Respect IR Invariants:** The hierarchical structure of operations, regions, and blocks, along with the non-negotiable requirement for terminators in SSACFG regions, forms the bedrock of IR validity. All transformations must preserve these invariants.  
2. **Use the Pattern Rewriter Exclusively:** All IR mutations must be channeled through the PatternRewriter API. This ensures that the pattern application driver can correctly track changes and manage the transformation process transactionally.  
3. **Analyze Before Mutating:** The "analyze-then-mutate" paradigm, centered on utilities like mlir::getUsedValuesDefinedAbove, is critical for separating the identification of dependencies from the mechanics of the rewrite, leading to more robust patterns.  
4. **Master the Correct APIs:** Choosing the right tool is paramount. inlineRegionBefore is the high-level, "smart" API for transplanting region logic. For terminator modification, the "insert-then-erase" sequence is a mandatory safety pattern.  
5. **Choose the Right Tool (C++ vs. PDLL):** Leverage the imperative power of C++ for patterns with complex algorithmic logic and the declarative conciseness of PDLL for complex structural graph matches.  
6. **Model Semantics Idiomatically:** Use modern interfaces like MemoryEffectsOpInterface to dynamically and accurately model complex semantic properties, ensuring correct interaction with the entire ecosystem of generic compiler passes.

By internalizing these principles and following the step-by-step procedures outlined in this guide, compiler engineers can confidently implement complex, structural IR transformations, unlocking the full potential of MLIR's multi-level rewriting capabilities to build the next generation of sophisticated and correct compilers.

#### **Referenzen**

1. source\_documents.pdf  
2. Pattern Rewriting : Generic DAG-to-DAG Rewriting \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/PatternRewriter/](https://mlir.llvm.org/docs/PatternRewriter/)  
3. mlir/docs/PatternRewriter.md · doe \- GitLab, Zugriff am August 22, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/PatternRewriter.md)  
4. llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp at main \- GitHub, Zugriff am August 22, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp)  
5. Deprecations & Current Refactoring \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/deprecation/](https://mlir.llvm.org/deprecation/)  
6. mlir::ConversionPatternRewriter Class Reference \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1ConversionPatternRewriter.html](https://mlir.llvm.org/doxygen/classmlir_1_1ConversionPatternRewriter.html)  
7. include/mlir/Transforms/RegionUtils.h File Reference \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/doxygen/RegionUtils\_8h.html](https://mlir.llvm.org/doxygen/RegionUtils_8h.html)  
8. Frontend Pattern Language \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/OpenMeetings/2021-11-04-PDLL-Pattern-Frontend-Language.pdf](https://mlir.llvm.org/OpenMeetings/2021-11-04-PDLL-Pattern-Frontend-Language.pdf)  
9. mlir/docs/PDLL.md · develop · Kevin Sala / llvm-project \- GitLab, Zugriff am August 22, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/develop/mlir/docs/PDLL.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/develop/mlir/docs/PDLL.md)  
10. MLIR — Defining Patterns with PDLL || Math ∩ Programming, Zugriff am August 22, 2025, [https://www.jeremykun.com/2024/08/04/mlir-pdll/](https://www.jeremykun.com/2024/08/04/mlir-pdll/)  
11. BladeDISC分享, Zugriff am August 22, 2025, [https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/bladedisc-intro-for-intel.pdf](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/bladedisc-intro-for-intel.pdf)  
12. Side Effects & Speculation \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)  
13. llvm-project/mlir/docs/Interfaces.md at main \- GitHub, Zugriff am August 22, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/Interfaces.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Interfaces.md)  
14. Dialect Conversion \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/DialectConversion/](https://mlir.llvm.org/docs/DialectConversion/)  
15. mlir/docs/DialectConversion.md · main · Kevin Sala / llvm-project \- GitLab, Zugriff am August 22, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/DialectConversion.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/DialectConversion.md)