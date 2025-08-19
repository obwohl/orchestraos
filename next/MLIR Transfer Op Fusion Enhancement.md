# **Technical Review and Enhancement Plan for Matmul-Add Fusion Pass**

## **1. Executive Summary**

This report presents a comprehensive technical review of the proposed MLIR pass for fusing orchestra.matmul and orchestra.add operations. The initial plan correctly identifies a valuable fusion opportunity and proposes a functional implementation pathway using the standard OpRewritePattern infrastructure. This demonstrates a solid foundational understanding of MLIR's rewrite mechanics.

However, the analysis reveals several areas requiring significant enhancement to meet production-quality standards. The proposed imperative C++ implementation, while viable, is verbose and does not leverage modern, more maintainable MLIR frameworks. It also overlooks critical edge cases concerning operator attributes, mixed-precision data types, and complex dataflow graph topologies. Furthermore, the plan contains API usages that are outdated according to the MLIR 20 specification and proposes a testing strategy that is insufficient to guarantee robustness.

The following core recommendations are put forth:

* **Strategic Pivot to a Declarative Approach:** The implementation should be refactored from a purely imperative C++ pattern to a declarative one using the **Pattern Description Language (PDLL)**. This state-of-the-art framework will produce a more concise, readable, and maintainable pattern, significantly reducing long-term engineering costs and aligning the project with MLIR best practices.  
* **Enhancement of Requirements:** The functional requirements must be expanded to explicitly define the pass's behavior for attribute propagation and conflict resolution, mixed-precision data types, and structural variants like broadcasts.  
* **Fortification of Testing:** The test suite must be systematically expanded to include complex test cases that probe the identified edge cases, including specific checks for correct diagnostic information preservation using FusedLoc.

Adopting these recommendations will result in a fusion pass that is not only correct but also robust, maintainable, and aligned with modern MLIR development principles. This will reduce future technical debt and increase the overall reliability of the compiler.

## **2. Revised Requirements Specification**

The following revised specification provides a complete, unambiguous, and technically precise set of requirements. It is intended to serve as the definitive guide for the implementation, covering all identified use cases and edge cases.

### **2.1. Core Fusion Pattern**

The pass shall identify and fuse a orchestra.matmul operation that is immediately followed by a orchestra.add operation. The fusion is applicable only when a result of the matmul operation is a direct operand to the add operation.

A critical precondition for this transformation is that the result of the matmul operation must have **exactly one use**, which must be the add operation targeted for fusion. This constraint is essential for ensuring correctness in complex Directed Acyclic Graphs (DAGs). Attempting to fuse an operation with multiple users without more sophisticated graph rewriting can lead to incorrect transformations by invalidating the inputs to other downstream operations. This principle aligns with established canonicalization strategies in MLIR, which often prioritize patterns that simplify the IR by matching on single-use values.1 The output of this transformation will be a single, new

orchestra.fused_matmul_add operation that semantically combines the original two.

### **2.2. Attribute Propagation and Merging Policy**

Operations in MLIR frequently carry attributes that dictate their semantics, such as fast-math flags or precision hints.3 A fusion pass must implement a clear and conservative policy for handling these attributes to prevent silent correctness bugs. When transforming operations, there are three possible policies for handling such metadata: block the transformation, implicitly drop the metadata, or propagate it.4 Dropping metadata is unsafe as it can alter program semantics. Therefore, a strict propagation policy is required.

The required attribute handling policy is as follows:

* **Identical Attributes:** If both the matmul and add operations possess an attribute with the same name and value (e.g., both have fastmath="allow"), this attribute shall be propagated to the new fused_matmul_add operation.  
* **Conflicting Attributes:** If the matmul and add operations have an attribute with the same name but different values (e.g., one has fastmath="allow" and the other has fastmath="none"), the fusion **must be blocked**. The pattern should fail to match in this scenario. This conservative approach prevents the compiler from making an arbitrary and potentially incorrect decision about which semantic behavior to preserve.  
* **Unique Attributes:** An attribute that is present on only one of the original operations shall be propagated to the fused operation, provided it does not conflict with any attribute on the other operation.  
* **Discardable Attributes:** While some dialects may define attributes that are purely for annotation and can be safely discarded, the default policy for this pass will be to propagate all attributes unless explicitly specified otherwise.

### **2.3. Data Type and Precision Handling**

The initial plan lacked specificity regarding data types. Modern machine learning models often employ mixed-precision arithmetic (e.g., f32, f16, bf16) to optimize performance. The fusion pass must handle these scenarios correctly.

* The pass must support fusion for any floating-point or integer element types for which the orchestra.matmul and orchestra.add operations are valid.  
* **Type Promotion:** The element type of the result of the new fused_matmul_add operation shall be determined by the result type of the original add operation, as it is the final operation in the semantic chain.  
* **Type Mismatch:** The pattern should only match if the operand and result types of the matmul and add operations are compatible according to the dialect's verification rules. For instance, if a matmul produces a tensor<16x16xf16> and the subsequent add expects a tensor<16x16xf32>, there would likely be an intermediate casting operation. This cast would break the "direct use" condition defined in section 2.1, and thus the pattern would not match. The pass must not attempt to resolve such type mismatches itself.

### **2.4. Structural and Semantic Variant Coverage**

Matrix multiplication and addition operations can have structural variants, such as transpositions, broadcasts, or strided memory layouts. These are often captured via attributes or distinct operations within dialects like linalg.5

* The fusion pattern must correctly capture any attributes related to transposition or memory layout from the matmul operation and propagate them to the new fused_matmul_add operation.  
* The add operation may involve a broadcast (e.g., adding a vector bias to a matrix). The pattern must be capable of matching this structure. The resulting orchestra.fused_matmul_add operation must have well-defined semantics for handling this broadcasted addition.

### **2.5. Graph Topology Constraints**

The "single use" constraint is the foundation for the initial implementation. However, a more advanced transformation is possible in the future.

* **Phase 1 (Current Scope):** The implementation must strictly enforce the single-use constraint as defined in section 2.1 for simplicity and guaranteed correctness.  
* **Phase 2 (Future Enhancement):** A potential future enhancement could relax this constraint. If the matmul result has multiple users, the pass could still perform fusion by creating the fused_matmul_add operation to replace the add op, while leaving the original matmul in place to satisfy its other users. A subsequent Dead Code Elimination (DCE) pass would then be responsible for removing the original matmul and add if all of their uses were eventually replaced. This more complex transformation aligns with the greedy, iterative nature of MLIR pass pipelines like the canonicalizer but should be deferred until the core functionality is proven.1

## **3. Revised & Enhanced Implementation Plan**

This section provides a state-of-the-art implementation blueprint that is robust, maintainable, and leverages the most powerful and idiomatic features of the MLIR framework. It begins by presenting the recommended declarative strategy, followed by a detailed correction of the original C++ proposal.

### **3.1. Recommended Strategy: Declarative Rewrites with PDLL**

The most effective and maintainable approach for implementing this fusion is to use MLIR's Pattern Description Language (PDLL). MLIR offers several methods for writing rewrite patterns, including imperative C++ (OpRewritePattern) 7, TableGen-based Declarative Rewrite Rules (DRR) 8, and PDLL.9 PDLL was specifically designed to overcome the structural limitations and awkward syntax of DRR, providing an intuitive, MLIR-native way to express patterns.9

For a structural DAG-to-DAG transformation like this fusion, PDLL is architecturally superior to an imperative C++ implementation for several reasons:

* **Readability and Conciseness:** The pattern's structure is expressed declaratively, mirroring the MLIR assembly format. This makes the transformation's intent immediately clear and far easier to verify by inspection than the equivalent C++ boilerplate.  
* **Maintainability:** As the dialect evolves, updating a concise PDLL pattern is significantly simpler and less error-prone than modifying complex C++ matching logic.  
* **Separation of Concerns:** PDLL enables a powerful hybrid approach. The structural *pattern matching* is defined declaratively, while complex, non-structural logic, such as the attribute merging policy, can be cleanly delegated to C++ helper functions using pdl.apply_native_constraint and pdl.apply_native_rewrite.10 This combines the clarity of declarative patterns with the power of imperative C++.

Adopting PDLL is not merely a stylistic choice; it is a strategic architectural decision. It aligns the project with the modern direction of MLIR development and invests in a more scalable and robust compiler development methodology. The patterns themselves become assets that are easier to test, compose, and reason about, significantly reducing long-term maintenance costs.

#### **PDLL Pattern Implementation (Illustrative)**

The following PDLL code illustrates the core of the fusion pattern.

Code-Snippet

// PDLL pattern for fusing orchestra.matmul -> orchestra.add  
pdl.pattern @FuseMatmulAdd : benefit(100) {  
  // Match the operations and capture their operands/results.  
  %matmul_result = pdl.result(0) of %matmul = pdl.operation "orchestra.matmul"(%a, %b);  
  %add = pdl.operation "orchestra.add"(%matmul_result, %c);

  // Apply constraints based on the requirements.  
  // 1. Matmul result must have one use.  
  pdl.apply_native_constraint "HasOneUse"([%matmul_result]);

  // 2. Attributes must be compatible (handled by a C++ helper).  
  // This constraint function will return failure if attributes conflict.  
  pdl.apply_native_constraint "AreAttributesCompatible"([%matmul, %add]);

  // Rewrite the matched pattern.  
  pdl.rewrite %add with "orchestra.fused_matmul_add"(%a, %b, %c) {  
    // A native rewriter handles attribute merging and location fusing.  
    %merged_attrs = pdl.apply_native_rewrite "MergeAttributesAndFuseLoc"([%matmul, %add]);  
    pdl.attributes = %merged_attrs;  
  }  
}

#### **Integrating Complex C++ Logic**

The PDLL pattern delegates complex logic to registered C++ functions:

* **Native Constraint Function (AreAttributesCompatible):** A C++ function must be implemented and registered with the PDL pattern set. This function will receive mlir::Operation* handles for the matmul and add ops, inspect their attributes according to the policy in section 2.2, and return success() if they are compatible or failure() otherwise. This cleanly encapsulates the attribute-checking logic.10  
* **Native Rewrite Function (MergeAttributesAndFuseLoc):** A C++ function will be implemented to take the two matched operations, create a mlir::FusedLoc (see section 3.3), merge their attributes into a new DictionaryAttr, and return this new attribute dictionary to the PDL rewriter.10

### **3.2. Analysis of Original C++ Plan and API Corrections (MLIR 20)**

While the PDLL approach is recommended, it is instructive to correct the originally proposed imperative C++ plan. A direct C++ implementation requires significant boilerplate and careful use of the PatternRewriter API to avoid common pitfalls. The following table details necessary corrections to align with modern MLIR 20 APIs and best practices.

| Original (Incorrect/Outdated) Usage | Corrected MLIR 20 Usage | Justification / Reference |
| :---- | :---- | :---- |
| rewriter.create<MyOp>(...) (assuming older syntax) | rewriter.create<orchestra::FusedOp>(loc,...) | Modern OpBuilder methods, inherited by PatternRewriter, require the mlir::Location (loc) as the first argument for all new operations.7 |
| Manual op->erase() and addOp->erase() | rewriter.replaceOp(addOp, fusedOp->getResults()); | All IR mutations must be performed via the PatternRewriter instance. replaceOp correctly handles SSA value replacement and erases the original op, ensuring the pattern driver's state remains valid.7 |
| op->getLoc() and addOp->getLoc() used separately | mlir::FusedLoc::get(rewriter.getContext(), {op->getLoc(), addOp->getLoc()}) | Using a FusedLoc preserves the source location history from both original operations, which is critical for debugging and diagnostics. This is a best practice.12 |
| op->getAttr("name") followed by manual casting | op->getAttrOfType<mlir::StringAttr>("name") | Using typed attribute getters like getAttrOfType is safer, less verbose, and avoids potential runtime errors from incorrect dyn_cast operations. |

### **3.3. Preserving Diagnostic Integrity: Location Handling**

When operations are fused, losing the original source code location information makes debugging the compiler's output extremely difficult. Error messages may point to a generated operation with no link back to the source code that produced it. MLIR's FusedLoc is the correct mechanism to solve this problem by creating a new location that references all original locations.12

The implementation, whether in a C++ helper for PDLL or directly in an imperative pattern, must explicitly construct a FusedLoc. This preserves the full history of the transformation in the IR itself.

C++

// Snippet for a C++ helper function to create a FusedLoc  
mlir::Location fuseLocations(mlir::Operation* op1, mlir::Operation* op2, mlir::PatternRewriter& rewriter) {  
    llvm::SmallVector<mlir::Location> locs = {op1->getLoc(), op2->getLoc()};  
    // Metadata can be used to indicate which pass performed the fusion.  
    auto fusionMetadata = mlir::StringAttr::get(rewriter.getContext(), "MatmulAddFusion");  
    return mlir::FusedLoc::get(rewriter.getContext(), locs, fusionMetadata);  
}

The testing strategy must then verify that the generated fused operation has the correct fused location attribute in the textual IR output. For example, a FileCheck directive would look for a line like: // CHECK: loc(fused<"MatmulAddFusion">["source.mlir":10:5, "source.mlir":11:5]).13

### **3.4. A Robust and Comprehensive Testing Strategy**

A production-quality pass requires a comprehensive test suite that verifies not only the intended transformations but also ensures the pass does not incorrectly transform valid IR. The MLIR testing ecosystem, based on lit and FileCheck, is the standard for this.16 The following test cases are mandatory:

1. **Basic Success Case:** A simple matmul followed by an add. The test must CHECK for the presence of the fused_matmul_add op and the correct FusedLoc.  
2. **Blocked by Multiple Uses:** A matmul whose result is used by both an add and another operation. The test must use CHECK-NOT to ensure the fused op is *not* generated.  
3. **Blocked by Conflicting Attributes:** A matmul and add with conflicting fastmath flags. This must also use CHECK-NOT to verify that fusion is correctly blocked.  
4. **Successful Attribute Propagation:** A matmul and add with identical or unique, non-conflicting attributes. The test must CHECK that the fused operation contains the correctly propagated attributes.  
5. **Mixed Precision with Cast:** A matmul producing an f16 tensor that is cast to f32 before being used by an add. The test must CHECK-NOT for fusion, as the intermediate cast breaks the direct-use precondition.  
6. **Broadcast Add:** A matmul producing a tensor<16x32xf32> and an add that consumes it along with a tensor<32xf32> (a broadcast). The test must CHECK that fusion occurs and the resulting fused op has the correct signature.  
7. **Complex DAG:** A chain of multiple operations where a valid fusion opportunity exists in the middle. The test must verify that only the correct pair is fused and the rest of the graph remains untouched.

### **3.5. Alternative Strategy: Best-Practice Imperative C++**

If the team is not prepared to adopt PDLL, a well-structured imperative C++ pattern remains a viable, though less ideal, alternative. This implementation must adhere to modern best practices.

The implementation would involve:

1. Creating a C++ class that inherits from OpRewritePattern<orchestra::AddOp>. The pattern should anchor on the add operation, as it is the consumer in the pattern.  
2. In the matchAndRewrite method, retrieve the first operand of the AddOp. Verify if it is defined by a orchestra::MatmulOp. If not, return failure().  
3. Check the "single use" constraint on the MatmulOp's result using the hasOneUse() method.18 If this check fails,  
   return failure().  
4. Implement the attribute compatibility logic from section 2.2 in a helper function. If the attributes are incompatible, return failure().  
5. If all checks pass, create the FusedLoc as described in section 3.3.  
6. Merge the attributes from both operations into a new DictionaryAttr.  
7. Create the new orchestra.fused_matmul_add operation using rewriter.create<...>().  
8. Replace the original AddOp with the results of the new fused op using rewriter.replaceOp(addOp,...). This single call correctly replaces the SSA uses and erases the original add and, if it becomes dead, the matmul.  
9. return success().

#### **Table 2: Comparison of Implementation Strategies**

The following table provides a clear summary of the trade-offs between the implementation strategies, justifying the recommendation for the PDLL-based approach.

| Metric | Original C++ Plan (Implied) | Enhanced Imperative C++ | Recommended PDLL Hybrid |
| :---- | :---- | :---- | :---- |
| **Readability** | Low | Medium | **High** (Pattern structure is explicit) |
| **Maintainability** | Low (Verbose, easy to introduce bugs) | Medium (Boilerplate is still significant) | **High** (Declarative, easy to modify) |
| **Development Effort** | Medium (High boilerplate, but familiar) | Medium (Requires careful API usage) | **Low** (Once PDLL is learned) |
| **Robustness** | Low (Misses many edge cases) | High (If implemented as per this plan) | **High** (Constraints are explicit) |
| **Performance** | N/A (Compile-time) | N/A (Compile-time) | N/A (Compile-time) |
| **Alignment w/ SOTA** | Low | Medium | **High** (Uses modern MLIR infrastructure) |