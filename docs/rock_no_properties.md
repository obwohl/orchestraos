

# **Architectural Review and Implementation Blueprint for the LinalgToRockPass**

## **1.0 Executive Summary**

### **1.1 Overview of Findings**

The proposal to implement a LinalgToRockPass correctly identifies a critical optimization pathway for the OrchestraOS Compiler. Lowering generic linear algebra operations to the AMD rock dialect is a necessary step to unlock the performance of specialized matrix core hardware on ROCm-supported GPUs. The high-level objective is sound and aligns perfectly with the project's goal of achieving performance portability across heterogeneous hardware.

However, a detailed analysis of the proposed implementation plan reveals significant architectural and API-level deficiencies. The current plan underestimates the complexities inherent in dialect lowering within the MLIR ecosystem, particularly concerning type safety, IR legality, and the parameterized nature of high-performance GPU operations. The proposed approach, based on a simple pattern rewrite, would result in a fragile, non-idiomatic, and likely incorrect implementation that would fail to handle real-world use cases involving mixed-precision types and varied data layouts.

### **1.2 Key Recommendations**

This report provides a corrected and enhanced blueprint for the implementation. The most critical recommendations are:

1. **Adopt the MLIR Dialect Conversion Framework:** The implementation must be re-architected to use MLIR's Dialect Conversion Framework. This is a non-negotiable strategic shift from the proposed simple pattern rewrite. This framework is the idiomatic and correct tool for managing the legality of operations and the complex, interdependent type conversions that are implicit in lowering from the linalg dialect on tensors to a GPU-level dialect operating on memory buffers.  
2. **Rigorously Define the rock.gemm Contract:** The proposed plan critically omits the definition of the target rock.gemm operation. The rocMLIR project is a sophisticated kernel generator, and its operations are highly parameterized to control performance tuning. The first and most important implementation task is to investigate the rocMLIR source code to determine the precise signature, arguments, and mandatory attributes of the rock.gemm operation. The lowering logic is entirely dependent on this contract.  
3. **Implement a Production-Grade Testing Strategy:** The proposal for a single test case is inadequate for a production-quality compiler pass. A comprehensive lit test suite is required. This suite must include a matrix of test cases covering various data types (f32, f16), matrix layouts (transpositions), negative cases (ops that should not be converted), and edge cases to ensure the pass is robust, correct, and does not introduce regressions.

### **1.3 Expected Outcome**

By adopting the architectural principles, corrected API usage, and robust methodologies detailed in this report, the developer will be equipped to implement a correct, maintainable, and high-performance LinalgToRockPass. The resulting implementation will align with modern MLIR best practices, effectively bridge the gap between high-level linear algebra and AMD's hardware-specific kernels, and constitute a significant step toward achieving the performance goals of the OrchestraOS Compiler on AMD GPUs.

## **2.0 Revised Requirements Specification**

The following section presents a corrected and enhanced version of the requirements specification. These criteria are designed to be unambiguous, comprehensive, and verifiable, ensuring the final feature is robust and meets the project's needs.

### **2.1 Feature Title**

linalg to rock Dialect Lowering Pass

### **2.2 Goal / User Story**

As a compiler engineer, I want to implement a new MLIR pass that lowers linalg.generic operations to the rock dialect from AMD's rocMLIR project, so that we can leverage AMD's matrix core hardware for GEMM and convolution operations.

### **2.3 Enhanced Acceptance Criteria**

The successful completion of this feature will be determined by the fulfillment of the following criteria:

* **AC1: Pass Implementation:** A new MLIR pass, LinalgToRockPass, is created. It must be implemented using the MLIR Dialect Conversion Framework, inherit from mlir::OperationPass, and be configured to operate on mlir::func::FuncOp operations.  
* **AC2: GEMM Conversion:** The pass must contain one or more mlir::OpConversionPatterns that successfully convert mlir::linalg::GenericOp operations into mlir::rock::GEMMOp operations, provided the source operation satisfies the full "rocMLIR contract."  
* **AC3: Expanded "rocMLIR Contract" Definition:** The contract for conversion is both structural and semantic. A linalg.generic operation is considered a convertible GEMM if and only if it meets all of the following conditions:  
  * Its iterator\_types attribute is an array of \["parallel", "parallel", "reduction"\].  
  * Its indexing\_maps attribute corresponds to a standard matrix multiplication (e.g., (m, n, k) \-\> (m, k), (m, n, k) \-\> (k, n), (m, n, k) \-\> (m, n)) or a valid transposition thereof.  
  * Its region contains a single block with a computational body consisting of a multiplication operation (arith.mulf), an addition operation (arith.addf), and a linalg.yield operation, representing a multiply-accumulate pattern. This ensures that ops with a GEMM-like structure but different semantics (e.g., element-wise products) are not incorrectly converted.  
  * It operates on tensor or memref types with supported floating-point element types: f32, f16, or bf16.  
* **AC4: Attribute Derivation:** The generated rock.gemm operation must be populated with all necessary hardware-specific tuning attributes (e.g., blockSize, mPerBlock, nPerBlock, kPerBlock). The strategy for deriving these attributes (e.g., from a static heuristic based on operand shapes, pass options, or a future cost model) must be defined and implemented. An initial implementation with safe, hardcoded defaults is acceptable, but the mechanism for parameterization must be in place.  
* **AC5: Pipeline Integration:** The pass is integrated into the LowerOrchestraToGPUPass pipeline. It must be added to the OpPassManager for the rocdl architecture target and be explicitly scheduled to run *before* the existing LowerOrchestraToROCDLPass. This ordering is crucial to ensure that GEMM operations are specialized before generic data movement and other lowering steps occur.  
* **AC6: Comprehensive Test Suite:** A new set of lit test files is created in the orchestra-compiler/tests/ directory. This suite must provide comprehensive coverage, including:  
  * **Positive Cases:** Successful conversion for standard GEMMs using f32 and f16 data types.  
  * **Layout Variations:** Successful conversion of linalg.generic operations that represent GEMM with transposed inputs (e.g., AT×B, A×BT, AT×BT).  
  * **Negative Cases:** linalg.generic operations that match the structural contract but have an incorrect computational body (e.g., element-wise multiplication) must be correctly identified and *not* converted.  
  * **Type Safety:** linalg.generic operations with unsupported data types (e.g., i32, f64) must be ignored by the pass.  
* **AC7: Regression Prevention:** The full existing project test suite must continue to pass after the integration of the new pass, ensuring no existing functionality is broken.

## **3.0 Revised & Enhanced Implementation Plan**

This section provides a detailed, corrected, and enhanced plan for implementing the LinalgToRockPass. It begins by establishing the correct architectural foundation, corrects proposed API usage, deconstructs the unknown target operation, provides a revised step-by-step guide, and outlines a robust testing strategy.

### **3.1 Foundational Strategy: Adopting the Dialect Conversion Framework**

The proposed implementation strategy of using a simple rewrite pattern is architecturally flawed for this task. The process of lowering from a high-level, abstract dialect like linalg to a hardware-specific dialect like rock is the canonical use case for MLIR's Dialect Conversion Framework.

A critical consideration in lowering is the management of type legality. GPU kernels, which the rock dialect targets, almost universally operate on explicit memory buffers (represented by memref types), not abstract, value-semantic tensor types. Therefore, a lowering from linalg on tensors to rock will implicitly require a type conversion from tensor to memref. A simple RewritePattern is unaware of this global transformation context. If it replaces an op that produces a tensor with one that operates on a memref, all downstream users of the original tensor result become invalid, leading to a corrupted IR state.

The Dialect Conversion Framework is MLIR's solution to this "type obstacle". It provides a holistic mechanism to manage the legality of operations and types across an entire function or module. It tracks type changes via a TypeConverter and can automatically insert "materialization" operations to bridge type mismatches between a newly converted operation and its legacy users. This ensures that the IR remains valid throughout the entire conversion process. For this reason, adopting the framework is not merely a best practice; it is a prerequisite for a correct and robust implementation.

The implementation must be built upon the following core components of the framework :

* **ConversionTarget:** A configuration object that defines the set of legal operations and dialects for the pass's output. For this pass, the rock dialect will be marked as Legal, while linalg.generic operations matching the GEMM contract will be marked as Illegal.  
* **RewritePatternSet:** A collection of conversion patterns. All patterns used within this framework must inherit from OpConversionPattern.  
* **OpConversionPattern:** A specialized rewrite pattern that is aware of the conversion process. It provides an OpAdaptor to access operands that have already been converted, which is essential for handling type changes correctly.3  
* **applyPartialConversion Driver:** The main function that orchestrates the conversion. This driver is chosen over applyFullConversion because the goal is to convert only a specific subset of linalg operations, leaving others untouched.

### **3.2 Corrected API Usage and Pass Architecture**

The proposed C++ code snippets for the pass and pattern definitions contain outdated or incorrect API usage. The following skeletons provide the correct, modern structure based on MLIR 20\.

#### **3.2.1 Pass Definition**

An MLIR pass should be defined using the PassWrapper Curiously Recurring Template Pattern (CRTP) utility, which provides necessary hooks and boilerplate.5 The pass should operate on

mlir::func::FuncOp, as the linalg.generic operation exists within a standard function before any GPU-specific outlining has occurred.

**Correct Pass Skeleton:**

C++

\#**include** "mlir/Pass/Pass.h"  
\#**include** "mlir/Dialect/Func/IR/FuncOps.h"  
\#**include** "mlir/Transforms/DialectConversion.h"

// Forward declarations for dialects this pass will introduce.  
namespace mlir {  
namespace rock {  
class RockDialect;  
} // namespace rock  
namespace arith {  
class ArithDialect;  
} // namespace arith  
} // namespace mlir

namespace {  
struct LinalgToRockPass : public mlir::PassWrapper\<LinalgToRockPass, mlir::OperationPass\<mlir::func::FuncOp\>\> {  
  MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(LinalgToRockPass)

  void getDependentDialects(mlir::DialectRegistry \&registry) const override {  
    // This pass creates operations from the 'rock' dialect.  
    // Declaring this dependency ensures the dialect is loaded and available.  
    registry.insert\<mlir::rock::RockDialect, mlir::arith::ArithDialect\>();  
  }

  mlir::StringRef getArgument() const final { return "lower-linalg-to-rock"; }  
  mlir::StringRef getDescription() const final { return "Lower linalg.generic GEMM ops to rock.gemm"; }

  void runOnOperation() override;  
};  
} // namespace

#### **3.2.2 Pattern Definition**

As established, all patterns used with the Dialect Conversion Framework must inherit from mlir::OpConversionPattern\<SourceOp\>.3 This base class provides the correct

matchAndRewrite signature, which includes an OpAdaptor argument for accessing converted operands.

**Correct Pattern Skeleton:**

C++

\#**include** "mlir/Dialect/Linalg/IR/Linalg.h"

class LinalgGEMMToRockGEMMPattern : public mlir::OpConversionPattern\<mlir::linalg::GenericOp\> {  
public:  
  using OpConversionPattern\<mlir::linalg::GenericOp\>::OpConversionPattern;

  mlir::LogicalResult  
  matchAndRewrite(mlir::linalg::GenericOp op, OpAdaptor adaptor,  
                  mlir::ConversionPatternRewriter \&rewriter) const override;  
};

### **3.3 Deconstructing the rock.gemm Operation Contract**

The single greatest risk and ambiguity in the proposed plan is the lack of a precise definition for the target rock.gemm operation. The plan's assumption of a simple builder signature is unfounded. The rocMLIR project is a compiler that generates high-performance GPU kernels for GEMM and convolution.7 Its MLIR dialects serve as the interface to this complex code generation backend.

Consequently, the rock.gemm operation is not a simple instruction but a highly parameterized configuration object. Its attributes are expected to control the entire code generation process, including tiling strategies, thread block configurations, and the use of hardware acceleration instructions like MFMA.7 The lowering pass is therefore not just translating an operation; it is making critical, performance-impacting decisions and encoding them as attributes on the

rock.gemm op.

The first practical step of implementation must be to locate and analyze the TableGen definition file for the rock dialect within the rocMLIR source code (version 6.4.3). This file, likely located at mlir/include/mlir/Dialect/Rock/IR/Rock.td, will provide the ground truth for the operation's C++ class name, builder method signature, operands, and attributes.

Based on an analysis of rocMLIR's purpose and common practices in GPU kernel generation, Table 1 presents a plausible, inferred structure for the rock.gemm operation. The developer must validate and refine this structure against the actual source code.

**Table 1: Plausible rock.gemm Target Operation Signature**

| Component | Name (Inferred) | Type (Inferred) | Description |
| :---- | :---- | :---- | :---- |
| **Operand** | A | Value (MemRef) | The left-hand side matrix, as a memref. |
| **Operand** | B | Value (MemRef) | The right-hand side matrix, as a memref. |
| **Operand** | C | Value (MemRef) | The output/accumulator matrix, as a memref. |
| **Attribute** | arch | StringAttr | Specifies the target GPU architecture (e.g., "gfx90a", "gfx1100") for architecture-specific tuning. |
| **Attribute** | num\_cu | IntegerAttr | The number of compute units on the target GPU, used for occupancy and scheduling heuristics. |
| **Attribute** | blockSize | IntegerAttr | The number of threads in a workgroup/thread block. A common value is 256\. |
| **Attribute** | mPerBlock | IntegerAttr | The tile size for the M dimension processed by a single thread block (e.g., 128). |
| **Attribute** | nPerBlock | IntegerAttr | The tile size for the N dimension processed by a single thread block (e.g., 128). |
| **Attribute** | kPerBlock | IntegerAttr | The tile size for the K (reduction) dimension processed in each step of the inner loop (e.g., 16). |
| **Attribute** | transposeA | BoolAttr | A flag indicating if matrix A should be treated as transposed during the computation. |
| **Attribute** | transposeB | BoolAttr | A flag indicating if matrix B should be treated as transposed during the computation. |
| **Result** | (none) | (none) | The operation likely has side effects, modifying the output buffer C in place, and thus has no results. |

### **3.4 A Step-by-Step Revised Implementation Guide**

This guide provides a corrected sequence of steps for implementing the pass using the Dialect Conversion Framework.

1. **Create Pass Files and Register:** As originally planned, create orchestra-compiler/lib/Orchestra/Transforms/LinalgToRock.cpp. Add the necessary declarations and registration calls for createLinalgToRockPass() in Passes.h and Passes.cpp.  
2. **Implement the runOnOperation Driver:** In LinalgToRock.cpp, implement the runOnOperation method for LinalgToRockPass. This is where the Dialect Conversion Framework is configured and invoked.  
   C++  
   void LinalgToRockPass::runOnOperation() {  
       mlir::ConversionTarget target(getContext());  
       mlir::RewritePatternSet patterns(\&getContext());  
       auto \*context \= \&getContext();

       // Define what constitutes a "legal" final state. We want rock ops,  
       // arith ops, and memref ops to be legal.  
       target.addLegalDialect\<mlir::rock::RockDialect, mlir::arith::ArithDialect,  
                              mlir::memref::MemRefDialect\>();

       // Any other op is legal if it's not a linalg.generic that we can convert.  
       target.addDynamicallyLegalOp\<mlir::linalg::GenericOp\>((mlir::linalg::GenericOp op) {  
           // isGEMM is a helper function that implements the full contract from AC3.  
           // The op is legal if it is NOT a GEMM we can handle.  
           return\!isGEMM(op);  
       });

       // Populate the pattern set.  
       patterns.add\<LinalgGEMMToRockGEMMPattern\>(context);

       // Run the partial conversion.  
       if (failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {  
           signalPassFailure();  
       }  
   }

3. **Implement the isGEMM Predicate:** In the same file, create a static helper function bool isGEMM(mlir::linalg::GenericOp op). This function will implement the full "rocMLIR Contract" as defined in AC3, checking iterator types, indexing maps, and the semantic content of the operation's region. This centralizes the matching logic.  
4. **Implement the Conversion Pattern's matchAndRewrite:** This is the core transformation logic.  
   C++  
   mlir::LogicalResult LinalgGEMMToRockGEMMPattern::matchAndRewrite(  
       mlir::linalg::GenericOp op, OpAdaptor adaptor,  
       mlir::ConversionPatternRewriter \&rewriter) const {

       // The ConversionTarget has already guaranteed this is a GEMM we want to convert.

       // Extract operands from the adaptor. These are the \*already converted\* values.  
       mlir::Value A \= adaptor.getOperands();  
       mlir::Value B \= adaptor.getOperands();  
       mlir::Value C \= adaptor.getOperands(); // This is the output buffer.

       // Analyze indexing maps to determine transposition.  
       bool transposeA \= needsTranspose(op.getIndexingMapsArray(),...);  
       bool transposeB \= needsTranspose(op.getIndexingMapsArray(),...);

       // Determine tuning parameters. Start with safe, hardcoded defaults.  
       // In the future, this logic could become a sophisticated heuristic.  
       int64\_t blockSize \= 256;  
       int64\_t mPerBlock \= 128;  
       int64\_t nPerBlock \= 128;  
       int64\_t kPerBlock \= 16;  
       std::string arch \= "gfx90a"; // This should be passed in as an option.

       // Create the new rock.gemm op. The builder signature used here is based on  
       // the plausible signature from Table 1 and MUST be validated.  
       rewriter.create\<mlir::rock::GEMMOp\>(  
           op.getLoc(),  
           A, B, C,  
           rewriter.getBoolAttr(transposeA),  
           rewriter.getBoolAttr(transposeB),  
           rewriter.getStringAttr(arch),  
           /\* num\_cu \*/ rewriter.getI32IntegerAttr(0), // Default/unused  
           rewriter.getI32IntegerAttr(blockSize),  
           rewriter.getI32IntegerAttr(mPerBlock),  
           rewriter.getI32IntegerAttr(nPerBlock),  
           rewriter.getI32IntegerAttr(kPerBlock)  
       );

       // The original op is replaced implicitly by not having its results remapped.  
       // We must erase the original op.  
       rewriter.eraseOp(op);  
       return mlir::success();  
   }

5. **Integrate the Pass into the Main Pipeline:** The developer's proposed integration point in LowerOrchestraToGPU.cpp is correct. The pass must be added to the OpPassManager for the rocdl architecture, and its position before LowerOrchestraToROCDLPass is critical.  
   C++  
   // In LowerOrchestraToGPU.cpp's runOnOperation method  
   if (gpuArch \== "rocdl") {  
       mlir::OpPassManager pm(mlir::func::FuncOp::getOperationName());  
       // Run this new pass first to specialize linalg.generic into rock.gemm.  
       pm.addPass(mlir::orchestra::createLinalgToRockPass());  
       // The existing pass then handles data movement and other lowerings.  
       pm.addPass(mlir::orchestra::createLowerOrchestraToROCDLPass());  
       //...  
       if (failed(runPipeline(pm, getOperation())))  
          return signalPassFailure();  
   }

6. **Build System Integration:** Add the new LinalgToRock.cpp file to the appropriate add\_library call in the relevant CMakeLists.txt file.

### **3.5 A Robust Testing Strategy for Production Readiness**

A single test case is insufficient for a compiler pass of this importance. A production-quality pass must be validated by a comprehensive test suite that verifies correctness, completeness, and precision.

* **Correctness:** Ensures that valid inputs are transformed into the correct, expected output.  
* **Completeness:** Ensures that all intended variations of valid input (e.g., different data types, layouts) are handled correctly.  
* **Precision:** Ensures that invalid or unsupported inputs are correctly identified and *not* transformed, preventing the generation of incorrect code. A silent failure that produces semantically wrong code is the most dangerous type of compiler bug.

A new test directory, orchestra-compiler/tests/Dialect/Rock/, should be created to house the lit tests for this pass. Table 2 outlines a recommended initial set of test cases.

**Table 2: Recommended lit Test Cases for LinalgToRockPass**

| Test File Name | Description | FileCheck Verification | Rationale |
| :---- | :---- | :---- | :---- |
| gemm\_f32\_nn.mlir | Basic C=A×B, f32, no transposes. | CHECK: rock.gemm, CHECK-NOT: linalg.generic | Verifies the fundamental "happy path" for the most common data type. |
| gemm\_f16\_nn.mlir | Basic C=A×B, f16, no transposes. | CHECK: rock.gemm, CHECK-NOT: linalg.generic | Ensures correctness for mixed-precision (f16) data types, common in AI models.10 |
| gemm\_f32\_tn.mlir | C=AT×B, f32, A is transposed via indexing maps. | CHECK: rock.gemm... transposeA \= true | Tests the pass's ability to correctly analyze indexing maps and handle data layouts. |
| gemm\_f32\_nt.mlir | C=A×BT, f32, B is transposed via indexing maps. | CHECK: rock.gemm... transposeB \= true | Verifies another critical layout variation. |
| negative\_wrong\_body.mlir | linalg.generic with GEMM structure but element-wise multiply body. | CHECK-NOT: rock.gemm, CHECK: linalg.generic | **Precision Test:** Ensures the pass checks the region body, preventing semantic errors. |
| negative\_wrong\_iterator.mlir | linalg.generic with GEMM body but incorrect iterator types (e.g., all parallel). | CHECK-NOT: rock.gemm, CHECK: linalg.generic | **Precision Test:** Verifies that the structural checks on iterator\_types are working. |
| negative\_unsupported\_type.mlir | linalg.generic GEMM on i32 data. | CHECK-NOT: rock.gemm, CHECK: linalg.generic | **Precision Test:** Confirms that unsupported data types are correctly ignored. |

### **3.6 Alternative Strategy: Declarative Rewrites with TableGen (DRR)**

MLIR provides a powerful declarative system, Table-driven Declarative Rewrite Rules (DRR), for specifying patterns in .td files.11 This approach can significantly reduce boilerplate C++ code for transformations that fit its DAG-to-DAG matching model.

A simplified version of the linalg.generic to rock.gemm pattern could be expressed declaratively as follows:

Code-Snippet

// In a new file, e.g., Orchestra/Transforms/LinalgToRock.td  
include "mlir/Dialect/Linalg/IR/LinalgBase.td"  
// Assume a Rock.td file defining the Rock\_GEMMOp exists  
include "RockDialect.td"

def LinalgGenericToRockGEMM : Pat\<  
  // Source Pattern: Match a linalg.generic op with specific traits  
  (LinalgGenericOp  
    (HasSingleReductionLoop), // Custom constraint checking iterators  
    $a, $b, $c,  
    /\* other properties like indexing\_maps \*/  
  ),  
  // Result Pattern: Replace with a rock.gemm op  
  (Rock\_GEMMOp $a, $b, $c,  
    // Attributes must be provided  
    (StrAttr "gfx90a"),  
    (I32Attr 256), // blockSize  
    (I32Attr 128), // mPerBlock  
    (I32Attr 128), // nPerBlock  
    (I32Attr 16\)   // kPerBlock  
  )  
\>;

#### **3.6.1 Analysis of Trade-offs**

* **Pros:**  
  * **Conciseness:** The .td format is significantly more compact than C++, making the core transformation logic easy to read and verify at a glance.11  
  * **Maintainability:** For a compiler with many simple, structural rewrites, centralizing them in .td files can improve organization and maintainability.  
* **Cons (and why C++ is superior for *this* task):**  
  * **Limited Expressiveness for Procedural Logic:** DRR excels at structural, DAG-based transformations. It is less suited for patterns that require complex, procedural C++ logic to either match an operation or construct the result.13  
  * The primary challenge of this pass is not matching the linalg.generic structure, but intelligently *deriving the tuning attributes* for rock.gemm. This logic is inherently procedural. It may involve querying a hardware information database, executing a cost model based on operand shapes, or applying complex heuristics. This type of logic is difficult or impossible to express cleanly within TableGen's declarative syntax. While DRR provides NativeCodeCall to escape to C++, heavy reliance on this feature negates the benefits of the declarative approach and leads to a less readable, hybrid implementation. The imperative C++ OpConversionPattern provides a natural and powerful environment for this complex, procedural decision-making.

#### **3.6.2 Recommendation**

While DRR is a valuable tool that the OrchestraOS team should consider for simpler canonicalization and peephole optimization patterns, the imperative C++ OpConversionPattern is the correct and more robust tool for this specific LinalgToRockPass. The need to programmatically derive performance-critical tuning attributes for the target rock.gemm operation strongly favors the expressiveness of C++.

#### **Referenzen**

1. mlir::OpConversionPattern\< SourceOp \> Class Template Reference, Zugriff am August 26, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1OpConversionPattern.html](https://mlir.llvm.org/doxygen/classmlir_1_1OpConversionPattern.html)  
2. \[Flang\] don't mix rewrite and conversion mlir APIs in the same pass · Issue \#83557 \- GitHub, Zugriff am August 26, 2025, [https://github.com/llvm/llvm-project/issues/83557](https://github.com/llvm/llvm-project/issues/83557)  
3. Pass Infrastructure \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/PassManagement/](https://mlir.llvm.org/docs/PassManagement/)  
4. mlir::PassWrapper\< PassT, BaseT \> Class Template Reference, Zugriff am August 26, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1PassWrapper.html](https://mlir.llvm.org/doxygen/classmlir_1_1PassWrapper.html)  
5. ROCm/rocMLIR \- GitHub, Zugriff am August 26, 2025, [https://github.com/ROCm/rocMLIR](https://github.com/ROCm/rocMLIR)  
6. rocmlir-rock \- MyNixOS, Zugriff am August 26, 2025, [https://mynixos.com/nixpkgs/package/rocmPackages.rocmlir](https://mynixos.com/nixpkgs/package/rocmPackages.rocmlir)  
7. Optimizing with Composable Kernel \- ROCm Documentation \- AMD, Zugriff am August 26, 2025, [https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html)  
8. Matrix Multiplication Background User's Guide \- NVIDIA Docs, Zugriff am August 26, 2025, [https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)  
9. Table-driven Declarative Rewrite Rule (DRR) \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)  
10. Glossary \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/getting\_started/Glossary/](https://mlir.llvm.org/getting_started/Glossary/)  
11. PDLL \- PDL Language \- MLIR, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/PDLL/](https://mlir.llvm.org/docs/PDLL/)