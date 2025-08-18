

# **Architectural Review and Implementation Guide: Lowering orchestra.transfer to the xegpu Dialect**

## **Executive Summary**

This report provides a comprehensive architectural review of the proposed plan to lower the orchestra.transfer operation to the Intel xegpu dialect within the OrchestraOS Compiler. The initial proposal demonstrates a solid foundational understanding of the task but requires significant enhancement in its architectural approach, requirements definition, and implementation details to meet the project's goal of creating a robust, hardware-aware compiler for complex AI workloads. The plan, in its current form, overlooks critical real-world use cases that would limit the feature's applicability and introduce substantial technical debt.

The most critical findings and recommendations are as follows:

1. **Architectural Refinement:** The proposal to implement the lowering within a single, monolithic LowerOrchestraToGPU pass is not scalable. It is strongly recommended to adopt a modular, vendor-specific pass architecture (e.g., LowerOrchestraToXeGPU, LowerOrchestraToNVGPU). This approach aligns with MLIR best practices, isolates target-specific complexities, and ensures the long-term maintainability of the compiler as support for new hardware is added.1  
2. **Requirement Enhancement for AI Workloads:** The acceptance criteria are critically underspecified. To be effective for its target domain, the lowering must be enhanced to handle fundamental features of modern AI models. This includes first-class support for non-contiguous memory layouts (strided memrefs) which arise from common tensor slicing operations, mixed-precision floating-point data types (f16, bf16) that are essential for performance on Intel GPUs, and awareness of GPU memory spaces (global, workgroup).3 These are not edge cases but core functional requirements.  
3. **API Precision and Implementation Guidance:** The developer's plan correctly identifies the necessary xegpu operations but notes ambiguity in their usage. This report provides precise, version-correct C++ builder syntax for xegpu.create\_nd\_tdesc, xegpu.load\_nd, and xegpu.store\_nd, resolving these uncertainties and detailing how to correctly manage dynamic tile offsets and non-contiguous memory layouts.6  
4. **Expanded Testing Strategy:** The proposed single "happy path" test is insufficient for validation. A comprehensive lit test suite is specified, including detailed test cases for strided memory access patterns, mixed-precision data types, asymmetric tiling, and memory space transfers. This robust strategy is necessary to ensure the correctness and reliability of the implementation across a wide range of scenarios.

Implementing these recommendations will mitigate the primary risks of the original plan, which include producing a functionally incomplete feature, creating a fragile and difficult-to-maintain pass structure, and lacking the necessary correctness guarantees for deployment in a production compiler. The revised plan outlined in this document provides a clear and robust path toward a high-quality, scalable, and correct implementation.

## **Revised Requirements Specification**

This section presents a complete, corrected, and enhanced version of the requirements specification. It is intended to supersede the original document and serve as the definitive standard for this feature's implementation and validation.

### **2.1. Feature Title**

Lowering orchestra.transfer to the Intel xegpu Dialect

### **2.2. Goal / User Story**

As a compiler developer for OrchestraOS, I want to translate the abstract orchestra.transfer operation into an efficient, concrete sequence of xegpu dialect operations. This will enable the compilation and hardware-aware optimization of data movement for complex AI workloads targeting Intel GPUs.

### **2.3. Enhanced Acceptance Criteria**

* **AC-1: Conversion Pattern Implementation:** A mlir::ConversionPattern targeting orchestra::TransferOp shall be implemented within a new, dedicated LowerOrchestraToXeGPU pass. The pattern must be registered with the MLIR dialect conversion framework.  
* **AC-2: Core Lowering Logic:** The pattern must rewrite a 2D orchestra.transfer operation into a tiled loop structure using the scf.for operation. The body of this loop must contain a sequence of xegpu operations, including xegpu.create\_nd\_tdesc, xegpu.load\_nd, and xegpu.store\_nd, to perform the memory copy.  
* **AC-3: Memory Synchronization:** An xegpu.fence operation must be inserted immediately following the copy loop to ensure that all memory writes performed within the loop are visible to subsequent operations within the correct synchronization scope.  
* **AC-4: Strided Layout Support:** The implementation must correctly handle memref operands with arbitrary, valid strided layouts, not only contiguous, row-major formats. Modern tensor computations frequently involve slicing and views, which result in memref types with non-identity affine maps or explicit strided layout attributes.3 The  
  xegpu.create\_nd\_tdesc operation is designed to encode this information via its strides parameters.6 The conversion pattern must correctly extract stride and offset information from the source and destination  
  MemRefType and propagate it to the xegpu.create\_nd\_tdesc builder to ensure correctness for non-contiguous memory transfers. Failure to support this is a fundamental functional gap, not a missing edge case.  
* **AC-5: Data Type Coverage:** The lowering must be verified to work correctly for memrefs with f32, f16, and bf16 element types. Intel GPUs provide hardware acceleration for 16-bit floating-point types, which are critical for the performance and memory footprint of AI workloads.4 The  
  xegpu dialect is designed to model these hardware capabilities.6 The implementation must be polymorphic with respect to these element types, generating  
  xegpu operations with the corresponding vector and memory types.  
* **AC-6: Memory Space Awareness:** The conversion pattern must correctly handle orchestra.transfer operations between different GPU memory spaces. The gpu dialect defines distinct memory spaces (\#gpu.address\_space\<global\>, \#gpu.address\_space\<workgroup\>, etc.) that are attached to memref types to model the GPU memory hierarchy.5 The pattern must inspect the  
  memorySpace attribute of the source and destination MemRefTypes. This information is critical for ensuring correct synchronization semantics (e.g., the scope of xegpu.fence) and is a prerequisite for future optimizations involving shared memory management.  
* **AC-7: Robustness and Diagnostics:** If the pattern encounters an orchestra.transfer configuration that it does not support (e.g., transfers of rank other than 2, unsupported element types), it must fail gracefully by returning mlir::failure(). It must also emit a clear, actionable diagnostic message to the user, pinpointing the unsupported operation using its source location (e.g., via op-\>emitError(...)).  
* **AC-8: Comprehensive Test Suite:** A comprehensive lit test suite must be created in a new file, orchestra-compiler/tests/lower-transfer-xegpu.mlir. This suite must contain distinct test cases that validate each of the acceptance criteria above, including basic correctness, all specified data types, various strided memory layouts, asymmetric tiling, and transfers between different memory spaces. All existing compiler tests must continue to pass after the changes are integrated.

## **Revised & Enhanced Implementation Plan**

This section provides a detailed, corrected, and deeply enhanced implementation plan. It addresses the architectural questions raised in the original proposal, provides precise API guidance, and outlines a robust strategy for testing and validation.

### **3.1. Strategic Architectural Considerations**

Before beginning implementation, two foundational architectural decisions must be made: the structure of the lowering pass itself and the choice of rewrite methodology. The initial proposal correctly identified these as key questions.

#### **3.1.1. Pass Architecture: Specialization over Generalization**

The query "Is it better to have separate passes... or one pass with conditional patterns?" points to a critical decision for the compiler's long-term health. A single LowerOrchestraToGPU pass containing conditional logic for each GPU vendor (if (target \== intel) {... } else if (target \== nvidia) {... }) is a common anti-pattern that leads to a monolithic, fragile, and difficult-to-maintain codebase. As support for new targets is added, the cyclomatic complexity increases, and developers must understand the intricacies of all backends to make a change to one.

MLIR's pass management infrastructure is explicitly designed to avoid this by encouraging modular, composable passes.1 Production-grade compilers leverage this by creating vendor-specific passes and pipelines (e.g.,

gpu-lower-to-nvvm-pipeline, amdgpu-emulate-atomics) that can be selectively applied.2

Recommendation:  
The implementation should create a new, dedicated pass, LowerOrchestraToXeGPU. The existing LowerOrchestraToGPU pass should be refactored into a pass pipeline. This pipeline will take a configuration option (e.g., a string option \--gpu-arch on orchestra-opt) and, based on its value, add the appropriate vendor-specific lowering pass to its execution list. This architecture offers superior modularity, separation of concerns, and scalability.

#### **3.1.2. Rewrite Strategy: The Right Tool for the Job**

The choice between an imperative C++ RewritePattern and a declarative approach like Table-driven Declarative Rewrite Rules (DRR) depends entirely on the complexity of the transformation.11

* **Declarative Rewrite Rules (DRR):** DRR provides a concise, readable syntax for expressing DAG-to-DAG transformations. It excels at patterns like opA(opB(x)) \-\> opC(x).11 However, its primary limitation is its difficulty in expressing transformations that create complex control flow or new regions with block arguments, such as loops.11 Generating an  
  scf.for loop, which has a region for its body and an induction variable as a block argument, is outside the scope of what DRR can express cleanly.  
* **Imperative C++ ConversionPattern:** The C++ PatternRewriter API provides fine-grained, imperative control over IR mutation. It includes methods to create new blocks, set insertion points, create operations with regions (rewriter.create\<scf::ForOp\>), and access the block arguments of those regions. This level of control is essential for lowering an operation to a loop-based implementation.

Recommendation:  
The proposed implementation correctly chooses to use a C++ class inheriting from mlir::ConversionPattern\<orchestra::TransferOp\>. This is the correct and only viable approach for this task due to the requirement of generating an scf.for loop.  
**Table 1: Comparison of Rewrite Strategies for orchestra.transfer Lowering**

| Criterion | Imperative C++ (ConversionPattern) | Declarative (DRR/TDRR) | Rationale for this Task |
| :---- | :---- | :---- | :---- |
| **Expressiveness** | **High.** Can create arbitrary IR structures, including control flow (scf.for), regions, and new blocks. | **Limited.** Primarily designed for DAG-to-DAG rewrites. Poor support for creating operations with regions or complex control flow.11 | The need to generate a tiled loop (scf.for) makes high expressiveness a mandatory requirement. |
| **Boilerplate Code** | **Moderate.** Requires C++ class definition, matchAndRewrite method, and explicit builder calls. | **Low.** Rules are defined concisely in TableGen, significantly reducing C++ boilerplate for simple patterns.11 | While DRR is less verbose for applicable patterns, its limitations make it unsuitable here. The C++ boilerplate is a necessary trade-off for the required functionality. |
| **Readability** | **Good.** Logic is contained within a single C++ method. Complex logic can be factored into helper functions. | **Excellent.** For simple patterns, the source and result DAGs are immediately clear. | The C++ implementation will be clear and readable, as the logic follows a standard pattern: setup, loop creation, loop body implementation, finalization. |
| **Maintainability** | **High.** Full power of C++ for abstraction and error handling. Easy to debug with standard tools. | **Moderate.** Changes are simple if they fit the declarative model. Complex constraints or logic require falling back to NativeCodeCall, mixing paradigms.14 | The vendor-specific pass architecture ensures maintainability by isolating this C++ logic from other backends. |
| **Suitability** | **Excellent.** The PatternRewriter API is explicitly designed for this type of structural transformation. | **Poor.** Unsuitable for lowering a single operation into a loop nest. | The imperative C++ approach is the correct tool for this specific transformation. |

### **3.2. Corrected Step-by-Step Implementation**

This revised plan provides a concrete, step-by-step guide incorporating the architectural decisions and enhanced requirements.

#### **Step 1: Scaffolding the Pass and Pattern**

1. **Create New File:** Create the file orchestra-compiler/lib/Orchestra/Transforms/LowerOrchestraToXeGPU.cpp.  
2. **Define the Pass:** In the new file, define the structure for the new pass. This involves creating a class that derives from mlir::PassWrapper and implementing the runOnOperation method.  
   C++  
   \#**include** "mlir/Pass/Pass.h"  
   \#**include** "mlir/Dialect/SCF/IR/SCF.h"  
   \#**include** "mlir/Dialect/XeGPU/IR/XeGPU.h"  
   //... other necessary includes...

   namespace {  
   struct LowerOrchestraToXeGPUPass  
       : public mlir::PassWrapper\<LowerOrchestraToXeGPUPass,  
                                  mlir::OperationPass\<mlir::gpu::GPUFuncOp\>\> {  
     MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(LowerOrchestraToXeGPUPass)

     void runOnOperation() override;  
     void getDependentDialects(mlir::DialectRegistry \&registry) const override {  
       registry.insert\<mlir::scf::SCFDialect, mlir::xegpu::XeGPUDialect\>();  
     }  
     //... pass options can be added here...  
   };  
   } // namespace

3. **Define the Conversion Pattern:** Inside the same file, define the OpConversionPattern for orchestra::TransferOp.  
   C++  
   class TransferOpLowering  
       : public mlir::OpConversionPattern\<orchestra::TransferOp\> {  
   public:  
     using OpConversionPattern\<orchestra::TransferOp\>::OpConversionPattern;

     mlir::LogicalResult  
     matchAndRewrite(orchestra::TransferOp op, OpAdaptor adaptor,  
                     mlir::ConversionPatternRewriter \&rewriter) const override;  
   };

**Rationale:** This structure establishes the new, vendor-specific pass (LowerOrchestraToXeGPUPass) that will operate on functions within a GPU module (gpu::GPUFuncOp). It declares its dependency on the scf and xegpu dialects, ensuring they are loaded in the context.1

#### **Step 2: Pre-computation and Loop Generation**

In the matchAndRewrite method, perform the initial setup and create the outer loop.

1. **Extract Operands and Types:** Get the source and destination memrefs from the adaptor. Get their MemRefType to access shape, layout, and element type information.  
2. **Validate Operands:** Check that the memrefs are 2D and that their element types are supported (f32, f16, bf16). If not, emit an error and return failure().  
3. **Define Tiling Parameters:** Define the tile sizes. These can be hardcoded constants initially (e.g., 32x32) and later exposed as pass options.  
4. **Calculate Loop Bounds:** Determine the upper bound for the loop. For a 2D copy tiled along the outer dimension, the loop will iterate from 0 to shape with a step equal to the tile height.  
5. **Create the Loop:** Use rewriter.create\<mlir::scf::ForOp\> to generate the loop.  
   C++  
   // Inside matchAndRewrite...  
   auto loc \= op.getLoc();  
   auto src \= adaptor.getSource();  
   auto dst \= adaptor.getDest();  
   auto srcType \= src.getType().cast\<mlir::MemRefType\>();  
   auto dstType \= dst.getType().cast\<mlir::MemRefType\>();

   // AC-4, AC-5: Validate shape and type  
   if (srcType.getRank()\!= 2) {  
     return op.emitError("only 2D transfers are currently supported");  
   }  
   //... add validation for element type...

   int64\_t tileHeight \= 32;  
   int64\_t tileWidth \= 32; // Assuming fixed tile size for now

   auto zero \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, 0);  
   auto outerDimSize \= rewriter.create\<mlir::memref::DimOp\>(loc, src, 0);  
   auto step \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, tileHeight);

   auto forOp \= rewriter.create\<mlir::scf::ForOp\>(loc, zero, outerDimSize, step);

**Rationale:** This step correctly sets up the iteration space for the tiled copy. It explicitly fetches the dimension size at runtime using memref.dim, ensuring support for dynamically shaped memrefs.

#### **Step 3: Loop Body Implementation (API Deep Dive)**

This is the core of the lowering, where the xegpu operations are generated inside the loop.

1. **Set Insertion Point:** Use a ConversionPatternRewriter::InsertionGuard or rewriter.setInsertionPointToStart(forOp.getBody()) to start inserting ops inside the loop's body.  
2. Get Induction Variable: The loop induction variable (iv) represents the current offset in the outer dimension.  
   mlir::Value iv \= forOp.getInductionVar();  
3. **Create Tensor Descriptors (xegpu.create\_nd\_tdesc):** This is the most critical operation. It requires providing the base memref, dynamic offsets, and static shape/stride information. The xegpu dialect documentation indicates that it can accept a mix of dynamic values (offsets) and static attributes (const\_offsets, etc.).6  
   C++  
   // Inside the loop body  
   mlir::OpBuilder::InsertionGuard guard(rewriter);  
   rewriter.setInsertionPointToStart(forOp.getBody());

   mlir::Value iv \= forOp.getInductionVar();  
   auto tileShape \= mlir::ArrayRef\<int64\_t\>{tileHeight, tileWidth};

   // For source descriptor  
   // Dynamic offsets: \[iv, 0\]. The outer dim offset is the loop IV.  
   SmallVector\<mlir::OpFoldResult\> srcOffsets;  
   srcOffsets.push\_back(iv);  
   srcOffsets.push\_back(rewriter.getIndexAttr(0));

   // The result type for the descriptor. The shape is the tile shape.  
   auto tdescType \= mlir::xegpu::TensorDescType::get(  
       tileShape, srcType.getElementType(), srcType.getMemorySpace());

   auto srcTdesc \= rewriter.create\<mlir::xegpu::CreateNdDescOp\>(  
       loc, tdescType, src, srcOffsets);

   // For destination descriptor  
   SmallVector\<mlir::OpFoldResult\> dstOffsets;  
   dstOffsets.push\_back(iv);  
   dstOffsets.push\_back(rewriter.getIndexAttr(0));

   auto dstTdesc \= rewriter.create\<mlir::xegpu::CreateNdDescOp\>(  
       loc, tdescType, dst, dstOffsets);

   *Note on Strides:* The simplified example above assumes a contiguous layout. To satisfy **AC-4**, the code must extract the strides from srcType and dstType and pass them to the CreateNdDescOp builder. The builder has overloads that accept static and dynamic stride values.7  
4. **Load and Store Data (xegpu.load\_nd, xegpu.store\_nd):** Use the created descriptors to perform the tiled load and store. The data is loaded into an MLIR vector type.  
   C++  
   // The vector type must match the tile shape and element type.  
   auto vectorType \= mlir::VectorType::get(tileShape, srcType.getElementType());

   auto loadedVector \= rewriter.create\<mlir::xegpu::LoadNdOp\>(loc, vectorType, srcTdesc);  
   rewriter.create\<mlir::xegpu::StoreNdOp\>(loc, loadedVector, dstTdesc);

**Rationale:** This section provides precise, actionable C++ code that correctly utilizes the xegpu dialect APIs. It demonstrates how the loop induction variable is used to compute dynamic offsets for the tensor descriptors, which is the fundamental mechanism for tiling. It also clarifies the relationship between the tile shape, the descriptor type, and the intermediate vector type.

#### **Step 4: Synchronization and Finalization**

1. **Insert Fence:** After the scf.for loop, insert the xegpu.fence operation.  
   C++  
   // After the forOp  
   rewriter.create\<mlir::xegpu::FenceOp\>(loc, mlir::xegpu::FenceScope::Workgroup);

   **Rationale:** The Workgroup scope is the most appropriate choice here. A data transfer is typically a collaborative effort by all threads in a workgroup to move a block of data from global memory to workgroup (shared) memory. This fence acts as a barrier, ensuring that no thread in the workgroup proceeds to use the data in shared memory until all threads have finished their portion of the store.15 This prevents read-after-write hazards.  
2. **Erase Original Operation:** Finally, remove the original orchestra.transfer op, as it has been fully replaced.  
   C++  
   rewriter.eraseOp(op);  
   return mlir::success();

   **Rationale:** Using rewriter.eraseOp(op) is correct because orchestra.transfer has no results. If it had results, rewriter.replaceOp(op, newValues) would be used instead.

#### **Step 5: Pass Registration and Pipeline Integration**

1. **Populate Patterns:** In the runOnOperation method of the LowerOrchestraToXeGPUPass, set up the ConversionTarget and populate the RewritePatternSet. The target should mark orchestra.transfer as illegal and all the generated ops (scf.for, xegpu.\*, etc.) as legal.  
   C++  
   void LowerOrchestraToXeGPUPass::runOnOperation() {  
     mlir::ConversionTarget target(getContext());  
     target.addIllegalOp\<orchestra::TransferOp\>();  
     target.addLegalDialect\<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,  
                             mlir::scf::SCFDialect, mlir::xegpu::XeGPUDialect\>();

     mlir::RewritePatternSet patterns(\&getContext());  
     patterns.add\<TransferOpLowering\>(\&getContext());

     if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))  
       signalPassFailure();  
   }

2. **Register the Pass:** Create a public function to construct the pass and register it so it can be invoked from orchestra-opt. This is typically done in a separate Passes.h and Passes.cpp file.

### **3.3. Alternative Strategy: Progressive Lowering via memref.copy**

An alternative to the direct orchestra.transfer \-\> xegpu lowering is a more gradual, two-stage approach that leverages standard dialects.

1. **Stage 1: Lower to memref.copy:** A high-level, target-agnostic pass lowers orchestra.transfer to the standard memref.copy operation. This pass would be very simple, essentially a 1-to-1 replacement. The memref.copy operation is a generic abstraction for copying data between two memrefs of the same shape and element type, and it explicitly supports different memory layouts.17  
2. **Stage 2: Lower memref.copy to xegpu:** A separate, hardware-specific pass would then be responsible for lowering memref.copy to the tiled xegpu implementation. This second pass would contain all the tiling logic and xegpu-specific details described in Section 3.2.

**Pros:**

* **Modularity and Reusability:** This approach decouples the Orchestra dialect from the specific xegpu backend. The memref.copy to xegpu lowering pass would be a general utility that could be reused for any operation that produces a memref.copy, not just orchestra.transfer.  
* **Simplified Dialect Lowering:** The pass to lower the Orchestra dialect becomes trivial, improving its maintainability. It only needs to know about the standard memref dialect, not every possible hardware backend.

**Cons:**

* **Loss of Semantic Information:** The primary goal of OrchestraOS is "global, hardware-aware optimization." The orchestra.transfer operation may carry specific semantic hints (e.g., via attributes) about the transfer's priority, expected latency, or optimal tiling strategy. Lowering to a generic memref.copy could lose this information, forcing the backend pass to rely on generic heuristics that may be suboptimal.  
* **Increased Pass Ordering Complexity:** This approach introduces another step in the compilation pipeline, making the overall pass ordering more complex and potentially fragile.18

Recommendation:  
For the stated goals of the OrchestraOS project, the direct lowering approach (Section 3.2) is superior because it provides a direct path for propagating hardware-specific optimization information from the high-level IR to the low-level backend. The alternative memref.copy strategy is a valid design pattern but is better suited for compilers where maximum modularity is prioritized over deep, cross-layer optimization.

### **3.4. A Robust Testing Strategy**

The initial plan to create a single test file is a good start, but it is insufficient to validate the full scope of the enhanced requirements. A comprehensive test suite is essential to ensure correctness, prevent regressions, and build confidence in the implementation. The following test cases should be implemented within orchestra-compiler/tests/lower-transfer-xegpu.mlir, with each test encapsulated in a separate gpu.func.

**Table 2: Expanded lit Test Cases for orchestra.transfer Lowering**

| Test Case Name | Description | Key IR Features to Validate |
| :---- | :---- | :---- |
| test\_basic\_f32 | A simple transfer of a 2D, contiguous memref\<256x256xf32\>. | Baseline correctness of the generated scf.for loop and xegpu ops. |
| test\_f16\_and\_bf16 | Two separate tests for f16 and bf16 element types. | Correct type propagation to xegpu::TensorDescType and vector.type. Validates **AC-5**. |
| test\_strided\_source | Transfer from a memref\<256x256xf32, strided\<\>\>, representing a subview of a larger matrix. | Correct extraction and usage of source strides in xegpu.create\_nd\_tdesc. Validates **AC-4**. |
| test\_strided\_destination | Transfer to a memref\<256x256xf32, strided\<\>\>. | Correct extraction and usage of destination strides. Validates **AC-4**. |
| test\_transpose\_copy | Transfer from memref\<128x256xf32, strided\<\>\> to memref\<256x128xf32, strided\<\>\>. | Handles complex, non-contiguous layouts for both source and destination simultaneously. |
| test\_asymmetric\_tiling | Transfer a memref\<260x250xf32\> with a tile size of 32x32. | Correct loop bound calculation and handling of partial tiles at the boundaries. |
| test\_global\_to\_workgroup | Transfer from memref\<... \#gpu.address\_space\<global\>\> to memref\<... \#gpu.address\_space\<workgroup\>\>. | Correct handling of memory space attributes in types. Validates **AC-6**. |
| test\_dynamic\_dims | Transfer a memref\<?x?xf32\>. | Correct use of memref.dim for loop bounds and dynamic shape propagation to xegpu ops where necessary. |

Each test should use FileCheck to precisely validate the structure of the generated IR. This includes checking the loop bounds, the types of the created xegpu ops, and the arguments passed to them, ensuring that the implementation adheres to the specification under a variety of conditions.

#### **Referenzen**

1. Pass Infrastructure \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/PassManagement/](https://mlir.llvm.org/docs/PassManagement/)  
2. Passes \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Passes/](https://mlir.llvm.org/docs/Passes/)  
3. MLIR Part 2 \- Memory in MLIR \- Stephen Diehl, Zugriff am August 18, 2025, [https://www.stephendiehl.com/posts/mlir\_memory/](https://www.stephendiehl.com/posts/mlir_memory/)  
4. Data Types — oneDNN v3.10.0 documentation, Zugriff am August 18, 2025, [https://uxlfoundation.github.io/oneDNN/dev\_guide\_data\_types.html](https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html)  
5. mlir/docs/Dialects/GPU.md · 5082acce4fd3646d5760c02b2c21d9cd2a1d7130 · llvm-doe / llvm-project \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md)  
6. 'xegpu' Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/XeGPU/](https://mlir.llvm.org/docs/Dialects/XeGPU/)  
7. \[Mlir-commits\] \[mlir\] \[MLIR\]\[XeGPU\] make offsets optional for create\_nd\_tdesc (PR \#148335) \- Mailing Lists, Zugriff am August 18, 2025, [https://lists.llvm.org/pipermail/mlir-commits/2025-July/118880.html](https://lists.llvm.org/pipermail/mlir-commits/2025-July/118880.html)  
8. RFC Strided MemRef Proposal (a.k.a Views) \- Google Groups, Zugriff am August 18, 2025, [https://groups.google.com/a/tensorflow.org/g/mlir/c/MaL8m2nXuio/m/a\_v07o9yBwAJ](https://groups.google.com/a/tensorflow.org/g/mlir/c/MaL8m2nXuio/m/a_v07o9yBwAJ)  
9. 'x86vector' Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/X86Vector/](https://mlir.llvm.org/docs/Dialects/X86Vector/)  
10. llvm-project/mlir/docs/Dialects/GPU.md at main \- GitHub, Zugriff am August 18, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md)  
11. Table-driven Declarative Rewrite Rule (DRR) \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)  
12. PDLL \- PDL Language \- MLIR, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/PDLL/](https://mlir.llvm.org/docs/PDLL/)  
13. Table-driven Declarative Rewrite Rule (DRR) \- GitLab, Zugriff am August 18, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/DeclarativeRewrites.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/DeclarativeRewrites.md)  
14. Quickstart tutorial to adding MLIR graph rewrite, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)  
15. User Guide for AMDGPU Backend — LLVM 22.0.0git documentation, Zugriff am August 18, 2025, [https://llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html)  
16. What's the difference between CUDA shared and global memory? \- Stack Overflow, Zugriff am August 18, 2025, [https://stackoverflow.com/questions/14093692/whats-the-difference-between-cuda-shared-and-global-memory](https://stackoverflow.com/questions/14093692/whats-the-difference-between-cuda-shared-and-global-memory)  
17. 'memref' Dialect \- MLIR \- LLVM, Zugriff am August 18, 2025, [https://mlir.llvm.org/docs/Dialects/MemRef/](https://mlir.llvm.org/docs/Dialects/MemRef/)  
18. The MLIR Transform Dialect \- arXiv, Zugriff am August 18, 2025, [https://arxiv.org/html/2409.03864v2](https://arxiv.org/html/2409.03864v2)