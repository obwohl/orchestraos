

# **Technical Specification and Implementation Guide: Lowering orchestra.transfer to the XeGPU Dialect**

## **Executive Summary**

This report provides a comprehensive architectural review and an enhanced implementation plan for the LowerOrchestraToXeGPU pass within the OrchestraOS Compiler project. The initial development plan represents a valid starting point for a proof-of-concept; however, it is critically insufficient for creating a robust, production-ready compiler component capable of handling the complexities of modern Artificial Intelligence and Machine Learning (AI/ML) workloads.

The analysis identifies four primary areas requiring significant enhancement:

1. **Critical Feature Gaps:** The original specification overlooks three fundamental requirements for a data transfer operation in a high-performance computing context. It fails to account for non-contiguous memory layouts (strided memrefs), mixed-precision floating-point data types (f16, bf16), and explicit GPU memory space management (global vs. workgroup). The absence of these features would render the pass incompatible with common optimization strategies like tiling and prevent it from generating high-performance code for contemporary GPU hardware.  
2. **API and Best Practice Corrections:** The proposed implementation plan references deprecated MLIR APIs (specifically vector.splat) and employs coding patterns that, while functional, do not align with current state-of-the-art MLIR development practices for safety and clarity. Adherence to the latest API versions and idiomatic patterns is essential for the long-term maintainability and stability of the compiler.  
3. **Architectural Refinement:** The proposed imperative C++ implementation strategy is appropriate for the complexity of the task, which involves control flow generation. However, the plan can be significantly improved by adopting a hybrid, builder-centric architectural pattern. This approach separates the high-level lowering logic from the low-level details of MLIR operation creation, resulting in a more readable, maintainable, and reusable codebase.  
4. **Insufficient Testing Strategy:** The reliance on a single test file is inadequate for validating the full scope of a data transfer operation. A comprehensive, multi-faceted test suite is required to ensure correctness across the newly introduced features and to protect against future regressions.

The recommendations detailed in this report are designed to guide the developer in transforming the LowerOrchestraToXeGPU pass from a basic prototype into a production-quality component. Implementing these enhancements will ensure the pass is correct, robust, and capable of generating efficient code for the diverse set of scenarios encountered in real-world AI/ML models, thereby de-risking and advancing the OrchestraOS project's support for Intel GPU targets.

## **Revised Requirements Specification**

To ensure the LowerOrchestraToXeGPU pass is a functional and valuable component of the OrchestraOS compiler, its requirements must be expanded to reflect the realities of modern GPU programming and AI/ML workloads. The initial specification addresses only the most trivial case and is therefore incomplete. This revised specification establishes a comprehensive set of acceptance criteria for a production-ready feature.

### **2.1 Feature Title**

Enable and Validate a Production-Ready LowerOrchestraToXeGPU Pass

### **2.2 Goal / User Story**

As a compiler developer, I want to implement and validate a robust LowerOrchestraToXeGPU pass and its comprehensive test suite, ensuring that the orchestra.transfer operation can be correctly lowered to the xegpu dialect for a wide range of real-world scenarios, including non-contiguous memory, mixed-precision data types, and transfers between different GPU memory spaces.

### **2.3 Enhanced Acceptance Criteria**

The successful completion of this task is defined by the following criteria:

1. The initial test file, orchestra-compiler/tests/lower-transfer-xegpu.mlir.disabled, must be renamed to orchestra-compiler/tests/lower-transfer-xegpu-base.mlir, and the test must pass.  
2. The OrchestraOS compiler project must build successfully with all new and re-enabled tests. The full test suite, executed via the check-orchestra CMake target, must complete and exit with a code of 0, indicating all tests have passed.  
3. **Strided Memory Support:** The pass must correctly lower orchestra.transfer operations where the source or destination memref operand has a non-trivial memory layout. This includes memref types with a strided\<...\> layout attribute specifying static or dynamic strides and offsets. This capability is essential for supporting tiled data access patterns, which are fundamental to optimizing performance in ML workloads.1 The lowering must correctly extract stride and offset information from the  
   memref type and propagate it to the appropriate fields of the xegpu.create\_nd\_tdesc operation.2  
4. **Mixed-Precision Support:** The pass must be generic with respect to its supported floating-point data types. It must correctly lower orchestra.transfer operations for memrefs with f32, f16, and bf16 element types. Modern GPU architectures, particularly those with specialized hardware like Intel's Xe Matrix Extensions (XMX), achieve peak performance by operating on lower-precision data types.3 A compiler pass that is not data-type-generic would fail to unlock this performance and would be unsuitable for state-of-the-art model compilation.4 The implementation must be polymorphic, avoiding hardcoded type assumptions.  
5. **GPU Memory Space Awareness:** The pass must correctly handle orchestra.transfer operations that move data between different GPU memory spaces. Specifically, it must recognize and preserve the memorySpace attribute on memref types, particularly for transfers involving default global memory and workgroup memory (also known as shared memory). Efficient GPU algorithms frequently rely on staging data from slow global memory to fast, on-chip workgroup memory to improve data locality and reduce latency.5 The lowering must propagate the memory space identifier (e.g.,  
   \#gpu.address\_space\<workgroup\>) from the memref type to the xegpu.create\_nd\_tdesc operation to ensure the hardware accesses the correct memory hierarchy.7  
6. **Idiomatic API Usage:** The final C++ implementation must exclusively use modern, idiomatic MLIR v20 APIs. This includes avoiding deprecated operations, employing safe type-casting paradigms, and utilizing builder patterns that are consistent with the latest LLVM/MLIR development practices. This ensures the code is forward-compatible, maintainable, and less prone to subtle bugs.

## **Revised & Enhanced Implementation Plan**

This section provides a detailed, authoritative guide for implementing the LowerOrchestraToXeGPU pass. It corrects and expands upon the initial proposal, incorporating best practices, addressing the enhanced requirements, and recommending a superior architectural pattern for long-term maintainability.

### **3.1 Prerequisite: API & Syntax Validation**

Before implementing the core logic, it is crucial to validate and correct the foundational API assumptions made in the initial plan. Using incorrect or outdated APIs can lead to subtle bugs, verifier failures, or maintenance issues.

#### **3.1.1 vector.splat vs. vector.broadcast**

The proposal to use mlir::vector::SplatOp to create a vector for the xegpu.update\_nd\_offset operation is incorrect. In recent versions of MLIR, vector.splat has been deprecated in favor of the more explicit and powerful vector.broadcast operation.8

* **Correction:** The vector.splat operation should be replaced with vector.broadcast. The latter more clearly expresses the semantic intent of broadcasting a scalar value to fill the elements of a vector.  
* **Idiomatic C++ Implementation:**  
  C++  
  // Get the required vector type, e.g., vector\<1xindex\>  
  auto vectorType \= mlir::VectorType::get({1}, rewriter.getIndexType());

  // Create the broadcast operation  
  auto vectorOffset \= rewriter.create\<mlir::vector::BroadcastOp\>(  
      loc, vectorType, scalarOffset);

* **Justification:** Using vector.broadcast aligns the implementation with modern MLIR standards, ensuring forward compatibility and improving the semantic clarity of the generated IR. It correctly models the transformation from a 0-dimensional value (a scalar) to a 1-dimensional value (a vector).8

#### **3.1.2 Safe Type Casting for xegpu.create\_nd\_tdesc**

The proposal to use mlir::cast on the memref operand is contextually appropriate but warrants a discussion of best practices. The mlir::cast utility performs a checked cast that will trigger a runtime assertion if the Value is not of the expected type. This is safe within a ConversionPattern's matchAndRewrite method because the pattern matching engine has already guaranteed that the root operation is of the correct type (orchestra.transfer), and that operation's verifier should have already guaranteed that its operands are of the correct type (MemRefType).

* **Validation:** The use of mlir::cast is acceptable in this context. However, developers should be aware of its semantics. In situations where a Value could be one of several types, mlir::dyn\_cast should be used instead, as it returns a nullptr on failure rather than asserting.  
* **Recommended C++ Pattern:**  
  C++  
  // Inside TransferOpLowering::matchAndRewrite  
  auto transferOp \= llvm::cast\<orchestra::TransferOp\>(op);  
  mlir::Value source \= transferOp.getSource();

  // The builder for xegpu::CreateNdDescOp expects a Value. If a more specific  
  // C++ type like TypedValue\<MemRefType\> were required by a helper function,  
  // this cast would be safe due to the op's verified properties.  
  // auto typedSource \= mlir::cast\<mlir::TypedValue\<mlir::MemRefType\>\>(source);

  // Pass the source Value directly to the builder.  
  rewriter.create\<mlir::xegpu::CreateNdDescOp\>(loc, /\*resultTypes=\*/..., source,...);

#### **3.1.3 xegpu.fence Scope Selection**

The initial choice of mlir::xegpu::FenceScope::Workgroup is a reasonable and safe default. This scope ensures that memory operations are visible to all threads within a GPU workgroup, which is the necessary guarantee for producer-consumer patterns that use workgroup (shared) memory.

* **Analysis:** The xegpu dialect provides several fence scopes to allow for fine-grained control over memory visibility. While Workgroup is often correct, other scopes might be applicable in more advanced scenarios. For this implementation, Workgroup provides the strongest and safest guarantee for coordinating threads after a data transfer.  
* **Recommendation:** The implementation should use mlir::xegpu::FenceScope::Workgroup. A comment should be added to the C++ code justifying this choice, noting that it provides the necessary memory visibility for subsequent computations within the workgroup that depend on the transferred data.  
  C++  
  // After the loop performing the tiled transfer.  
  // A workgroup-level fence ensures that all writes from the transfer are  
  // visible to all threads in the workgroup before proceeding.  
  rewriter.create\<mlir::xegpu::FenceOp\>(loc, mlir::xegpu::FenceScope::Workgroup);

### **3.2 Step-by-Step Task Breakdown (Revised)**

The implementation should proceed in a phased manner, starting with the simplest case and progressively adding support for the more complex features outlined in the revised requirements. This iterative approach simplifies debugging and ensures each feature is built on a solid foundation.

1. **Re-enable and Isolate the Base Test:**  
   * Rename orchestra-compiler/tests/lower-transfer-xegpu.mlir.disabled to orchestra-compiler/tests/lower-transfer-xegpu-base.mlir.  
   * Run the check-orchestra target and confirm that the test fails as expected. This validates the testing infrastructure.  
2. **Implement the "Happy Path" Lowering:**  
   * Focus exclusively on the simplest case: a orchestra.transfer of a contiguous, f32-element memref in the default (global) memory space.  
   * Implement the correct descriptor lifecycle:  
     * **Outside the loop:** Create a single base tensor descriptor using rewriter.create\<mlir::xegpu::CreateNdDescOp\>. This descriptor represents the entire source/destination memref.  
     * **Inside an scf.for loop:** In each loop iteration, create a new, updated descriptor for the current tile using rewriter.create\<mlir::xegpu::UpdateNdOffsetOp\>. The base descriptor is passed as an operand, along with a tile-specific offset.  
     * Use the tile-specific descriptor for the xegpu.load\_nd or xegpu.store\_nd operation within the loop.  
   * Apply the API corrections from section 3.1, specifically using vector.broadcast to create the offset vector for UpdateNdOffsetOp.  
   * Insert an xegpu.fence after the loop.  
   * **Goal:** Achieve a passing result for the lower-transfer-xegpu-base.mlir test.  
3. **Architect for Generality: Introduce Strided Layout Support:**  
   * Refactor the pass to correctly handle memref operands that have a StridedLayoutAttr.  
   * **Logic:**  
     1. In the matchAndRewrite function, get the MemRefType of the orchestra.transfer operand.  
     2. Check if the layout is a StridedLayoutAttr using memrefType.getLayout().dyn\_cast\<StridedLayoutAttr\>().  
     3. If it is, the xegpu.create\_nd\_tdesc operation must be constructed differently. This operation has dedicated operands for dynamic shapes, strides, and offsets.2  
     4. The pass must extract the stride and offset values from the MemRefType. For dynamic strides/offsets (represented by ShapedType::kDynamic), the pass must find the corresponding SSA values that define them, which are typically passed as operands to the function or created by ops like memref.extract\_strided\_metadata. These SSA values must be passed to the CreateNdDescOp builder.  
   * **Create New Test:** Add a new test file, lower-transfer-xegpu-strided.mlir, containing test cases with various static and dynamic strided layouts to validate this logic.  
4. **Generalize Data Types: Introduce Mixed-Precision Support:**  
   * Refactor the pass to be agnostic to the element type of the memref.  
   * **Logic:**  
     1. Avoid any hardcoded C++ types like mlir::Float32Type.  
     2. Extract the element type directly from the operand's MemRefType using memrefType.getElementType().  
     3. This mlir::Type object should be used throughout the pass when creating new operations or types. The xegpu dialect operations are designed to be polymorphic and will correctly handle various floating-point and integer types. The key is to ensure this polymorphism is not broken by hardcoded assumptions in the lowering pass.  
   * **Create New Test:** Add lower-transfer-xegpu-mixedprec.mlir with transfers of memref\<...xf16\> and memref\<...xbf16\> to ensure type propagation is correct.  
5. **Incorporate Memory Hierarchy: Add GPU Memory Space Awareness:**  
   * Modify the pass to correctly handle memrefs located in different GPU memory spaces.  
   * **Logic:**  
     1. Extract the memory space attribute from the MemRefType using memrefType.getMemorySpace().  
     2. The xegpu.create\_nd\_tdesc builder has a parameter for specifying the memory space. The extracted attribute must be passed to this builder.  
     3. If no memory space is specified on the memref type, it defaults to global memory. The pass must handle this default case correctly.  
   * **Create New Test:** Add lower-transfer-xegpu-workgroup.mlir to test transfers from a default-space memref to a memref\<..., \#gpu.address\_space\<workgroup\>\>, verifying that the resulting CreateNdDescOp has the correct memory space attribute.  
6. **Final Verification:**  
   * After all features are implemented and their respective tests are passing, run the entire check-orchestra suite one final time to ensure that no regressions have been introduced in other parts of the compiler.

### **3.3 Deep Dive: Architectural and Strategic Considerations**

A robust implementation requires more than just correct code; it requires sound architectural decisions that promote maintainability and extensibility.

#### **3.3.1 Comparative Analysis of Rewrite Strategies**

The choice of a rewrite strategy is a critical architectural decision in an MLIR-based compiler. The two primary options are imperative C++ patterns and declarative, TableGen-based rules (DRR/PDLL).

* **Imperative C++ (ConversionPattern):** This approach involves subclassing mlir::ConversionPattern and implementing the logic in the matchAndRewrite method.  
  * **Strengths:** It offers maximum flexibility and programmatic control. This is essential for complex transformations that are not simple graph-to-graph replacements. Specifically, generating control flow structures like scf.for loops, which contain regions, is straightforward in C++ but difficult or impossible in declarative formats.9 Furthermore, imperative logic is required to inspect  
    memref types, extract layout attributes, and perform the calculations needed to derive loop bounds and tile offsets.  
  * **Weaknesses:** This approach can be verbose, requiring significant boilerplate code to create and configure each MLIR operation. A monolithic matchAndRewrite function can quickly become difficult to read and maintain.  
* **Declarative Rewrite Rules (DRR/PDLL):** This approach defines patterns in .td files using a specialized syntax.10  
  * **Strengths:** DRR is exceptionally concise and readable for simple DAG-to-DAG rewrites.10 It eliminates boilerplate and makes the transformation's intent immediately obvious from the pattern definition.  
  * **Weaknesses:** DRR is fundamentally unsuited for this specific lowering task. Its limitations include poor support for matching or generating operations with regions (like scf.for) and an inability to express the complex, algorithmic logic needed to process strided layout attributes or calculate loop structures.10 The lowering of  
    orchestra.transfer is an algorithmic conversion, not a simple structural replacement.  
* **Conclusion:** A pure DRR/PDLL approach is not a viable option for this task. The requirement to generate a tiled loop structure (scf.for) necessitates the power and flexibility of an imperative C++ ConversionPattern.

#### **3.3.2 Alternative Strategy: A Hybrid, Builder-Centric Approach**

While the core logic must be imperative, the implementation can be vastly improved by adopting a hybrid architectural pattern that combines the control of C++ with the clarity of a declarative-style builder. This is a state-of-the-art practice for writing complex MLIR passes.

* **Proposal:** Instead of implementing all logic within the TransferOpLowering::matchAndRewrite method, create a dedicated helper class or a set of free functions (e.g., in a XeGPUTileUtils.h file) that act as builders for common xegpu code patterns. The main rewrite pattern then becomes a high-level orchestrator that calls these builders.  
* **Example Structure:**  
  C++  
  // In a helper file, e.g., XeGPUTileUtils.h  
  namespace mlir {  
  namespace orchestra {

  /// A helper structure to return the results of a tiled load generation.  
  struct TiledLoadResult {  
    scf::ForOp loop; // The generated scf.for loop.  
    Value lastTileData; // The data loaded in the final iteration.  
  };

  /// Generates a tiled load from a source memref using XeGPU operations.  
  /// Encapsulates the creation of descriptors, loops, and load operations.  
  FailureOr\<TiledLoadResult\> createTiledLoad(  
      PatternRewriter \&rewriter, Location loc, Value sourceMemRef,  
      /\* other params like tile sizes \*/);

  } // namespace orchestra  
  } // namespace mlir

  // In LowerOrchestraToXeGPU.cpp  
  LogicalResult TransferOpLowering::matchAndRewrite(...) const {  
    //... initial setup and checks...

    auto tiledLoadResult \= createTiledLoad(rewriter, loc, source,...);  
    if (failed(tiledLoadResult)) {  
      return failure();  
    }

    //... logic to store the loaded data...

    rewriter.eraseOp(op);  
    return success();  
  }

* **Advantages of the Hybrid Approach:**  
  * **Separation of Concerns:** The main rewrite pattern is responsible for the high-level logic: matching the operation, deciding if a tiled lowering is appropriate, and orchestrating the calls to the builders. The builder functions encapsulate the complex, low-level details of creating the specific sequence of xegpu and scf operations.  
  * **Improved Readability and Maintainability:** The matchAndRewrite method becomes significantly shorter and easier to understand, as its logic is expressed at a higher level of abstraction. Debugging is simplified because the complex op-creation logic is isolated in a single, well-defined location.  
  * **Reusability:** The builder functions, such as createTiledLoad, are decoupled from the specific orchestra.transfer operation. They can be reused by other passes in the future that may also need to generate tiled memory accesses for the XeGPU target.

This hybrid, builder-centric approach is the strongly recommended architecture. It represents a mature and scalable design pattern that leverages the strengths of the imperative framework while promoting code quality and long-term maintainability.

### **3.4 A Comprehensive Testing Strategy**

A robust compiler pass requires an equally robust and comprehensive test suite. A single test file is insufficient to validate the expanded requirements. The testing strategy must be modular, with dedicated tests for each distinct feature. These tests will be written in .mlir files and use lit with FileCheck to verify the correctness of the generated IR.12 The following table outlines the recommended test suite.

| Proposed Test File | Objective | Key IR Features to Test | FileCheck Verification Points |
| :---- | :---- | :---- | :---- |
| lower-transfer-xegpu-base.mlir | Validates the fundamental lowering for a simple, contiguous f32 memref. | orchestra.transfer memref\<...xf32\> | Correct scf.for loop bounds, xegpu.create\_nd\_tdesc outside loop, xegpu.update\_nd\_offset inside loop, and a final xegpu.fence. |
| lower-transfer-xegpu-strided.mlir | Ensures correctness for transfers involving non-contiguous memory. | memref\<...xf32, strided\<\[?,?\], offset:?\>\> | xegpu.create\_nd\_tdesc is created with the correct (potentially dynamic) stride and offset SSA value operands. |
| lower-transfer-xegpu-mixedprec.mlir | Verifies support for f16 and bf16 data types. | memref\<...xf16\>, memref\<...xbf16\> | All generated xegpu ops, descriptors, and loaded vector types correctly use f16 or bf16 element types. |
| lower-transfer-xegpu-workgroup.mlir | Tests transfers to and from workgroup (shared) memory. | memref\<..., \#gpu.address\_space\<workgroup\>\> | The xegpu.create\_nd\_tdesc operation has the correct memory space attribute corresponding to workgroup. |
| lower-transfer-xegpu-edgecases.mlir | Validates behavior with zero-sized and single-element transfers. | memref\<0x...xf32\>, memref\<1x...xf32\> | The pass handles zero-sized inputs gracefully (e.g., generates no loops or memory operations) and correctly handles non-tiled single-element cases. |

This structured testing approach ensures that each critical feature is validated in isolation, simplifying debugging and providing clear documentation of the pass's capabilities. It establishes a strong foundation of regression testing that will be invaluable as the OrchestraOS compiler continues to evolve.

#### **Referenzen**

1. RFC Strided MemRef Proposal (a.k.a Views) \- Google Groups, Zugriff am August 20, 2025, [https://groups.google.com/a/tensorflow.org/g/mlir/c/MaL8m2nXuio/m/a\_v07o9yBwAJ](https://groups.google.com/a/tensorflow.org/g/mlir/c/MaL8m2nXuio/m/a_v07o9yBwAJ)  
2. 'xegpu' Dialect \- MLIR, Zugriff am August 20, 2025, [https://mlir.llvm.org/docs/Dialects/XeGPU/](https://mlir.llvm.org/docs/Dialects/XeGPU/)  
3. MLIR-based code generation for GPU tensor cores \- YouTube, Zugriff am August 20, 2025, [https://www.youtube.com/watch?v=3LLzHKeL2hs](https://www.youtube.com/watch?v=3LLzHKeL2hs)  
4. 8\. Lowering — TPU-MLIR 1.1 documentation \- SOPHGO, Zugriff am August 20, 2025, [https://doc.sophgo.com/sdk-docs/v23.05.01/docs\_latest\_release/docs/tpu-mlir/developer\_manual\_en/html/08\_lowering.html](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/developer_manual_en/html/08_lowering.html)  
5. Comco: A MLIR-Based Intermediate Representation for CUDA Kernel Fusion \- Department of Computer Science : University of Rochester, Zugriff am August 20, 2025, [https://ftp.cs.rochester.edu/\~sree/courses/csc-290-571/fall-2024/static/final-reports/CSC290NappoFinal.pdf](https://ftp.cs.rochester.edu/~sree/courses/csc-290-571/fall-2024/static/final-reports/CSC290NappoFinal.pdf)  
6. llvm-project/mlir/docs/Dialects/GPU.md at main \- GitHub, Zugriff am August 20, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/GPU.md)  
7. mlir/docs/Dialects/GPU.md · 5082acce4fd3646d5760c02b2c21d9cd2a1d7130 · llvm-doe / llvm-project \- GitLab, Zugriff am August 20, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Dialects/GPU.md)  
8. 'vector' Dialect \- MLIR, Zugriff am August 20, 2025, [https://mlir.llvm.org/docs/Dialects/Vector/](https://mlir.llvm.org/docs/Dialects/Vector/)  
9. 'scf' Dialect \- MLIR \- LLVM, Zugriff am August 20, 2025, [https://mlir.llvm.org/docs/Dialects/SCFDialect/](https://mlir.llvm.org/docs/Dialects/SCFDialect/)  
10. Table-driven Declarative Rewrite Rule (DRR) \- MLIR, Zugriff am August 20, 2025, [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)  
11. Table-driven Declarative Rewrite Rule (DRR) \- GitLab, Zugriff am August 20, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/DeclarativeRewrites.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/doe/mlir/docs/DeclarativeRewrites.md)  
12. Testing guide \- IREE, Zugriff am August 20, 2025, [https://iree.dev/developers/general/testing-guide/](https://iree.dev/developers/general/testing-guide/)