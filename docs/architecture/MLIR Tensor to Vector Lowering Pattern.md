

# **A Canonical Approach to Lowering Tensor Operations to Vector Intrinsics in MLIR**

## **Executive Report: The Canonical Tensor-to-Vector Lowering Pattern**

The canonical pattern for lowering a high-level, tensor-based operation (e.g., rock.gemm) to a low-level, vector-based hardware intrinsic (e.g., amdgpu.mfma) is a structured, four-stage process. This pattern is necessitated by the fundamental semantic and type-system gap between abstract memory representations (tensor) and concrete hardware register representations (vector) within the Multi-Level Intermediate Representation (MLIR) framework. A direct, one-to-one replacement of the tensor operation is infeasible; instead, the compiler must generate explicit Intermediate Representation (IR) to manage the data movement from memory to registers, perform the computation, and write the result back to memory.

The four essential stages of this canonical lowering pattern are:

1. **Tiling and Loop Generation:** The high-level operation is first decomposed into a nest of loops, typically using the scf.for operation. This loop nest iterates over the tensor operands in fixed-size blocks, or "tiles." The dimensions of these tiles are chosen specifically to match the operational dimensions of the target hardware intrinsic. This tiling strategy transforms a single large computation into a series of smaller, manageable computations that fit within the hardware's register files and execution units.  
2. **Memory-to-Register Data Movement:** Inside the body of the innermost loop, vector.load operations are generated. These operations explicitly transfer a tile of data from the source tensors (which represent data in memory) into newly created vector-typed SSA values. These vector values are the MLIR representation of data held in hardware SIMD registers.  
3. **Hardware Intrinsic Invocation:** With the data for a given tile now loaded into vectors, the target hardware intrinsic can be invoked. For example, an amdgpu.mfma operation is created, taking the vectors loaded in the previous step as its source operands. The operation performs a fused multiply-accumulate on the register-resident data. The resulting vector is typically carried as an iteration argument (iter\_arg) through the reduction loop, accumulating the result of each tile computation.  
4. **Register-to-Memory Data Persistence:** Once the reduction loop completes, the final accumulated vector holds the result for the corresponding output tile. A vector.store operation is then generated to write this vector value from its virtual register representation back into the appropriate slice of the destination tensor in memory.

This pattern is more than a mere sequence of IR generation steps; it is the physical manifestation in the IR of crossing a critical abstraction boundary within the compiler. The process makes the implicit hardware behavior of a "load-compute-store" cycle an explicit and optimizable part of the compiler's representation. High-level tensor operations are defined mathematically, free from the constraints of physical hardware. In contrast, low-level vector intrinsics are thin wrappers around specific machine instructions that operate on a finite set of registers. By forcing the generation of explicit loops and memory operations, MLIR's progressive lowering philosophy ensures that the compiler makes no unsafe assumptions about data locality or movement. This explicitness allows subsequent optimization passes to analyze and transform the data movement strategy—for instance, by introducing software prefetching or applying memory layout transformations—before final code generation, which is a cornerstone of building high-performance, reliable compilers.

## **Conceptual Foundations: Bridging the Memory-Register Divide in MLIR**

The persistent "tensor-to-vector type mismatch" error that often arises during compiler development is not a superficial bug but a direct consequence of MLIR's foundational design philosophy: progressive lowering through a hierarchy of abstractions. Understanding this hierarchy is essential to grasping why the canonical pattern of explicit looping and data movement is the correct and necessary solution for converting tensor-based operations to vector intrinsics.

### **The Abstraction Hierarchy: Tensor, MemRef, and Vector**

MLIR provides several core types to represent structured data, each with distinct semantics tailored to a specific level of abstraction in the compilation pipeline.

* **The tensor Type:** At the highest level of abstraction, the tensor type represents a multi-dimensional array as an abstract, immutable value. Operations on tensors are defined in a purely mathematical sense, detached from the concerns of memory layout, pointers, or aliasing. A tensor value does not have an explicit memory address; it is a pure computational object that enables powerful, high-level optimizations like algebraic simplification and operator fusion in a side-effect-free environment. The rock.gemm operation, which consumes and produces tensors, exists at this abstract level.  
* **The memref Type:** As compilation proceeds toward a concrete hardware target, the abstract tensor representation must be mapped to physical memory. The memref (memory reference) type serves as this crucial bridge. A memref represents a pointer to a region of memory, augmented with metadata describing its shape, strides, and layout. It is a mutable, buffer-like type that makes memory access explicit and introduces the possibility of side effects.  
* **The vector Type:** The vector type exists at a still lower level of abstraction, representing data that is held directly in hardware registers—specifically, SIMD (Single Instruction, Multiple Data) or SIMT (Single Instruction, Multiple Thread) registers. A vector is effectively a "virtual register," an in-flight value that can be directly manipulated by a single machine instruction. Operations like amdgpu.mfma are defined to operate on vector types precisely because they are thin wrappers around hardware instructions that operate on the GPU's register files, not on main memory.

The core of the lowering problem is that a tensor represents data *in memory* (abstractly), while a vector represents data *in registers*. A hardware instruction cannot directly operate on a tensor any more than a CPU's ALU can directly add two arrays stored in RAM without first executing load instructions to bring their elements into registers.

### **Progressive Lowering as a Guiding Principle**

The journey from rock.gemm to amdgpu.mfma is a classic example of MLIR's progressive lowering methodology. The compiler's task is to systematically transform high-level, target-agnostic representations into progressively more concrete, target-specific ones. The type mismatch error is a deliberate design feature that signals a level of abstraction is being crossed. The lowering pattern's purpose is not to "fix" this mismatch but to *implement the semantics of the transition*.

This design places a "burden of proof" on the compiler engineer. Unlike systems that might implicitly handle data movement, MLIR's strict type system requires the lowering pattern to explicitly prove that the transition from the memory domain to the register domain is being handled correctly and safely. A high-level tensor operation carries no information about how its data is laid out in memory or how it should be moved. A low-level vector intrinsic has strict requirements that its operands reside in registers. An implicit conversion would hide immense complexity and could introduce subtle bugs related to memory access patterns or out-of-bounds reads. By forcing the creation of loops and vector.load/store operations, MLIR compels the compiler pass to make an explicit statement in the IR: "I will iterate over the tensor in this specific tile order (scf.for), I will load this exact slice of data (vector.load with specific indices), I will compute on it, and I will store it back to this precise location (vector.store)." This explicit "proof" makes the transformation verifiable, debuggable, and, most importantly, optimizable, which is a cornerstone of building reliable, high-performance compilation systems.

## **Anatomy of the Lowering Pattern: A Code-Driven Deconstruction**

The C++ code provided in the source material implements the complete, sophisticated transformation from a rock.gemm operation to a tiled loop nest utilizing the amdgpu.mfma intrinsic. Each component of the mlir::OpConversionPattern plays a critical role in correctly and efficiently bridging the abstraction gap. This section presents the full implementation and deconstructs each stage of the process in detail.

C++

\#**include** "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"  
\#**include** "mlir/Dialect/Arith/IR/Arith.h"  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
\#**include** "mlir/Dialect/Vector/IR/VectorOps.h"  
\#**include** "mlir/IR/Builders.h"  
\#**include** "mlir/IR/BuiltinTypes.h"  
\#**include** "mlir/Transforms/DialectConversion.h"  
// Assuming the existence of the Rock dialect and GemmOp definition.  
\#**include** "path/to/Orchestra/RockDialect.h"

namespace {  
// This OpConversionPattern provides the complete lowering logic for  
// converting a \`rock::GemmOp\` into a tiled loop nest that uses the  
// \`amdgpu::MFMAOp\` intrinsic.  
class GemmLoweringPattern : public mlir::OpConversionPattern\<rock::GemmOp\> {  
public:  
  using OpConversionPattern\<rock::GemmOp\>::OpConversionPattern;

  mlir::LogicalResult  
  matchAndRewrite(rock::GemmOp gemmOp, OpAdaptor adaptor,  
                  mlir::ConversionPatternRewriter \&rewriter) const override {  
    mlir::Location loc \= gemmOp.getLoc();  
    mlir::MLIRContext \*context \= getContext();

    //===------------------------------------------------------------------===//  
    // 1\. Define Tiling and Vectorization Parameters  
    //===------------------------------------------------------------------===//  
    // These parameters must be chosen based on the target amdgpu.mfma  
    // intrinsic. For this example, we assume a 32x32x2 f32 MFMA variant,  
    // which is common on CDNA-class hardware.  
    // M, N, K are the dimensions of the GEMM operation.  
    // The tile sizes mTile, nTile, kTile correspond to the dimensions  
    // handled by a single mfma instruction.  
    constexpr int64\_t mTileSize \= 32;  
    constexpr int64\_t nTileSize \= 32;  
    constexpr int64\_t kTileSize \= 2; // For f32, MFMA often has a small K dim.

    // Define the vector types that the target amdgpu.mfma intrinsic expects.  
    // These types are hardware-dependent.  
    auto f32Type \= rewriter.getF32Type();  
    auto vectorAType \= mlir::VectorType::get({mTileSize}, f32Type);  
    auto vectorBType \= mlir::VectorType::get({nTileSize}, f32Type);  
    auto vectorAccumulatorType \=  
        mlir::VectorType::get({mTileSize, nTileSize}, f32Type);

    //===------------------------------------------------------------------===//  
    // 2\. Get Operands and Tensor Shapes  
    //===------------------------------------------------------------------===//  
    mlir::Value matrixA \= adaptor.getMatrixA();  
    mlir::Value matrixB \= adaptor.getMatrixB();  
    mlir::Value matrixC \= adaptor.getMatrixC(); // The output tensor

    auto tensorAType \= matrixA.getType().cast\<mlir::RankedTensorType\>();  
    auto tensorBType \= matrixB.getType().cast\<mlir::RankedTensorType\>();  
    int64\_t dimM \= tensorAType.getShape();  
    int64\_t dimK \= tensorAType.getShape();  
    int64\_t dimN \= tensorBType.getShape();

    //===------------------------------------------------------------------===//  
    // 3\. Generate the Loop Nest for Tiling  
    //===------------------------------------------------------------------===//  
    // Create constant values for loop bounds and steps.  
    mlir::Value c0 \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, 0);  
    mlir::Value mBound \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, dimM);  
    mlir::Value nBound \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, dimN);  
    mlir::Value kBound \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, dimK);  
    mlir::Value mStep \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, mTileSize);  
    mlir::Value nStep \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, nTileSize);  
    mlir::Value kStep \= rewriter.create\<mlir::arith::ConstantIndexOp\>(loc, kTileSize);

    // We will build a 3-level loop nest: M, N, and K.  
    // The K loop is the innermost reduction loop.  
    auto outerLoop \= rewriter.create\<mlir::scf::ForOp\>(  
        loc, c0, mBound, mStep, /\*iterArgs=\*/mlir::ValueRange{matrixC},  
        \[&\](mlir::OpBuilder \&builder, mlir::Location loopLoc, mlir::Value m\_iv,  
            mlir::ValueRange mIterArgs) {  
          mlir::Value cIntermediate \= mIterArgs;  
          auto middleLoop \= builder.create\<mlir::scf::ForOp\>(  
              loopLoc, c0, nBound, nStep,  
              /\*iterArgs=\*/mlir::ValueRange{cIntermediate},  
              \[&\](mlir::OpBuilder \&builder, mlir::Location loopLoc,  
                  mlir::Value n\_iv, mlir::ValueRange nIterArgs) {  
                mlir::Value cInnerIntermediate \= nIterArgs;

                // Load the initial accumulator tile from the C tensor.  
                mlir::Value accVector \= builder.create\<mlir::vector::LoadOp\>(  
                    loopLoc, vectorAccumulatorType, cInnerIntermediate,  
                    mlir::ValueRange{m\_iv, n\_iv});

                // The innermost loop performs the reduction over the K dimension.  
                auto innerLoop \= builder.create\<mlir::scf::ForOp\>(  
                    loopLoc, c0, kBound, kStep,  
                    /\*iterArgs=\*/mlir::ValueRange{accVector},  
                    \[&\](mlir::OpBuilder \&builder, mlir::Location loopLoc,  
                        mlir::Value k\_iv, mlir::ValueRange kIterArgs) {  
                      mlir::Value currentAcc \= kIterArgs;

                      // Load a tile from matrix A.  
                      // Note: This is a simplified load. For a real MFMA, the data  
                      // might need to be loaded in a specific layout.  
                      mlir::Value vecA \= builder.create\<mlir::vector::LoadOp\>(  
                          loopLoc, vectorAType, matrixA,  
                          mlir::ValueRange{m\_iv, k\_iv});

                      // Load a tile from matrix B.  
                      mlir::Value vecB \= builder.create\<mlir::vector::LoadOp\>(  
                          loopLoc, vectorBType, matrixB,  
                          mlir::ValueRange{k\_iv, n\_iv});

                      // Call the amdgpu.mfma intrinsic.  
                      // Attributes must match the chosen instruction variant.  
                      auto mfmaResult \= builder.create\<mlir::amdgpu::MFMAOp\>(  
                          loopLoc, vectorAccumulatorType, vecA, vecB, currentAcc,  
                          /\*m=\*/builder.getI32IntegerAttr(mTileSize),  
                          /\*n=\*/builder.getI32IntegerAttr(nTileSize),  
                          /\*k=\*/builder.getI32IntegerAttr(kTileSize),  
                          /\*blocks=\*/builder.getI32IntegerAttr(1),  
                          /\*cbsz=\*/builder.getI32IntegerAttr(0),  
                          /\*abid=\*/builder.getI32IntegerAttr(0),  
                          /\*blgp=\*/nullptr,  
                          /\*reducePrecision=\*/nullptr,  
                          /\*negateA=\*/nullptr,  
                          /\*negateB=\*/nullptr,  
                          /\*negateC=\*/nullptr);

                      builder.create\<mlir::scf::YieldOp\>(loopLoc,  
                                                        mfmaResult.getDestC());  
                    });  
                mlir::Value finalAcc \= innerLoop.getResult(0);

                // Store the final accumulated vector back into the C tensor.  
                mlir::Value updatedC \= builder.create\<mlir::vector::StoreOp\>(  
                    loopLoc, finalAcc, cInnerIntermediate,  
                    mlir::ValueRange{m\_iv, n\_iv});

                builder.create\<mlir::scf::YieldOp\>(loopLoc, updatedC);  
              });  
          builder.create\<mlir::scf::YieldOp\>(loopLoc, middleLoop.getResult(0));  
        });

    //===------------------------------------------------------------------===//  
    // 4\. Finalize the Lowering  
    //===------------------------------------------------------------------===//  
    // Replace the original rock.gemm operation with the result of the loop nest.  
    rewriter.replaceOp(gemmOp, outerLoop.getResult(0));  
    return mlir::success();  
  }  
};  
} // anonymous namespace

### **3.1 Tiling and Loop Generation**

The first step in the lowering process is to decompose the large, abstract matrix multiplication into a series of smaller, fixed-size computations that match the capabilities of the target hardware. This technique is known as tiling. The amdgpu.mfma instruction is designed to operate on small matrices (e.g., 32×32) whose elements are distributed across the registers of a GPU wavefront. The tiling parameters in the code (mTileSize \= 32, nTileSize \= 32, kTileSize \= 2\) are chosen to align with a specific variant of this instruction.

The pattern then generates a three-level scf.for loop nest to iterate over these tiles. The outer loops iterate over the M and N dimensions of the GEMM, while the innermost loop performs the reduction over the K dimension. The loop bounds (mBound, nBound, kBound) are set to the full dimensions of the source tensors, and the step sizes (mStep, nStep, kStep) are set to the tile dimensions. This configuration ensures that each iteration of the loop nest processes exactly one tile of the overall computation. The use of iterArgs is critical: the output tensor C is passed as a loop-carried variable through the outer M and N loops, allowing each tile computation to update it. Similarly, the accumulator vector is passed through the innermost K loop, enabling the chain of multiply-accumulate operations.

### **3.2 Data Movement (Memory-to-Register)**

Inside the loop body, the abstract tensor data must be materialized into vector registers. The vector dialect's vector.load operation is used for this purpose. It is a straightforward operation designed for loading a contiguous slice of memory into a vector.

The crucial step is the construction of the indices parameter for each vector.load call. For example, the load from matrixA uses mlir::ValueRange{m\_iv, k\_iv}. Here, m\_iv and k\_iv are the induction variables from the outer M-loop and the inner K-loop, respectively. This directly connects the control flow of the loops to the data flow from memory. As the loops iterate, the induction variables change, causing the vector.load operation to read successive tiles from the source tensors. This explicit indexing is what ensures that each iteration processes the correct slice of the input data, bridging the gap between the loop's iteration space and the tensor's memory space.

### **3.3 Hardware Intrinsic Invocation**

With the data for a given tile now loaded into vector-typed SSA values (vecA and vecB), the target hardware intrinsic can be invoked. The amdgpu.mfma operation in MLIR is a high-level wrapper that corresponds to a family of Matrix Fused Multiply-Add instructions on AMD GPUs.

The operation's attributes are critical for selecting the correct hardware instruction variant. The attributes m, n, and k directly configure the dimensions of the matrix multiplication to be performed by the instruction. These must align precisely with the tile sizes chosen in the first step and the vector types created for the loaded data. The operands vecA, vecB, and currentAcc must be of specific mlir::VectorType dictated by the hardware architecture. The currentAcc operand serves as the input accumulator, and the operation returns the updated accumulator value. This result is then yielded (scf.yield) to become the input accumulator for the next iteration of the K-loop, forming the reduction chain.

The following table provides a conceptual map that clarifies how the high-level semantics of the rock.gemm operation are progressively decomposed and mapped onto the tiled execution model required by the amdgpu.mfma intrinsic.

**Table 3.3.1: Mapping rock.gemm Semantics to amdgpu.mfma Tiled Execution**

| Conceptual Layer | Component | Description | MLIR Representation |
| :---- | :---- | :---- | :---- |
| High-Level Op | rock.gemm operands | A, B, and C matrices. | \`tensor\<M×K×f32\>, tensor\<K×N×f32\>, tensor\<M×N×f32\> |
| Tiling Strategy | Outer Loops | Iterate over the entire M, N space in tile-sized steps. | scf.for %m\_tile \=..., scf.for %n\_tile \=... |
|  | Inner Loop | Reduction loop over the K dimension for each tile. | scf.for %k\_tile \=... iter\_args(%acc \=...) |
| Data Movement | Load A-tile | Load a tile of matrix A from memory into a vector. | %vec\_a \= vector.load %A\[%m\_tile, %k\_tile\]... |
|  | Load B-tile | Load a tile of matrix B from memory into a vector. | %vec\_b \= vector.load %B\[%k\_tile, %n\_tile\]... |
| Hardware Intrinsic | amdgpu.mfma | Perform a matrix multiply-accumulate on the register data. | %acc\_next \= amdgpu.mfma %vec\_a, %vec\_b, %acc... |
| Data Persistence | Store C-tile | Store the final accumulated vector back to the C matrix tile. | vector.store %acc\_final, %C\[%m\_tile, %n\_tile\]... |

### **3.4 Data Persistence (Register-to-Memory)**

After the innermost reduction loop over the K-dimension completes, the finalAcc value holds the fully computed result for the output tile corresponding to the current m\_iv and n\_iv indices. This data, which exists only in virtual registers, must be written back to memory to persist the result.

This is accomplished using the vector.store operation, which is the inverse of vector.load. The call builder.create\<mlir::vector::StoreOp\>(loc, finalAcc, cInnerIntermediate, mlir::ValueRange{m\_iv, n\_iv}) generates the operation. The finalAcc vector is the data to be stored, cInnerIntermediate is the output tensor passed down through the loops, and the indices m\_iv and n\_iv specify the top-left corner of the destination tile in the output tensor. This operation completes the data flow for a single tile, updating the output tensor in memory with the computed result before the outer loops proceed to the next tile.

## **Framework Integration: Applying the Pattern within a Dialect Conversion Pass**

The GemmLoweringPattern contains the core transformation logic, but it must be executed within a controlled environment to ensure the entire IR is converted systematically and correctly. MLIR's Dialect Conversion framework is the standard mechanism for managing such transformations, guaranteeing that all operations are legalized according to a defined target.

### **The Dialect Conversion Framework**

This framework consists of three main components that work in concert to drive the lowering process:

1. **mlir::ConversionTarget**: This class defines the "goal" state of the IR after the pass completes. The engineer specifies which dialects and operations are considered Legal. Any operation not marked as legal is considered illegal and must be converted by a pattern.  
2. **mlir::RewritePatternSet**: This is a collection of rewrite patterns, such as the GemmLoweringPattern, that the framework can apply to transform illegal operations into legal ones.  
3. **mlir::applyPartialConversion / applyFullConversion**: These driver functions orchestrate the conversion. They iteratively apply patterns from the RewritePatternSet to all illegal operations within a given scope (e.g., a function) until the entire IR conforms to the rules defined in the ConversionTarget.

The process is implemented within a class that inherits from mlir::OperationPass. The configuration involves first defining the target: the source rock dialect is marked Illegal, forcing the framework to find a pattern to eliminate any rock operations. The target dialects—amdgpu, scf, vector, and arith—are marked Legal, as they constitute the desired output representation. Next, the GemmLoweringPattern is added to a RewritePatternSet. Finally, applyPartialConversion is called to execute the transformation.

This framework is more than just a tool for applying rewrites; it is a powerful mechanism for enforcing compiler correctness and promoting a modular design. A naive compiler might apply rules haphazardly, potentially leaving the IR in an inconsistent, "hybrid" state where, for example, a high-level tensor operation feeds a low-level vector operation—an illegal combination. The ConversionTarget establishes a strict contract: "At the end of this pass, no operations from the rock dialect shall exist." The conversion driver acts as the engine to enforce this contract. If it cannot find a pattern to legalize an illegal operation, the pass fails, preventing the generation of invalid code. This enforcement mechanism encourages engineers to design passes that perform complete abstraction transitions (e.g., from Dialect A to Dialect B), leading to a more modular, understandable, and maintainable pass pipeline where each stage has a clear, verifiable goal.

## **Production-Grade Considerations and Strategic Architectural Recommendations**

While the canonical pattern provides a complete solution for the core lowering problem, a production-grade compiler must address additional real-world complexities and should be architected for long-term maintainability and extensibility.

### **5.1 Handling Dynamic Shapes and Imperfect Tiling**

The example pattern assumes that the tensor dimensions are static and perfectly divisible by the tile sizes. In practice, tensors often have dynamic dimensions (represented by ? in the type), or their static dimensions may not be a multiple of the tile size. A naive implementation of the pattern would read or write out of bounds in these "remainder" or "tail" iterations.

The correct and robust way to handle this in MLIR is with masking. The vector dialect provides a suite of operations specifically for this purpose:

* **vector.create\_mask**: This operation can be used to create a mask vector (e.g., \`vector\<32×i1\>) based on the dynamic boundaries of the tensor. Elements within the valid bounds are set to 1 (true), and those outside are set to 0 (false).  
* **vector.maskedload / vector.maskedstore**: These operations perform a load or store, but only for the vector lanes where the corresponding element in the mask vector is true. For disabled lanes, a pass-through value (for load) or no-op (for store) is used.

Incorporating masking into the lowering pattern makes it robust to arbitrary tensor shapes, a critical requirement for any general-purpose compiler. The logic involves calculating the valid iteration space for each tile and generating a mask that is passed to the memory operations to prevent illegal access.

### **5.2 Strategic Recommendation: Adopt the linalg Dialect as a "Narrow Waist"**

Writing a direct, monolithic lowering from a high-level, domain-specific dialect like rock to a low-level hardware dialect like amdgpu is a valid but strategically suboptimal approach. This methodology creates a tightly coupled, brittle pipeline that is difficult to maintain, extend, and optimize.

A more robust and powerful architecture involves using the linalg (Linear Algebra) dialect as a common intermediate representation. The linalg dialect is designed to be the "narrow waist" of the MLIR tensor compiler stack. It represents structured tensor operations like matrix multiplication (linalg.matmul) in a generic, abstract form using indexing\_maps and iterator\_types. A superior lowering pipeline would be a two-stage process:

1. **rock.gemm \-\> linalg.matmul**: This first pass is a simple, high-level conversion. The domain-specific rock.gemm operation is rewritten as a generic linalg.matmul operation. This pass is easy to write and maintain, as it remains entirely within the abstract tensor domain.  
2. **linalg.matmul \-\> Tiled Loops \+ amdgpu.mfma**: This second stage leverages the extensive, pre-existing ecosystem of transformations that operate on the linalg dialect. MLIR provides powerful, reusable patterns and passes for tiling, fusing, and vectorizing linalg operations. By targeting linalg, a compiler can reuse this entire infrastructure instead of re-implementing it from scratch.

This architectural shift provides immense "compiler engineering leverage." It is not just a good design choice; it is a strategic decision that acts as a force multiplier for the entire compiler team by fundamentally changing the scaling properties of the development effort. Consider a compiler with *M* front-end operations (e.g., gemm, conv, attention) and *N* hardware back-ends (e.g., AMD GPU, NVIDIA GPU, CPU). A monolithic, direct-lowering approach requires implementing and maintaining *M \\times N* lowering pathways. The complexity is multiplicative. Adding a new back-end requires writing *M* new lowerings; adding a new front-end operation requires writing *N* new lowerings.

By introducing linalg as a narrow waist, the problem is decomposed. The team writes *M* lowerings from the front-end dialects to linalg and *N* lowerings from linalg to the back-ends. The complexity becomes additive (*M \+ N*). This architectural shift dramatically reduces the work required to extend the compiler. The return on investment for each new linalg-based optimization or lowering is amortized across all *M* front-end operations and all *N* back-ends. This massively increases development velocity and makes the entire compiler project more scalable and sustainable in the long term, representing the most critical strategic recommendation for building modern compilers with MLIR.