

# **Report on the Idiomatic Construction of xegpu::CreateNdDescOp in MLIR**

## **1\. Executive Summary: Resolving the TypedValue Mismatch**

The C++ build error encountered during the lowering of the orchestra.transfer operation is a direct consequence of the MLIR C++ API's robust, statically-enforced type system. The error, no matching function for call to... build(...), indicates a C++ type mismatch where a generic mlir::Value is provided for an argument that requires a specific mlir::TypedValue\<mlir::MemRefType\>. This is not a bug or an API flaw, but rather a deliberate design choice that leverages the C++ type system to enforce the invariants of an MLIR operation at compile time, preventing the construction of invalid Intermediate Representation (IR).

The canonical and idiomatic solution is to perform an explicit, checked downcast on the mlir::Value handle using the MLIR framework's own Run-Time Type Information (RTTI) utilities. Specifically, the mlir::cast template function should be used to convert the generic mlir::Value into the required mlir::TypedValue\<mlir::MemRefType\>. This action asserts the programmer's knowledge that the underlying SSA value is indeed a memref, satisfying the strict contract of the xegpu::CreateNdDescOp builder method.

The following annotated code snippet demonstrates the correct implementation pattern, which resolves the build error and adheres to MLIR development best practices.

C++

// Assumes 'builder', 'loc', 'source', 'iv', and 'jv' are defined.  
// 'source' is the mlir::Value representing the source memref.

// 1\. Retrieve the MLIR type from the generic value handle.  
auto sourceType \= source.getType().cast\<mlir::MemRefType\>();

// 2\. Define the result type for the tensor descriptor operation.  
SmallVector\<int64\_t\> tileShape \= {16, 16};  
auto tensorDescType \= xegpu::TensorDescType::get(tileShape, sourceType.getElementType(),  
                                               /\*array\_length=\*/1,  
                                               /\*boundary\_check=\*/true,  
                                               xegpu::MemorySpace::Global,  
                                               /\*sg\_map=\*/mlir::Attribute());

// 3\. Prepare the dynamic offsets as a SmallVector of OpFoldResult.  
SmallVector\<OpFoldResult\> offsets;  
offsets.push\_back(iv);  
offsets.push\_back(jv);

// 4\. \*\*\* THE IDIOMATIC SOLUTION \*\*\*  
// The builder's contract requires a TypedValue\<MemRefType\> to ensure type  
// safety at compile time. Use mlir::cast to perform a checked downcast from  
// the generic mlir::Value to the specific C++ handle type. This cast  
// asserts in debug builds that 'source' is of the expected IR type.  
auto typedSource \= mlir::cast\<mlir::TypedValue\<mlir::MemRefType\>\>(source);

// 5\. Create the operation using the correctly-typed C++ handle.  
// This call now matches the generated builder signature and will compile.  
auto source\_tdesc \= builder.create\<xegpu::CreateNdDescOp\>(  
    loc,  
    tensorDescType,  
    typedSource,  
    offsets);

This report will further deconstruct the underlying principles of the MLIR C++ API, the role of TableGen in defining these strict contracts, and the proper use of MLIR's RTTI system, providing a comprehensive understanding that extends beyond this specific issue.

## **2\. The C++ API Dichotomy: Understanding mlir::Value and mlir::TypedValue**

The root of the compilation failure lies in the fundamental design of MLIR's C++ object model, which carefully distinguishes between generic, type-erased handles and specific, statically-typed handles for IR constructs. This design mirrors a core principle inherited from the broader LLVM project: providing both maximum flexibility for generic transformations and maximum type safety for specific, constrained operations. Understanding the roles of mlir::Value and mlir::TypedValue is essential for any developer working with the MLIR C++ API.

### **2.1. mlir::Value: The Generic SSA Value Handle**

In the MLIR system, every SSA value—whether it is an argument to a Block or the result of an Operation—is represented in the C++ API by the mlir::Value class.1 This class is a lightweight, value-semantic wrapper around a pointer to the internal IR object (

ValueImpl).3

From the perspective of the C++ type system, mlir::Value is a "type-erased" handle. It can point to any kind of SSA value in the IR, such as a memref\<16x16xf32\>, a vector\<4xi32\>, or a custom orchestra.graph type. This generality is a powerful feature, as it allows compiler passes and transformation utilities to be written generically. An algorithm that, for example, analyzes use-def chains can operate on mlir::Value objects without needing to know their specific MLIR types at compile time.

To determine the underlying IR type of a mlir::Value, one must query it at runtime using the getType() method, which returns an instance of mlir::Type.3 This

mlir::Type object can then be inspected or cast to a more specific type class (e.g., MemRefType, VectorType) to make decisions. This runtime nature is fundamental to the flexibility of a multi-level, extensible IR framework like MLIR.4

### **2.2. mlir::TypedValue\<T\>: Enforcing Compile-Time Type Safety**

While mlir::Value provides flexibility, many operations have strict semantic requirements for their operands. For example, the xegpu.create\_nd\_tdesc operation is defined to operate on a memory region, so its source operand must be a memref.5 To enforce such invariants as early as possible—at compile time rather than at runtime—MLIR provides the

mlir::detail::TypedValue\<Ty\> struct template.6

TypedValue\<Ty\> is a C++ class that inherits from mlir::Value but adds a crucial piece of static information: it represents an SSA Value that is known by the programmer to correspond to the MLIR IR type Ty.6 For instance,

mlir::TypedValue\<mlir::MemRefType\> is a C++ type that can only represent an SSA value whose MLIR type is memref.

This specialization allows dialect authors to define C++ builder methods with highly specific and safe signatures. When an operation's definition specifies that an operand must be a memref, the auto-generated builder can require a mlir::TypedValue\<mlir::MemRefType\> as its C++ argument. This shifts the burden of type verification from a runtime check inside a pass to a standard C++ compile-time check, producing a clear build error if the contract is violated. This is precisely the situation encountered in the user's query.

The following table provides a clear comparison between these two critical C++ handle types.

| Feature | mlir::Value | mlir::TypedValue\<mlir::MemRefType\> |
| :---- | :---- | :---- |
| **C++ Type** | class Value | struct TypedValue\<MemRefType\> : public Value |
| **Represents** | Any SSA value in the IR. | An SSA value known to be a memref. |
| **Type Knowledge** | Runtime (dynamic), via getType(). | Compile-time (static), enforced by the C++ type system. |
| **getType() Return** | mlir::Type | Returns mlir::Type, but can be safely cast to mlir::MemRefType. |
| **Typical Use Case** | Generic pass logic, operand lists, graph traversals. | Type-safe builders, APIs for memref-specific operations. |
| **Conversion Method** | Downcast via mlir::cast or mlir::dyn\_cast. | Implicit upcast to mlir::Value. |

### **2.3. The Role of LLVM-Style RTTI**

A critical aspect of this system is how conversions between mlir::Value and mlir::TypedValue are handled. The LLVM project, for reasons of performance, code size, and flexibility, disables standard C++ RTTI features like dynamic\_cast and typeid.7 Instead, it provides a custom, opt-in, and highly efficient RTTI system based on template functions:

isa\<\> for type checking, cast\<\> for checked downcasting, and dyn\_cast\<\> for safe downcasting that can fail gracefully.8

This custom RTTI is the mechanism that connects the generic mlir::Value world with the specific mlir::TypedValue world. An attempt to use static\_cast or dynamic\_cast to convert a mlir::Value to a mlir::TypedValue will fail because the relationship between these types is managed by LLVM's RTTI logic, not by standard C++ virtual tables.7 The

mlir::cast and mlir::dyn\_cast functions are aware of the internal structure of MLIR's class hierarchies and can correctly perform the required checks and conversions. Therefore, when faced with a type mismatch like the one in the query, the correct tool is always one of MLIR's own casting utilities.

## **3\. Anatomy of an Op Builder: Deconstructing the CreateNdDescOp Signature**

The specific C++ signature of the xegpu::CreateNdDescOp builder method is not arbitrary; it is the direct result of a declarative, model-driven process that forms the foundation of dialect definition in MLIR. Understanding this process reveals why the build error occurs and why it is a feature of a well-designed API.

### **3.1. From TableGen to C++: The Genesis of the API**

MLIR dialects are primarily defined using TableGen, a configuration language used throughout LLVM to generate boilerplate C++ code from high-level, declarative specifications.9 Operation definitions, including their names, arguments, results, attributes, and semantic traits, are written in

.td (TableGen definition) files. The primary specification for a dialect's operations is known as the Operation Definition Specification (ODS).11

When the MLIR project is built, a tool named mlir-tblgen processes these .td files. For each operation definition, it generates a corresponding C++ class (e.g., xegpu::CreateNdDescOp) that inherits from mlir::Op. Crucially, it also generates a set of convenient build methods for constructing instances of this operation using an mlir::OpBuilder.9

The signatures of these generated build methods are determined by the constraints placed on the operation's arguments in the .td file. If an operand in the .td file is defined with a specific type constraint—for example, specifying that it must be a MemRef of any element type—mlir-tblgen will translate this IR-level constraint into a C++-level type requirement. It achieves this by using mlir::TypedValue\<mlir::MemRefType\> in the generated C++ function signature instead of the generic mlir::Value. This automated process ensures that the C++ API for creating an operation directly reflects the semantic rules of that operation. The source of this generation for the XeGPU dialect is located in files like mlir/include/mlir/Dialect/XeGPU/IR/XeGPUOps.td.14

### **3.2. A Contract of Types: Analyzing the Build Error**

The build error message provides a clear diagnosis of the issue:  
error: no matching function for call to ‘mlir::xegpu::CreateNdDescOp::build(...)’  
note: candidate: ‘static void mlir::xegpu::CreateNdDescOp::build(..., mlir::TypedValue\<mlir::MemRefType\>,...)  
note: no known conversion for argument 4 from ‘mlir::Value’ to ‘mlir::TypedValue\<mlir::MemRefType\>’  
This sequence confirms the process described above. The dialect author for XeGPU specified in the .td file that the source operand for create\_nd\_tdesc must be a memref. Consequently, mlir-tblgen generated a build method whose C++ signature enforces this contract by requiring a mlir::TypedValue\<mlir::MemRefType\>.

The user's code attempts to call this method with a variable of type mlir::Value. In C++, an implicit conversion from a derived class to a base class (an upcast, e.g., from TypedValue to Value) is safe and allowed. However, an implicit conversion from a base class to a derived class (a downcast, from Value to TypedValue) is not allowed because it is a potentially unsafe operation—not every Value is a memref. The C++ compiler correctly identifies that there is no implicit conversion rule and flags this as an error.

This compile-time failure is the intended behavior. It acts as a powerful safeguard, preventing a developer from accidentally passing an incorrect type of value to an operation builder. It forces the developer to explicitly assert their knowledge of the value's type, leading to more robust and correct compiler passes. The error is not a problem to be worked around, but a signal to be correctly addressed by fulfilling the API's contract.

## **4\. The Canonical Solution and Implementation**

With a clear understanding of the underlying API design, the path to the correct solution becomes evident. It involves using the appropriate MLIR RTTI tool to satisfy the C++ type contract of the generated builder method. This approach is not only correct but is also the established and idiomatic pattern seen throughout the MLIR codebase.

### **4.1. The Correct Tool: mlir::cast and mlir::dyn\_cast**

MLIR provides two primary template functions for downcasting its C++ value handles, both defined in llvm/Support/Casting.h and pervasively used in the project.8

* **mlir::cast\<T\>(Value v)**: This function performs a *checked* downcast. It is used when the programmer is certain that the Value v is of the target type T. In debug builds (-DNDEBUG is not defined), mlir::cast contains an assertion that verifies the type correctness at runtime. If the cast is invalid, the program will terminate with an assertion failure, immediately pinpointing the incorrect assumption. In release builds, this assertion is compiled out, and the cast typically resolves to a static\_cast, incurring minimal to no performance overhead. This is the ideal tool for the situation described in the query, as the source value is known to be a memref.  
* **mlir::dyn\_cast\<T\>(Value v)**: This function performs a *safe* downcast. It checks if the Value v is of the target type T. If the cast is valid, it returns the value cast to T. If the cast is invalid, it returns a null or empty instance of T (e.g., a null TypedValue). This is used when the type of a value is uncertain and the code needs to handle different types of values in different ways, typically via an if statement checking the result of the dyn\_cast.

For the task of creating the xegpu::CreateNdDescOp, where the source operand is guaranteed by the logic of the lowering pass to be a memref, mlir::cast is the most appropriate and idiomatic choice. It clearly communicates the programmer's intent and provides a valuable safety net during development.

### **4.2. Corrected Code with In-Depth Annotation**

The following C++ code provides the complete and correct implementation for creating the xegpu::CreateNdDescOp within a lowering pass. Each step is annotated to explain its purpose and its role in the overall construction.

C++

// This code is assumed to be within a method of an MLIR pass, where an  
// mlir::OpBuilder \`builder\`, an mlir::Location \`loc\`, and the mlir::Value  
// handles \`source\`, \`iv\`, and \`jv\` are available.

// 1\. Obtain the MLIR type of the source memref. This is a runtime query on  
//    the generic mlir::Value handle. We use.cast\<\>() here on the mlir::Type  
//    itself to get a C++ handle to the specific MemRefType.  
auto sourceType \= source.getType().cast\<mlir::MemRefType\>();  
if (\!sourceType) {  
  // It is good practice to handle potential type mismatches gracefully.  
  // In a real compiler, this would likely emit an error.  
  return;  
}

// 2\. Define the properties of the tensor descriptor we want to create. This  
//    involves specifying the shape of the tile, the element type (derived  
//    from the source memref), and other hardware-specific parameters for XeGPU.  
SmallVector\<int64\_t\> tileShape \= {16, 16};  
auto tensorDescType \= xegpu::TensorDescType::get(  
    tileShape, sourceType.getElementType(),  
    /\*array\_length=\*/1,  
    /\*boundary\_check=\*/true,  
    xegpu::MemorySpace::Global,  
    /\*sg\_map=\*/mlir::Attribute());

// 3\. Prepare the dynamic offsets for the tile. The loop induction variables  
//    \`iv\` and \`jv\` are mlir::Value handles, which can be directly placed  
//    into an OpFoldResult container. OpFoldResult is a variant type that  
//    can hold either a mlir::Value or an mlir::Attribute.  
SmallVector\<OpFoldResult\> offsets;  
offsets.push\_back(iv);  
offsets.push\_back(jv);

// 4\. \*\*\* THE SOLUTION: Perform the necessary C++ type cast. \*\*\*  
// The builder for CreateNdDescOp requires a TypedValue\<MemRefType\> as its  
// source operand to satisfy its generated C++ signature. We use mlir::cast  
// to perform a checked downcast from the generic mlir::Value handle \`source\`  
// to the specific C++ handle \`typedSource\`. This line is the key to  
// resolving the build error.  
auto typedSource \= mlir::cast\<mlir::TypedValue\<mlir::MemRefType\>\>(source);

// 5\. Create the xegpu.create\_nd\_tdesc operation. The builder call now  
//    perfectly matches one of the auto-generated signatures. The C++ compiler  
//    is satisfied, and the operation is correctly inserted into the IR.  
auto source\_tdesc \= builder.create\<xegpu::CreateNdDescOp\>(  
    loc,  
    tensorDescType,  
    typedSource, // Pass the correctly-typed C++ handle  
    offsets);

// The \`source\_tdesc\` variable is now an mlir::Value that can be used in  
// subsequent operations, such as an xegpu.load\_nd.

### **4.3. Precedent in the MLIR Codebase**

This casting pattern is not an obscure trick but the standard, canonical way of handling such type requirements in MLIR. A review of the LLVM/MLIR source tree reveals numerous examples that validate this approach.

For instance, in the mlir/lib/Conversion/VectorToXeGPU/VectorToXeGPU.cpp file, patterns for lowering vector operations to the XeGPU dialect can be found. While some examples may obtain a correctly-typed handle directly from another operation that returns a TypedValue, the principle of passing a typed handle to the builder is upheld.15

More explicit examples of this exact casting pattern are present in other parts of the codebase. For example, the mesh dialect transformations shown in mlir/doxygen/Mesh/Transforms/Transforms.cpp\_source.html demonstrate the use of cast\<TypedValue\<IndexType\>\> to convert a generic Value (or an OpFoldResult first cast to a Value) into the specific TypedValue required by a function or builder.16 These examples serve as definitive evidence that

mlir::cast\<TypedValue\<...\>\> is the intended and idiomatic solution for the problem at hand.

## **5\. Advanced Topics and Alternative Methodologies**

While using the generated OpBuilder methods with the correct types is the preferred approach, MLIR provides lower-level mechanisms for operation construction. It is also important to consider the stability of the API in question.

### **5.1. Low-Level Construction via OperationState**

The mlir::OpBuilder create methods are convenient wrappers around a more fundamental construction mechanism. Any operation in MLIR can be created by manually populating an mlir::OperationState object and passing it to builder.createOperation(state). This approach offers maximum flexibility and can be considered an "escape hatch" when the generated builders are not suitable for a particular dynamic use case.

This method would also resolve the build error, as the OperationState::addOperands method accepts a ValueRange, which is a generic container for mlir::Value objects. This bypasses the C++ type-checking of the specific builder method.

The process would be as follows:

1. **Initialize OperationState**: Create an instance of mlir::OperationState, providing the Location and the full operation name as a string ("xegpu.create\_nd\_tdesc").  
2. **Add Operands**: Add the source value and the offsets values to the state using state.addOperands(...).  
3. **Add Result Types**: Add the tensorDescType to the state using state.addTypes(...).  
4. **Create Operation**: Call builder.createOperation(state) to get the final operation.

C++

// Alternative low-level construction (for illustration)  
mlir::OperationState state(loc, "xegpu.create\_nd\_tdesc");  
state.addOperands(source);  
state.addOperands({iv, jv});  
state.addTypes(tensorDescType);  
mlir::Operation \*op \= builder.createOperation(state);  
auto source\_tdesc \= op-\>getResult(0);

While this code compiles and produces the correct IR, it comes with significant trade-offs. It is far more verbose, less readable, and crucially, it sacrifices the compile-time type safety that the generated builders provide. A typo in the operation name string would lead to a runtime error instead of a compile-time one. This method should be reserved for situations where operations are being constructed with a structure that is only known at runtime, and it is not the recommended solution for this particular problem.

### **5.2. API Conventions and Stability**

The user's query touched upon whether this issue could be due to recent API changes or subtle conventions in MLIR v20. The analysis confirms that this is not the case. The design pattern of using mlir::Value for generic handles, mlir::TypedValue for type-safe APIs, and mlir::cast to bridge the two has been a stable and fundamental convention in MLIR for many versions.

This pattern is a cornerstone of MLIR's C++ API design, providing a balance between the dynamic nature of the IR and the static safety of C++. It is highly unlikely to change. Recent patches to the XeGPU dialect, such as the one noted in a June 2024 commit log for patching dynamic descriptor creation, relate to the internal implementation logic of the operations, not their public C++ builder interfaces.17 Therefore, the solution presented here is robust and expected to be correct for the foreseeable future.

## **6\. Conclusion and Engineering Best Practices**

The investigation into the xegpu::CreateNdDescOp build error reveals a core principle of MLIR C++ development: the framework's API is intentionally designed to leverage the C++ type system to enforce IR correctness. The build error is not an obstacle but a feature, a compile-time guardrail that prevents the creation of semantically invalid operations. The resolution lies in understanding and correctly using the tools provided by MLIR to manage the two parallel type systems at play: the dynamic types of the IR itself and the static types of the C++ host environment.

Based on this comprehensive analysis, the following engineering best practices are recommended for developers working within the MLIR C++ ecosystem:

1. **Trust the Compiler Error**: When an OpBuilder method demands a mlir::TypedValue\<T\>, it is a definitive signal that the underlying MLIR operation has a specific type constraint defined in its ODS specification. The error is a request to fulfill this API contract.  
2. **Use MLIR RTTI for Casting**: The idiomatic and correct way to bridge the C++ type gap between the generic mlir::Value and a specific mlir::TypedValue\<T\> is to use MLIR's own RTTI utilities. Use mlir::cast\<T\>() when the type is known and the assumption should be asserted. Use mlir::dyn\_cast\<T\>() when the type is uncertain and requires a runtime check with a fallback path. Never use C++'s built-in static\_cast or dynamic\_cast for MLIR value or operation handles.  
3. **Inspect Generated Headers for Ground Truth**: When facing ambiguity about an operation's C++ API, the most reliable source of information is the auto-generated header file for the dialect (e.g., XeGPUOps.h.inc in the build directory). This file contains the exact C++ signatures of all available builder methods, providing the ground truth for the API contract that must be satisfied.  
4. **Consult Upstream Examples**: The LLVM/MLIR project repository is the ultimate reference for idiomatic API usage. The mlir/lib/Conversion/ and mlir/test/Dialect/ directories are invaluable resources containing numerous examples of how core developers write lowering passes and interact with dialect APIs. These examples often provide clear precedents for solving common problems.

By internalizing these principles, a compiler engineer can navigate the MLIR C++ API more effectively, writing code that is not only correct but also safer, more maintainable, and aligned with the framework's core design philosophy.
