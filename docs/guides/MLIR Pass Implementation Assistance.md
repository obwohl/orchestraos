

# **Mastering Idiomatic MLIR for GPU Lowering: A Guide to Conversion Patterns and Pass Management**

## **Introduction: Mastering Idiomatic MLIR for Heterogeneous Lowering**

The task of compiling high-level abstractions to specialized, heterogeneous hardware represents a significant challenge in modern compiler engineering. Projects like OrchestraOS, which leverage MLIR to target architectures such as Intel GPUs via the xegpu dialect, operate at the frontier of this domain. The process of lowering, or translating, operations from a custom dialect to a hardware-specific one is where the architectural principles of MLIR become most critical for ensuring correctness, maintainability, and performance.1

The emergence of C++ compilation errors during the implementation of such lowering passes is a common and often instructive experience. These errors are rarely simple syntactical mistakes; instead, they frequently signal a mismatch between the developer's implementation and the idiomatic patterns established by the MLIR framework. Such issues provide a valuable opportunity to delve deeper into the core philosophies that underpin MLIR's design: the sophisticated interplay between static and dynamic typing in its value hierarchy, the dual C++ and in-IR representation of compile-time constants like enumerations, and the strict architectural requirements imposed by the parallel-first pass management system.

This report provides an exhaustive analysis of three specific C++ compilation challenges encountered during the development of a lowering pattern for an orchestra.transfer operation. It moves beyond simple fixes to offer a foundational understanding of the canonical MLIR patterns that resolve these issues. The subsequent sections will deconstruct each problem, explain the underlying design rationale within the MLIR framework, and provide robust, idiomatic C++ code solutions. The objective is not merely to resolve the immediate compilation failures but to equip the compiler developer with the principles needed to architect more effective and resilient MLIR passes for any complex lowering task.

## **Part I: Type-Safe Operand Handling in Conversion Patterns**

The first compilation error, a failure to convert mlir::Value to mlir::TypedValue\<mlir::MemRefType\>, highlights a fundamental aspect of the dialect conversion framework: how it manages types that are in a state of transition. Understanding the distinction between the generic mlir::Value and the type-safe mlir::TypedValue is key to correctly handling operands within a conversion pattern.

### **1.1 The MLIR Value Hierarchy: Value, OpResult, BlockArgument, and TypedValue**

At the core of MLIR's Intermediate Representation (IR) is a graph-like structure where operations are nodes and SSA values are the connecting edges.3 The C++ class that represents these edges is

mlir::Value. It is the fundamental, generic base class for any SSA value in the system. An instance of mlir::Value can be one of two concrete things: the result of an operation (mlir::OpResult) or an argument to a basic block (mlir::BlockArgument).4 The

mlir::Value class itself is intentionally designed to be generic; its getType() method returns an instance of mlir::Type, which can be any type registered in the system, from a built-in integer to a complex, dialect-specific tensor type.

To enhance type safety and improve the ergonomics of the C++ API, MLIR introduced mlir::TypedValue\<T\>. This class is a template-based wrapper around mlir::Value that provides compile-time guarantees about the underlying IR type.6 For example, a function signature that accepts a

mlir::TypedValue\<mlir::MemRefType\> communicates a clear requirement to the C++ compiler. This was a deliberate API evolution designed to reduce the prevalence of runtime checks and verbose getType().cast\<...\>() call chains, thereby making compiler code safer, more self-documenting, and easier to read. As noted in the design review for this feature, TypedValue allows an API to directly express what it expects or provides, shifting error detection from runtime to compile time where possible.6

### **1.2 The Role of OpAdaptor in Dialect Conversion**

The context in which an operand is accessed significantly affects its C++ representation. Within a standard mlir::RewritePattern, the generated OpAdaptor provides convenient, statically-typed accessors for an operation's operands. However, the situation is more complex within an mlir::OpConversionPattern, which is the context of the user's problem.

The OpConversionPattern is a specialized rewrite pattern designed to work within the dialect conversion framework.7 A key component of this framework is the

TypeConverter, which defines rules for mapping types from a source representation to a target one. When OpConversionPattern::matchAndRewrite is invoked, the OpAdaptor argument it receives does not provide direct access to the original operation's operands. Instead, it provides mlir::Value handles to the operands *after* they have been processed by the conversion driver. These operands may have already been legalized by other patterns or had their types changed by the TypeConverter.8

This dynamic nature is the source of the apparent type erasure. The dialect conversion framework cannot know at C++ compile time what the final, legalized type of any given operand will be. For instance, a memref\<16x16xf32\> might be converted to a memref\<16x16xf32, \#gpu.address\_space\<workgroup\>\>, or it might be completely unrolled into a set of scalar values, or it might be replaced by a packed vector type. To accommodate this boundless flexibility, the OpAdaptor in OpConversionPattern must return the most generic handle possible: mlir::Value. The framework intentionally erases the static C++ type information because the underlying IR type is mutable during the conversion process. The responsibility for verifying the type at a specific point in the lowering is thus shifted to the pattern writer, whose logic dictates what types are expected for the newly generated code.9 The implicit conversion from

mlir::Value to mlir::TypedValue fails because it would violate this fundamental design principle of the conversion framework.

### **1.3 The Canonical Solution: Safe Casting with dyn\_cast**

Given that the OpAdaptor provides a generic mlir::Value, the pattern writer must safely verify and convert it to the specific type required by subsequent op builders. The canonical and safest mechanism for this in the LLVM and MLIR ecosystem is mlir::dyn\_cast.

mlir::dyn\_cast is a templated casting operator that performs a runtime check on the type of the object being cast. If the mlir::Value holds a value of the target type (e.g., its getType() returns a MemRefType), dyn\_cast will return a valid, populated TypedValue. If the cast is invalid, it will return a null or empty value (in the case of TypedValue, an instance for which the boolean operator evaluates to false) without causing a program crash.11 This behavior is ideal for rewrite patterns, as it allows the pattern to gracefully fail by returning

mlir::failure() if it encounters an unexpected or unsupported IR structure.

The following code snippet demonstrates the correct, idiomatic solution to the user's problem:

C++

\#**include** "mlir/IR/BuiltinTypes.h"  
\#**include** "mlir/IR/Value.h"

// Inside the matchAndRewrite method of your ConversionPattern\<orchestra::TransferOp\>  
// LogicalResult matchAndRewrite(orchestra::TransferOp op,   
//                               orchestra::TransferOp::Adaptor adaptor,  
//                               ConversionPatternRewriter \&rewriter) const override {

  // The adaptor provides the \*converted\* source operand as a generic mlir::Value.  
  mlir::Value sourceOperand \= adaptor.getSource();

  // Use dyn\_cast to safely convert the generic Value to a TypedValue.  
  // This performs a runtime check on the underlying IR type.  
  auto sourceMemRef \= mlir::dyn\_cast\<mlir::TypedValue\<mlir::MemRefType\>\>(sourceOperand);

  // If the cast fails, it means the operand is not a MemRefType.  
  // This is an unexpected state for this pattern, so we fail gracefully.  
  if (\!sourceMemRef) {  
    return rewriter.notifyMatchFailure(op, "source operand was not a MemRefType");  
  }

  // At this point, 'sourceMemRef' is a valid TypedValue\<MemRefType\> and can be  
  // passed to builders that require this specific C++ type. For example:  
  // rewriter.create\<mlir::xegpu::CreateNdDescOp\>(loc,..., sourceMemRef,...);

  //... rest of the lowering logic...  
// }

An alternative is mlir::cast, which functions similarly but will trigger an assertion and terminate the compiler on a failed cast. While useful in contexts where a type is guaranteed by an invariant, mlir::dyn\_cast is generally the superior choice within the fallible context of a matchAndRewrite function, promoting the creation of more robust and resilient compiler passes.

## **Part II: Constructing Operations with Enum Attributes**

The second compilation error, no matching function for call to ‘mlir::xegpu::FenceOp::build(...)’, points to a common misunderstanding of how compile-time constants, particularly enumerations, are represented in MLIR's IR and manipulated via its C++ APIs. The solution lies in recognizing the distinction between a C++ enum class and its corresponding in-IR EnumAttr representation.

### **2.1 The Dual Representation of Enums in MLIR**

MLIR's Operation Definition Specification (ODS), which uses TableGen, is the primary mechanism for defining dialect components.14 When an enumeration is defined in a dialect's

.td file, the mlir-tblgen backend generates two distinct but related C++ entities.1 This dual representation is fundamental to how MLIR bridges the gap between static C++ compiler logic and the dynamic, extensible IR.

For an enum like FenceScope in the xegpu dialect, the generated components are:

1. **A C++ enum class**: This is a standard, type-safe C++ enumeration, for example, mlir::xegpu::FenceScope. It provides enumerators like mlir::xegpu::FenceScope::Workgroup. This C++ type is intended for use in the compiler's internal logic, such as in if statements, switch cases, or as a parameter to helper functions. It exists only at C++ compile time and has no direct representation in the MLIR IR.  
2. **An mlir::Attribute Subclass**: This is a class that inherits from mlir::Attribute, for example, mlir::xegpu::FenceScopeAttr. This class serves as a wrapper for the C++ enum value and is its canonical representation *within the IR*. All constant parameters of an operation, including strings, integers, and enums, are stored in the operation's attribute dictionary as instances of mlir::Attribute subclasses.16

The user's compilation error arises from attempting to pass the C++ entity (mlir::xegpu::FenceScope::Workgroup) to an operation builder that expects the IR entity (mlir::xegpu::FenceScopeAttr). The builder's function is to construct the in-memory representation of an mlir::Operation, which consists of mlir::Value operands and mlir::Attribute parameters. It has no knowledge of raw C++ types like enum class values.

### **2.2 The Builder Pattern and Attribute Creation**

The OpBuilder and ConversionPatternRewriter classes provide templated create\<OpT\>(...) methods for instantiating operations.18 The signature of each

create method is generated by TableGen to directly correspond to the arguments defined for that operation in its .td file. If an operation argument is defined as an EnumAttr (e.g., FenceScopeAttr:$scope), the corresponding parameter in the C++ create method will have the type mlir::xegpu::FenceScopeAttr.

To create an instance of an attribute class, the canonical approach is to use its static get(...) factory method. This method is also generated by TableGen for all attribute definitions.16 The

get method is responsible for creating the attribute and "uniquing" it within the MLIRContext. This means that if two calls to get are made with the same parameters, they will both return a handle to the same single, immutable instance in memory. For an enum attribute, the get method typically takes the MLIRContext\* and the C++ enum class value as arguments, performing the necessary wrapping to create the IR attribute.

### **2.3 Solution: Building xegpu.fence Correctly**

To resolve the build error, the C++ enum class value must first be wrapped in its corresponding Attribute class using the ::get() method. This attribute object can then be passed to the xegpu::FenceOp builder.

The exact signature of the xegpu::FenceOp builder is defined by its entry in the XeGPUOps.td file. Based on the dialect's documentation and common patterns, it likely takes a FenceScopeAttr and potentially a MemorySpaceAttr.1 The following code provides the precise C++ implementation for creating an

xegpu.fence operation with a workgroup scope.

C++

\#**include** "mlir/Dialect/XeGPU/IR/XeGPUOps.h"

// Inside matchAndRewrite, where 'rewriter' is a ConversionPatternRewriter  
// and 'loc' is the desired mlir::Location.

// 1\. Get the MLIRContext. It is available from the rewriter.  
mlir::MLIRContext \*context \= rewriter.getContext();

// 2\. Define the desired C++ enum value.  
constexpr auto desiredScope \= mlir::xegpu::FenceScope::Workgroup;

// 3\. Create the IR Attribute wrapper for the enum.  
//    The C++ enum is \`mlir::xegpu::FenceScope\`.  
//    The corresponding Attribute class is \`mlir::xegpu::FenceScopeAttr\`.  
//    Use the static \`::get()\` factory method to create the attribute.  
auto fenceScopeAttr \= mlir::xegpu::FenceScopeAttr::get(context, desiredScope);

// 4\. (Hypothetical) If the fence op also required a memory space, the  
//    same pattern would be used. For example, for global memory:  
//  
//    constexpr auto desiredSpace \= mlir::xegpu::MemorySpace::Global;  
//    auto memorySpaceAttr \= mlir::xegpu::MemorySpaceAttr::get(context, desiredSpace);

// 5\. Call the op builder with the created Attribute instance(s).  
//    The exact signature is defined in the generated XeGPUOps.h.inc file.  
//    Assuming it takes only a scope attribute:  
rewriter.create\<mlir::xegpu::FenceOp\>(loc, fenceScopeAttr);

// If it took both scope and space (a hypothetical example):  
// rewriter.create\<mlir::xegpu::FenceOp\>(loc, fenceScopeAttr, memorySpaceAttr);

This pattern of "get C++ value \-\> wrap in Attribute via ::get() \-\> pass to builder" is universal across MLIR for all attribute types, including integers, strings, arrays, and enums. Internalizing this workflow is fundamental to correctly creating and manipulating operations in C++.

## **Part III: Canonical Structure for MLIR Passes with Options**

The third compilation error, "use of deleted function," is a classic C++ issue that, in the context of MLIR, reveals a deep architectural requirement of the pass management system. The error indicates that the pass class is not copy-constructible, a property mandated by the pass manager to support features like multi-threading. The root cause is an incorrect structuring of pass options.

### **3.1 The Core Conflict: Pass Copyability vs. Option State**

The MLIR pass manager is a sophisticated piece of infrastructure designed for high-performance, parallel compilation. To enable transformations on different parts of the IR concurrently (e.g., running a function pass on multiple functions at once), the pass manager reserves the right to create copies of a pass instance.20 Consequently, a fundamental constraint on all MLIR passes is that they

**must be copy-constructible**.22

The user's error stems from defining a Pass::Option as a direct member variable of their pass class. The Pass::Option and Pass::ListOption classes are more than simple data members; they are stateful objects that, upon construction, register themselves with the llvm::cl command-line parsing infrastructure via an internal mlir::detail::PassOptions object.24 This registration and the associated parsing state are tied to a specific pass instance.

Copying this state is a non-trivial and potentially unsafe operation. A simple member-wise copy would lead to multiple registrations of the same command-line flag or other undefined behavior. To prevent this, the MLIR framework's designers made a crucial safety decision: the mlir::detail::PassOptions class explicitly deletes its copy constructor and copy assignment operator.25 According to C++ language rules, if a class has a member variable whose copy constructor is deleted, the compiler will implicitly delete the copy constructor of the containing class.26

This creates an irreconcilable conflict: the pass manager requires the pass to be copyable, but the inclusion of Pass::Option as a member makes it non-copyable. The "use of deleted function" error is the C++ compiler correctly enforcing this architectural constraint. The solution, therefore, must involve decoupling the non-copyable option state from the copyable pass object.

### **3.2 The Idiomatic Solution: Decoupling Options with a PassOptions Struct**

The canonical pattern for managing pass options in MLIR is to separate the option definitions from the pass's transformation logic. This is achieved by defining all Option and ListOption members within a dedicated struct that inherits from mlir::PassOptions. The pass class itself then becomes stateless with respect to the option *definitions*, although it will hold the parsed option *values*.

This separation can be implemented in two ways:

1. **Declaratively via TableGen (.td file):** This is the modern, recommended, and most robust approach. The pass is defined in a .td file, which references the C++ options struct. The mlir-tblgen tool then generates a ...Base class that correctly handles all the boilerplate for option management, construction, and registration.22  
2. **Manually in C++ (using PassWrapper):** For passes that cannot be defined declaratively, the developer must manage the options struct manually. The pass constructor typically takes a const reference to the options struct and copies the necessary values into its own members.

The TableGen approach is strongly preferred as it reduces boilerplate, ensures consistency with the rest of the MLIR ecosystem, and automatically integrates with tools like mlir-opt for command-line argument parsing and documentation generation.28

### **3.3 Solution: A Complete Implementation Example (TableGen-based)**

The following steps outline the complete, idiomatic implementation of a pass with command-line options using the declarative TableGen-based approach.

#### **Step 1: Define the Options Struct in a C++ Header**

First, create a header file (e.g., LowerOrchestraToGPUPass.h) to define the options struct. This struct will contain the Option members.

C++

// In include/MyDialect/Passes/LowerOrchestraToGPUPass.h

\#**ifndef** MYDIALECT\_PASSES\_LOWERORCHESTRATOGPUPASS\_H  
\#**define** MYDIALECT\_PASSES\_LOWERORCHESTRATOGPUPASS\_H

\#**include** "mlir/Pass/Pass.h"  
\#**include** "llvm/ADT/StringRef.h"  
\#**include** "llvm/Support/CommandLine.h"

namespace mlir {  
namespace orchestra {

// This struct holds the options. It inherits from mlir::PassOptions  
// and contains the Option/ListOption members.  
struct LowerOrchestraToGPUPassOptions : public PassOptions {  
  LowerOrchestraToGPUPassOptions() \= default;

  // The pass option is a member of the struct, not the pass class.  
  Option\<std::string\> targetChipset{  
      \*this, "target-chipset",  
      llvm::cl::desc("Specify the target Intel GPU chipset (e.g., pvc, dg2)."),  
      llvm::cl::init("pvc")};  
    
  // Example of another option  
  Option\<bool\> enableTiling{  
      \*this, "enable-tiling",  
      llvm::cl::desc("Enable tile-based memory operations."),  
      llvm::cl::init(true)};  
};

} // namespace orchestra  
} // namespace mlir

\#**endif** // MYDIALECT\_PASSES\_LOWERORCHESTRATOGPUPASS\_H

#### **Step 2: Define the Pass Declaratively in TableGen**

Next, define the pass in your dialect's Passes.td file. This definition links the pass to the C++ options struct.

Code-Snippet

// In include/MyDialect/Passes.td

include "mlir/Pass/PassBase.td"

// Define the pass, anchoring it to run on ModuleOp.  
def LowerOrchestraToGPU : Pass\<"lower-orchestra-to-gpu", "mlir::ModuleOp"\> {  
  let summary \= "Lowers Orchestra dialect to the xegpu dialect.";  
  let description \=;  
    
  // The command-line argument for the pass.  
  let arg \= "lower-orchestra-to-gpu";

  // Associate the C++ options struct with this pass definition.  
  // This tells TableGen how to construct the pass.  
  let options \=;

  // Specify the C++ class name for the pass implementation.  
  let className \= "::mlir::orchestra::LowerOrchestraToGPUPass";

  // The constructor signature that TableGen will expect.  
  let constructor \= "::mlir::orchestra::createLowerOrchestraToGPUPass()";  
}

*Note: A more advanced setup can have TableGen generate the factory function and pass options directly to the constructor, but this setup is common and clear.*

#### **Step 3: Implement the Pass in C++**

Now, implement the pass logic in a .cpp file. The pass class will inherit from a ...Base class that is automatically generated by TableGen from the .td definition. This base class provides the necessary infrastructure for handling the options.

C++

// In lib/MyDialect/LowerOrchestraToGPU.cpp

\#**include** "MyDialect/Passes/LowerOrchestraToGPUPass.h"  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
\#**include** "mlir/Dialect/XeGPU/IR/XeGPU.h"  
\#**include** "mlir/Transforms/DialectConversion.h"  
// Other necessary includes...

// Include the TableGen-generated header for pass definitions.  
// This defines the LowerOrchestraToGPUPassBase class.  
\#**define** GEN\_PASS\_DEF\_LOWERORCHESTRATOGPU  
\#**include** "MyDialect/Passes.h.inc"

namespace mlir {  
namespace orchestra {

// The pass class now inherits from the generated Base class.  
// This Base class correctly handles the options struct and makes the  
// pass copy-constructible.  
class LowerOrchestraToGPUPass  
    : public impl::LowerOrchestraToGPUPassBase\<LowerOrchestraToGPUPass\> {  
public:  
  // The default constructor is sufficient as the Base class handles options.  
  LowerOrchestraToGPUPass() \= default;

  void runOnOperation() override {  
    // Access options via the members automatically provided by the Base class.  
    // These members are named after the option names in the.td file.  
    llvm::StringRef chipset \= targetChipset;  
    bool tilingEnabled \= enableTiling;

    llvm::dbgs() \<\< "Lowering Orchestra to XeGPU for chipset: " \<\< chipset \<\< "\\n";  
    if (\!tilingEnabled) {  
      // Option can control pass behavior.  
      return;  
    }

    mlir::ConversionTarget target(getContext());  
    target.addLegalDialect\<scf::SCFDialect, xegpu::XeGPUDialect\>();  
    //... setup other target legality...  
    target.addIllegalOp\<orchestra::TransferOp\>();

    mlir::RewritePatternSet patterns(\&getContext());  
    // Populate patterns, potentially passing option values to them.  
    // For example:  
    // patterns.add\<TransferOpLowering\>(patterns.getContext(), chipset);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {  
      signalPassFailure();  
    }  
  }  
};

// Factory function for creating an instance of the pass.  
// This is the constructor referenced in the Passes.td file.  
std::unique\_ptr\<Pass\> createLowerOrchestraToGPUPass() {  
  return std::make\_unique\<LowerOrchestraToGPUPass\>();  
}

} // namespace orchestra  
} // namespace mlir

#### **Step 4: Register the Pass**

Finally, ensure the pass is registered so it can be used by tools like mlir-opt. This is typically done in a central Passes.cpp file using a macro that expands the TableGen definitions.

C++

// In lib/MyDialect/Passes.cpp

// Include the TableGen-generated registration code.  
\#**define** GEN\_PASS\_REGISTRATION  
\#**include** "MyDialect/Passes.h.inc"

void registerMyDialectPasses() {  
  // This function is generated by the macro above.  
  ::registerPasses();  
}

### **3.4 Comparison of Pass Option Architectures**

The transition from an incorrect, stateful pass design to the canonical, decoupled architecture is a critical step in mastering MLIR pass development. The following table summarizes the key differences and benefits of the idiomatic approach.

| Feature | Anti-Pattern: Pass::Option as Member | Better: Manual C++ PassOptions Struct | Canonical: TableGen-Defined Pass |
| :---- | :---- | :---- | :---- |
| **Copy-Constructible** | **No** (Causes compiler error) | Yes | **Yes** (Handled automatically) |
| **State Management** | State is tightly coupled with the pass instance, preventing copies. | State is decoupled into a POD-like struct, but requires manual handling. | State is decoupled and managed by robust, auto-generated code. |
| **Boilerplate Code** | Minimal, but fundamentally incorrect. | Requires manual constructor, field copying, and registration logic. | **Minimal**, as most boilerplate is auto-generated by mlir-tblgen. |
| **Robustness** | Brittle. Fails to compile and is incompatible with multi-threading. | Good, but relies on the developer to correctly implement the pattern. | **Excellent**. The idiomatic and safest approach, fully compatible with all pass manager features. |
| **Tool Integration** | Poor. Command-line parsing can be difficult to manage. | Good. Options can be parsed, but registration is manual. | **Excellent**. Auto-generates command-line flags, help text, and registration for mlir-opt. |

## **Conclusion: Principles for Robust MLIR Development**

The process of diagnosing and resolving the C++ compilation errors encountered in the OrchestraOS project reveals three foundational principles for robust and idiomatic MLIR development. Moving beyond the immediate code fixes, these principles provide a durable framework for architecting effective compiler passes.

First, **embrace the type system's duality**. The MLIR C++ API provides both the generic mlir::Value for maximum flexibility in dynamic contexts and the specific mlir::TypedValue for compile-time safety. The dialect conversion framework intentionally operates with the generic mlir::Value to accommodate in-flight type transformations. The developer's role is to bridge this gap at the appropriate moments using safe, runtime-checked mechanisms like mlir::dyn\_cast. Trusting the type system involves understanding both its power and its necessary boundaries.

Second, **understand the role of code generation**. TableGen is not merely a declarative syntax but a powerful code generator that defines the contract between the IR and the C++ compiler logic. The dual representation of enumerations as both a C++ enum class and an IR EnumAttr is a prime example of this. Recognizing that op builders operate on the IR-level Attribute representation is crucial for correctly constructing operations. This principle extends to all dialect components defined in .td files.

Finally, **decouple state from transformation logic**. The MLIR pass manager is architected for parallelism, which imposes a strict requirement of copy-constructibility on all passes. The canonical solution of separating pass options into a dedicated PassOptions struct is a direct consequence of this design. This decoupling makes passes stateless with respect to their configuration, aligning them with the functional paradigm favored by modern compiler frameworks and ensuring their compatibility with advanced features like multi-threaded execution.

By internalizing these principles, a compiler engineer moves from simply writing code that compiles to architecting passes that are scalable, maintainable, and fully aligned with the powerful, extensible philosophy of the MLIR framework. The journey through these compilation errors, while challenging, ultimately equips the developer with a more profound and practical mastery of modern compiler construction.