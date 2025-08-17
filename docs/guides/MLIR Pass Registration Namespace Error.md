

# **A Comprehensive Guide to Manual C++ Pass Registration in MLIR: Resolving Namespace and Linkage Errors**

## **Introduction**

This report addresses a common yet nuanced challenge in advanced MLIR development: the manual, C++-native implementation and registration of compiler passes. The compilation error ‘...’ should have been declared inside ‘...’ is not merely a syntax issue. It is a direct consequence of a fundamental C++ design pattern for building scalable, maintainable libraries—a pattern that MLIR's C++ API leverages extensively. By forgoing the declarative TableGen system for defining passes, a developer gains fine-grained control at the cost of needing to manually adhere to these architectural conventions.1 This document will not only provide the immediate code fix for the

LowerOrchestraToGPUPass but will also deconstruct the underlying principles of C++ linkage, MLIR's registration machinery, and the critical separation of public API from private implementation. A thorough understanding of these concepts is essential for navigating such challenges with expertise in the future.

---

## **Section 1: The Root Cause: C++ Namespace and Declaration/Definition Matching**

The compilation error at the heart of this issue stems from foundational C++ principles governing how symbols are declared, defined, and linked across different parts of a program. These rules are independent of MLIR and are paramount to the construction of any large-scale C++ project.

### **The Unbreakable Contract of Declarations**

A function declaration placed in a header file, such as the one for createLowerOrchestraToGPUPass in orchestra-compiler/include/Orchestra/Transforms/Passes.h, acts as a legally binding contract with the C++ compiler and linker. This contract specifies the function's complete signature, including its return type, parameter types, its name, and, critically, the namespace in which it resides. When this header is included in a source file, that source file becomes aware of this contract and expects a definition matching it precisely to be available at link time.

### **Namespace as an Integral Part of Identity**

The core of the error lies in a misunderstanding of how C++ namespaces function. A namespace is not just a convenient grouping mechanism for code; it is an inseparable part of a symbol's unique identity. The declaration:

C++

namespace mlir {  
namespace orchestra {  
std::unique\_ptr\<mlir::Pass\> createLowerOrchestraToGPUPass();  
} // namespace orchestra  
} // namespace mlir

establishes a promise to provide a function with the fully qualified name mlir::orchestra::createLowerOrchestraToGPUPass. Any subsequent definition for this function *must* be defined within that exact nested namespace to satisfy the contract. When the compiler processes the definition, it mangles the function name to include its namespace and signature, creating a unique symbol for the linker. A definition in a different namespace (or the global namespace) will result in a different mangled name, thus failing to fulfill the original promise.

### **Dissecting the Error Message**

The compiler's message, ‘std::unique\_ptr\<mlir::Pass\> mlir::orchestra::createLowerOrchestraToGPUPass()’ should have been declared inside ‘mlir::orchestra’, is precise. The compiler has encountered a definition for a function that it believes is intended to be mlir::orchestra::createLowerOrchestraToGPUPass, but the definition itself is not correctly scoped within the mlir::orchestra namespace in the .cpp file. For instance, defining the function at the global scope:

C++

// In LowerOrchestraToGPU.cpp, at the global scope  
std::unique\_ptr\<mlir::Pass\> mlir::orchestra::createLowerOrchestraToGPUPass() {  
  //...  
}

is incorrect because this syntax is for defining a member function of a class, not a free function within a namespace. The correct approach is to open the namespaces and define the function inside them. Attempts to place the definition in an anonymous namespace or the global namespace without proper qualification fail because they define a new, distinct function with a different fully qualified name (e.g., (anonymous)::createLowerOrchestraToGPUPass or ::createLowerOrchestraToGPUPass). The compiler correctly identifies this mismatch between the promise made in the header and the implementation provided in the source file, flagging it as a compilation error to prevent an inevitable "undefined symbol" error at link time.

---

## **Section 2: Architectural Patterns for MLIR Pass Implementation without TableGen**

Having established the C++ rules, it is now possible to apply them to the specific context of MLIR. The framework's C++ API encourages specific idiomatic patterns for creating passes manually. These conventions are not arbitrary; they are designed to promote encapsulation, modularity, and maintainability, mirroring the very architecture that the TableGen-based declarative system would generate automatically.1

### **Subsection 2.1: Encapsulating Implementation with Anonymous Namespaces**

The pass class itself, LowerOrchestraToGPUPass, is an implementation detail. Its complete definition, including its private members and the implementation of its runOnOperation method, is not needed by any other part of the compiler. The only code that needs to know about the concrete class is the factory function within the same source file that instantiates it.

The standard C++ idiom for this level of encapsulation is the anonymous namespace (namespace {}). Any symbol defined within an anonymous namespace is given *internal linkage* by the compiler. This means the symbol is private to its translation unit (the .cpp file) and is completely invisible to the linker and other object files. Using an anonymous namespace for the pass class definition is a powerful technique that prevents symbol collisions across the project and enforces a clean separation between the pass's internal workings and its public interface.2 This is a documented and recommended practice within the MLIR community for manual pass creation.

### **Subsection 2.2: Exposing the Public API via Factory Functions**

While the pass class itself is kept private, the rest of the compiler infrastructure needs a controlled way to create an instance of it. This is the designated role of the factory function, createLowerOrchestraToGPUPass. In contrast to the pass class, this function must be part of the public API.

Therefore, the factory function must have *external linkage*, allowing it to be called from other translation units—specifically, from the code that orchestrates pass registration (e.g., Passes.cpp). To achieve external linkage and satisfy the contract from the header file, it must be defined outside of any anonymous namespace and, as established in Section 1, it must be defined within the mlir::orchestra namespace.

This architectural pattern—a private implementation class exposed only through a public factory function—is central to the MLIR C++ API. An examination of the code generated by mlir-tblgen for declarative passes reveals this same structure: the generated header file exposes only the create...Pass() function, hiding the auto-generated pass class entirely.1 By writing the pass manually, one is simply recreating this robust and scalable pattern by hand. The factory function's purpose is to return a

std::unique\_ptr\<mlir::Pass\>, an opaque base class pointer that abstracts away the concrete implementation type from the rest of the system, which then uses this instance to populate a PassManager.4

### **Subsection 2.3: Connecting to the Pass Machinery: Registration**

With the pass class and its factory function correctly defined, the final step is to make the MLIR pass management system aware of its existence. There are two primary patterns for manual pass registration in C++.

#### **Pattern 1: Static Registration for Command-Line Tools**

The first pattern, often seen in simpler examples or standalone tools, uses a static global object to trigger registration. This is accomplished with the mlir::PassRegistration template class.5 A line of code like the following is placed at the global scope in the pass's

.cpp file:

C++

static mlir::PassRegistration\<MyPass\> pass(  
    "command-line-arg", "Description of the pass.");

The constructor for this pass object is executed during the program's static initialization phase, before the main function begins. This constructor calls mlir::registerPass, adding the pass to a global registry.5 This makes the pass available to tools like

mlir-opt or, in this case, orchestra-opt, which can then invoke it using its command-line argument (--command-line-arg).2 The primary advantage of this method is its simplicity; merely linking the object file containing the pass is sufficient to make it available.

#### **Pattern 2: Programmatic Registration for Libraries**

The second pattern, which is employed by the Orchestra codebase, offers more explicit control. Instead of relying on static initializers, it uses a dedicated registration function (e.g., registerLoweringToGPUPasses()) that is called explicitly from a central initialization point in the application (e.g., from main.cpp via registerOrchestraPasses).

This programmatic approach is preferable for larger, more complex systems or when the compiler components are intended to be used as a library. It avoids the potential pitfalls of static initialization order and gives the application developer precise control over which passes are available and when they are registered.

Inside this registration function, one calls mlir::registerPass directly. This function is overloaded, but the relevant version takes a PassAllocatorFunction, which is essentially a std::function\<std::unique\_ptr\<Pass\>()\>.6 This is a perfect fit for the factory pattern: the registration function provides a lambda that simply calls the public

create...Pass() factory function. This pattern is also explicitly documented for passes that are not default-constructible or require custom setup logic.7 The Orchestra project's use of a

registerOrchestraPasses function signals a clear architectural choice in favor of this more robust, programmatic registration model. Therefore, any new pass must conform to this model by providing the necessary factory and registration functions, rather than attempting to use the static registration pattern.

---

## **Section 3: The Corrected Implementation for LowerOrchestraToGPU.cpp**

Based on the principles of C++ namespacing and MLIR's architectural patterns, the following is the complete and correct implementation for the orchestra-compiler/lib/Orchestra/Transforms/LowerOrchestraToGPU.cpp file. The code is heavily annotated to connect each element back to the concepts discussed in the preceding sections.

### **The Full, Annotated Code**

C++

//===- LowerOrchestraToGPU.cpp \- Orchestra to GPU Dialect Lowering Pass \---===//  
//  
// Part of the Orchestra Project, under the Apache License v2.0 with LLVM Exceptions.  
// See https://llvm.org/LICENSE.txt for license information.  
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception  
//  
//===----------------------------------------------------------------------===//  
//  
// This file implements a pass to lower Orchestra operations to the GPU dialect.  
//  
//===----------------------------------------------------------------------===//

\#**include** "Orchestra/Transforms/Passes.h"  
\#**include** "mlir/Conversion/GPUCommon/GPUCommonPass.h"  
\#**include** "mlir/Dialect/GPU/IR/GPUDialect.h"  
\#**include** "mlir/Dialect/Func/IR/FuncOps.h"  
\#**include** "mlir/IR/BuiltinOps.h"  
\#**include** "mlir/Pass/Pass.h"  
\#**include** "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Use an anonymous namespace to encapsulate the pass implementation.  
// This gives the LowerOrchestraToGPUPass class internal linkage, making it  
// invisible to other source files and preventing symbol collisions.  
namespace {

// The pass class definition. It inherits from the CRTP base class  
// provided by MLIR for defining passes that operate on a specific operation  
// type. In this case, it's a generic pass on the top-level ModuleOp.  
struct LowerOrchestraToGPUPass  
    : public mlir::PassWrapper\<LowerOrchestraToGPUPass,  
                               mlir::OperationPass\<mlir::ModuleOp\>\> {  
  // A required virtual method to provide the command-line argument used to  
  // invoke the pass from tools like orchestra-opt. The MLIR pass registry  
  // uses this argument as a unique key.\[6\]  
  MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(LowerOrchestraToGPUPass)

  mlir::StringRef getArgument() const final {  
    return "lower-orchestra-to-gpu";  
  }

  // A required virtual method to provide a short description of the pass.  
  // This is used for help messages and documentation.\[7, 8\]  
  mlir::StringRef getDescription() const final {  
    return "Lower Orchestra constructs to the GPU dialect";  
  }

  // The core logic of the pass is implemented in this method.  
  // It is called by the PassManager for each ModuleOp in the IR.  
  void runOnOperation() override {  
    mlir::ModuleOp module \= getOperation();  
    mlir::MLIRContext\* context \= \&getContext();

    // Pass logic implementation would go here.  
    // For example, setting up rewrite patterns and applying them.  
    // mlir::RewritePatternSet patterns(context);  
    //... populate patterns...  
    // if (failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns))))  
    //   signalPassFailure();  
  }  
};

} // end anonymous namespace

// The public-facing factory function. This is the sole entry point into this  
// file from the rest of the compiler. It MUST be defined inside the  
// mlir::orchestra namespace to match its declaration in the corresponding  
// header file (orchestra-compiler/include/Orchestra/Transforms/Passes.h).  
namespace mlir {  
namespace orchestra {

std::unique\_ptr\<mlir::Pass\> createLowerOrchestraToGPUPass() {  
  return std::make\_unique\<LowerOrchestraToGPUPass\>();  
}

} // namespace orchestra  
} // namespace mlir

### **Detailed Walkthrough**

1. **Includes**: The necessary headers are included, most importantly Orchestra/Transforms/Passes.h (which contains the public declaration of createLowerOrchestraToGPUPass) and mlir/Pass/Pass.h (which provides the core pass infrastructure).8  
2. **Anonymous Namespace**: The entire LowerOrchestraToGPUPass class definition is wrapped in namespace {}. This correctly makes the class an implementation detail private to this file.  
3. **Pass Class Definition**: The class LowerOrchestraToGPUPass inherits from mlir::PassWrapper, which is a utility that simplifies the boilerplate for creating passes. It is templated on the concrete pass type and the base MLIR pass class, in this case mlir::OperationPass\<mlir::ModuleOp\>, indicating this pass runs on ModuleOp operations.  
4. **Required Methods**: The class provides overrides for getArgument() and getDescription(). These virtual methods from the base Pass class are essential for the pass registration and command-line tooling to function correctly. The mlir::registerPass function will fail if a pass does not provide a command-line argument.6  
5. **Factory Function**: The createLowerOrchestraToGPUPass function is defined. Crucially, it is placed inside namespace mlir { namespace orchestra {... } }. This ensures its definition matches the declaration in the header, resolving the original compilation error. Its implementation is simple: it heap-allocates an instance of the private LowerOrchestraToGPUPass class and returns it as a std::unique\_ptr\<mlir::Pass\>, successfully abstracting the concrete type.

### **The Registration Function**

The user's description indicates that Passes.cpp calls a function named registerLoweringToGPUPasses. This function connects the factory to the MLIR pass registry. While its location could be in either LowerOrchestraToGPU.cpp or a central Passes.cpp, its implementation would look as follows:

C++

// This function must be declared in Passes.h and can be defined here or  
// in a central Passes.cpp file.  
void mlir::orchestra::registerLoweringToGPUPasses() {  
  mlir::registerPass(  
      // This lambda serves as the PassAllocatorFunction required by  
      // mlir::registerPass. It captures no state and, when invoked by the  
      // pass registry, calls our public factory function.  
     () \-\> std::unique\_ptr\<mlir::Pass\> {  
        return createLowerOrchestraToGPUPass();  
      });  
}

This registration function completes the programmatic pattern. It calls mlir::registerPass and provides a lambda that acts as the required allocator function.6 When the registry needs to create an instance of the "lower-orchestra-to-gpu" pass, it will invoke this lambda, which in turn calls the factory.

---

## **Section 4: A Tale of Two Passes: Reconciling LowerOrchestraToGPU with LowerOrchestraToStandard**

A key point of confusion is the existence of another pass, LowerOrchestraToStandard.cpp, which compiles successfully despite having a different structure. An analysis of its described structure reveals that it uses the alternative static registration pattern, highlighting an important architectural distinction within the project.

### **Deductive Analysis of LowerOrchestraToStandard.cpp**

Given the information that this pass has its class in an anonymous namespace and lacks a create function, its structure can be deduced with high confidence. It almost certainly uses the static mlir::PassRegistration object to achieve self-registration, a pattern well-suited for standalone command-line tools.2

### **Inferred Structure**

The code for LowerOrchestraToStandard.cpp likely resembles the following:

C++

// Inferred structure of LowerOrchestraToStandard.cpp

\#**include** "mlir/Pass/Pass.h"  
//... other includes...

namespace {  
// The pass class is still encapsulated as an implementation detail.  
struct LowerOrchestraToStandardPass  
    : public mlir::PassWrapper\<LowerOrchestraToStandardPass,  
                               mlir::OperationPass\<mlir::ModuleOp\>\> {  
  MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(LowerOrchestraToStandardPass)

  mlir::StringRef getArgument() const final {  
    return "lower-orchestra-to-std";  
  }

  mlir::StringRef getDescription() const final {  
    return "Lower Orchestra to the Standard dialect";  
  }

  void runOnOperation() override { /\*... implementation... \*/ }  
};  
} // end anonymous namespace

// The key difference: a static PassRegistration object.  
// This object's constructor is called during static initialization, before  
// main(), automatically registering the pass with the global registry.  
// No public header declaration or factory function is needed.\[5\]  
static mlir::PassRegistration\<LowerOrchestraToStandardPass\> pass(  
    "lower-orchestra-to-std", "Lower Orchestra to Standard dialect");

This pass is entirely self-contained. It does not expose any functions in a public header. The act of linking its corresponding object file into the final orchestra-opt binary is sufficient to ensure its registration and availability from the command line.

### **Comparative Analysis**

The presence of these two different patterns is not a sign of inconsistency but rather a likely indicator of the project's architectural evolution. The LowerOrchestraToStandard pass was probably implemented when the project's primary target was the standalone orchestra-opt tool, for which static registration is simple and effective. The new LowerOrchestraToGPU pass is being added under a more mature, library-oriented architecture, reflected by the explicit registration functions in Passes.h and Passes.cpp. This new architecture provides the control necessary to use the compiler's components in different contexts beyond a single command-line tool. The compilation error encountered is effectively enforcing adherence to this newer, more robust design.

The following table makes the distinction between the two patterns clear:

| Feature | LowerOrchestraToStandard Pattern (Inferred) | LowerOrchestraToGPU Pattern (Required) |
| :---- | :---- | :---- |
| **Primary Use Case** | Self-registration for command-line tools (orchestra-opt) where linking the pass is sufficient. | Programmatic registration in a library context, allowing explicit control over which passes are available. |
| **Key Component** | static mlir::PassRegistration\<...\> object.2 | std::unique\_ptr\<mlir::Pass\> create...Pass() factory function.1 |
| **Header Declaration** | No public function declaration needed in Passes.h. The pass is self-contained. | create...Pass() and register...Passes() must be declared in Passes.h as part of the library's public API. |
| **Namespace Logic** | Entire implementation, including the registration object, can be static or in an anonymous namespace. | Pass class in anonymous namespace; factory and registration functions in the public mlir::orchestra namespace. |
| **Registration Trigger** | Static initialization (before main runs). The linker including the object file is the trigger. | Explicit function call (registerOrchestraPasses() from main). |

---

## **Conclusion: Best Practices for Robust MLIR Pass Development**

The analysis of this compilation error provides a clear roadmap for robust, manual pass development in MLIR. By adhering to the framework's C++ idioms and established software engineering principles, developers can create modular, maintainable, and scalable compiler infrastructure. The following best practices synthesize the findings of this report:

* **Embrace the API/Implementation Divide**: Always place the full definition of a pass class inside an anonymous namespace to give it internal linkage. This encapsulates the implementation details. Expose functionality to the rest of the system only through a minimal, well-defined public API, typically a factory function (create...Pass) that returns a std::unique\_ptr\<mlir::Pass\>.  
* **Match Declarations and Definitions Precisely**: A C++ header file establishes a contract. Ensure that any function declared in a header is defined in the exact same namespace in its corresponding source file. This is fundamental to the C++ compilation and linking model.  
* **Choose the Right Registration Pattern**: Understand the architectural context of the project. If the system uses an explicit, centralized registration function (e.g., registerAllPasses()), new passes must follow the programmatic pattern by providing a factory and a registration function. If passes are being added to a simple, standalone tool, the static mlir::PassRegistration object is a viable and simpler alternative.  
* **Leverage the MLIR C++ API Contracts**: When writing a pass class manually, ensure all required virtual methods from the mlir::Pass base classes are implemented. This includes getArgument(), getDescription(), and runOnOperation(). These methods are essential contracts used by the pass registry, pass manager, and command-line tooling.6  
* **View TableGen as an Automator**: Recognize that manual C++ implementation requires the developer to perform the same architectural separation that mlir-tblgen would automate. The pattern of a hidden implementation class exposed by a public factory function is the goal in both approaches. Understanding this underlying pattern makes both manual and TableGen-driven development more intuitive and effective.1
