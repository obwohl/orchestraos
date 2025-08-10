

# **Architecting an Out-of-Tree MLIR Dialect: A Definitive Guide to CMake Structure and Dependency Management**

## **Introduction**

The Multi-Level Intermediate Representation (MLIR) framework provides an extensible infrastructure for building reusable compilers, with dialects serving as the core mechanism for defining new intermediate representations (IRs).1 While the framework itself is powerful, navigating the intricacies of its build system, particularly for out-of-tree projects, presents a common and significant challenge for developers. A frequent point of failure arises from the interaction between MLIR's custom CMake functions and the fundamental operating principles of the CMake build system itself.

This report addresses a specific, yet illustrative, build error: The dependency target "..." of target "..." does not exist. This message, encountered during the CMake configuration stage, is not a superficial syntax error. Instead, it signals a deeper architectural misunderstanding of how CMake discovers, defines, and manages dependencies across different scopes and files. The error typically occurs when the generation of C++ code from TableGen definition files and the compilation of C++ source files that consume this generated code are attempted within the same CMakeLists.txt scope.

The objective of this report is to provide a definitive, best-practice solution to this problem. It will deconstruct the MLIR build system's interaction with CMake, establish the canonical project structure that resolves the dependency issue, and provide a complete, production-ready implementation for a standalone dialect. The analysis will move beyond a simple fix, offering a detailed exploration of the underlying mechanics to equip developers with a robust mental model for architecting out-of-tree MLIR projects. By understanding not just the "what" but the "why" of the prescribed structure, developers can avoid common pitfalls and leverage the full power of the MLIR ecosystem, from compilation to automated testing and documentation.

## **Section 1: The Root of the Error: Understanding CMake Target Generation and Scoping**

The error message The dependency target... does not exist is a configuration-time failure, indicating that CMake was unable to resolve a dependency relationship while parsing the project's CMakeLists.txt files. This is not a C++ compilation or linking error but a failure in constructing the build dependency graph itself. The root cause lies in the fundamental way CMake processes build scripts, specifically concerning target definition and visibility across different scopes. The idiomatic MLIR project structure is explicitly designed to work in harmony with this process, and deviating from it, as in the user's case, leads directly to this class of error.

### **CMake's Two-Phase Process and Sequential Evaluation**

To comprehend the failure, it is essential to understand that CMake operates in two distinct phases:

1. **Configuration Phase:** In this initial phase, CMake executes the CMakeLists.txt scripts, starting from the project root. It processes the files sequentially, command by command, to build an internal representation of the project. This representation includes a directed acyclic graph (DAG) of all build targets (e.g., libraries, executables, custom commands) and the dependencies between them. It is during this phase that functions like add\_library(), add\_executable(), and MLIR's add\_mlir\_dialect() are invoked, and their resulting targets are registered within CMake's internal state. The error in question occurs here, when a target's dependency is referenced before that dependency is known to CMake.  
2. **Generation Phase:** After the configuration phase completes successfully, CMake uses the internal dependency graph to generate the actual build files for the chosen backend build system (e.g., Ninja build files or Unix Makefiles). These generated files contain the precise commands and rules needed to compile and link the project.

The user's error arises because both the definition of the TableGen generation target (OrchestraOpsIncGen) and its consumption as a dependency for the C++ library (obj.Orchestra) occur within the same CMakeLists.txt file. While CMake processes a single file from top to bottom, the complex nature of custom commands and target properties means that the full definition of a target may not be considered "finalized" and globally visible for dependency resolution at the exact moment another target within the same file references it. The MLIR build functions are complex macros that perform many internal steps, and relying on simple top-to-bottom ordering within a single file is fragile and not the intended usage pattern.

### **The Role of add\_subdirectory() in Enforcing Order and Scope**

The canonical solution to this problem hinges on a core CMake command: add\_subdirectory(). This command is far more than an organizational tool for tidying up a project; it is a critical mechanism for controlling CMake's evaluation order and managing target visibility scopes.2

When CMake encounters add\_subdirectory(directory), its execution flow is as follows:

1. It pauses processing the current CMakeLists.txt file.  
2. It descends into the specified directory.  
3. It fully processes the CMakeLists.txt file found within that subdirectory, from top to bottom. This includes defining all targets, setting all properties, and recursively processing any further add\_subdirectory() calls within that file.  
4. Only after the subdirectory's CMakeLists.txt has been completely processed and all its targets are fully defined and registered in CMake's internal state does CMake resume processing the parent CMakeLists.txt file at the line immediately following the add\_subdirectory() call.

This sequential, scoped evaluation model is the key to resolving the dependency error. By placing the TableGen target definition in one subdirectory (e.g., include/) and the library that depends on it in another (e.g., lib/), and then calling them in order from the root CMakeLists.txt, a strict and reliable ordering is enforced.

### **Target Visibility and the Canonical Structure**

The standard MLIR project structure leverages this behavior deliberately. A typical root CMakeLists.txt contains the following sequence 2:

CMake

\#... project setup...  
add\_subdirectory(include)  
add\_subdirectory(lib)  
\#... add tests, tools, etc....

This structure guarantees the following sequence of events during the configuration phase:

1. CMake starts processing the root CMakeLists.txt.  
2. It encounters add\_subdirectory(include). It pauses and processes include/Orchestra/CMakeLists.txt completely.  
3. Inside include/Orchestra/CMakeLists.txt, the add\_mlir\_dialect() command is executed. This defines the TableGen custom target (e.g., MLIROrchestraOpsIncGen) and registers it with CMake.  
4. Once include/Orchestra/CMakeLists.txt is finished, CMake returns to the root file. The MLIROrchestraOpsIncGen target is now a fully known, globally visible entity within the build configuration.  
5. CMake proceeds to the next line, add\_subdirectory(lib). It pauses and processes lib/Orchestra/CMakeLists.txt.  
6. Inside lib/Orchestra/CMakeLists.txt, the add\_mlir\_dialect\_library() command is executed. It references MLIROrchestraOpsIncGen as a dependency. Because this target was already defined and finalized in the previous step, CMake can successfully find it and establish the dependency link in its internal graph.

The error the user is experiencing is a direct result of short-circuiting this process. By placing both commands in the same file, they attempt to create and resolve a dependency within the same evaluation scope, leading to a race condition where the dependency is referenced before it is guaranteed to be visible. The structural separation into include and lib directories is, therefore, not merely a convention but a direct, mechanical solution to the target visibility problem imposed by CMake's fundamental processing model.

## **Section 2: The MLIR Build System Unveiled: add\_mlir\_dialect and add\_mlir\_dialect\_library**

The MLIR build system provides a suite of specialized CMake functions within the AddMLIR.cmake module to streamline the creation of dialects. The two most fundamental functions at the heart of this process are add\_mlir\_dialect() and add\_mlir\_dialect\_library(). Understanding their distinct, non-overlapping responsibilities is crucial for correctly structuring a project. The user's error stems from conflating these roles and mismanaging the dependency flow between them.

### **Dissecting add\_mlir\_dialect(): The Code Generator**

The add\_mlir\_dialect() function is exclusively concerned with **code generation**. Its primary purpose is to orchestrate the invocation of the mlir-tblgen utility on a dialect's TableGen (.td) definition files.

* **Primary Role:** When called as add\_mlir\_dialect(OrchestraOps orchestra), this function takes the base name of the .td file (OrchestraOps) and the dialect's C++ namespace (orchestra) as arguments.3 It then sets up a series of  
  mlir-tblgen invocations with different backend generators. These generators parse OrchestraOps.td and produce several C++ code fragments, known as "include files" because they are meant to be \#include'd into handwritten C++ code.5 These generated files typically include:  
  * OrchestraOps.h.inc: C++ declarations for all operations defined in the dialect.  
  * OrchestraOps.cpp.inc: C++ definitions for the operations (e.g., builders, parsers, printers).  
  * OrchestraOpsDialect.h.inc: The C++ declaration for the dialect class itself.  
  * OrchestraOpsDialect.cpp.inc: C++ definitions for the dialect class methods.  
  * If types or attributes are defined, corresponding ...Types.h.inc and ...Types.cpp.inc files are also generated.5  
* **Key Artifact \- The IncGen Target:** The most important output of add\_mlir\_dialect() from a CMake perspective is the creation of a custom target. This target represents the action of running all the necessary mlir-tblgen commands. By a strong and consistent convention within the LLVM/MLIR build system, this target is named by prepending MLIR and appending IncGen to the base name of the .td file.7 Therefore, for  
  add\_mlir\_dialect(OrchestraOps...) the generated target is named **MLIROrchestraOpsIncGen**. This is a critical point of correction for the user, who was attempting to depend on the non-existent OrchestraOpsIncGen target. This IncGen target does not compile any C++ code; it only generates the .inc header files.

### **Dissecting add\_mlir\_dialect\_library(): The C++ Compiler**

In contrast, the add\_mlir\_dialect\_library() function is responsible for **compilation**. It is a specialized wrapper around the more general add\_llvm\_library() function, tailored for building the shared or static library that contains the dialect's implementation.8

* **Primary Role:** This function takes a list of C++ source files (.cpp) and compiles them into a library target. For the Orchestra dialect, this would be the OrchestraDialect.cpp file, which contains the handwritten logic for registering the dialect, implementing its interfaces, and including the generated code from TableGen.  
* **The DEPENDS Keyword:** This is the most critical parameter for resolving the user's issue. The DEPENDS argument is passed directly to the underlying add\_llvm\_library() call. It explicitly informs CMake that the compilation of the C++ sources for this library cannot begin until the targets listed under DEPENDS have been successfully built.8 In this case, the library must depend on the  
  MLIROrchestraOpsIncGen target to ensure that all .inc files are generated *before* the C++ compiler is invoked on OrchestraDialect.cpp, which includes them.  
* **The LINK\_LIBS Keyword:** This parameter specifies link-time dependencies on other MLIR or LLVM libraries. Any dialect will at a minimum need to link against MLIRIR to get access to core IR classes like Operation, Dialect, MLIRContext, etc. If the dialect's implementation uses operations or types from other dialects (e.g., arith.constant), it must also link against those dialect libraries (e.g., MLIRArith).8 The  
  PUBLIC specifier ensures that any downstream target linking against the Orchestra library will also be automatically linked against its public dependencies, satisfying the transitive dependency requirements.

### **Why DEPENDS is Superior to add\_dependencies()**

One might be tempted to use a separate add\_dependencies(Orchestra MLIROrchestraOpsIncGen) call after defining the library. However, the official MLIR documentation and LLVM developer discussions strongly advise against this, recommending the DEPENDS keyword instead.8 The reason is rooted in the complexity of the LLVM build infrastructure.

The add\_llvm\_library() function is a powerful macro that, depending on the global CMake configuration (e.g., BUILD\_SHARED\_LIBS=ON or LLVM\_LINK\_LLVM\_DYLIB=ON), may create multiple underlying build targets from a single invocation. For instance, it might create a static library target Orchestra, a shared library target, and an object library target named obj.Orchestra.9

An external add\_dependencies() call might only attach the dependency to the primary Orchestra target. It is not guaranteed to propagate this dependency to the other implicitly created targets like obj.Orchestra. If a subsequent build step then uses obj.Orchestra, it would do so without the necessary dependency on the TableGen output, leading to the exact "target does not exist" or, more subtly, a compilation failure due to missing headers.

By passing the dependency list directly into the function via the DEPENDS keyword, the add\_llvm\_library() macro is made fully aware of the dependency. It can then ensure that this dependency is correctly and robustly propagated to *every single target* it creates internally. This makes the build configuration resilient to changes in global build settings and is the sanctioned, idiomatic way to express dependencies in the MLIR build system.

## **Section 3: The Canonical Solution: Structuring Your Project for Success**

Having established the technical reasons for the build failure—namely, CMake's evaluation order and the distinct roles of the MLIR build functions—the solution becomes clear: the project must be restructured to align with these principles. The community-standard include/lib directory structure is not an arbitrary choice but a deliberate and necessary design pattern for out-of-tree dialects. This structure mechanically resolves the technical CMake problem while simultaneously promoting sound software engineering principles.

### **The Interface/Implementation Analogy**

The best way to conceptualize the canonical project structure is as a physical manifestation of the fundamental C++ software engineering principle of separating interface from implementation. This separation enhances modularity, clarifies intent, and simplifies dependency management.

* **include/: The Public Interface.** This directory is intended to contain the public-facing contract of the dialect. For an MLIR dialect, this includes two types of files 7:  
  1. **Public C++ Headers (.h):** These are the headers that downstream users of your dialect will include in their own code to interact with your operations, types, and attributes. For example, include/Orchestra/OrchestraDialect.h.  
  2. **TableGen Definitions (.td):** These files are the declarative definition of the dialect's IR components. They define the structure, syntax, and properties of operations, types, and attributes.5 As they define the public API of the dialect (e.g., what operations exist and what arguments they take), they conceptually belong with the public interface. For example,  
     include/Orchestra/OrchestraOps.td.  
* **lib/: The Private Implementation.** This directory contains the private implementation details of the dialect that consumers do not need to see or depend on directly.3 This includes:  
  1. **C++ Source Files (.cpp):** These files contain the handwritten C++ logic that brings the dialect to life. This includes the main dialect registration function (initialize()), implementations of operation verifiers, custom parsers/printers, and the crucial \#include directives for the TableGen-generated .cpp.inc files. For example, lib/Orchestra/OrchestraDialect.cpp.

This separation is not merely aesthetic. It directly maps to the build process. The CMakeLists.txt in the include directory is responsible only for code generation from the public interface definitions. The CMakeLists.txt in the lib directory is responsible for compiling the private implementation, which naturally depends on the code generated from the interface.

### **Refactoring the orchestra-compiler Project**

To fix the user's project, the following refactoring is required:

1. **Create an include/Orchestra Directory:** Move the public header OrchestraDialect.h and the TableGen file OrchestraOps.td into this new directory.  
2. **Create a CMakeLists.txt in include/Orchestra:** This file will contain only the add\_mlir\_dialect() command.  
3. **Clean up lib/Orchestra:** This directory should now only contain the C++ implementation file, OrchestraDialect.cpp, and its corresponding CMakeLists.txt.  
4. **Update the lib/Orchestra/CMakeLists.txt:** This file will now contain only the add\_mlir\_dialect\_library() command, with the correct dependency on the TableGen target defined in the include directory.  
5. **Update the Root CMakeLists.txt:** This file must be configured to process the subdirectories in the correct order: add\_subdirectory(include) followed by add\_subdirectory(lib).

### **The CMake Configuration Blueprint**

The relationship between file location, CMake commands, and core responsibility can be summarized in the following blueprint. This table serves as a high-density, actionable guide that distills the architectural solution into a quick reference, reinforcing the separation of concerns that is key to a successful build.

| File Path | Key CMake Command(s) | Core Responsibility |
| :---- | :---- | :---- |
| orchestra-compiler/CMakeLists.txt | project(...) find\_package(MLIR REQUIRED CONFIG) list(APPEND CMAKE\_MODULE\_PATH...) add\_subdirectory(include) add\_subdirectory(lib) | **Project Orchestration.** Handles top-level project setup, finds external dependencies like LLVM and MLIR, and most importantly, orchestrates the build by ensuring the include directory is processed *before* the lib directory. |
| orchestra-compiler/include/Orchestra/CMakeLists.txt | add\_mlir\_dialect(OrchestraOps orchestra) | **Code Generation.** Defines the TableGen target (MLIROrchestraOpsIncGen) which is responsible for invoking mlir-tblgen to generate all necessary C++ header and implementation fragments (.h.inc, .cpp.inc) from the OrchestraOps.td definition file. |
| orchestra-compiler/lib/Orchestra/CMakeLists.txt | add\_mlir\_dialect\_library(Orchestra... DEPENDS MLIROrchestraOpsIncGen LINK\_LIBS PUBLIC MLIRIR) | **Compilation & Linking.** Compiles the C++ source files (.cpp) into the final Orchestra library. It explicitly declares a build-time dependency on the successful completion of the MLIROrchestraOpsIncGen target and a link-time dependency on core MLIR libraries like MLIRIR. |

By adhering to this structure, the build process becomes robust, predictable, and aligned with the design of both CMake and the MLIR build system.

## **Section 4: A Complete, Production-Ready Implementation for the 'Orchestra' Dialect**

With the theoretical and structural foundations established, this section provides the complete, corrected, and annotated source code required to build the Orchestra dialect. This implementation not only fixes the user's immediate CMake error but also addresses the subsequent linker errors that inevitably arise from incorrectly including the TableGen-generated artifacts. The code is based on the best practices demonstrated in the official MLIR standalone examples and templates.12

### **The Corrected Project Structure**

First, the project's file layout should be as follows:

orchestra-compiler/  
├── CMakeLists.txt  
├── include/  
│   └── Orchestra/  
│       ├── CMakeLists.txt  
│       ├── OrchestraDialect.h  
│       └── OrchestraOps.td  
└── lib/  
    └── Orchestra/  
        ├── CMakeLists.txt  
        └── OrchestraDialect.cpp

### **Root CMakeLists.txt**

This file orchestrates the entire build. It finds dependencies, sets up include paths, and processes the include and lib subdirectories in the correct order.

**File: orchestra-compiler/CMakeLists.txt**

CMake

\# Require a modern CMake version. MLIR itself requires at least 3.20.0.  
cmake\_minimum\_required(VERSION 3.20.0)

\# Define the project name and languages.  
project(OrchestraCompiler LANGUAGES CXX C)

\# Set the C++ standard. MLIR requires C++17.  
set(CMAKE\_CXX\_STANDARD 17)  
set(CMAKE\_CXX\_STANDARD\_REQUIRED ON)

\# Find the MLIR package provided by the Debian/Ubuntu installation.  
\# The user specified version 18.1. The CONFIG mode looks for MLIRConfig.cmake.  
\# The user must point CMake to the right location, e.g., by setting:  
\# \-DMLIR\_DIR=/usr/lib/llvm-18/lib/cmake/mlir  
find\_package(MLIR 18.1 REQUIRED CONFIG)  
message(STATUS "Found MLIR\_DIR: ${MLIR\_DIR}")  
message(STATUS "Found LLVM\_DIR: ${LLVM\_DIR}")

\# Add the MLIR and LLVM CMake module directories to the search path.  
\# This is essential for finding AddMLIR.cmake, AddLLVM.cmake, and TableGen.cmake.  
list(APPEND CMAKE\_MODULE\_PATH "${MLIR\_CMAKE\_DIR}")  
list(APPEND CMAKE\_MODULE\_PATH "${LLVM\_CMAKE\_DIR}")

\# Include the necessary MLIR and LLVM CMake modules.  
include(AddMLIR)  
include(AddLLVM)  
include(TableGen)  
include(HandleLLVMOptions)

\# Add the MLIR and LLVM include directories to the project's include path.  
include\_directories(${MLIR\_INCLUDE\_DIRS})  
include\_directories(${LLVM\_INCLUDE\_DIRS})

\# Add the project's own include directory and the build directory's include  
\# directory (for generated headers) to the path.  
include\_directories(${PROJECT\_SOURCE\_DIR}/include)  
include\_directories(${PROJECT\_BINARY\_DIR}/include)

\# This is the core of the solution. Process the 'include' directory first  
\# to define the TableGen targets. Then, process the 'lib' directory, which  
\# can now safely depend on the targets from 'include'.  
add\_subdirectory(include)  
add\_subdirectory(lib)

### **Interface Definition (include/)**

The include directory contains the public API and its declarative TableGen definition.

**File: orchestra-compiler/include/Orchestra/CMakeLists.txt**

CMake

\# This file's sole responsibility is code generation.  
\# It defines the custom target that runs mlir-tblgen on OrchestraOps.td.  
\# The first argument, "OrchestraOps", is the basename of the.td file.  
\# The second argument, "orchestra", is the C++ namespace for the dialect.  
\# This command creates the target "MLIROrchestraOpsIncGen".  
add\_mlir\_dialect(OrchestraOps orchestra)

**File: orchestra-compiler/include/Orchestra/OrchestraDialect.h**

C++

\#**ifndef** ORCHESTRA\_ORCHESTRADIRECT\_H  
\#**define** ORCHESTRA\_ORCHESTRADIRECT\_H

\#**include** "mlir/IR/Dialect.h"

// This includes the C++ class declaration for the dialect,  
// which is generated by TableGen from OrchestraOps.td.  
\#**include** "Orchestra/OrchestraOpsDialect.h.inc"

\#**endif** // ORCHESTRA\_ORCHESTRADIRECT\_H

**File: orchestra-compiler/include/Orchestra/OrchestraOps.td**

Code-Snippet

\#ifndef ORCHESTRA\_OPS\_TD  
\#define ORCHESTRA\_OPS\_TD

include "mlir/IR/OpBase.td"

// Define the dialect itself. This provides the C++ namespace, summary, etc.  
def Orchestra\_Dialect : Dialect {  
  let name \= "orchestra";  
  let cppNamespace \= "::orchestra";  
  let summary \= "A dialect for a hypothetical orchestra compiler.";  
  let description \=;  
}

// A base class for all operations in this dialect.  
class Orchestra\_Op\<string mnemonic, list\<Trait\> traits \=\> :  
    Op\<Orchestra\_Dialect, mnemonic, traits\>;

// Example operation definition.  
def MyOp : Orchestra\_Op\<"my\_op"\> {  
  let summary \= "An example operation.";  
  let description \=;

  let arguments \= (ins);  
  let results \= (outs);  
}

\#endif // ORCHESTRA\_OPS\_TD

### **Implementation (lib/)**

The lib directory contains the C++ implementation and the CMake script to compile it into a library.

**File: orchestra-compiler/lib/Orchestra/CMakeLists.txt**

CMake

\# This file's responsibility is to compile the C++ sources into a library.  
add\_mlir\_dialect\_library(Orchestra  
  \# List of C++ source files to compile.  
  OrchestraDialect.cpp

  \# CRITICAL: This establishes the build-time dependency.  
  \# It ensures that the.inc files are generated before OrchestraDialect.cpp is compiled.  
  \# Note the target name is MLIROrchestraOpsIncGen, not OrchestraOpsIncGen.  
  DEPENDS  
  MLIROrchestraOpsIncGen

  \# Establish link-time dependencies.  
  \# All dialects need MLIRIR. Add others as needed (e.g., MLIRParser).  
  \# PUBLIC visibility propagates this link dependency to any target that links against Orchestra.  
  LINK\_LIBS  
  PUBLIC  
  MLIRIR  
  MLIRSupport  
  MLIRParser  
)

**File: orchestra-compiler/lib/Orchestra/OrchestraDialect.cpp**

C++

\#**include** "Orchestra/OrchestraDialect.h"

// Include the header for the operation definitions.  
// While not strictly required by this file, it's good practice.  
// The real magic happens with the.cpp.inc includes below.  
\#**include** "mlir/Dialect/Arith/IR/Arith.h" // Example of depending on another dialect  
\#**include** "mlir/IR/Builders.h"  
\#**include** "mlir/IR/BuiltinTypes.h"  
\#**include** "mlir/IR/OpImplementation.h"

using namespace mlir;  
using namespace orchestra;

// CRITICAL: This includes the TableGen-generated implementation of the  
// dialect class, such as the \`initialize\` method shell.  
\#**include** "Orchestra/OrchestraOpsDialect.cpp.inc"

// Dialect initialization hook.  
// This is where the dialect registers its operations, types, and attributes.  
void OrchestraDialect::initialize() {  
  // The addOperations\<\>() call is generated by the.cpp.inc file above.  
  // It takes a template pack of all operations to register.  
  addOperations\<  
    // This macro expands to a comma-separated list of all operation  
    // classes defined in the.td file. Omitting this is a common cause  
    // of "dialect does not have registered operation" runtime errors.  
    \#**define** GET\_OP\_LIST  
    \#**include** "Orchestra/OrchestraOps.h.inc"  
  \>();  
}

### **Satisfying the Linker: The Role of .cpp.inc**

A successful CMake configuration and C++ compilation are only half the battle. The next common failure point is at the final linking stage, with errors like undefined reference to 'orchestra::MyOp::getOperationName()'. This occurs because while the header files (.h and .h.inc) provide the *declarations* of the operation classes, the *definitions* (the actual implementation of their methods) are located in the .cpp.inc files generated by TableGen.5

The C++ source code must explicitly include these generated definitions to make them available to the linker. In the example above, this is accomplished by two key includes in OrchestraDialect.cpp:

1. \#include "Orchestra/OrchestraOpsDialect.cpp.inc": This provides the implementation for the OrchestraDialect class itself.  
2. \#define GET\_OP\_LIST followed by \#include "Orchestra/OrchestraOps.h.inc": This is a standard MLIR pattern used inside the initialize() method. The addOperations template function needs a list of the C++ types of the operations to register. The GET\_OP\_LIST macro configures the subsequent include to expand to exactly this list of types.14 While this include is a  
   .h.inc, its behavior is controlled by the macro to provide the necessary information for registration. The actual definitions of the operations are often included implicitly by the dialect's .cpp.inc or must be included separately if custom logic is added. A common pattern to ensure all definitions are available is to also include the op definitions file: \#include "Orchestra/OrchestraOps.cpp.inc".

By following this complete structure, a developer can successfully configure, compile, and link a standalone out-of-tree MLIR dialect, having resolved both the initial CMake dependency error and the subsequent linker errors.

## **Section 5: Beyond the Build: Best Practices for a Healthy Dialect Ecosystem**

Successfully compiling a dialect library is the foundational first step. However, creating a professional-grade, maintainable dialect requires embracing the broader ecosystem provided by MLIR. This includes establishing a robust testing framework, automating documentation generation, and correctly managing dependencies on other dialects. The MLIR build system, through CMake, offers first-class support for these critical activities, turning the TableGen definition file into a "single source of truth" that drives code, tests, and documentation simultaneously.

### **Establishing a Test Suite with lit**

Testing is paramount in compiler development. MLIR uses the LLVM Integrated Tester (lit) in conjunction with the FileCheck utility as its primary testing framework.15 This approach allows developers to write tests directly in

.mlir files, specifying transformation pipelines and asserting the output is correct.

To integrate testing into the orchestra-compiler project:

1. **Create a test/ Directory:** All test files will reside here. A CMakeLists.txt file and a lit configuration file are needed.  
2. **Create a Test Tool:** A standalone opt-like tool is typically created to load the custom dialect and run passes. This tool (e.g., orchestra-opt) is what the lit tests will invoke. This involves adding another subdirectory (e.g., tools/orchestra-opt) with its own sources and CMakeLists.txt that builds an executable and links it against the Orchestra library.  
3. **Configure lit:** The test/ directory needs two configuration files 13:  
   * lit.cfg.py: A Python script that configures the lit test runner, defining available features and substitutions.  
   * lit.site.cfg.py.in: A template that CMake processes to substitute build-specific paths (like the path to the orchestra-opt executable and FileCheck) into the final lit.site.cfg.py in the build directory.  
4. **Write Tests:** Create .mlir files containing // RUN: directives. These directives are shell commands that lit executes. A typical test invokes orchestra-opt on the test file (%s) and pipes the output to FileCheck, which verifies that the output contains specific patterns defined by // CHECK: lines in the same file.15

**Example: test/simple.mlir**

MLIR

// RUN: orchestra-opt %s | FileCheck %s

// CHECK: "orchestra.my\_op"  
module {  
  func.func @test() {  
    "orchestra.my\_op"() : () \-\> ()  
    return  
  }  
}

During CMake configuration, the LLVM\_EXTERNAL\_LIT variable must be set to point to a valid llvm-lit or lit executable, which is used by the check-orchestra target to run the test suite.13

### **Automating Documentation with add\_mlir\_doc**

One of MLIR's most powerful features is its ability to generate high-quality documentation directly from the TableGen source. This ensures that documentation never becomes stale and always reflects the actual implementation.19 This is achieved using the

add\_mlir\_doc function.

This command should be added to the CMakeLists.txt file responsible for TableGen processing, which is include/Orchestra/CMakeLists.txt.

**Example: include/Orchestra/CMakeLists.txt (with documentation)**

CMake

add\_mlir\_dialect(OrchestraOps orchestra)

\# Generate markdown documentation for the dialect as a whole.  
\# This uses the 'summary' and 'description' from the 'Dialect' definition in the.td file.  
add\_mlir\_doc(OrchestraDialect  
  \-gen-dialect-doc  
  OrchestraDialect  
  Dialects/  
)

\# Generate markdown documentation for each operation.  
\# This uses the 'summary' and 'description' from each 'Op' definition.  
add\_mlir\_doc(OrchestraOps  
  \-gen-op-doc  
  OrchestraOps  
  Dialects/  
)

This configuration creates a new build target, typically mlir-doc, which, when built, will generate OrchestraDialect.md and OrchestraOps.md in the build directory.4 This practice makes the

.td file the single source of truth for the dialect's interface, from which C++ code and documentation are derived, enforcing consistency and simplifying maintenance.

### **Managing Inter-Dialect Dependencies**

Dialects rarely exist in isolation. A common workflow involves lowering operations from a high-level custom dialect to operations in lower-level, standard dialects like arith, scf, or func. When the C++ implementation of a pass or operation in the Orchestra dialect needs to create or manipulate operations from the arith dialect, a link-time dependency must be established.

This is managed through the LINK\_LIBS parameter of the add\_mlir\_dialect\_library function.

**Example: lib/Orchestra/CMakeLists.txt (with arith dependency)**

CMake

add\_mlir\_dialect\_library(Orchestra  
  OrchestraDialect.cpp  
  MyOrchestraToArithPass.cpp \# A new file implementing the lowering pass

  DEPENDS  
  MLIROrchestraOpsIncGen

  LINK\_LIBS  
  PUBLIC  
  \# Core libraries  
  MLIRIR  
  MLIRSupport  
  MLIRParser  
  \# Pass infrastructure  
  MLIRPass  
  \# Dependency on the Arith dialect library  
  MLIRArith  
)

By adding MLIRArith to the LINK\_LIBS list, the Orchestra library will be linked against the library containing the arith dialect's implementation.8 This resolves any symbols related to

arith operations (e.g., mlir::arith::ConstantOp) that are referenced in the Orchestra C++ code. Using PUBLIC linkage is generally recommended for dialect dependencies, as it ensures that any executable linking against Orchestra (like orchestra-opt) will also be correctly linked against MLIRArith, satisfying the transitive dependency.

## **Conclusion**

The build error The dependency target... does not exist is a common but solvable impediment for developers new to the out-of-tree MLIR ecosystem. The analysis in this report has demonstrated that the error is not an idiosyncratic flaw in MLIR but a direct consequence of violating the fundamental evaluation and scoping rules of the CMake build system. The resolution lies not in a simple command-line flag or a minor syntax correction, but in adopting a canonical project architecture that is deliberately designed to align with these rules.

The key findings can be synthesized into a set of core principles:

1. **Embrace Structural Separation:** The division of a dialect project into distinct include/ and lib/ directories is the cornerstone of a successful build. This structure enforces a strict, predictable evaluation order where interface-level code generation targets are fully defined before implementation-level compilation targets that depend on them are processed. This is the direct and robust solution to the user's initial problem.  
2. **Respect Functional Boundaries:** The MLIR CMake functions add\_mlir\_dialect and add\_mlir\_dialect\_library have clear and separate purposes. The former is exclusively for TableGen code generation, while the latter is for C++ compilation. Understanding this distinction is critical for placing them in the correct CMakeLists.txt files.  
3. **Utilize Correct Dependency Mechanisms:** The DEPENDS keyword within add\_mlir\_dialect\_library is the sanctioned and most robust mechanism for expressing a build-time dependency on TableGen outputs. It guarantees that the dependency is correctly propagated across all underlying targets created by the LLVM build system, ensuring build consistency across different configurations.  
4. **Acknowledge the Code-Build Symbiosis:** A correct build configuration is necessary but not sufficient. The C++ implementation must correctly \#include the generated .inc files to provide the necessary class definitions to the compiler and linker, preventing subsequent "undefined reference" errors.

By internalizing these principles, developers can move beyond foundational build issues and begin to leverage the more powerful, ecosystem-level features of MLIR. The TableGen definition file (.td) emerges as a single source of truth, from which not only C++ code but also comprehensive documentation and verifiable test cases can be automatically derived. This integrated approach minimizes boilerplate, prevents drift between implementation and documentation, and ultimately enables the development of complex, robust, and maintainable compilers. The prescribed structure is therefore not a constraint but an enabling framework for productive and professional compiler engineering with MLIR.

#### **Referenzen**

1. FOSDEM 23 \- How to Build your own MLIR Dialect, Zugriff am August 8, 2025, [https://archive.fosdem.org/2023/schedule/event/mlirdialect/attachments/slides/5740/export/events/attachments/mlirdialect/slides/5740/How\_to\_Build\_your\_own\_MLIR\_Dialect.pdf](https://archive.fosdem.org/2023/schedule/event/mlirdialect/attachments/slides/5740/export/events/attachments/mlirdialect/slides/5740/How_to_Build_your_own_MLIR_Dialect.pdf)  
2. CMakeLists.txt \- jmgorius/mlir-standalone-template \- GitHub, Zugriff am August 8, 2025, [https://github.com/jmgorius/mlir-standalone-template/blob/main/CMakeLists.txt](https://github.com/jmgorius/mlir-standalone-template/blob/main/CMakeLists.txt)  
3. Step-by-step Guide to Adding a New Dialect in MLIR \- Perry Gibson, Zugriff am August 8, 2025, [https://gibsonic.org/blog/2024/01/11/new\_mlir\_dialect.html](https://gibsonic.org/blog/2024/01/11/new_mlir_dialect.html)  
4. Hello,World with MLIR (2) \- The First Cry of Atom, Zugriff am August 8, 2025, [https://www.lewuathe.com/2020-12-25-hello,world-with-mlir-(2)/](https://www.lewuathe.com/2020-12-25-hello,world-with-mlir-\(2\)/)  
5. MLIR Dialects in Catalyst \- PennyLane Documentation, Zugriff am August 8, 2025, [https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html)  
6. Defining Dialect Attributes and Types \- MLIR, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)  
7. mlir/docs/Tutorials/CreatingADialect.md · 16f349251fabacfdba4acac3b25baf0e6890c1a0 · llvm-doe / llvm-project · GitLab, Zugriff am August 8, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/16f349251fabacfdba4acac3b25baf0e6890c1a0/mlir/docs/Tutorials/CreatingADialect.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/16f349251fabacfdba4acac3b25baf0e6890c1a0/mlir/docs/Tutorials/CreatingADialect.md)  
8. Creating a Dialect \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/Tutorials/CreatingADialect/](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)  
9. mlir/cmake/modules/AddMLIR.cmake · llvmorg-13.0.1-rc1 \- GitLab, Zugriff am August 8, 2025, [https://mirrors.git.embecosm.com/mirrors/llvm-project/-/blob/llvmorg-13.0.1-rc1/mlir/cmake/modules/AddMLIR.cmake?ref\_type=tags](https://mirrors.git.embecosm.com/mirrors/llvm-project/-/blob/llvmorg-13.0.1-rc1/mlir/cmake/modules/AddMLIR.cmake?ref_type=tags)  
10. mlir/lib/Dialect/Tensor/IR/CMakeLists.txt \- GitLab, Zugriff am August 8, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/llvmorg-14.0.6/mlir/lib/Dialect/Tensor/IR/CMakeLists.txt?ref\_type=tags](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/llvmorg-14.0.6/mlir/lib/Dialect/Tensor/IR/CMakeLists.txt?ref_type=tags)  
11. D73130 \[MLIR\] Add support for libMLIR.so \- LLVM Phabricator archive, Zugriff am August 8, 2025, [https://reviews.llvm.org/D73130](https://reviews.llvm.org/D73130)  
12. D77133 \[mlir\] Add an out-of-tree dialect example \- LLVM Phabricator archive, Zugriff am August 8, 2025, [https://reviews.llvm.org/D77133](https://reviews.llvm.org/D77133)  
13. jmgorius/mlir-standalone-template: An out-of-tree MLIR dialect template. \- GitHub, Zugriff am August 8, 2025, [https://github.com/jmgorius/mlir-standalone-template](https://github.com/jmgorius/mlir-standalone-template)  
14. llvm-project/mlir/examples/standalone/lib/Standalone/StandaloneDialect.cpp at main, Zugriff am August 8, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/lib/Standalone/StandaloneDialect.cpp](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/lib/Standalone/StandaloneDialect.cpp)  
15. Testing Guide \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/getting\_started/TestingGuide/](https://mlir.llvm.org/getting_started/TestingGuide/)  
16. Testing guide \- IREE, Zugriff am August 8, 2025, [https://iree.dev/developers/general/testing-guide/](https://iree.dev/developers/general/testing-guide/)  
17. mlir/examples/standalone · develop · Kevin Sala / llvm-project \- GitLab, Zugriff am August 8, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/tree/develop/mlir/examples/standalone?ref\_type=heads](https://git.cels.anl.gov/ksalapenades/llvm-project/-/tree/develop/mlir/examples/standalone?ref_type=heads)  
18. mlir/examples/standalone · llvmorg-16.0.3 · tracing-llvm / llvm \- Gricad-gitlab, Zugriff am August 8, 2025, [https://www.gricad-gitlab.univ-grenoble-alpes.fr/tracing-llvm/llvm/-/tree/llvmorg-16.0.3/mlir/examples/standalone?ref\_type=tags](https://www.gricad-gitlab.univ-grenoble-alpes.fr/tracing-llvm/llvm/-/tree/llvmorg-16.0.3/mlir/examples/standalone?ref_type=tags)  
19. Defining Dialects \- MLIR \- LLVM, Zugriff am August 8, 2025, [https://mlir.llvm.org/docs/DefiningDialects/](https://mlir.llvm.org/docs/DefiningDialects/)