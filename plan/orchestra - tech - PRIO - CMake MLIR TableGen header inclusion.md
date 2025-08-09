

# **Configuring CMake for Out-of-Tree MLIR Passes with TableGen-Generated Dialect Headers**

## **Executive Summary**

This report provides a comprehensive guide for configuring CMake to build out-of-tree MLIR passes that rely on custom, TableGen-generated dialect headers. It navigates the complexities of MLIR's build system, TableGen's code generation, and CMake's dependency management, offering a robust, maintainable, and scalable solution. The core methodology involves the correct utilization of find\_package(MLIR), leveraging MLIR's specialized CMake macros such as add\_mlir\_dialect, mlir\_tablegen, and add\_mlir\_dialect\_library, and meticulously managing include paths for generated headers through target\_include\_directories with generator expressions. Adhering to these practices facilitates independent development and seamless integration of custom MLIR components.

## **1\. Introduction to Out-of-Tree MLIR Development**

### **1.1. Rationale for Out-of-Tree Development**

Developing MLIR components, including dialects, passes, and tools, outside the main LLVM/MLIR monorepository offers significant advantages for developers. This "out-of-tree" approach enables independent versioning of custom components, which is crucial for managing project lifecycles separate from the upstream MLIR development cycle. It also leads to reduced build times during iterative development, as developers only compile their specific components rather than the entire LLVM/MLIR project. Furthermore, out-of-tree development simplifies integration into existing projects that may not require the full LLVM infrastructure. This modularity also allows for experimentation with new features or domain-specific optimizations without the overhead and rigorous review process associated with direct upstream contributions. MLIR's design inherently supports this modular, out-of-tree development, positioning it as a highly extensible framework.1 The very existence of specialized CMake modules designed for MLIR component integration, rather than solely relying on manual, low-level CMake scripting, underscores this fundamental architectural choice. This design choice facilitates the growth of the MLIR ecosystem and fosters independent innovation.

### **1.2. Overview of MLIR's Build System Philosophy within the LLVM Ecosystem**

MLIR, as a part of the broader LLVM project, leverages CMake as its meta-build system.2 CMake does not directly build the project; instead, it generates the necessary build files (e.g., Makefiles, Ninja files, Visual Studio project files) for the chosen underlying build tool.2 This cross-platform capability is a cornerstone of the LLVM ecosystem's build philosophy.

Crucially, MLIR extends CMake's capabilities by providing a suite of custom CMake modules, such as AddMLIR.cmake and TableGen.cmake.3 These modules are specifically designed to simplify the configuration and integration of MLIR-specific components, particularly those that involve TableGen for code generation. These specialized modules are indispensable for correctly integrating out-of-tree projects, as they abstract away much of the underlying complexity of managing TableGen dependencies and MLIR-specific build rules. The

find\_package(MLIR REQUIRED CONFIG) command, a standard practice in out-of-tree MLIR projects, is not merely about locating MLIR libraries; it is fundamentally about discovering and integrating the *build system infrastructure* provided by MLIR itself.3 This command sets crucial variables like

MLIR\_DIR, MLIR\_INCLUDE\_DIRS, and MLIR\_CMAKE\_DIR, which are then used to include MLIR's specialized CMake macros. This mechanism is vital because it enables developers to build against a locally compiled MLIR (from a build tree) during development and seamlessly transition to a system-wide installed MLIR (from an install tree) for deployment or distribution. This dual-tree support is a foundational aspect of CMake and LLVM/MLIR's design, ensuring that out-of-tree components remain functional across different development and deployment environments.

## **2\. Understanding TableGen and MLIR Code Generation**

### **2.1. The Role of TableGen in MLIR**

TableGen serves as a pivotal tool in the MLIR ecosystem, providing a generic language and associated tooling for maintaining and processing domain-specific information.5 Its role is central to MLIR's declarative approach to defining compiler infrastructure. MLIR extensively utilizes TableGen's Operation Definition Specification (ODS) to declaratively define core IR entities such as operations, types, attributes, and passes.5 This declarative methodology significantly reduces the amount of boilerplate C++ code that developers would otherwise need to write and maintain manually.

From these .td (TableGen Definition) files, mlir-tblgen generates substantial C++ code, including template specializations, accessor methods for operation parameters, parsing and printing logic for IR entities, and verification methods.5 This automated code generation streamlines development, enhances consistency, and minimizes the risk of errors associated with manual implementation of repetitive patterns.

### **2.2. Types of Files Generated by mlir-tblgen**

When mlir-tblgen processes .td files, it produces a variety of C++ .inc files. These files are not standalone compilation units but are specifically designed to be included within hand-written C++ source and header files. The common types of generated files and their typical purposes include:

* **\<Dialect\>Ops.h.inc**: Contains the C++ declarations for operations defined in the dialect's ODS.  
* **\<Dialect\>Ops.cpp.inc**: Provides the C++ definitions for operations, often included with specific macros like \#define GET\_OP\_LIST and \#define GET\_OP\_CLASSES to instantiate operation classes and register them with the dialect.10  
* **\<Dialect\>Types.h.inc**: Declares C++ classes for custom types within the dialect.7  
* **\<Dialect\>Types.cpp.inc**: Defines methods for custom types, such as parsing and printing.7  
* **\<Dialect\>AttrDefs.h.inc**: Declares C++ classes for custom attributes.9  
* **\<Dialect\>AttrDefs.cpp.inc**: Defines methods for custom attributes.9  
* **\<Dialect\>OpsInterfaces.h.inc and \<Dialect\>OpsInterfaces.cpp.inc**: Provide declarations and definitions for operation interfaces, enabling polymorphic behavior across different operations.11

These .inc files are typically included in hand-written C++ files such as FooDialect.cpp or FooOps.cpp to integrate the generated boilerplate with custom logic.7

### **2.3. Core MLIR CMake Macros: add\_mlir\_dialect(), mlir\_tablegen(), add\_mlir\_dialect\_library()**

MLIR's CMake modules provide several high-level macros that streamline the integration of TableGen-defined components:

* **add\_mlir\_dialect(name)**: This macro serves as the primary entry point for declaring an MLIR dialect within the CMake build system. It automatically sets up the necessary TableGen rules to generate the core operation and dialect declarations (e.g., \<name\>Ops.h.inc, \<name\>Ops.cpp.inc). A significant aspect of this macro is its creation of a public dependency target, typically named MLIR\<name\>IncGen.7 This target is crucial for managing build order, ensuring that the TableGen generation step is completed  
  *before* any C++ compilation that relies on these generated headers. This explicit dependency prevents common "file not found" or "undeclared identifier" errors during compilation.  
* **mlir\_tablegen(output\_file\_base\_name command\_args...)**: This macro acts as a specialized wrapper around LLVM's generic tablegen macro, tailored for MLIR-specific code generation.12 While  
  add\_mlir\_dialect handles the default operation and dialect declarations, mlir\_tablegen is used for generating more specific components, such as attribute definitions (using flags like \-gen-attrdef-decls and \-gen-attrdef-defs) or pass declarations.9 This layered approach to TableGen integration allows for common patterns to be abstracted by  
  add\_mlir\_dialect, while providing fine-grained control over specific code generation needs via mlir\_tablegen.  
* **add\_mlir\_library(name)**: This macro is designed for declaring a generic library that can be compiled and linked within the MLIR ecosystem, often becoming part of libMLIR.so or linked by MLIR tools.14 It functions as a thin wrapper around  
  add\_llvm\_library.  
* **add\_mlir\_dialect\_library(name)**: This macro is specifically intended for libraries associated with a custom dialect. Beyond defining the library, it appends the library's name to the global MLIR\_DIALECT\_LIBS property.14 This global list is particularly useful for tools like  
  mlir-opt, which can then automatically discover and link against all available dialects, simplifying the process of building a comprehensive MLIR tool.11 This demonstrates a higher-level abstraction provided by MLIR's CMake system for managing the ecosystem of dialects.  
* **add\_mlir\_conversion\_library(name...)**: Similar to add\_mlir\_dialect\_library, this macro declares a library specifically for dialect conversions, collecting its name into MLIR\_CONVERSION\_LIBS.14

### **Table 1: Key MLIR CMake Functions and Their Purpose**

| CMake Function | Purpose |
| :---- | :---- |
| add\_mlir\_dialect(name) | Declares an MLIR dialect, setting up TableGen rules for operations and dialect declarations. Creates a dependency target (e.g., MLIR\<name\>IncGen) to ensure generated files are available before compilation. |
| mlir\_tablegen(output...) | Invokes mlir-tblgen for specific code generation tasks beyond basic dialect/operation declarations, such as attributes, types, or passes. Provides granular control over TableGen output. |
| add\_mlir\_library(name...) | Declares a generic MLIR-related library, often used for passes or utilities that are not strictly tied to a single dialect. |
| add\_mlir\_dialect\_library(...) | Declares a library specifically for a custom dialect. It automatically adds the library to the MLIR\_DIALECT\_LIBS global property, making it discoverable and linkable by MLIR tools (e.g., mlir-opt) without explicit individual linking. |
| add\_mlir\_conversion\_library(...) | Declares a library specifically for dialect conversions, adding it to the MLIR\_CONVERSION\_LIBS global property. |
| add\_public\_tablegen\_target(name) | Creates a public CMake target for TableGen output dependencies. This is used when mlir\_tablegen is invoked directly for specific generated files (e.g., attributes, types) to ensure that targets depending on these files can correctly specify their build order requirements. |

## **3\. Setting Up the Top-Level Out-of-Tree CMake Project**

### **3.1. Initial CMakeLists.txt Structure**

The foundation of any out-of-tree MLIR project begins with a well-structured top-level CMakeLists.txt file. This file establishes the project's basic configuration and locates the necessary MLIR infrastructure.

The file should start by specifying the minimum required CMake version. A version of 3.20.0 or higher is recommended for compatibility with modern MLIR builds.2

CMake

cmake\_minimum\_required(VERSION 3.20.0)

Next, the project is defined, specifying the languages used. For MLIR development, C++ and C are typically required:

CMake

project(MyMLIRProject LANGUAGES CXX C)

It is also essential to explicitly set the C++ standard to conform to. MLIR generally requires C++17 3:

CMake

set(CMAKE\_CXX\_STANDARD 17 CACHE STRING "C++ standard to conform to")  
set(CMAKE\_POSITION\_INDEPENDENT\_CODE ON) \# Often required for shared libraries

A critical step for out-of-tree development is locating the MLIR installation. This is achieved using find\_package(MLIR REQUIRED CONFIG).3 This command searches for

MLIRConfig.cmake within the CMake package registry and sets a series of variables, including MLIR\_DIR (the path to the MLIR installation or build directory), MLIR\_INCLUDE\_DIRS (a list of include directories for MLIR headers), and MLIR\_CMAKE\_DIR (the path to MLIR's CMake modules). Informative messages can be added to confirm the successful location of MLIR:

CMake

find\_package(MLIR REQUIRED CONFIG)  
message(STATUS "Using MLIRConfig.cmake in: ${MLIR\_DIR}")  
message(STATUS "Using LLVMConfig.cmake in: ${LLVM\_DIR}") \# LLVM\_DIR is often set by MLIR's config

### **3.2. Including Essential MLIR CMake Modules**

Once MLIR has been located, its custom CMake modules must be made available to the project. This is achieved by appending MLIR\_CMAKE\_DIR to CMAKE\_MODULE\_PATH.3 This step is critical because it allows CMake to find and include MLIR's specialized macros, without which the entire TableGen integration and MLIR-specific build processes would be significantly more complex or impossible.

CMake

list(APPEND CMAKE\_MODULE\_PATH "${MLIR\_CMAKE\_DIR}")

Following this, core LLVM and MLIR CMake utilities are included. include(AddLLVM) provides fundamental macros like add\_llvm\_library, while include(TableGen) makes the generic tablegen macro available.3 Finally,

include(AddMLIR) brings in MLIR-specific macros such as add\_mlir\_dialect and add\_mlir\_library.3

CMake

include(AddLLVM)  
include(TableGen)  
include(AddMLIR)

### **3.3. Managing Global Include Directories and C++ Standards**

While target\_include\_directories is generally preferred for specifying include paths on a per-target basis, it is common practice in top-level MLIR projects to include global directories for core MLIR and LLVM headers. This ensures that all source files can readily locate fundamental headers without requiring explicit per-file path specifications.

CMake

include\_directories(${LLVM\_INCLUDE\_DIRS})  
include\_directories(${MLIR\_INCLUDE\_DIRS})  
include\_directories(${PROJECT\_SOURCE\_DIR}) \# Include your project's source root  
include\_directories(${PROJECT\_BINARY\_DIR}) \# Include your project's binary root

### **Table 3: Essential CMake Variables for Paths**

Understanding the various CMake path variables is crucial for correctly configuring an out-of-tree MLIR project, especially when dealing with generated files. These variables indicate different locations within the source and build trees, and their values dynamically change depending on the CMakeLists.txt file currently being processed. TableGen-generated headers are typically placed in the *binary directory* corresponding to where their add\_mlir\_dialect or mlir\_tablegen command was invoked. Misunderstanding these paths can lead to "file not found" errors.

| CMake Variable | Description |  |
| :---- | :---- | :---- |
| CMAKE\_SOURCE\_DIR | The full path to the root of the top-level source tree of the entire project. This variable remains constant throughout the CMake configuration process for a given project. |  |
| CMAKE\_BINARY\_DIR | The full path to the root of the top-level build tree. This is where all generated build artifacts for the entire project are stored. |  |
| CMAKE\_CURRENT\_SOURCE\_DIR | The full path to the source directory containing the CMakeLists.txt file currently being processed by CMake. Its value changes as add\_subdirectory() commands are processed.16 |  |
| CMAKE\_CURRENT\_BINARY\_DIR | The full path to the build directory that corresponds to the CMAKE\_CURRENT\_SOURCE\_DIR. This is where build artifacts (including TableGen-generated files) for the current subdirectory are placed.16 Its value also changes with | add\_subdirectory().18 |
| MLIR\_DIR | The path to the MLIR installation or build directory that find\_package(MLIR) successfully located. |  |
| MLIR\_INCLUDE\_DIRS | A CMake list variable containing paths to the public include directories of MLIR. |  |
| MLIR\_CMAKE\_DIR | The path to the directory containing MLIR's custom CMake modules (e.g., AddMLIR.cmake). |  |

## **4\. Structuring Your Out-of-Tree MLIR Pass and Dialect**

### **4.1. Recommended Directory Layout for a Pass and its Custom Dialect**

A well-organized directory structure is fundamental for maintainability and clarity in out-of-tree MLIR development. Adopting a layout that mirrors MLIR's internal organization for dialects and passes can significantly improve familiarity for developers accustomed to the LLVM ecosystem.11 This structure promotes modularity and logical separation of concerns.

A recommended structure for a custom MLIR project might appear as follows:

my-mlir-project/  
├── CMakeLists.txt              (Top-level project configuration)  
├── include/  
│   └── MyDialect/  
│       ├── CMakeLists.txt    (For dialect TableGen definitions)  
│       ├── MyDialect.h       (Hand-written dialect public header)  
│       ├── MyDialect.td       (Dialect base definition and properties)  
│       └── MyOps.td          (Operation definitions for MyDialect)  
├── lib/  
│   └── MyDialect/  
│       ├── CMakeLists.txt    (For dialect library sources and pass implementations)  
│       ├── MyDialect.cpp     (Dialect C++ implementation, includes generated.inc files)  
│       └── MyPass.cpp        (MLIR Pass C++ implementation)  
├── tools/  
│   └── my-mlir-tool/  
│       ├── CMakeLists.txt    (For a custom mlir-opt-like executable)  
│       └── my-mlir-tool.cpp  (Main source for the custom tool)  
└── test/  
    └── MyDialect/  
        └── CMakeLists.txt    (For integration tests using Lit)

This structure aligns closely with the typical organization of in-tree MLIR dialects 11 and is reflected in out-of-tree examples.1

### **4.2. Using add\_subdirectory() for Modular CMake Configuration**

The add\_subdirectory() command is instrumental in creating a hierarchical build system, allowing distinct parts of a project to be configured and built independently.18 When

add\_subdirectory(source\_dir \[binary\_dir\]) is invoked, CMake processes the CMakeLists.txt file located in source\_dir.

A critical aspect of add\_subdirectory() for MLIR projects is its handling of the binary\_dir argument. If binary\_dir is not explicitly specified, it defaults to a path that mirrors the source\_dir within the *current output directory*.18 This behavior is fundamental to understanding where TableGen-generated output will reside. For instance, if the top-level

CMakeLists.txt is in my-mlir-project/ and the build directory is my-mlir-project/build/, then add\_subdirectory(include/MyDialect) will cause the CMakeLists.txt in my-mlir-project/include/MyDialect/ to be processed, and any generated files (such as those from add\_mlir\_dialect or mlir\_tablegen) will be placed in my-mlir-project/build/include/MyDialect/.10 This implicit mapping is predictable but necessitates careful path management when specifying include directories for compilation units that depend on these generated headers.

For out-of-tree projects, it is common to include the custom dialect and pass components using add\_subdirectory() calls in the top-level CMakeLists.txt:

CMake

add\_subdirectory(include/MyDialect)  
add\_subdirectory(lib/MyDialect)  
add\_subdirectory(tools/my-mlir-tool)  
add\_subdirectory(test/MyDialect)

This modular approach simplifies the overall CMake configuration and promotes a cleaner separation of concerns within the build system.

## **5\. Configuring CMake for TableGen-Generated Dialect Headers**

### **5.1. Defining the Custom Dialect (FooDialect.td, FooOps.td)**

The declarative definition of an MLIR dialect begins with TableGen files. These files specify the dialect's core properties and the operations it supports.

A primary TableGen file, often named MyDialect.td or FooBase.td, defines the dialect itself. This file typically includes mlir/IR/OpBase.td and declares the dialect's name and C++ namespace 10:

Code-Snippet

// include/MyDialect/MyDialect.td  
\#ifndef MY\_DIALECT\_BASE  
\#define MY\_DIALECT\_BASE

include "mlir/IR/OpBase.td"

def MyDialect\_Dialect : Dialect {  
  let name \= "mydialect";  
  let cppNamespace \= "::mlir::mydialect";  
  let description \=;  
}

\#endif // MY\_DIALECT\_BASE

Operations within the dialect are defined in a separate TableGen file, such as MyOps.td or FooOps.td. This file specifies the operations' arguments, results, traits, and can include custom builders or parsers.5

Code-Snippet

// include/MyDialect/MyOps.td  
\#ifndef MY\_DIALECT\_OPS  
\#define MY\_DIALECT\_OPS

include "MyDialect/MyDialect.td" // Include the dialect base definition  
include "mlir/IR/BuiltinTypes.td"  
include "mlir/IR/OpBase.td"

// Example operation definition  
def MyDialect\_AddOp : MyDialect\_Op\<"add",\> {  
  let summary \= "Addition operation";  
  let description \= \[{  
    Performs element-wise addition of two operands.  
  }\];  
  let arguments \= (ins AnyType:$lhs, AnyType:$rhs);  
  let results \= (outs AnyType:$result);  
}

\#endif // MY\_DIALECT\_OPS

For custom types and attributes, additional TableGen files like MyTypes.td and MyAttrs.td are created, often including mlir/IR/AttrTypeBase.td.9

### **5.2. Generating Dialect-Specific Headers using add\_mlir\_dialect() and mlir\_tablegen()**

Within the include/MyDialect/CMakeLists.txt file, the MLIR CMake macros are used to trigger the TableGen code generation.

The add\_mlir\_dialect() macro is used to process the primary operations TableGen file. This command automatically sets up the generation of MyOps.h.inc, MyOps.cpp.inc, MyDialect.h.inc, and MyDialect.cpp.inc, among others. It also creates the MLIRMyOpsIncGen target, which serves as a dependency for any C++ compilation that requires these generated files.7

CMake

\# include/MyDialect/CMakeLists.txt  
add\_mlir\_dialect(MyOps mydialect)

For custom attributes and types, which are not implicitly handled by add\_mlir\_dialect(), explicit invocations of mlir\_tablegen() are necessary. These commands specify the output file names and the specific generator flags for mlir-tblgen.9

add\_public\_tablegen\_target() is then used to create a public CMake target for these generated files, allowing other targets to declare dependencies on them.

CMake

\# For custom attributes (if defined in MyAttrs.td)  
set(LLVM\_TARGET\_DEFINITIONS MyAttrs.td)  
mlir\_tablegen(MyAttrDefs.h.inc \-gen-attrdef-decls \-attrdefs-dialect=mydialect)  
mlir\_tablegen(MyAttrDefs.cpp.inc \-gen-attrdef-defs \-attrdefs-dialect=mydialect)  
add\_public\_tablegen\_target(MLIRMyAttrDefsIncGen)

\# For custom types (if defined in MyTypes.td)  
set(LLVM\_TARGET\_DEFINITIONS MyTypes.td)  
mlir\_tablegen(MyTypes.h.inc \-gen-typedef-decls \-typedefs-dialect=mydialect)  
mlir\_tablegen(MyTypes.cpp.inc \-gen-typedef-defs \-typedefs-dialect=mydialect)  
add\_public\_tablegen\_target(MLIRMyTypesIncGen)

### **5.3. Ensuring Correct Include Paths for Generated Files**

The generated .h.inc files (e.g., MyOps.h.inc, MyAttrDefs.h.inc) are placed in the *binary directory* that corresponds to the CMakeLists.txt file where add\_mlir\_dialect() or mlir\_tablegen() was invoked.10 If

add\_subdirectory(include/MyDialect) was used in the top-level CMakeLists.txt, these files will reside in ${CMAKE\_BINARY\_DIR}/include/MyDialect/.

To ensure that the C++ compiler can find these generated headers, their containing directory must be added to the include paths of any target that uses them. This is achieved using target\_include\_directories().21

The target\_include\_directories() command allows specifying include paths with different scopes:

* **PRIVATE**: The include directories are used only when compiling the target itself.  
* **PUBLIC**: The include directories are used when compiling the target itself *and* are propagated to any other target that links against it.  
* **INTERFACE**: The include directories are *only* propagated to targets that link against this target. This is typically used for header-only libraries or for specifying usage requirements without directly compiling sources.

Crucially, to support both development (building from source) and deployment (using an installed package), **Generator Expressions** ($\<BUILD\_INTERFACE:...\> and $\<INSTALL\_INTERFACE:...\>) must be employed.21 This is a fundamental aspect of creating relocatable packages in CMake.

For the dialect library itself (e.g., MLIRMyDialect), which will include its own generated headers, the PUBLIC scope with generator expressions is appropriate:

CMake

\# In lib/MyDialect/CMakeLists.txt for target MLIRMyDialect  
target\_include\_directories(MLIRMyDialect PUBLIC  
    $\<BUILD\_INTERFACE:${CMAKE\_CURRENT\_BINARY\_DIR}/../include/MyDialect\> \# Path to generated headers in build tree  
    $\<INSTALL\_INTERFACE:include/MyDialect\> \# Path to installed headers relative to CMAKE\_INSTALL\_PREFIX  
)

The path ${CMAKE\_CURRENT\_BINARY\_DIR}/../include/MyDialect is used because if lib/MyDialect/CMakeLists.txt is being processed, CMAKE\_CURRENT\_BINARY\_DIR points to build/lib/MyDialect. The generated headers for MyDialect are in build/include/MyDialect, requiring the ../include/MyDialect relative path. The INSTALL\_INTERFACE path include/MyDialect assumes that during installation, the generated headers (or the hand-written headers that include them) will be installed under ${CMAKE\_INSTALL\_PREFIX}/include/MyDialect. This careful use of generator expressions ensures that the project compiles correctly in both the build tree and after installation, enabling the creation of shareable, out-of-tree MLIR components that function correctly regardless of their installation location.

### **5.4. Establishing Build Dependencies on TableGen Targets**

Any target (e.g., your MLIR pass library or a tool that uses your dialect) that includes the TableGen-generated headers *must* declare a build dependency on the corresponding TableGen generation target. This ensures that mlir-tblgen runs and produces the .h.inc files before the C++ compiler attempts to include them.

For operations and dialect declarations, the dependency is typically MLIRMyOpsIncGen (created by add\_mlir\_dialect).11 For attributes or types generated via explicit

mlir\_tablegen calls, the dependency would be MLIRMyAttrDefsIncGen or MLIRMyTypesIncGen (created by add\_public\_tablegen\_target).

CMake

\# In lib/MyDialect/CMakeLists.txt for target MLIRMyDialect  
add\_mlir\_dialect\_library(MLIRMyDialect  
    SOURCES  
        MyDialect.cpp  
        MyPass.cpp  
    DEPENDS  
        MLIRMyOpsIncGen \# Ensures ops and dialect headers are generated  
        MLIRMyAttrDefsIncGen \# Ensures attribute headers are generated  
        MLIRMyTypesIncGen \# Ensures type headers are generated  
    \#... other configurations  
)

This explicit dependency management is vital for maintaining a correct build order in complex projects involving code generation.

### **Table 2: Common TableGen Generated Files and Their Usage**

Understanding which TableGen files are generated and where they should be included in C++ source files is a common point of confusion for new MLIR developers. This table provides a quick reference to clarify the typical inclusion patterns for dialect-related generated files.

| Generated File (.inc suffix) | C++ Inclusion Point | Purpose |
| :---- | :---- | :---- |
| MyDialect.h.inc | include/MyDialect/MyDialect.h | Contains the C++ class declaration for the dialect itself. |
| MyDialect.cpp.inc | lib/MyDialect/MyDialect.cpp | Provides C++ method definitions for the dialect class. |
| MyOps.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyOps.h | Contains C++ class declarations for all operations defined in MyOps.td. |
| MyOps.cpp.inc | lib/MyDialect/MyDialect.cpp (with \#define GET\_OP\_LIST and \#define GET\_OP\_CLASSES) | Contains C++ method definitions for operations, used to register operations with the dialect and instantiate their classes. |
| MyTypes.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyTypes.h | Contains C++ class declarations for custom types. |
| MyTypes.cpp.inc | lib/MyDialect/MyDialect.cpp or lib/MyDialect/MyTypes.cpp | Provides C++ method definitions for custom types (e.g., parsing, printing). |
| MyAttrDefs.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyAttrs.h | Contains C++ class declarations for custom attributes. |
| MyAttrDefs.cpp.inc | lib/MyDialect/MyDialect.cpp or lib/MyDialect/MyAttrs.cpp | Provides C++ method definitions for custom attributes. |
| MyOpsInterfaces.h.inc | include/MyDialect/MyDialect.h or relevant interface headers | Contains C++ class declarations for operation interfaces. |
| MyOpsInterfaces.cpp.inc | lib/MyDialect/MyDialect.cpp or relevant interface source files | Provides C++ method definitions for operation interfaces. |

## **6\. Building and Linking the Out-of-Tree MLIR Pass**

### **6.1. Defining the MLIR Pass Library**

The implementation of the custom dialect and MLIR pass typically resides within a library target. In lib/MyDialect/CMakeLists.txt, a library is defined using add\_mlir\_dialect\_library() for dialect-specific code or add\_mlir\_library() for more general MLIR passes.11

add\_mlir\_dialect\_library() is generally preferred when the library contains dialect-specific implementations, as it automatically registers the library with the global MLIR\_DIALECT\_LIBS property, making it discoverable by MLIR tools.

The library definition includes its source files, dependencies on TableGen targets, and links to necessary MLIR components. LINK\_COMPONENTS is used for linking against standard LLVM/MLIR components (e.g., Core), while LINK\_LIBS is used for specific libraries or other CMake targets.11 The

PUBLIC keyword ensures that linking requirements are propagated transitively to targets that link against this library.

CMake

\# lib/MyDialect/CMakeLists.txt  
add\_mlir\_dialect\_library(MLIRMyDialect  
    SOURCES  
        MyDialect.cpp  
        MyPass.cpp  
    DEPENDS  
        MLIRMyOpsIncGen \# Dependency on generated operations and dialect  
        MLIRMyAttrDefsIncGen \# Dependency on generated attributes  
        MLIRMyTypesIncGen \# Dependency on generated types  
    LINK\_COMPONENTS  
        Core \# Link against core MLIR libraries  
    LINK\_LIBS  
        PUBLIC MLIRIR \# For MLIR core infrastructure  
        MLIRPass \# For pass management infrastructure  
        MLIRTransforms \# For common MLIR transformations  
        MLIRSupport \# For general MLIR support utilities  
        \#... other required MLIR/LLVM libraries as needed  
)

The use of add\_mlir\_dialect\_library effectively establishes an implicit contract: by using this macro, the dialect library becomes part of a global list (MLIR\_DIALECT\_LIBS) that can be automatically linked by MLIR tools, streamlining the build process for complex MLIR applications.

### **6.2. Linking Against the Custom Dialect Library and Core MLIR Components**

Any executable or library that intends to use your custom dialect or pass must link against the MLIRMyDialect library. For a custom mlir-opt-like tool, the add\_mlir\_tool() macro simplifies this process. This macro is provided by MLIR's CMake system and handles much of the boilerplate for creating MLIR-based executables.

CMake

\# tools/my-mlir-tool/CMakeLists.txt  
add\_mlir\_tool(my-mlir-tool  
    SOURCES  
        my-mlir-tool.cpp  
    LINK\_LIBS  
        MLIRMyDialect \# Link against your custom dialect and pass library  
        MLIRIR \# Core MLIR infrastructure  
        MLIRPass \# For pass management  
        MLIRTransforms \# For common MLIR transformations  
        MLIRSupport \# For general MLIR support utilities  
        \#... other required MLIR/LLVM libraries as needed  
)

It is important to link only the necessary core MLIR components (e.g., MLIRIR, MLIRPass, MLIRTransforms, MLIRSupport) rather than relying on a monolithic libMLIR.so or an overly comprehensive list of libraries.19 This practice, highlighted by the upstream MLIR project, reduces binary size, improves build times, and enhances maintainability by avoiding unnecessary dependencies.

### **6.3. Registering the Custom Dialect and Pass within an MLIR Tool**

Even after correctly configuring CMake and linking the libraries, the custom dialect and pass must be explicitly registered with the MLIRContext and PassManager within the C++ source code of the MLIR tool (e.g., my-mlir-tool.cpp). This ensures that the MLIR framework can discover and utilize your custom components at runtime.

First, include the public header for your dialect:

C++

// tools/my-mlir-tool/my-mlir-tool.cpp  
\#**include** "MyDialect/MyDialect.h" // Assuming this header includes MyDialect.h.inc and MyOps.h.inc  
\#**include** "MyDialect/MyPass.h"   // Assuming this header declares your pass  
\#**include** "mlir/InitAllDialects.h" // For registering all MLIR core dialects  
\#**include** "mlir/InitAllPasses.h"   // For registering all MLIR core passes  
\#**include** "mlir/Tools/mlir-opt/MlirOptMain.h"

Within the main function or an initialization routine of your tool, register your dialect and pass:

C++

// tools/my-mlir-tool/my-mlir-tool.cpp  
int main(int argc, char \*\*argv) {  
  mlir::DialectRegistry registry;  
  mlir::registerAllDialects(registry); // Register all core MLIR dialects  
  mlir::registerAllPasses(); // Register all core MLIR passes

  // Register your custom dialect  
  registry.insert\<mlir::mydialect::MyDialect\>(); \[10\]

  // Register your custom pass  
  mlir::PassRegistration\<mlir::mydialect::MyPass\>();

  return mlir::asMainReturnCode(  
      mlir::MlirOptMain(argc, argv, "My MLIR custom tool", registry));  
}

This explicit registration is a crucial runtime step that complements the CMake build configuration, enabling the MLIR framework to properly load and execute operations and passes from your out-of-tree components.

## **7\. Example CMakeLists.txt Snippets**

This section provides complete, minimal examples of the CMakeLists.txt files for an out-of-tree MLIR project, demonstrating the concepts discussed.

### **7.1. Top-Level CMakeLists.txt for the Out-of-Tree Project**

This file sets up the overall project, finds the MLIR installation, and includes the subdirectories for the custom dialect and tool.

CMake

\# my-mlir-project/CMakeLists.txt  
cmake\_minimum\_required(VERSION 3.20.0)  
project(MyMLIRProject LANGUAGES CXX C)

set(CMAKE\_CXX\_STANDARD 17 CACHE STRING "C++ standard to conform to")  
set(CMAKE\_POSITION\_INDEPENDENT\_CODE ON)

\# Find MLIR installation  
find\_package(MLIR REQUIRED CONFIG)  
message(STATUS "Using MLIRConfig.cmake in: ${MLIR\_DIR}")  
message(STATUS "Using LLVMConfig.cmake in: ${LLVM\_DIR}")

\# Add MLIR's CMake modules to the module path  
list(APPEND CMAKE\_MODULE\_PATH "${MLIR\_CMAKE\_DIR}")

\# Include essential LLVM/MLIR CMake utilities  
include(AddLLVM)  
include(TableGen)  
include(AddMLIR)

\# Globally include LLVM and MLIR headers (optional, but common)  
include\_directories(${LLVM\_INCLUDE\_DIRS})  
include\_directories(${MLIR\_INCLUDE\_DIRS})  
include\_directories(${PROJECT\_SOURCE\_DIR}/include) \# For hand-written public headers

\# Add subdirectories for the dialect, library, and tool  
add\_subdirectory(include/MyDialect)  
add\_subdirectory(lib/MyDialect)  
add\_subdirectory(tools/my-mlir-tool)  
add\_subdirectory(test/MyDialect) \# Optional: if you have tests

### **7.2. CMakeLists.txt for the Custom Dialect (include/MyDialect/CMakeLists.txt)**

This file handles the TableGen generation for the dialect's operations, attributes, and types, and sets up the include paths for the generated headers.

CMake

\# my-mlir-project/include/MyDialect/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(include/MyDialect)

\# Define the dialect and generate operation/dialect declarations  
\# This creates MLIRMyOpsIncGen target and MyOps.h.inc, MyOps.cpp.inc etc.  
add\_mlir\_dialect(MyOps mydialect)

\# For custom attributes (if MyAttrs.td exists)  
\# These will be generated in ${CMAKE\_CURRENT\_BINARY\_DIR} (e.g., build/include/MyDialect)  
set(LLVM\_TARGET\_DEFINITIONS MyAttrs.td)  
mlir\_tablegen(MyAttrDefs.h.inc \-gen-attrdef-decls \-attrdefs-dialect=mydialect)  
mlir\_tablegen(MyAttrDefs.cpp.inc \-gen-attrdef-defs \-attrdefs-dialect=mydialect)  
add\_public\_tablegen\_target(MLIRMyAttrDefsIncGen)

\# For custom types (if MyTypes.td exists)  
set(LLVM\_TARGET\_DEFINITIONS MyTypes.td)  
mlir\_tablegen(MyTypes.h.inc \-gen-typedef-decls \-typedefs-dialect=mydialect)  
mlir\_tablegen(MyTypes.cpp.inc \-gen-typedef-defs \-typedefs-dialect=mydialect)  
add\_public\_tablegen\_target(MLIRMyTypesIncGen)

\# Define a target for the public headers of the dialect  
\# This ensures that other targets can link against this and get the correct include paths  
add\_library(MyDialectHeaders INTERFACE)  
target\_include\_directories(MyDialectHeaders INTERFACE  
    $\<BUILD\_INTERFACE:${CMAKE\_CURRENT\_BINARY\_DIR}\> \# Generated headers in build tree  
    $\<INSTALL\_INTERFACE:include/MyDialect\> \# Installed headers relative to prefix  
)

### **7.3. CMakeLists.txt for the MLIR Pass Library (lib/MyDialect/CMakeLists.txt)**

This file defines the library containing the C++ implementations of the dialect and the custom pass, linking against necessary MLIR components and ensuring dependencies on generated headers.

CMake

\# my-mlir-project/lib/MyDialect/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(lib/MyDialect)

\# Define the dialect library, including its C++ sources  
\# and dependencies on the TableGen generated headers  
add\_mlir\_dialect\_library(MLIRMyDialect  
    SOURCES  
        MyDialect.cpp  
        MyPass.cpp  
    DEPENDS  
        MLIRMyOpsIncGen \# Ensure operations are generated before compilation  
        MLIRMyAttrDefsIncGen \# Ensure attributes are generated  
        MLIRMyTypesIncGen \# Ensure types are generated  
    LINK\_COMPONENTS  
        Core \# Link against core MLIR components  
    LINK\_LIBS  
        PUBLIC MLIRIR \# Core MLIR infrastructure  
        MLIRPass \# Pass management framework  
        MLIRTransforms \# Common MLIR transformations  
        MLIRSupport \# General MLIR support utilities  
        MyDialectHeaders \# Propagate include paths for hand-written and generated headers  
)

## **8\. Troubleshooting Common Configuration Issues**

Developing with MLIR and CMake can present several common challenges, particularly when integrating out-of-tree components and TableGen-generated code. Understanding the root causes of these issues is key to efficient resolution.

### **8.1. "Not a member of" or "undeclared identifier" errors**

These errors typically indicate that the C++ compiler cannot locate the necessary declarations for classes, functions, or members that are expected to be present, often originating from TableGen-generated files.

* **Cause**: The most frequent cause is that the include path to the TableGen-generated .h.inc files is not correctly added to the target's compilation flags, or the generation step itself has not completed before compilation.  
* **Diagnosis**:  
  * **Missing target\_include\_directories**: Verify that the directory containing the generated .h.inc files (e.g., ${CMAKE\_BINARY\_DIR}/include/MyDialect/) is correctly added to the PUBLIC or PRIVATE include paths of the target that is failing to compile.16 The behavior of  
    add\_subdirectory() and CMAKE\_CURRENT\_BINARY\_DIR dictates where these files are placed.  
  * **Missing DEPENDS on TableGen target**: Ensure that the C++ library or executable target explicitly declares a dependency on the TableGen generation target (e.g., MLIRMyOpsIncGen, MLIRMyAttrDefsIncGen).11 Without this dependency, CMake might attempt to compile C++ files before  
    mlir-tblgen has produced the required headers.  
  * **Incorrect \#include path in C++ source**: Double-check that the \#include directives in your C++ files use the correct relative path for the generated headers (e.g., \#include "MyDialect/MyOps.h.inc").  
  * **TableGen generation failure**: Examine the CMake output for any errors or warnings during the mlir-tblgen execution. If TableGen itself fails, the .h.inc files will not be created, leading to subsequent compilation errors. TableGen's diagnostics for unimplemented functions or contract violations can sometimes be subtle, requiring careful review of the TableGen documentation or source code.6  
* **Remedy**: Systematically verify and correct target\_include\_directories paths and ensure all necessary DEPENDS are specified on the relevant TableGen generation targets.

### **8.2. Linker errors (e.g., "undefined reference to...")**

Linker errors occur when the compiled code requires symbols (functions, variables) from a library that is not being linked or is linked incorrectly.

* **Cause**: The most common reason is that the necessary MLIR core libraries, your custom dialect library, or other required dependencies are not specified in the linking step.  
* **Diagnosis**:  
  * **Missing LINK\_LIBS or LINK\_COMPONENTS**: Review the target\_link\_libraries() calls for your executable or library. Ensure all required MLIR components (e.g., MLIRIR, MLIRPass, MLIRTransforms) and your custom dialect library (MLIRMyDialect) are explicitly listed.11  
  * **Incorrect PUBLIC/PRIVATE/INTERFACE scope**: If your custom pass library depends on your dialect library, and a tool links against your pass library, the dialect library must be linked with PUBLIC scope to ensure its symbols are transitively available to the tool.24  
  * **Dialect/Pass not registered at runtime**: Even if all libraries are correctly linked, the MLIR framework needs to be explicitly informed about your custom dialect and pass. Verify that registry.insert\<mlir::mydialect::MyDialect\>(); and mlir::PassRegistration\<mlir::mydialect::MyPass\>(); (or similar registration calls) are present and correctly executed in your tool's main function.10  
* **Remedy**: Meticulously review all target\_link\_libraries calls, ensuring that all required MLIR components and your custom libraries are linked with the appropriate scopes. Confirm that dialect and pass registration occurs correctly at runtime.

### **8.3. CMake configuration errors (e.g., find\_package failures, unknown command)**

These errors indicate fundamental issues with CMake's ability to set up the build environment.

* **Cause**: CMake cannot locate the MLIR installation or its custom modules, or MLIR-specific macros are being used before their definitions are included.  
* **Diagnosis**:  
  * **find\_package(MLIR) fails**: This typically means MLIR is not installed on the system, or CMake cannot find it. Ensure MLIR is installed and that the CMAKE\_PREFIX\_PATH environment variable is set to point to its installation directory (e.g., export CMAKE\_PREFIX\_PATH=/path/to/llvm-project/build/install).  
  * **Unknown command add\_mlir\_dialect (or similar MLIR macros)**: This error signifies that CMake has not yet processed the AddMLIR.cmake module, which defines these macros. The most common oversight is forgetting to add ${MLIR\_CMAKE\_DIR} to CMAKE\_MODULE\_PATH or failing to include(AddMLIR) after setting the module path.3  
* **Remedy**: Verify that MLIR is correctly installed and that the CMAKE\_PREFIX\_PATH is set appropriately. Ensure the correct order of list(APPEND CMAKE\_MODULE\_PATH...) and include(...) commands in your top-level CMakeLists.txt to make MLIR's custom macros available early in the configuration process.

## **9\. Conclusion and Best Practices**

### **Summary of Key Considerations for Out-of-Tree MLIR Development**

Developing MLIR components out-of-tree leverages MLIR's inherent modularity and CMake's powerful build system capabilities. The process fundamentally relies on a clear understanding of TableGen's code generation, MLIR's custom CMake macros, and precise path management. The core considerations include:

* **Modular Design**: MLIR's architecture is designed to support extensible components, allowing for independent development and integration.  
* **TableGen's Role**: TableGen is central to MLIR, significantly reducing boilerplate C++ code for operations, types, and attributes through declarative definitions.  
* **MLIR-Specific CMake Macros**: Macros like add\_mlir\_dialect(), mlir\_tablegen(), and add\_mlir\_dialect\_library() are indispensable for correctly integrating TableGen-generated code and managing MLIR-specific build rules.  
* **Path Management with Generator Expressions**: Correctly specifying include paths for generated headers using target\_include\_directories() with $\<BUILD\_INTERFACE\> and $\<INSTALL\_INTERFACE\> generator expressions is crucial for ensuring that the project builds and functions correctly in both development (build tree) and deployment (install tree) environments. This ensures the relocatability of the built components.  
* **Explicit Dependencies**: Establishing clear DEPENDS relationships on TableGen generation targets is vital to enforce the correct build order, preventing compilation errors due to missing generated files.

### **Recommendations for Maintainability and Scalability**

To ensure the long-term maintainability and scalability of out-of-tree MLIR projects, the following best practices are recommended:

* **Mirror MLIR's Internal Structure**: Adopting a directory layout similar to MLIR's in-tree organization (e.g., include/MyDialect, lib/MyDialect, tools/my-mlir-tool) enhances familiarity for developers and improves overall project clarity and maintainability.11  
* **Leverage add\_subdirectory() for Modularity**: Break down the project into logical CMake subdirectories. This modular approach simplifies the management of build rules, dependencies, and source files, making the project easier to navigate and extend.20  
* **Consistently Use CMake Generator Expressions**: Always employ $\<BUILD\_INTERFACE\> and $\<INSTALL\_INTERFACE\> for specifying include directories and other usage requirements. This practice is fundamental for creating robust and relocatable packages that can be easily distributed and used across different systems without requiring specific directory structures.21  
* **Be Explicit with Dependencies**: Clearly define DEPENDS for all TableGen-generated files and specify LINK\_LIBS/LINK\_COMPONENTS for all required libraries. This explicit declaration of dependencies makes the build graph transparent and helps prevent subtle build failures.  
* **Minimize Linked Libraries**: When building custom MLIR tools, avoid over-linking. Explicitly link only the MLIR components and other libraries that are strictly necessary for the tool's functionality. This practice reduces the size of the final executable and can significantly improve build times, as highlighted by efforts in upstream MLIR examples.19  
* **Stay Updated with MLIR Documentation**: The MLIR API and CMake best practices can evolve. Regularly consult the official MLIR documentation and analyze upstream examples to stay informed of the latest recommendations and potential API breaking changes.10

By adhering to these principles, developers can effectively configure CMake for out-of-tree MLIR passes that depend on TableGen-generated dialect headers, fostering a robust and extensible MLIR development workflow.