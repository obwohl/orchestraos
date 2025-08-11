

# **Resolving Linker-Based Symbol Elision for Statically Registered MLIR Dialects**

## **I. Introduction: The Anatomy of a Static Registration Linking Failure**

### **Problem Statement**

This report addresses a critical and subtle linker issue encountered within the orchestra-compiler project. The primary symptom is the failure of a custom tool, orchestra-opt, to recognize and load its associated Orchestra MLIR dialect. The root cause has been identified as linker-induced symbol elision: the C++ symbols corresponding to the dialect's custom operations, which are essential for their registration with the MLIR infrastructure, are being stripped from the final executable during the link stage. This occurs even though the dialect is correctly compiled into a static library, libOrchestra.a, and linked into the executable using the canonical \--whole-archive linker flag, the very tool designed to prevent this exact problem.

### **The Static Registration Pattern**

The issue originates from a common and powerful C++ software design pattern known as static, or self-registration. This pattern is used extensively throughout the LLVM and MLIR ecosystems to create modular and extensible systems.1 In the context of an MLIR dialect, each custom operation (or type, or attribute) is accompanied by a static global object. The constructor of this object is responsible for registering the operation with the MLIR framework's central registry. This process happens automatically before the execution of the

main function, as part of the C++ static initialization phase.3

The elegance of this pattern is also its primary vulnerability with respect to static linking. A static library (a .a file) is not a single linked unit but rather an archive of individual object files (.o files).5 When linking an executable against a static library, the linker's default behavior is to perform an optimization: it scans the archive and includes only those object files that satisfy an "undefined symbol" reference from the code already included in the link.7 The registration objects, being global variables, are not explicitly referenced by any function call from the main executable's code. Their constructors are called by the C++ runtime, not by user code. Consequently, the linker sees the object files containing these registration objects as "unused" and discards them to reduce the final executable size. This elision prevents the registration constructors from ever running, leading to runtime errors indicating the dialect was not loaded.8

### **Thesis of this Report**

The observed failure of the \--whole-archive flag is not indicative of a bug in the linker or the build tools. Rather, it is a symptom of a nuanced but critical breakdown in the communication between the CMake build system and the linker. The GNU linker (ld) is a stateful, command-line-driven tool whose behavior is acutely sensitive to the order of its arguments.7 The failure suggests that CMake, in the process of translating the

target\_link\_libraries directive into a final linker command, is arranging the arguments in a sequence that inadvertently nullifies the effect of \--whole-archive on the intended libOrchestra.a library. This report will deconstruct this interaction, provide a catalog of robust alternative solutions that bypass this fragility, and establish a protocol for definitive diagnosis and introspection of the linker's behavior.

## **II. In-Depth Analysis of the \--whole-archive Anomaly**

The failure of \--whole-archive is perplexing because it is the designated solution for the static registration problem.11 Its ineffectiveness in this context points not to a flaw in the flag itself, but to a misunderstanding or violation of the strict rules governing its use. The investigation must therefore focus on the precise mechanics of the linker's command-line processing.

### **The Primacy of Linker Command-Line Order**

The GNU linker does not process its command-line arguments as a declarative set of options. Instead, it operates like a state machine, parsing arguments sequentially from left to right. Many of its options are modal, meaning they alter the linker's behavior for all subsequent arguments until the mode is changed back.9

The \--whole-archive and \--no-whole-archive flags are prime examples of this modal behavior. The \--whole-archive option instructs the linker to abandon its selective inclusion logic and instead incorporate every object file from any archive that follows it on the command line. This state persists until the linker encounters a \--no-whole-archive option, which restores the default behavior.13 Therefore, the correct invocation for linking a single static library with this mechanism is an atomic, ordered triplet:

1. The \-Wl,--whole-archive flag to enable the mode.  
2. The library to be linked, e.g., libOrchestra.a.  
3. The \-Wl,--no-whole-archive flag to disable the mode immediately afterward, preventing it from unintentionally affecting other libraries that gcc or clang might add to the link line automatically (like libstdc++ or libgcc).12

Any deviation from this flag-library-flag sequence will lead to failure. If the library does not immediately follow the \--whole-archive flag, the linker will correctly apply the whole-archive logic to whatever *does* follow it, and not to the intended library. The flag is not being "ignored"; it is being applied precisely as designed, but to the wrong (or no) arguments.7

### **CMake's Translation Layer and Potential for Reordering**

The root of the problem most likely lies in how CMake translates the abstract dependency information from target\_link\_libraries into a concrete command line for the compiler driver (g++ or clang++). The command target\_link\_libraries is designed to aggregate link items from various sources, including direct specification, transitive dependencies from other targets, and usage requirements.17

The provided CMake snippet is:

CMake

target\_link\_libraries(orchestra-opt PRIVATE  
 ...  
  "-Wl,--whole-archive"  
  Orchestra  
  "-Wl,--no-whole-archive"  
)

Here, "-Wl,--whole-archive", Orchestra, and "-Wl,--no-whole-archive" are provided as three separate arguments in a list. While CMake often preserves the relative order of items within a single command, its generator may categorize these items. It could perceive "-Wl,--whole-archive" as a generic linker flag and group it with other flags, while treating Orchestra (which resolves to a library path like \-L/path/to \-lOrchestra or a direct path to libOrchestra.a) as a library item. This can lead to a final link command where the arguments are reordered, for example:  
g++... \-Wl,--whole-archive \-Wl,--no-whole-archive... /path/to/libOrchestra.a...  
In this reordered command, the \--whole-archive mode is enabled and immediately disabled before the linker ever sees libOrchestra.a, rendering the entire mechanism ineffective.

A critical piece of evidence for this behavior comes from a reported issue in a similar build environment, where passing linker flags and libraries as separate list items failed. The solution was to combine them into a single, comma-delimited string for the \-Wl flag 20:

"-Wl,--whole-archive,path/to/libfoo.a,--no-whole-archive"  
This syntax forces the compiler driver to pass \--whole-archive, path/to/libfoo.a, and \--no-whole-archive as a single, contiguous block of arguments to the linker. This effectively makes the triplet atomic from CMake's perspective, preventing it from inserting other arguments and breaking the required sequence. This strongly suggests that the problem is one of argument separation during CMake's generation phase.

### **Toolchain-Specific Quirks**

While no specific bugs matching this exact toolchain configuration (Ubuntu 22.04, LLVM 18.1, CMake 3.22.1, GNU ld) were found, linker behavior can be complex. An issue involving wasm-ld demonstrated that unexpected interactions with flag ordering can occur, though these are typically rooted in the linker's documented argument processing rules rather than being outright bugs.21 The current problem is almost certainly a manifestation of the linker's documented, strict, order-dependent design being inadvertently violated by the build system generator.

### **Integrity of the Static Archive (libOrchestra.a)**

For completeness, the integrity of the static archive itself must be considered. A static library is created by the ar archiver utility.5 For the linker to use the archive efficiently, it must contain a symbol table index. This index is created by the

ranlib command or, more commonly today, by the s modifier to ar (e.g., ar \-rcs).22

A missing or corrupt index would typically lead to "undefined reference" errors for symbols that *are* explicitly referenced, as the linker would be unable to find the object file containing their definition. It is less likely to be the cause of \--whole-archive failing, because that flag's purpose is to override the linker's symbol-driven decision-making process entirely. The problem described is that the linker is actively *choosing* to discard object files, not that it *cannot find* them. Therefore, while ensuring libOrchestra.a is correctly created with an index is a necessary piece of due diligence, the primary focus of the investigation should remain on the final linker command line generated by CMake.

## **III. A Compendium of Alternative Strategies for Forcing Symbol Preservation**

Given the fragility of the \--whole-archive approach, exploring alternative strategies is prudent. These methods range from minor source-code annotations to significant architectural changes, offering different trade-offs in terms of portability, maintainability, and implementation effort.

### **Source-Level Annotation: \_\_attribute\_\_((used))**

A direct and targeted approach is to instruct the compiler and linker that a specific symbol must be preserved, regardless of whether it appears to be referenced. The \_\_attribute\_\_((used)) attribute, a common extension in GCC and Clang, serves this purpose.23

When applied to a variable, this attribute ensures that the variable is emitted into the object file's symbol table even if it is static and seemingly unused. This creates a symbol that the linker is much less likely to garbage collect. This attribute is perfectly suited for marking the static registration objects in the MLIR dialect.

#### **Implementation**

The change is made directly in the C++ source files where the dialect's operations are implemented.

C++

// In, for example, lib/Orchestra/OrchestraOps.cpp

// Assume this is the static object responsible for registering 'my\_op'  
static mlir::DialectOpRegistration\<MyOp\> op\_registration \_\_attribute\_\_((used));

This modification requires no changes to the CMakeLists.txt files. It directly annotates the problematic symbols, providing a clear signal to the toolchain to retain them. The primary drawback is its lack of portability; it is a compiler-specific extension and will not work with compilers like MSVC, which uses a different mechanism (\#pragma comment(linker, "/INCLUDE:...")). Snippets 53 through 54 discuss C++ attributes more broadly but confirm that

\_\_attribute\_\_((used)) is the correct tool for this specific task.

### **Build-System Refactoring: The OBJECT Library Approach**

A more robust and modern CMake-native solution is to change the way the dialect's source files are handled by the build system. Instead of bundling them into a static archive, they can be compiled into a collection of object files known as an OBJECT library. These object files can then be linked directly into the final orchestra-opt executable.

This approach completely sidesteps the logic of selective linking from a static archive. When object files are provided directly on the linker command line, they are included unconditionally.

#### **CMake Integration**

This requires a two-part change to the CMakeLists.txt files.

1. **Modify the library definition:** In lib/Orchestra/CMakeLists.txt, change the library type from STATIC to OBJECT.  
   CMake  
   \# In lib/Orchestra/CMakeLists.txt

   \# Before:  
   \# add\_library(Orchestra STATIC ${ORCHESTRA\_SOURCES})

   \# After:  
   add\_library(Orchestra OBJECT ${ORCHESTRA\_SOURCES})  
   \# Any target\_include\_directories or other properties remain the same.

2. **Modify the executable's link step:** In orchestra-opt/CMakeLists.txt, replace the Orchestra library target with a generator expression that refers to its object files.  
   CMake  
   \# In orchestra-opt/CMakeLists.txt

   \# Remove the \--whole-archive flags and the library name  
   target\_link\_libraries(orchestra-opt PRIVATE  
    ...  
     \# Instead of "-Wl,--whole-archive" Orchestra "-Wl,--no-whole-archive", use:  
     $\<TARGET\_OBJECTS:Orchestra\>  
    ...  
   )

This approach is highly recommended. It is portable across platforms and linkers, as it relies on fundamental CMake features rather than linker-specific flags.25 While

OBJECT libraries have some complexities regarding the propagation of transitive dependencies, this is not a concern when they are consumed directly by an executable target.26

### **Direct Linker Control: Employing Linker Scripts**

For maximum control over the linker's behavior, a linker script can be used. This is a powerful but highly platform-specific mechanism. A linker script is a file containing explicit commands that guide the linking process. The KEEP() directive can be used to forcefully include input sections from specific object files, overriding any garbage collection logic.28

#### **Minimal Example Script**

A linker script can be crafted to target only the necessary object files. For example, if the registration code resides in OrchestraDialect.cpp and the various \*Ops.cpp files, the script could be:

Code-Snippet

/\* orchestra.ld \*/  
/\* This script explicitly tells the linker to not discard code or data \*/  
/\* from the object files that define the Orchestra dialect and its operations. \*/  
KEEP(\*(\*OrchestraDialect.cpp.o))  
KEEP(\*(\*OrchestraOps.cpp.o))

The wildcards ensure that this rule applies to the object files generated from any source file ending in OrchestraOps.cpp.

#### **CMake Integration**

The linker script is passed to the linker using the \-T flag. This is best done using target\_link\_options in CMake.

CMake

\# In orchestra-opt/CMakeLists.txt  
target\_link\_options(orchestra-opt PRIVATE "-T${CMAKE\_CURRENT\_SOURCE\_DIR}/orchestra.ld")

This approach provides surgical control but comes at a high cost. It makes the build process dependent on the GNU ld linker and its specific script syntax, reducing portability. It also introduces a level of indirection that can make the build harder to understand and maintain for developers unfamiliar with linker scripts.29 It should be considered a solution of last resort.

### **Architectural Pivot: Dynamic Dialect Loading**

A completely different approach is to embrace dynamic linking. The Orchestra dialect can be compiled as a shared library (.so file) instead of a static one. The orchestra-opt tool can then load this dialect at runtime as a plugin. This entirely avoids the problem of static linker elision.

MLIR is designed with this workflow in mind. Tools like mlir-opt provide a \--load-dialect-plugin command-line option to load dialects from shared libraries.31

#### **CMake Integration and Usage**

1. **Build a shared library:** In lib/Orchestra/CMakeLists.txt, change the library type to SHARED. Additional properties may be needed to control symbol visibility.  
   CMake  
   \# In lib/Orchestra/CMakeLists.txt  
   add\_library(Orchestra SHARED ${ORCHESTRA\_SOURCES})

2. **Runtime loading:** The orchestra-opt tool would no longer be linked against libOrchestra. Instead, it would be invoked with a command-line flag:  
   Bash  
   /path/to/orchestra-opt \--load-dialect-plugin=/path/to/libOrchestra.so \[other options\]

This represents a significant architectural shift. It provides excellent modularity but introduces the complexities of managing shared library dependencies, paths (e.g., rpath), and ensuring ABI (Application Binary Interface) compatibility between the tool and the plugin.33 This is a powerful and valid architecture for MLIR, but its adoption should be considered for its broader benefits, not merely as a workaround for a static linking issue.1

### **The Manual Fallback: Explicit Symbol Referencing**

The most traditional C++ workaround for this problem is to manually create an explicit dependency that the linker can follow.36 This involves creating a single function within the static library that makes a dummy reference to a symbol from each object file that needs to be preserved. This function is then called from

main(), forcing the linker to include the entire chain of dependencies.

#### **Implementation**

1. **Create a registration function in the library:**  
   C++  
   // In a new file, e.g., lib/Orchestra/OrchestraRegistration.cpp  
   \#**include** "mlir/IR/Dialect.h"

   namespace mlir {  
   namespace orchestra {  
     // In each Op's.cpp file, define a dummy function:  
     // void referenceMyOp1() {}  
     // Then declare them here.  
     void referenceMyOp1();  
     void referenceMyOp2();  
     //... for every operation...

     // This function creates the explicit reference.  
     void ensureRegistration() {  
       referenceMyOp1();  
       referenceMyOp2();  
       //...  
     }  
   } // namespace orchestra  
   } // namespace mlir

2. **Call the function from the executable:**  
   C++  
   // In orchestra-opt/main.cpp  
   namespace mlir {  
   namespace orchestra {  
     void ensureRegistration();  
   } // namespace orchestra  
   } // namespace mlir

   int main(int argc, char \*\*argv) {  
     // This call creates the link-time dependency.  
     mlir::orchestra::ensureRegistration();

     mlir::DialectRegistry registry;  
     // The dialect should now be registered and available.  
     //... rest of main...  
   }

This method is guaranteed to work and is completely portable. However, it imposes a significant maintenance burden. Every time a developer adds a new operation to the dialect, they must remember to update the ensureRegistration function. This is error-prone and undermines the "automatic" nature of the self-registration pattern.

## **IV. A Protocol for Advanced Linker Introspection and Debugging**

To move from hypothesis to a definitive diagnosis, direct inspection of the build process and its artifacts is required.

### **Revealing the Linker's Actions**

The first step is to observe the exact command line that CMake generates for the linker.

* **CMake Verbose Output:** The most straightforward method is to invoke the build system in verbose mode. For Makefile-based generators, this is done by setting the VERBOSE environment variable.37  
  Bash  
  \# From the build directory  
  make VERBOSE=1 orchestra-opt

  Alternatively, configuring the project with \-DCMAKE\_VERBOSE\_MAKEFILE=ON achieves the same result for the entire build.39 This will cause  
  make to print the full, unabbreviated g++... command used to link orchestra-opt. This command is the ground truth and should be inspected carefully to verify the order of the \--whole-archive flags and the libOrchestra.a library.  
* **Linker Tracing:** For an even deeper view, the linker can be instructed to trace its own execution. Passing the \--trace flag via \-Wl,--trace in target\_link\_options will cause ld to print a detailed log of every file it opens and every symbol it resolves. This is invaluable for understanding its decision-making process.

### **Forensic Symbol Table Analysis**

Comparing the symbols present in the library with those in the final executable provides definitive proof of elision. The standard binutils tools—nm, objdump, and readelf—are essential for this analysis.40

#### **The Diagnostic Process**

1. **Baseline (Object File):** First, inspect a single object file known to contain a registration symbol. Use nm with the \-C flag to demangle C++ names.  
   Bash  
   nm \-C lib/Orchestra/CMakeFiles/Orchestra.dir/OrchestraOps.cpp.o | grep 'op\_registration'

   This should show the symbol for the static registration object, confirming it was correctly compiled.  
2. **Archive Check:** Next, inspect the entire static library archive.  
   Bash  
   nm \-C lib/libOrchestra.a | grep 'op\_registration'

   This confirms that the object file was correctly added to the archive and its symbols are present.  
3. **Final Executable Check:** Finally, inspect the linked executable.  
   Bash  
   nm \-C bin/orchestra-opt | grep 'op\_registration'

   In the failing scenario, this command will produce no output, proving that the symbol was elided by the linker.  
4. **Advanced Inspection:** For more detail, readelf \-sW \<file\> provides a richer view of the symbol table, including symbol type, binding (GLOBAL, WEAK, LOCAL), and visibility.41  
   objdump \-tT \<file\> is another excellent tool for this purpose.44 Comparing the output of these tools on  
   libOrchestra.a and orchestra-opt will reveal precisely which symbols are being discarded.45 The key is to look for symbols with  
   GLOBAL or WEAK binding in the .data or .bss sections of the library that are absent from the executable.

### **Constructing a Minimal Reproducible Example (MRE)**

To rapidly test potential solutions without the overhead of the full MLIR project, constructing a Minimal Reproducible Example (MRE) is invaluable.50 An MRE isolates the core problem into the smallest possible self-contained project.

#### **MRE Implementation**

The following file structure creates a perfect analogue of the symbol elision problem.

* **lib/foo.cpp**: Mimics the registration object.  
  C++  
  \#**include** \<iostream\>  
  struct Registrar {  
      Registrar() {  
          std::cout \<\< " Registrar constructed. Symbol was NOT elided." \<\< std::endl;  
      }  
  };  
  // This static object's constructor is the payload.  
  static Registrar r;

* **lib/CMakeLists.txt**: Creates the static library.  
  CMake  
  cmake\_minimum\_required(VERSION 3.22)  
  project(FooLib CXX)  
  add\_library(foo STATIC foo.cpp)

* **main.cpp**: A minimal executable that does nothing.  
  C++  
  int main() {  
      // This main function makes no reference to anything in libfoo.  
      return 0;  
  }

* **CMakeLists.txt**: The top-level file that links them.  
  CMake  
  cmake\_minimum\_required(VERSION 3.22)  
  project(MRE CXX)  
  add\_subdirectory(lib)  
  add\_executable(main\_app main.cpp)

  \# This reproduces the problematic link command.  
  target\_link\_libraries(main\_app PRIVATE  
      "-Wl,--whole-archive"  
      foo  
      "-Wl,--no-whole-archive"  
  )

When this MRE is compiled and run, the message from the Registrar constructor will not be printed, confirming that the symbol was elided. This small test case provides a fast and reliable environment for verifying the solutions proposed in this report.3

## **V. Synthesis and Strategic Recommendations**

The analysis indicates that the root cause of the symbol elision is almost certainly a linker command-line ordering issue created by CMake's build generation process. The following recommendations provide a clear path to resolution, prioritizing robust, maintainable, and modern solutions.

### **Diagnostic Flowchart**

A systematic approach to diagnosis is recommended:

1. Execute the build with make VERBOSE=1 to expose the full linker command.  
2. Examine the command. Does libOrchestra.a (or \-lOrchestra) appear immediately after \-Wl,--whole-archive?  
   * **If NO:** The diagnosis is confirmed as a CMake argument reordering problem.  
     * **Immediate Fix:** Attempt to use the comma-separated linker flag syntax within target\_link\_libraries: "-Wl,--whole-archive,/path/to/libOrchestra.a,--no-whole-archive".  
     * **Recommended Long-Term Fix:** Refactor the build to use the OBJECT library approach (Section III.B). This is the most robust and idiomatic CMake solution.  
   * **If YES:** The command line appears correct, yet the failure persists. This is a highly improbable scenario but suggests a more obscure toolchain interaction.  
     * **Action:** Proceed to more forceful methods. The \_\_attribute\_\_((used)) source-level annotation (Section III.A) is the next logical step, as it directly targets the symbols in question. As a last resort, a linker script (Section III.C) can provide absolute control.

### **Comparative Analysis of Symbol Preservation Strategies**

To aid in selecting the most appropriate solution for the orchestra-compiler project, the following table compares the primary strategies across several key metrics.

| Strategy | Implementation Complexity | Build-Time Impact | Portability | Maintainability | Architectural Intrusiveness |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **\--whole-archive (Corrected)** | Low | Minimal | High (if flags correct) | High | None |
| **\_\_attribute\_\_((used))** | Low | None | Low (GCC/Clang only) | Medium (Attribute clutter) | Low |
| **OBJECT Library** | Low | Minimal | High | High | Low |
| **Linker Script** | Medium | None | Very Low (Linker-specific) | Low (Opaque, brittle) | Medium |
| **Dynamic Linking** | High | Increased | High | Medium (ABI/path issues) | High |
| **Explicit Referencing** | Medium | None | Very High | Very Low (Manual, error-prone) | Medium |

### **Quick Reference: Key Introspection Commands**

This table consolidates the essential commands for executing the debugging protocol outlined in Section IV.

| Command | Purpose | What to Look For |
| :---- | :---- | :---- |
| make VERBOSE=1 | Display the exact link command. | The order of \-Wl,--whole-archive, libOrchestra.a, and \-Wl,--no-whole-archive. |
| nm \-C \<file\> | List demangled symbols. | Presence/absence of registration symbols (e.g., ...op\_registration...). |
| readelf \-sW \<file\> | Detailed symbol table view. | Symbol type (e.g., OBJECT), binding (GLOBAL vs. LOCAL), and section index. |
| objdump \-tT \<file\> | Alternative symbol table view. | Compare symbols in .a vs. final executable. |
| objdump \-d \<file\> | Disassemble executable sections. | Confirm if code from the dialect's object files is present. |

### **Final Recommendations**

1. **Highest Recommendation: OBJECT Library Approach.** This strategy is the most effective and durable solution. By refactoring libOrchestra to be an OBJECT library, the project leverages a modern CMake feature that is designed for this exact use case. It eliminates the reliance on fragile, order-dependent linker flags, thereby increasing the build's robustness and portability. This should be the primary course of action.  
2. **Alternative Recommendation: Correcting \--whole-archive.** If modifying the library type from STATIC to OBJECT is undesirable due to project constraints, the next best solution is to ensure the \--whole-archive flag is passed correctly. This likely involves using the comma-separated \-Wl,... syntax to make the flag-library-flag group atomic to CMake. This preserves the existing project structure while fixing the immediate defect.  
3. **Contingency Recommendation: \_\_attribute\_\_((used)).** If the above build-system-level solutions prove infeasible, applying the \_\_attribute\_\_((used)) annotation directly to the static registration objects in the C++ source is a targeted and effective fix. Its main trade-off is the loss of compiler portability.

The remaining strategies—linker scripts and explicit manual referencing—are strongly discouraged due to their high maintenance cost, brittleness, and lack of portability. Dynamic linking is a valid architectural direction but represents a much larger decision that should be evaluated on its own merits rather than as a simple fix for this linking issue.

#### **Referenzen**

1. FAQ \- MLIR \- LLVM, Zugriff am August 12, 2025, [https://mlir.llvm.org/getting\_started/Faq/](https://mlir.llvm.org/getting_started/Faq/)  
2. MLIR Dialects in Catalyst \- PennyLane Documentation, Zugriff am August 12, 2025, [https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html)  
3. Death by static initialization \- zeux.io, Zugriff am August 12, 2025, [https://zeux.io/2010/10/10/death-by-static-initialization/](https://zeux.io/2010/10/10/death-by-static-initialization/)  
4. C++ \- Initialization of Static Variables \- pablo arias, Zugriff am August 12, 2025, [https://pabloariasal.github.io/2020/01/02/static-variable-initialization/](https://pabloariasal.github.io/2020/01/02/static-variable-initialization/)  
5. ar(1) \- Linux manual page \- man7.org, Zugriff am August 12, 2025, [https://man7.org/linux/man-pages/man1/ar.1.html](https://man7.org/linux/man-pages/man1/ar.1.html)  
6. Easily Create Shared Libraries with CMake (Part 1\) \- Alexander's Programming Tips, Zugriff am August 12, 2025, [https://blog.shaduri.dev/easily-create-shared-libraries-with-cmake-part-1](https://blog.shaduri.dev/easily-create-shared-libraries-with-cmake-part-1)  
7. Library order in static linking \- Eli Bendersky's website, Zugriff am August 12, 2025, [https://eli.thegreenplace.net/2013/07/09/library-order-in-static-linking](https://eli.thegreenplace.net/2013/07/09/library-order-in-static-linking)  
8. Not registering a new type with the dialect in MLIR results in error "LLVM ERROR: can't create type... because storage uniquer isn't initialized, Zugriff am August 12, 2025, [https://discourse.llvm.org/t/not-registering-a-new-type-with-the-dialect-in-mlir-results-in-error-llvm-error-cant-create-type-because-storage-uniquer-isnt-initialized-the-dialect-was-likely-not-loaded/4500/1](https://discourse.llvm.org/t/not-registering-a-new-type-with-the-dialect-in-mlir-results-in-error-llvm-error-cant-create-type-because-storage-uniquer-isnt-initialized-the-dialect-was-likely-not-loaded/4500/1)  
9. ld \- The GNU linker \- Ubuntu Manpage, Zugriff am August 12, 2025, [https://manpages.ubuntu.com/manpages/lunar/man1/ld.1.html](https://manpages.ubuntu.com/manpages/lunar/man1/ld.1.html)  
10. Options (LD) \- Sourceware, Zugriff am August 12, 2025, [https://sourceware.org/binutils/docs/ld/Options.html](https://sourceware.org/binutils/docs/ld/Options.html)  
11. ld linker question: the \--whole-archive option \- Stack Overflow, Zugriff am August 12, 2025, [https://stackoverflow.com/questions/805555/ld-linker-question-the-whole-archive-option](https://stackoverflow.com/questions/805555/ld-linker-question-the-whole-archive-option)  
12. ld(1) \- Linux manual page \- man7.org, Zugriff am August 12, 2025, [https://man7.org/linux/man-pages/man1/ld.1.html](https://man7.org/linux/man-pages/man1/ld.1.html)  
13. ld.lld — ELF linker from the LLVM project \- Ubuntu Manpage, Zugriff am August 12, 2025, [https://manpages.ubuntu.com/manpages/bionic/man1/ld.lld-8.1.html](https://manpages.ubuntu.com/manpages/bionic/man1/ld.lld-8.1.html)  
14. The GNU linker, Zugriff am August 12, 2025, [https://www.eecs.umich.edu/courses/eecs373/readings/Linker.pdf](https://www.eecs.umich.edu/courses/eecs373/readings/Linker.pdf)  
15. Using LD, the GNU linker, Zugriff am August 12, 2025, [https://astro.uni-bonn.de/\~sysstw/CompMan/gnu/ld.html](https://astro.uni-bonn.de/~sysstw/CompMan/gnu/ld.html)  
16. Using LD, the GNU linker \- Invocation \- Utah Math Department, Zugriff am August 12, 2025, [https://www.math.utah.edu/docs/info/ld\_2.html](https://www.math.utah.edu/docs/info/ld_2.html)  
17. target\_link\_libraries — CMake 3.0.2 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/v3.0/command/target\_link\_libraries.html](https://cmake.org/cmake/help/v3.0/command/target_link_libraries.html)  
18. target\_link\_libraries — CMake 3.2.3 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/v3.2/command/target\_link\_libraries.html](https://cmake.org/cmake/help/v3.2/command/target_link_libraries.html)  
19. target\_link\_libraries — CMake 4.1.0 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/latest/command/target\_link\_libraries.html](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)  
20. \[bug\] CMakeDeps/CMakeToolchain generators put quotation marks around exelinkflags in linker call if they contain spaces · Issue \#16634 · conan-io/conan \- GitHub, Zugriff am August 12, 2025, [https://github.com/conan-io/conan/issues/16634](https://github.com/conan-io/conan/issues/16634)  
21. wasm-ld should ignore .rlib files in archives · Issue \#55786 · llvm/llvm-project \- GitHub, Zugriff am August 12, 2025, [https://github.com/llvm/llvm-project/issues/55786](https://github.com/llvm/llvm-project/issues/55786)  
22. ranlib(1) \- Linux manual page \- man7.org, Zugriff am August 12, 2025, [https://man7.org/linux/man-pages/man1/ranlib.1.html](https://man7.org/linux/man-pages/man1/ranlib.1.html)  
23. Variable Attributes \- Using the GNU Compiler Collection (GCC), Zugriff am August 12, 2025, [https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Variable-Attributes.html](https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Variable-Attributes.html)  
24. attribute\_\_((used)) function attribute \- ARM Compiler v5.06 for uVision armcc User Guide, Zugriff am August 12, 2025, [https://developer.arm.com/documentation/dui0375/latest/Compiler-specific-Features/--attribute----used---function-attribute](https://developer.arm.com/documentation/dui0375/latest/Compiler-specific-Features/--attribute----used---function-attribute)  
25. add\_library — CMake 4.1.0 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/latest/command/add\_library.html](https://cmake.org/cmake/help/latest/command/add_library.html)  
26. How can I "link" a CMake object library to another CMake object library? \- Stack Overflow, Zugriff am August 12, 2025, [https://stackoverflow.com/questions/75339783/how-can-i-link-a-cmake-object-library-to-another-cmake-object-library](https://stackoverflow.com/questions/75339783/how-can-i-link-a-cmake-object-library-to-another-cmake-object-library)  
27. Object Libraries \- propagate target\_link\_libraries to final target (\#18090) · Issue \- GitLab, Zugriff am August 12, 2025, [https://gitlab.kitware.com/cmake/cmake/-/issues/18090](https://gitlab.kitware.com/cmake/cmake/-/issues/18090)  
28. ld \- What does KEEP mean in a linker script? \- Stack Overflow, Zugriff am August 12, 2025, [https://stackoverflow.com/questions/9827157/what-does-keep-mean-in-a-linker-script](https://stackoverflow.com/questions/9827157/what-does-keep-mean-in-a-linker-script)  
29. stm32-cmake/examples/custom-linker-script/CMakeLists.txt at master \- GitHub, Zugriff am August 12, 2025, [https://github.com/ObKo/stm32-cmake/blob/master/examples/custom-linker-script/CMakeLists.txt](https://github.com/ObKo/stm32-cmake/blob/master/examples/custom-linker-script/CMakeLists.txt)  
30. Linker Script Generation \- ESP32-C3 \- — ESP-IDF Programming Guide v5.4.2 documentation \- Espressif Systems, Zugriff am August 12, 2025, [https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-guides/linker-script-generation.html](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-guides/linker-script-generation.html)  
31. D147053 Implement Pass and Dialect plugins for mlir-opt, Zugriff am August 12, 2025, [https://reviews.llvm.org/D147053](https://reviews.llvm.org/D147053)  
32. \[MLIR\] Enabling external dialects to be shared libs for C API users \#108253 \- GitHub, Zugriff am August 12, 2025, [https://github.com/llvm/llvm-project/issues/108253](https://github.com/llvm/llvm-project/issues/108253)  
33. Creating a Dialect \- MLIR, Zugriff am August 12, 2025, [https://mlir.llvm.org/docs/Tutorials/CreatingADialect/](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)  
34. AIRCC \- The AIR Compiler Driver | AMD AIR MLIR Dialect, Zugriff am August 12, 2025, [https://xilinx.github.io/mlir-air/aircc.html](https://xilinx.github.io/mlir-air/aircc.html)  
35. Chapter 2: Emitting Basic MLIR \- LLVM, Zugriff am August 12, 2025, [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)  
36. Whole archive and self registration : r/cpp \- Reddit, Zugriff am August 12, 2025, [https://www.reddit.com/r/cpp/comments/1j1daij/whole\_archive\_and\_self\_registration/](https://www.reddit.com/r/cpp/comments/1j1daij/whole_archive_and_self_registration/)  
37. VERBOSE — CMake 4.1.0 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/latest/envvar/VERBOSE.html](https://cmake.org/cmake/help/latest/envvar/VERBOSE.html)  
38. cmake(1) — CMake 4.1.0 Documentation, Zugriff am August 12, 2025, [https://cmake.org/cmake/help/latest/manual/cmake.1.html](https://cmake.org/cmake/help/latest/manual/cmake.1.html)  
39. With CMAKE\_VERBOSE\_MAKEFILE on, CMake only enables verbosity for compilation, NOT linking (\#16649) · Issue \- GitLab, Zugriff am August 12, 2025, [https://gitlab.kitware.com/cmake/cmake/-/issues/16649](https://gitlab.kitware.com/cmake/cmake/-/issues/16649)  
40. llvm-objdump \- LLVM's object file dumper \- ROCm Documentation, Zugriff am August 12, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/CommandGuide/llvm-objdump.html](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/CommandGuide/llvm-objdump.html)  
41. readelf (GNU Binary Utilities) \- Sourceware, Zugriff am August 12, 2025, [https://sourceware.org/binutils/docs/binutils/readelf.html](https://sourceware.org/binutils/docs/binutils/readelf.html)  
42. nm \- Display symbol table of object, library, or executable files \- IBM, Zugriff am August 12, 2025, [https://www.ibm.com/docs/en/zos/2.4.0?topic=scd-nm-display-symbol-table-object-library-executable-files](https://www.ibm.com/docs/en/zos/2.4.0?topic=scd-nm-display-symbol-table-object-library-executable-files)  
43. llvm-readelf \- GNU-style LLVM Object Reader — LLVM 22.0.0git documentation, Zugriff am August 12, 2025, [https://llvm.org/docs/CommandGuide/llvm-readelf.html](https://llvm.org/docs/CommandGuide/llvm-readelf.html)  
44. llvm-objdump \- LLVM's object file dumper, Zugriff am August 12, 2025, [https://llvm.org/docs/CommandGuide/llvm-objdump.html](https://llvm.org/docs/CommandGuide/llvm-objdump.html)  
45. objdump(1) \- Linux manual page \- man7.org, Zugriff am August 12, 2025, [https://man7.org/linux/man-pages/man1/objdump.1.html](https://man7.org/linux/man-pages/man1/objdump.1.html)  
46. linux \- compare executable or object file \- Stack Overflow, Zugriff am August 12, 2025, [https://stackoverflow.com/questions/28808063/compare-executable-or-object-file](https://stackoverflow.com/questions/28808063/compare-executable-or-object-file)  
47. Lab 03 \- Executables. Static Analysis \[CS Open CourseWare\], Zugriff am August 12, 2025, [https://ccom.uprrp.edu/\~rarce/ccom4995/ref/buc/Lab%2003%20-%20Executables.%20Static%20Analysis%20\[CS%20Open%20CourseWare\].html](https://ccom.uprrp.edu/~rarce/ccom4995/ref/buc/Lab%2003%20-%20Executables.%20Static%20Analysis%20[CS%20Open%20CourseWare].html)  
48. How to compare two executable or lib.a file? : r/cpp\_questions \- Reddit, Zugriff am August 12, 2025, [https://www.reddit.com/r/cpp\_questions/comments/ncknyg/how\_to\_compare\_two\_executable\_or\_liba\_file/](https://www.reddit.com/r/cpp_questions/comments/ncknyg/how_to_compare_two_executable_or_liba_file/)  
49. How objdump disassemble elf binary \- Unix & Linux Stack Exchange, Zugriff am August 12, 2025, [https://unix.stackexchange.com/questions/343013/how-objdump-disassemble-elf-binary](https://unix.stackexchange.com/questions/343013/how-objdump-disassemble-elf-binary)  
50. Minimal reproducible example \- Wikipedia, Zugriff am August 12, 2025, [https://en.wikipedia.org/wiki/Minimal\_reproducible\_example](https://en.wikipedia.org/wiki/Minimal_reproducible_example)  
51. Linker error with static member variable · Fekir's Blog, Zugriff am August 12, 2025, [https://fekir.info/post/linker-error-with-static-member-variable/](https://fekir.info/post/linker-error-with-static-member-variable/)  
52. CMake does not find symbols in static lib that is linked to other static lib \- Stack Overflow, Zugriff am August 12, 2025, [https://stackoverflow.com/questions/43047871/cmake-does-not-find-symbols-in-static-lib-that-is-linked-to-other-static-lib](https://stackoverflow.com/questions/43047871/cmake-does-not-find-symbols-in-static-lib-that-is-linked-to-other-static-lib)  
53. C++ Attributes (Using the GNU Compiler Collection (GCC)), Zugriff am August 12, 2025, [https://gcc.gnu.org/onlinedocs/gcc/C\_002b\_002b-Attributes.html](https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Attributes.html)  
54. Attribute specifier sequence (since C++11) \- cppreference.com \- C++ Reference, Zugriff am August 12, 2025, [https://en.cppreference.com/w/cpp/language/attributes.html](https://en.cppreference.com/w/cpp/language/attributes.html)