

# **A Comprehensive Guide to MLIR Pass Development: Definition, Registration, and Debugging**

## **Section 1: The Canonical Methodology for Defining and Registering a C++ MLIR Pass**

The successful integration of a custom pass into an MLIR-based compiler toolchain hinges on a precise and canonical sequence of steps. This sequence ensures not only that the pass compiles but also that it is correctly registered with the pass management infrastructure, linked into the final executable, and made available for invocation. This section details the authoritative methodology for defining and registering a C++ MLIR pass that operates on gpu.func operations, with a focus on the programmatic registration model employed by tools like orchestra-opt.

### **1.1 The Anatomy of an Operation-Specific Pass Class**

The foundation of any transformation is the pass class itself. Its structure must conform to the expectations of the MLIR pass manager to enable correct scheduling and execution.

#### **Core Structure**

A pass designed to operate on a specific MLIR operation, such as mlir::gpu::GPUFuncOp, must inherit from the mlir::OperationPass\<OpT\> template class, where OpT is the target operation type.1 For modern MLIR development, it is canonical to wrap this inheritance using the Curiously Recurring Template Pattern (CRTP) utility

mlir::PassWrapper\<T,...\>. This wrapper automatically provides essential boilerplate hooks for command-line options, statistics, and interaction with the pass manager, significantly simplifying the pass definition.3

The canonical class signature for the OrchestraBranchProfiler pass is therefore:

C++

namespace {  
struct OrchestraBranchProfilerPass  
    : public mlir::PassWrapper\<OrchestraBranchProfilerPass,  
                               mlir::OperationPass\<mlir::gpu::GPUFuncOp\>\> {  
  //... Pass implementation...  
};  
} // end anonymous namespace

#### **TypeID for Robust RTTI**

Within the class definition, it is a critical best practice to define a stable, unique type identifier using the MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID macro.

C++

struct OrchestraBranchProfilerPass  
    : public mlir::PassWrapper\<OrchestraBranchProfilerPass,  
                               mlir::OperationPass\<mlir::gpu::GPUFuncOp\>\> {  
  MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(OrchestraBranchProfilerPass)  
  //...  
};

MLIR uses a TypeID system for its own form of runtime type information (RTTI), which is more efficient and robust than standard C++ RTTI. This macro explicitly defines the TypeID for the pass class within the class body itself. This approach is superior to the fallback method, which relies on type names and can be fragile in complex builds involving shared libraries, templates, or types defined in anonymous namespaces. Explicitly defining the TypeID prevents subtle linker or runtime errors related to type mismatches that are notoriously difficult to debug.5

#### **Namespace Strategy and Symbol Visibility**

The pass class definition should be placed within an anonymous namespace. This is a standard C++ idiom that grants the class internal linkage, confining its symbol to the current translation unit (the .cpp file). This practice effectively prevents symbol name collisions with other passes or components across a large compiler project.2 The pass is then exposed to the wider MLIR framework not by its class name, but through a controlled public API consisting of a factory function and a registration call, as detailed in Section 1.4.

### **1.2 Implementing Essential Pass Interfaces**

A pass class must override several key virtual methods from its base classes to establish its public contract with the pass manager and command-line tools.

#### **runOnOperation()**

This method is the primary entry point for the pass's transformation logic. The pass manager invokes this function for each operation that matches the pass's target type (gpu::GPUFuncOp in this case). Inside this method, the target operation can be retrieved via getOperation(). The implementation for OrchestraBranchProfiler would then walk the regions of the gpu::GPUFuncOp to find and instrument all scf.if operations.

#### **getArgument() and getDescription()**

These two methods define the command-line interface for the pass.

* getArgument() must return a unique string literal that will be used as the command-line flag to invoke the pass. This string is the key used by the pass registry to look up the pass. For this scenario, it should be "orchestra-branch-profiler".9  
* getDescription() returns a string literal that serves as the help text for the pass when a user runs orchestra-opt \--help.

A mismatch between the string returned by getArgument() and the flag used on the command line is a frequent source of "silent failure." The pass registry is effectively a map from the argument string to a pass factory function. When a tool like orchestra-opt parses its arguments, it performs a lookup in this map. If the lookup fails due to a typo or mismatch, it is not treated as an error. The tool simply concludes that the user did not request that specific pass and continues execution without adding it to the pipeline. This behavior is correct from the pass manager's perspective but can be a source of confusion for developers.

### **1.3 Declaring Dialect Dependencies: A Critical Prerequisite for Correctness**

A pass that creates new operations, types, or attributes must explicitly declare the dialects to which those entities belong. This is a non-negotiable requirement for ensuring the correctness and thread-safety of the compilation pipeline.

#### **The Role of getDependentDialects()**

The getDependentDialects(mlir::DialectRegistry \&registry) virtual method must be overridden to register these dependencies.4 For the

OrchestraBranchProfiler pass, which inserts memref.alloca and memref.atomic\_rmw, there is a hard dependency on the MemRef dialect. Since the pass also operates on scf.if and will likely create arith.constant operations for counter initialization, it also depends on the SCF and Arith dialects.

The correct implementation is as follows:

C++

void getDependentDialects(mlir::DialectRegistry \&registry) const override {  
  registry.insert\<mlir::memref::MemRefDialect, mlir::scf::SCFDialect,  
                  mlir::arith::ArithDialect\>();  
}

Failure to declare these dependencies can lead to severe runtime errors. Before executing a pass pipeline, especially in a multi-threaded context, the pass manager first queries all scheduled passes for their dialect dependencies. It then ensures all required dialects are loaded into the MLIRContext. This preemptive loading prevents race conditions where multiple threads might attempt to load the same dialect simultaneously. If a pass attempts to create an operation from a dialect that has not been loaded, the compiler will crash, often with an error message like "can't create type because storage uniquer isn't initialized: the dialect was likely not loaded.".14 This error indicates that the necessary infrastructure for creating and uniquing entities of that dialect was not present in the context. The

getDependentDialects mechanism is therefore fundamental to MLIR's robust, multi-threaded architecture.

### **1.4 The Programmatic Registration and Linking Pattern**

With the pass class fully defined, the final step is to make it known to the orchestra-opt tool. This involves creating a factory function, registering it, and ensuring the build system links the pass's code into the final executable.

#### **The Factory Function and Centralized Registration**

The canonical pattern for programmatic registration involves two key components:

1. **A Public Factory Function:** A function that creates an instance of the pass should be declared in a public header file. This function serves as the sole, controlled entry point for instantiating the pass.  
   C++  
   // In Passes.h  
   std::unique\_ptr\<mlir::Pass\> createOrchestraBranchProfilerPass();

2. **Centralized Registration Call:** The implementation of the factory function and the call to mlir::registerPass should reside in the pass's .cpp file. The registration itself should be invoked from a central function within the tool (e.g., a function named registerOrchestraPasses() that is called from main.cpp).  
   C++  
   // In OrchestraBranchProfiler.cpp  
   std::unique\_ptr\<mlir::Pass\> createOrchestraBranchProfilerPass() {  
     return std::make\_unique\<OrchestraBranchProfilerPass\>();  
   }

   // In a central registration file, e.g., OrchestraPasses.cpp  
   void registerOrchestraPasses() {  
     mlir::registerPass(createOrchestraBranchProfilerPass);  
   }

This explicit function call chain (main \-\> registerOrchestraPasses \-\> mlir::registerPass) is crucial because it creates a non-discardable symbol reference to the pass's object file. Older registration methods that relied on a global static PassRegistration object were susceptible to a classic C++ linker issue: if no other code directly referenced a symbol from an object file containing such a static object, an optimizing linker was free to discard the entire object file. This would prevent the static object's constructor—which performs the registration—from ever running, causing the pass to be silently unavailable at runtime.2 The programmatic registration model is the correct and robust solution to this "static initializer problem."

#### **Build System Integration (CMake)**

If a pass is correctly defined and registered programmatically but still fails to appear in the tool, the issue is almost certainly in the build system configuration. The linker must be explicitly told to include the object file containing the pass. Using MLIR's idiomatic CMake functions is the recommended approach.

1. Define a library for the pass implementation:  
   CMake  
   add\_mlir\_library(OrchestraBranchProfilerPass  
     OrchestraBranchProfiler.cpp  
     \# Other sources...  
   )

2. Explicitly link the main tool executable against this library:  
   CMake  
   add\_executable(orchestra-opt main.cpp)  
   target\_link\_libraries(orchestra-opt  
     PRIVATE  
     OrchestraBranchProfilerPass  
     \# Other libraries...  
   )

Failure to include the target\_link\_libraries dependency breaks the explicit call chain, allowing the linker to discard the pass's object file and leading to the silent failure mode where the pass is compiled but never registered.

## **Section 2: Navigating the Pass Manager for Nested GPU Operations**

Understanding the hierarchical nature of the MLIR pass manager is essential for correctly applying transformations to operations nested deep within the IR structure, such as a gpu.func inside a gpu.module. A pass that is correctly defined and registered will still fail to run if the pass pipeline is not constructed to match the IR's nesting.

### **2.1 The Principle of Anchoring and Nesting**

The MLIR pass manager is not a flat list of passes; it is a tree-like structure of OpPassManager instances. Each OpPassManager is "anchored" to a specific operation type (e.g., builtin.module) and is responsible for scheduling passes on operations of that type.4

To run a pass on a nested operation, the pass pipeline must mirror the structural nesting of the IR. A gpu.func operation typically resides within a gpu.module operation, which in turn is inside the top-level builtin.module. Therefore, to reach the gpu.func, the pass pipeline must first be anchored on builtin.module, then nest into an OpPassManager for gpu.module, and finally, within that, schedule the pass that operates on gpu.func.1

In C++, this nesting is achieved using the nest\<OpT\>() method on an OpPassManager instance.17 For example:

C++

mlir::PassManager pm(context);  
// Nest into gpu.module operations  
auto \&gpuModulePM \= pm.nest\<mlir::gpu::GPUModuleOp\>();  
// Add a pass that runs on gpu.func operations within each gpu.module  
gpuModulePM.addPass(createOrchestraBranchProfilerPass());

### **2.2 Constructing the Correct Command-Line Pipeline**

The \--pass-pipeline command-line argument provides a powerful textual domain-specific language (DSL) that directly corresponds to the C++ nest\<\>() API. The syntax op-anchor-name(pass1, pass2, nested-op-anchor(...)) allows for the complete specification of a nested pipeline from the command line.4

Given that the OrchestraBranchProfilerPass is defined as OperationPass\<gpu::GPUFuncOp\>, the correct command-line invocation for orchestra-opt must reflect the builtin.module \-\> gpu.module \-\> gpu.func nesting. The correct command is:

orchestra-opt \--pass-pipeline="builtin.module(gpu.module(orchestra-branch-profiler))" my\_gpu\_code.mlir

This command instructs the pass manager to:

1. Start with a pass manager anchored on builtin.module.  
2. Within that, create a nested pass manager anchored on gpu.module. This nested manager will iterate over all gpu.module operations found inside the top-level module.  
3. For each gpu.module it finds, it will run the orchestra-branch-profiler pass. Because this pass is typed to operate on gpu::GPUFuncOp, the pass manager will automatically apply it to all gpu.func operations within that gpu.module.

The user's original invocation, orchestra-opt \--orchestra-branch-profiler, is the primary source of the silent failure. This command is interpreted as a request to add the orchestra-branch-profiler pass directly to the top-level pass manager, which is by default anchored on builtin.module. When this top-level manager executes, it retrieves the pass and inspects its C++ type, discovering that it is an OperationPass\<gpu::GPUFuncOp\>. The manager then proceeds to iterate over the operations it is responsible for—in this case, the single top-level ModuleOp. For this ModuleOp, it performs a type check: "Is this operation a gpu::GPUFuncOp?". The answer is no. Consequently, the pass is never scheduled to run on any operation. The process completes successfully without error because the pass manager correctly enforced its scheduling rules based on a misspecified, non-nested pipeline.

## **Section 3: A Systematic Methodology for Diagnosing Silent Pass Failures**

When an MLIR pass compiles but fails to run, a systematic diagnostic process is required to efficiently pinpoint the root cause. This process should begin with the simplest checks for registration and progress toward more detailed introspection of the pass manager and build system. MLIR provides a rich set of command-line flags designed for this purpose, enabling full diagnosis often without modifying the compiler's source code.

### **3.1 Step 1: Confirming Registration and Availability**

The first step is to verify that the pass has been successfully registered with the tool's pass registry and is available for invocation.

* **The Litmus Test:** Run orchestra-opt \--help. The output should contain a list of available passes. The argument specified in getArgument() (--orchestra-branch-profiler) and the description from getDescription() should be present in this list. If they are missing, the problem is almost certainly a registration or linking issue.  
* The Definitive Check: The most authoritative way to check for registration is with the \--list-passes flag:  
  orchestra-opt \--list-passes  
  This command directly queries the global PassRegistry and prints the argument of every registered pass.20 If  
  orchestra-branch-profiler is not in this list, the problem is definitively in the registration or linking stage (as described in Section 1.4). No further pipeline debugging is useful until this foundational issue is resolved.

### **3.2 Step 2: Tracing Pass Manager Execution and Pipeline Parsing**

Once registration is confirmed, the next step is to ensure the pass manager is correctly parsing the command-line arguments and constructing the intended pipeline.

* **\--dump-pass-pipeline:** This flag prints the textual representation of the OpPassManager hierarchy that was constructed from the command-line arguments.21 Running  
  orchestra-opt \--orchestra-branch-profiler \--dump-pass-pipeline would show a flat pipeline, while running with the correct \--pass-pipeline="..." syntax would show the required nested structure. This immediately reveals any misconfiguration of the pipeline.  
* **\--debug and \--debug-only=pass-manager:** For maximum verbosity, these flags enable LLVM\_DEBUG print statements within the pass manager's source code.19 The resulting output provides a detailed execution trace, showing the pass manager walking the IR, nesting into regions, and making scheduling decisions. It will explicitly state why a pass is being scheduled on a given operation or, crucially, why it is being skipped.

### **3.3 Step 3: Observing the IR Transformation Flow**

If the pass manager appears to be constructing the correct pipeline, the next step is to observe the state of the IR as it flows through that pipeline. This confirms whether the pass is actually being executed.

* **IR Printing Flags:** MLIR provides a suite of flags for dumping the IR at various points during execution. The most useful for this scenario is \-mlir-print-ir-before-all.1

  orchestra-opt \--pass-pipeline="builtin.module(gpu.module(orchestra-branch-profiler))" \-mlir-print-ir-before-all my\_gpu\_code.mlir  
* **Diagnostic Procedure:** The output of this command will be interleaved with comments indicating which pass is about to run. The developer should look for a line like // \---- IR Dump Before OrchestraBranchProfilerPass \---- //. If this line appears, it is definitive proof that the pass manager has successfully scheduled the pass for execution. Any remaining problem must lie within the runOnOperation implementation itself. If this line is absent, the pass manager is still skipping the pass for some reason, pointing back to an issue with pipeline anchoring or pass type compatibility. Other useful flags include \-mlir-print-ir-after-all and the less verbose \-mlir-print-ir-after-change, which only prints the IR if a pass has modified it.19

### **3.4 Step 4: Investigating Build System and Linker Issues**

As established in previous sections, if the \--list-passes check fails, the problem is almost always related to the build system and linker. The programmatic registration model relies on an unbroken chain of function calls from main to the pass registration function, which in turn requires the executable to be linked against the library containing the pass. The following checklist helps diagnose such issues:

1. Is the pass's .cpp file correctly listed as a source in an add\_mlir\_library(...) or add\_library(...) command in CMakeLists.txt?  
2. Is the library target from the previous step listed as a PRIVATE or PUBLIC dependency in the target\_link\_libraries(...) command for the orchestra-opt executable?  
3. Are MLIR's idiomatic CMake functions, such as add\_mlir\_library, being used where appropriate? These functions correctly handle the propagation of dependencies and flags, and deviating from them can lead to subtle build failures.24

The suite of debugging flags provided by MLIR reflects a core design philosophy: the compiler should be an observable and introspectable system. By systematically applying these tools, developers can diagnose complex issues like silent pass failures without resorting to invasive printf debugging, leading to a more efficient and robust development process.

## **Section 4: Reference Implementation and Diagnostic Summary**

This section consolidates the preceding analysis into a complete, actionable solution. It provides a canonical C++ implementation for the OrchestraBranchProfilerPass, example build system and tool integration code, and a summary table for rapid diagnostics.

### **4.1 Complete OrchestraBranchProfilerPass.cpp Example**

The following code provides a complete, well-commented reference implementation for the pass, incorporating all best practices discussed in Section 1\.

C++

\#**include** "mlir/Dialect/Arith/IR/Arith.h"  
\#**include** "mlir/Dialect/GPU/IR/GPUDialect.h"  
\#**include** "mlir/Dialect/MemRef/IR/MemRef.h"  
\#**include** "mlir/Dialect/SCF/IR/SCF.h"  
\#**include** "mlir/IR/Builders.h"  
\#**include** "mlir/IR/Dialect.h"  
\#**include** "mlir/Pass/Pass.h"

\#**include** \<memory\>

using namespace mlir;

namespace {

// Inherit from PassWrapper and OperationPass\<gpu::GPUFuncOp\> to define a pass  
// that operates specifically on GPU kernel functions.  
struct OrchestraBranchProfilerPass  
    : public PassWrapper\<OrchestraBranchProfilerPass,  
                         OperationPass\<gpu::GPUFuncOp\>\> {  
  // MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID provides a stable RTTI anchor.  
  MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID(OrchestraBranchProfilerPass)

  // Returns the command-line argument used to invoke the pass.  
  StringRef getArgument() const final {  
    return "orchestra-branch-profiler";  
  }

  // Returns a short description of the pass for \--help.  
  StringRef getDescription() const final {  
    return "Instruments scf.if operations in GPU functions to profile branch "  
           "divergence.";  
  }

  // Declares dependencies on dialects whose ops will be created by this pass.  
  // This is crucial for multi-threaded pass execution.  
  void getDependentDialects(DialectRegistry \&registry) const override {  
    registry.insert\<memref::MemRefDialect, scf::SCFDialect,  
                    arith::ArithDialect\>();  
  }

  // The main entry point for the pass logic.  
  void runOnOperation() override {  
    gpu::GPUFuncOp funcOp \= getOperation();  
    OpBuilder builder(\&getContext());

    // 1\. Create a counter buffer in workgroup memory at the start of the function.  
    // This buffer will have one counter for each scf.if (then/else pair).  
    // For simplicity, we allocate a fixed-size buffer here. A real implementation  
    // would first count the number of scf.if ops.  
    int numIfOps \= 0;  
    funcOp.walk(\[&\](scf::IfOp ifOp) { numIfOps++; });  
    if (numIfOps \== 0) {  
      return; // Nothing to instrument.  
    }

    builder.setInsertionPointToStart(\&funcOp.front());  
    auto memrefType \= MemRefType::get(  
        {numIfOps \* 2}, builder.getI32Type(), {},  
        gpu::AddressSpaceAttr::get(\&getContext(), gpu::AddressSpace::Workgroup));  
    Value counterBuffer \= builder.create\<memref::AllocaOp\>(funcOp.getLoc(), memrefType);

    // 2\. Walk all scf.if operations and instrument them.  
    int ifOpIdx \= 0;  
    funcOp.walk(\[&\](scf::IfOp ifOp) {  
      Location loc \= ifOp.getLoc();

      // Instrument the 'then' block.  
      builder.setInsertionPointToStart(ifOp.getThenBlock());  
      Value thenIndex \= builder.create\<arith::ConstantIndexOp\>(loc, ifOpIdx \* 2);  
      builder.create\<memref::AtomicRMWOp\>(  
          loc, builder.getI32Type(), memref::AtomicRMWKind::addi,  
          builder.create\<arith::ConstantIntOp\>(loc, 1, 32),  
          counterBuffer, ValueRange{thenIndex});

      // Instrument the 'else' block (if it exists).  
      if (ifOp.hasElse()) {  
        builder.setInsertionPointToStart(ifOp.getElseBlock());  
        Value elseIndex \= builder.create\<arith::ConstantIndexOp\>(loc, ifOpIdx \* 2 \+ 1);  
        builder.create\<memref::AtomicRMWOp\>(  
            loc, builder.getI32Type(), memref::AtomicRMWKind::addi,  
            builder.create\<arith::ConstantIntOp\>(loc, 1, 32),  
            counterBuffer, ValueRange{elseIndex});  
      }  
      ifOpIdx++;  
    });  
  }  
};

} // namespace

// Public factory function for creating the pass.  
std::unique\_ptr\<Pass\> createOrchestraBranchProfilerPass() {  
  return std::make\_unique\<OrchestraBranchProfilerPass\>();  
}

### **4.2 Example CMakeLists.txt and main.cpp Integration**

#### **CMakeLists.txt**

CMake

\# Define a library for your pass. add\_mlir\_library is an idiomatic helper  
\# that correctly sets up dependencies on MLIR components.  
add\_mlir\_library(OrchestraBranchProfilerPass  
  OrchestraBranchProfiler.cpp  
  DEPENDS  
  MLIRGPU  
  MLIRMemRef  
  MLIRSCF  
  MLIRArith  
  MLIRPass  
  MLIRIR  
)

\# In the definition for your main tool...  
add\_executable(orchestra-opt  
  main.cpp  
  \# other tool sources...  
)

\# Link the tool against the pass library. This is the crucial step  
\# that prevents the linker from discarding the pass implementation.  
target\_link\_libraries(orchestra-opt  
  PRIVATE  
  OrchestraBranchProfilerPass  
  \# other libraries...  
  MLIRSupport  
  LLVMSupport  
)

#### **main.cpp (and supporting files)**

C++

// In a header file like Orchestra/Passes.h  
\#**include** "mlir/Pass/Pass.h"  
\#**include** \<memory\>

namespace orchestra {  
std::unique\_ptr\<mlir::Pass\> createOrchestraBranchProfilerPass();

// A function to register all custom passes.  
void registerOrchestraPasses();  
} // namespace orchestra

// In a source file like Orchestra/Passes.cpp  
\#**include** "Orchestra/Passes.h"

void orchestra::registerOrchestraPasses() {  
  // This call connects the factory function to the global pass registry.  
  mlir::registerPass(orchestra::createOrchestraBranchProfilerPass);  
}

// In your main.cpp for orchestra-opt  
\#**include** "Orchestra/Passes.h"  
\#**include** "mlir/InitAllDialects.h"  
\#**include** "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char \*\*argv) {  
  mlir::DialectRegistry registry;  
  // Register all standard dialects.  
  mlir::registerAllDialects(registry);  
  // Register custom dialects if you have them.  
  // registry.insert\<orchestra::OrchestraDialect\>();

  // Register your custom passes. This call ensures they are available  
  // to the pass manager.  
  orchestra::registerOrchestraPasses();

  return mlir::asMainReturnCode(  
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\\n", registry));  
}

### **4.3 Diagnostic Tables**

The following tables provide a condensed reference for diagnosing silent pass failures and understanding the components of a C++ pass.

**Table 1: Systematic Diagnosis of Silent Pass Failures**

| Symptom | Probable Cause | Diagnostic Command | Interpretation and Next Steps |
| :---- | :---- | :---- | :---- |
| Pass argument not found in \--help or \--list-passes output. | **Linker Error / Registration Failure:** The pass's object file was discarded by the linker, or the registration function was never called. | orchestra-opt \--list-passes | If the pass argument (orchestra-branch-profiler) is missing, the cause is linking/registration. **Next Step:** Verify CMakeLists.txt for correct add\_mlir\_library and target\_link\_libraries calls. Ensure registerOrchestraPasses() is called from main. |
| Pass is registered, but runOnOperation is never called (verified with prints or debugger). | **Incorrect Pass Pipeline:** The pass is being scheduled on the wrong operation type due to improper nesting. | ... \--pass-pipeline="..." \--dump-pass-pipeline | Compare the dumped pipeline with the expected nested structure (builtin.module(gpu.module(...))). A flat pipeline confirms this is the cause. **Next Step:** Correct the command-line invocation to use the nested \--pass-pipeline syntax. |
| Pass is scheduled (confirmed with \-mlir-print-ir-before-all), but IR is not modified. | **Bug in runOnOperation Logic:** The pass logic contains a flaw, or an invariant is not met, causing it to do no work. | ... \-mlir-print-ir-before-all and ... \-mlir-print-ir-after-all | Confirm the pass is scheduled by finding its "before" dump. If the "after" dump is unchanged, the issue is internal to the pass. **Next Step:** Use a debugger or LLVM\_DEBUG macros within runOnOperation to trace its logic. |
| Crash with "storage uniquer isn't initialized" error. | **Missing Dialect Dependency:** The pass is creating an operation from a dialect that was not declared in getDependentDialects. | Review runOnOperation code. | Identify all dialects of newly created operations (memref, arith, etc.). **Next Step:** Add the missing dialects to the registry.insert\<...\>() call in getDependentDialects. |

**Table 2: Anatomy of a C++ OperationPass for Programmatic Registration**

| Component | Purpose | Common Pitfalls |
| :---- | :---- | :---- |
| class MyPass : public PassWrapper\<..., OperationPass\<OpT\>\> | Defines the pass, specifying its target operation type OpT. | Targeting the wrong operation type (ModuleOp instead of gpu::GPUFuncOp). Forgetting PassWrapper adds boilerplate. |
| MLIR\_DEFINE\_EXPLICIT\_INTERNAL\_INLINE\_TYPE\_ID | Provides a stable, unique type identifier for the pass. | Forgetting this macro, which can lead to subtle RTTI issues in complex builds. |
| runOnOperation() | Contains the core transformation logic for the pass. | Not handling all IR cases correctly; creating invalid IR; forgetting to signalPassFailure() on unrecoverable errors. |
| getArgument() | Returns the unique command-line flag for the pass. | Mismatch between this string and the flag used in \--pass-pipeline or on the command line. |
| getDependentDialects() | Declares dialects of any newly created MLIR entities. | Forgetting to declare a dialect, leading to "storage uniquer" crashes, especially in multi-threaded mode. |
| createMyPass() | A public factory function to instantiate the pass. | Defining this function inside an anonymous namespace, making it invisible to the registration call. |
| mlir::registerPass(...) | Adds the pass factory to the global pass registry. | Forgetting to call this from a function that is guaranteed to be linked into the main executable. |
| CMakeLists.txt Integration | Links the pass implementation into the final tool executable. | Forgetting to add the pass library to target\_link\_libraries, causing the linker to discard the pass code entirely. |

