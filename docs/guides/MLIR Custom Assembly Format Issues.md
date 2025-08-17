# **A Comprehensive Guide to Defining Custom Assembly Formats in MLIR for Operations with Regions and Variadic Results**

## **Introduction: Navigating MLIR's Assembly Format Mechanisms**

The Multi-Level Intermediate Representation (MLIR) framework is designed for extensibility, allowing compiler engineers to define custom dialects with unique operations, types, and attributes. A critical aspect of this process is defining a human-readable textual representation for these custom constructs. This textual form is not merely for debugging; it enables reliable round-tripping of the Intermediate Representation (IR), which is essential for writing test cases and understanding the behavior of compiler transformations. MLIR provides two primary pathways for defining the assembly format of a custom operation: a declarative approach using the Operation Definition Specification (ODS) framework in TableGen, and a programmatic approach requiring direct C++ implementation.  
The declarative path, specified via the assemblyFormat string in an operation's TableGen definition, offers a concise and powerful method for defining syntax for common operational structures. It leverages a dedicated grammar to bind syntactic elements to an operation's arguments, results, and regions, thereby automating the generation of parsing and printing logic. This approach significantly reduces boilerplate code and centralizes the operation's definition. However, its power is predicated on strict adherence to its underlying grammar, and deviations often result in cryptic errors from the mlir-tblgen utility.  
Conversely, the programmatic path provides ultimate flexibility for operations with highly irregular or context-sensitive syntax. By either setting the hasCustomAssemblyFormat \= 1 flag or, more modernly, by implementing the OpAsmOpInterface, developers can write bespoke C++ code to control every character of an operation's textual form. This complete control comes at the cost of increased implementation complexity, particularly in ensuring that the C++ print and parse method signatures precisely match those expected by the code generated from the TableGen definitions.  
The challenge of defining a custom assembly format for operations that combine features like regions and variadic results often pushes developers to this declarative-programmatic crossroads. The issue at hand, specifically the expected literal, variable, directive, or optional group error, does not stem from a fundamental limitation in MLIR's capabilities. Rather, it originates from a subtle but critical misunderstanding of the declarative assemblyFormat grammar. This grammar is not a free-form language but a structured system designed to customize MLIR's canonical generic operation form. This report will first deconstruct this grammar to provide a correct and robust declarative solution. Subsequently, it will furnish the definitive C++ signatures and implementation patterns required for the programmatic approach, equipping developers with a comprehensive toolkit for mastering MLIR assembly formats.

## **Mastering the Declarative Assembly Format Grammar**

The mlir-tblgen error expected literal, variable, directive, or optional group indicates a fundamental syntactic violation at the beginning of the assemblyFormat string. The root cause of this error is the declarative format's implicit structure, which is designed to be a customizable overlay on MLIR's generic operation printing format. Understanding this structure is paramount to correctly authoring any custom assembly format.

### **The Unspoken Structure: Mirroring the Generic Form**

Every MLIR operation, regardless of its custom format, can be represented in a generic form that directly maps to its in-memory representation. This canonical structure is:  
"dialect.op\_name" ( ssa-operands ) attribute-dictionary : functional-type (regions)\`  
The declarative assemblyFormat grammar is designed to parse and print customizations of this sequence. The parser expects elements to appear in an order that logically follows this structure: operands are typically specified before attributes, attributes before the type signature, and the type signature before any large, brace-enclosed regions.  
Placing a directive like functional-type at the beginning of the format string violates this expected sequence. The parser, upon encountering the operation's mnemonic, does not anticipate an immediate declaration of the full type signature. Instead, it expects to first parse the elements that *determine* the signature—namely, the operands. This logical dependency in parsing dictates the required ordering of directives within the assemblyFormat string. An examination of well-formed operations, such as func.call, confirms this pattern, where the functional-type directive appears at the end of the signature, prefixed by a colon, after all operands and attributes have been specified.

### **Core Components and Their Ordering**

The assemblyFormat string is composed of a sequence of space-separated elements, which fall into three primary categories: literals, variables, and directives.

* **Literals:** These are keywords or punctuation enclosed in backticks, such as \`schedule\` or \`-\>\`. They serve as fixed syntactic markers that guide the parser and improve readability.  
* **Variables:** These are identifiers prefixed with a dollar sign, such as $results or $body. They create a binding between a syntactic element in the textual format and a component of the operation defined in TableGen (an operand, an attribute, a result, or a region).  
* **Directives:** These are special keywords that invoke built-in parsing and printing logic for complex or standard parts of an operation. The essential directives for this context are:  
  * functional-type(operands, results): This directive parses or prints the full functional type signature of the operation, in the form (operand\_types) \-\> (result\_types). It takes two arguments: the TableGen name of the operand group and the result group. For an operation with no operands, an empty operand group must still be defined in the arguments list and passed to the directive. As established, this directive should appear late in the format string, typically just before the region directive.  
  * attr-dict and attr-dict-with-keyword: This directive handles any attributes of the operation that are not explicitly bound to a variable elsewhere in the assemblyFormat. It provides a mechanism for ensuring all attributes are round-tripped. The \-with-keyword variant will print the attributes keyword before the dictionary if it is not empty, enhancing readability.  
  * region: This directive parses or prints the operation's attached regions. For an operation with a single region, like orchestra.schedule, this directive will handle the entire {...} block.

The conventional and syntactically valid ordering for these components is:

1. Operation mnemonic (often as a literal, or implicitly handled).  
2. Operand variables (e.g., $operands).  
3. Attribute variables (e.g., $target).  
4. The attr-dict directive for remaining attributes.  
5. The functional-type directive, usually preceded by a literal colon.  
6. The region directive.

### **Handling Variadic and Optional Elements with Optional Groups**

A crucial feature of the assembly format grammar is its ability to handle elements that may not be present in every instance of an operation, such as variadic operand lists or optional attributes. This is achieved through the "optional group," denoted by (...) ?.  
For an optional group to be parsed unambiguously, it must contain an "anchor"—a variable marked with a caret (^). The presence of the value bound to the anchor variable during printing determines whether the entire group is printed. Conversely, during parsing, the parser first attempts to parse the group; if successful, the anchor variable is considered present.  
This mechanism is indispensable for variadic operands and results. A variadic list can be empty, and the syntax associated with it (e.g., parentheses around an operand list or a \-\> for a result list) should only be printed if the list is non-empty. The func.return operation provides a canonical example of this pattern for its variadic operands: its format includes ($operands^ : type($operands))?. Here, $operands is the anchor. If there are any operands to print, the entire group—including the list of operands and their types—is printed. If the operand list is empty, nothing is printed. This anchoring mechanism is the key to creating robust and clean syntax for operations with variadic components.

## **Definitive TableGen Solutions for the Orchestra Dialect**

Applying the grammatical principles outlined above, it is possible to resolve the mlir-tblgen errors and construct correct, declarative assembly formats for both orchestra.schedule and orchestra.task.

### **Root Cause Analysis of the Original Error**

The user's various attempts failed due to specific violations of the assemblyFormat grammar:

* "functional-type(ins, results) attr-dict-with-keyword region": This is the primary error. As detailed previously, functional-type is a directive that describes the operation's signature and cannot be the leading element in the format string. The parser expects the format to begin with elements that identify and constitute the operation, such as its operands or key attributes, before the full type signature is specified.  
* "attr-dict-with-keyword, region": This fails because the comma is not a valid top-level separator between directives. The grammar expects a space-separated sequence of elements.  
* "functional-type((), results)": This fails for the same reason as the first attempt. Additionally, an optional group () must be quantified with ? to be syntactically valid and must not appear as a top-level element in this manner.

### **Corrected Definition for orchestra.schedule**

The orchestra.schedule operation has no operands, variadic results, a single region, and a standard attribute dictionary. A correct assembly format must respect the canonical ordering of these components. The functional-type directive requires handles for both operands and results. Even though there are no operands, an empty ins group must be defined in the arguments list to satisfy the directive's signature.  
The corrected TableGen definition is as follows:  
`def Orchestra_ScheduleOp : Orchestra_Op<"schedule",> {`  
  `let summary = "Container for a physically scheduled DAG of tasks.";`  
  `let description =;`

  `// Define an empty operand group named 'ins' for use in the assemblyFormat.`  
  `let arguments = (ins);`

  `let results = (outs Variadic<AnyType>:$results);`  
  `let regions = (region AnyRegion:$body);`

  `let hasCanonicalizer = 1;`  
  `let hasVerifier = 1;`

  `// Correct assemblyFormat following the canonical order:`  
  `// 1. attr-dict: Prints any non-ODS-defined attributes.`  
  `// 2. functional-type: Prints the full type signature, e.g., "() -> (i32, f32)".`  
  `// 3. region: Prints the attached region body.`  
  `let assemblyFormat = "attr-dict functional-type(ins, results) region";`  
`}`

This format is clean, unambiguous, and directly maps to the operation's structure. It instructs the parser to first look for an optional attribute dictionary, then the full functional type signature, and finally the mandatory region. This will produce IR such as:  
`orchestra.schedule -> (f32, i64) {`  
  `//... region body...`  
`}`

`orchestra.schedule attributes { some.attr = true } {`  
  `//... region body...`  
`}`

### **Corrected Definition for orchestra.task**

The orchestra.task operation is more complex, featuring variadic operands, a named DictionaryAttr ($target), variadic results, and a region. The assembly format should give the named attribute a distinct syntactic position to improve readability, while also handling the variadic operands and the remaining attributes.  
The corrected TableGen definition places the variadic operands in parentheses, followed by a keyword-driven syntax for the mandatory $target attribute. The attr-dict directive will then print any other attributes, followed by the functional type and the region.  
`def Orchestra_TaskOp : Orchestra_Op<"task",> {`  
  `let summary = "An asynchronous unit of computation assigned to a resource.";`  
  `let description = [{`  
    `Encapsulates an atomic unit of computation assigned to a specific`  
    ``hardware resource. Its `target` attribute provides a flexible mechanism``  
    `for specifying fine-grained placement constraints.`  
  `}];`

  `let arguments = (ins Variadic<AnyType>:$operands,`  
                       `DictionaryAttr:$target);`  
  `let results = (outs Variadic<AnyType>:$results);`  
  `let regions = (region AnyRegion:$body);`

  `let hasVerifier = 1;`

  `// Correct assemblyFormat with explicit operand and named attribute parsing:`  
  ``// 1. `(` $operands `)`: Prints the variadic operands list.``  
  ``// 2. `target` `=` $target: A custom keyword-based format for the target attribute.``  
  `// 3. attr-dict: Handles any other attributes.`  
  `// 4. functional-type: Prints the full signature.`  
  `// 5. region: Prints the region body.`  
  `let assemblyFormat = [{`  
    `` `(` $operands `)` `target` `=` $target attr-dict ``  
    `functional-type($operands, results) region`  
  `}];`  
`}`

This format provides a highly readable and structured representation. An example of the resulting IR would be:  
`orchestra.task(%val1, %val2) target = {device = "cpu", core = 0} -> (f64) {`  
  `//... task body...`  
`}`

This structure is robust and follows the best practices demonstrated by standard MLIR dialects.

## **Canonical Examples from the MLIR Standard Dialects**

To validate the proposed solutions, it is instructive to analyze how existing, complex operations in the MLIR codebase handle similar features. The scf.for and memref.generic\_atomic\_rmw operations serve as excellent case studies.

### **Case Study 1: scf.for (Structured Control Flow)**

The scf.for operation is a prime example of an operation with a complex, keyword-driven custom format that includes both fixed and variadic operands (the loop-carried variables, or initArgs), as well as a region. Its results correspond directly to the variadic initArgs, making it an analogue for an operation with variadic results.  
Its assemblyFormat is defined in SCFOps.td as: $inductionVar \=$lowerBoundto$upperBoundstep $step (init ($initArgs^))? region attr-dict  
This definition reveals several advanced techniques:

1. **Keyword-Driven Syntax:** The format uses literals like to and step to create a domain-specific, readable syntax, rather than relying on a generic parenthesized operand list.  
2. **Optional Group for Variadic Operands:** The loop-carried variables ($initArgs) are enclosed in an optional group: (init ($initArgs^))?. The caret on $initArgs designates it as the anchor. The init(...) clause is only printed if the $initArgs list is non-empty, resulting in a clean syntax for loops both with and without loop-carried variables.  
3. **Implicit Functional Type:** Notably, scf.for does not use the functional-type directive. Instead, its result types are inferred from the types of the $initArgs operands. This is possible because the operation implements the InferTypeOpInterface, which provides a C++ hook to compute result types based on operand types and attributes. While this is a valid and powerful technique, for the Orchestra dialect, using the explicit functional-type directive is simpler and more direct.

### **Case Study 2: memref.generic\_atomic\_rmw**

The memref.generic\_atomic\_rmw operation is a good example of a more standard structure that combines operands, a region, and a result type signature. Its assemblyFormat is:  
$memref \[$indices\]region attr-dict: type($result)  
This format reinforces the canonical ordering principle. The operands ($memref, $indices) are specified first. The region follows immediately after. Finally, the trailing type signature, consisting of a colon and the result's type (printed via the type($result) directive), concludes the definition. This "signature at the end" pattern is a recurring theme in well-defined MLIR operations and is the key to resolving the user's original parsing error.

## **The Programmatic Fallback: Custom C++ Parsers and Printers**

While the declarative assemblyFormat is sufficient for the Orchestra dialect as currently described, there are scenarios where a programmatic C++ implementation is necessary. This is the case when the desired syntax is highly irregular, requires context-sensitive parsing logic, or needs to print derived information not directly available in the operation's ODS definition.

### **Choosing the Right Mechanism**

MLIR provides two primary ways to enable a custom C++ implementation:

1. **hasCustomAssemblyFormat \= 1:** This is the legacy mechanism. Setting this flag in the TableGen def instructs mlir-tblgen to generate declarations for a static parse method and a member print method, which must then be implemented in the operation's C++ file.  
2. **OpAsmOpInterface:** This is the modern, preferred approach. It is an interface that provides hooks for various assembly-related customizations, including full control over parsing and printing, as well as suggesting names for SSA values (getAsmResultNames) and blocks (getAsmBlockNames). To use it, one adds the OpAsmOpInterface trait to the operation's definition in TableGen (often via DeclareOpInterfaceMethods\<OpAsmOpInterface\>) and implements the corresponding C++ methods. This approach is more extensible and aligns better with MLIR's interface-driven design philosophy.

### **Definitive C++ Method Signatures (for OpAsmOpInterface)**

The most common point of failure when implementing custom parsers and printers is a mismatch between the C++ method signatures in the implementation file and the declarations generated by TableGen. The MLIR ODS documentation provides a definitive mapping from the types used in an operation's arguments and results lists to the C++ parameter types expected by the parse and print functions. Adhering to this mapping is non-negotiable for a successful build.  
When using OpAsmOpInterface, the required methods to implement for custom parsing and printing are:  
`// In the operation's C++ class definition (e.g., OrchestraOps.cpp)`

`// The 'parse' method is static and populates an OperationState.`  
`static mlir::ParseResult MyOp::parse(mlir::OpAsmParser &parser,`  
                                    `mlir::OperationState &result);`

`// The 'print' method is a const member function.`  
`void MyOp::print(mlir::OpAsmPrinter &printer) const;`

The parameters to these functions are fixed. The logic inside must interact with the OpAsmParser and OpAsmPrinter objects to consume and produce the textual format. The OperationState in the parse method is a temporary representation of the operation being built, to which parsed operands, attributes, types, and regions are added.  
The following table provides a canonical mapping from ODS argument types to the C++ types used to handle them within the parse and print methods.

| ODS Argument/Result Type | parse C++ Handling | print C++ Handling |
| :---- | :---- | :---- |
| SomeAttr:$name | SomeAttr attr; parser.parseAttribute(attr, "name", result.attributes); | printer.printAttribute(getName()); |
| OptionalAttr\<T\>:$name | T attr; if (succeeded(parser.parseOptionalAttribute(...))) {... } | if (getName()) { printer.printAttribute(getName()); } |
| AnyType:$operand | OpAsmParser::UnresolvedOperand operand; parser.parseOperand(operand); | printer.printOperand(getOperand()); |
| Variadic\<AnyType\>:$operands | SmallVector\<OpAsmParser::UnresolvedOperand\> operands; parser.parseOperandList(operands); | printer.printOperands(getOperands()); |
| Variadic\<AnyType\>:$results | SmallVector\<Type\> resultTypes; parser.parseTypeList(resultTypes); result.addTypes(resultTypes); | printer.printTypes(getOperation()-\>getResultTypes()); |
| AnyRegion:$body | Region \*body \= result.addRegion(); parser.parseRegion(\*body, {}); | printer.printRegion(getBody(), /\*printEntryBlockArgs=\*/false); |

### **A Complete Implementation Guide for orchestra.schedule**

Should a programmatic approach be necessary for orchestra.schedule, the following C++ implementation demonstrates the correct usage of the parser and printer APIs.  
First, update the TableGen definition to use the interface:  
`// In OrchestraOps.td`  
`include "mlir/Interfaces/OpAsmInterface.td"`

`def Orchestra_ScheduleOp : Orchestra_Op<"schedule",`  
   `> {`  
  `//... arguments, results, regions...`  
  `// Remove assemblyFormat and add the interface.`  
`}`

Next, provide the C++ implementation:  
`// In OrchestraOps.cpp`  
`#include "mlir/IR/FunctionImplementation.h" // For helper functions`

`// --- ScheduleOp ---`

`void Orchestra_ScheduleOp::print(mlir::OpAsmPrinter &p) {`  
  `// Print any attributes that are not part of the ODS definition.`  
  `p.printOptionalAttrDict((*this)->getAttrs());`

  `// Print the result types if they are present.`  
  `if (getNumResults() > 0) {`  
    `p << " -> ";`  
    `p.printFunctionalType(getOperation());`  
  `}`

  `// Print the region, without its entry block arguments (since it has none).`  
  `p.printRegion(getBody(), /*printEntryBlockArgs=*/false);`  
`}`

`mlir::ParseResult Orchestra_ScheduleOp::parse(mlir::OpAsmParser &parser,`  
                                              `mlir::OperationState &result) {`  
  `// Parse the optional attribute dictionary.`  
  `if (parser.parseOptionalAttrDict(result.attributes))`  
    `return mlir::failure();`

  `// Parse the optional result type list.`  
  `if (parser.parseOptionalArrow()) {`  
    `// If '->' is present, parse the function signature.`  
    `if (mlir::function_interface_impl::parseFunctionSignature(`  
            `parser, /*allowVariadic=*/false, result.types)`  
           `.failed()) {`  
      `return mlir::failure();`  
    `}`  
  `}`

  `// Parse the region.`  
  `auto *body = result.addRegion();`  
  `if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))`  
    `return mlir::failure();`

  `return mlir::success();`  
`}`

### **Architectural Insights from func.func**

The func.func operation stands as the canonical example of a complex operation with a fully custom C++ parser and printer. Its TableGen definition in FuncOps.td simply states let hasCustomAssemblyFormat \= 1;. The entire logic for parsing and printing the symbol name (@my\_func), argument lists with attributes (%arg0: i32 {attr}), result lists, and the region body is handled in C++ helper functions like parseFunctionOp and printFunctionOp found in FunctionImplementation.cpp. A study of this implementation reveals the patterns necessary for handling highly sophisticated syntax, such as parsing nested attribute dictionaries on arguments and results, which would be impossible to express in the declarative format. This serves as a valuable reference for developers whose dialects evolve to require such complexity.

## **Synthesis and Strategic Recommendations**

The investigation into defining a custom assembly format for MLIR operations with regions and variadic results yields a clear set of conclusions and strategic advice for dialect developers.

### **On Fundamental Limitations in MLIR v20**

There are no fundamental bugs or architectural limitations in MLIR v20 (or later versions) that prevent the definition of a declarative assembly format for an operation that possesses both a region and variadic results. The assemblyFormat system is fully capable of handling this combination. The difficulties and mlir-tblgen errors encountered are consistently attributable to syntactic misunderstandings of the declarative grammar, specifically regarding the required ordering of directives and the proper use of optional groups for variadic elements. The system is robust, but it demands precision.

### **Declarative vs. Programmatic: A Decision Framework**

The choice between the declarative assemblyFormat and a programmatic C++ implementation via OpAsmOpInterface is a key architectural decision in dialect design. The following framework can guide this choice:

* **Use Declarative assemblyFormat when:**  
  * The desired syntax can be mapped to the canonical MLIR structure: op-name operands attributes : type region. This structure is flexible and can be customized with keywords and optional groups to achieve a high degree of readability.  
  * All components of the operation (operands, results, attributes, regions) are explicitly defined in the ODS arguments, results, and regions lists, making them available as variables in the format string.  
  * The primary goal is to minimize boilerplate C++ code, maximize maintainability, and leverage the full power of TableGen's code generation. This should be the default choice for most operations.  
* **Use Programmatic C++ (OpAsmOpInterface) when:**  
  * The syntax is highly irregular and cannot be expressed as a linear sequence of the standard components (e.g., the keyword-interspersed format of scf.for).  
  * Parsing requires complex, context-dependent logic, such as inferring an attribute's value from an operand's type in a non-trivial way.  
  * The printed form must be customized in ways that deviate from the in-memory representation, such as eliding default attributes or printing derived information.  
  * Additional assembly-related behaviors are required, such as providing custom names for SSA values (getAsmResultNames), which is only available through the OpAsmOpInterface.

For the Orchestra dialect, the corrected declarative formats presented in this report are not only functional but also idiomatic and maintainable. They should be the preferred solution. The programmatic C++ approach remains a powerful fallback, and the C++ signatures and implementation patterns provided herein serve as a definitive guide should the dialect's syntactic requirements become more complex in the future.

#### **Quellenangaben**

1\. MLIR Language Reference, https://mlir.llvm.org/docs/LangRef/ 2\. llvm-project-with-mlir/mlir/g3doc/LangRef.md at master \- GitHub, https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/LangRef.md 3\. Defining Dialect Attributes and Types \- MLIR, https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/ 4\. Operation Definition Specification (ODS) \- MLIR \- LLVM, https://mlir.llvm.org/docs/DefiningDialects/Operations/ 5\. Interfaces \- MLIR \- LLVM, https://mlir.llvm.org/docs/Interfaces/ 6\. Customizing Assembly Behavior \- MLIR \- LLVM, https://mlir.llvm.org/docs/DefiningDialects/Assembly/ 7\. mlir/test/mlir-tblgen/op-format-invalid.td · 2c7829e676dfd6a33f7c9955ea930f51aca37e20 · llvm-doe / llvm-project · GitLab, https://code.ornl.gov/llvm-doe/llvm-project/-/blob/2c7829e676dfd6a33f7c9955ea930f51aca37e20/mlir/test/mlir-tblgen/op-format-invalid.td 8\. Debugging Tips \- MLIR \- LLVM, https://mlir.llvm.org/getting\_started/Debugging/ 9\. Chapter 2: Emitting Basic MLIR \- LLVM, https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/ 10\. 'func' Dialect \- MLIR, https://mlir.llvm.org/docs/Dialects/Func/ 11\. llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.td at main ..., https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td 12\. \[Mlir-commits\] \[mlir\] 0748639 \- \[mlir\]\[ods\] Optional Attribute or Type Parameters \- Mailing Lists, https://lists.llvm.org/pipermail/mlir-commits/2022-February/006502.html 13\. 1 TableGen Programmer's Reference — LLVM 22.0.0git documentation, https://llvm.org/docs/TableGen/ProgRef.html 14\. D74681 \[mlir\]\[DeclarativeParser\] Add basic support for optional groups in the assembly format. \- LLVM Phabricator archive, https://reviews.llvm.org/D74681 15\. D95109 \[mlir\]\[OpFormatGen\] Add support for anchoring optional groups with types \- LLVM Phabricator archive, https://reviews.llvm.org/D95109 16\. 'scf' Dialect \- MLIR \- LLVM, https://mlir.llvm.org/docs/Dialects/SCFDialect/ 17\. 'memref' Dialect \- MLIR \- LLVM, https://mlir.llvm.org/docs/Dialects/MemRef/ 18\. mlir/docs/OpDefinitions.md · 89bb0cae46f85bdfb04075b24f75064864708e78 · llvm-doe / llvm-project · GitLab, https://code.ornl.gov/llvm-doe/llvm-project/-/blob/89bb0cae46f85bdfb04075b24f75064864708e78/mlir/docs/OpDefinitions.md 19\. include/mlir/Interfaces/FunctionImplementation.h Source File \- LLVM, https://mlir.llvm.org/doxygen/FunctionImplementation\_8h\_source.html 20\. mlir/docs/DefiningDialects/Operations.md · e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e · llvm-doe / llvm-project \- GitLab, https://code.ornl.gov/llvm-doe/llvm-project/-/blob/e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e/mlir/docs/DefiningDialects/Operations.md