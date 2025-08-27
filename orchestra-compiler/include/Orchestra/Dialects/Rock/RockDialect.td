#ifndef ORCHESTRA_DIALECT_ROCK_DIALECT_TD
#define ORCHESTRA_DIALECT_ROCK_DIALECT_TD
include "mlir/IR/DialectBase.td"
def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::rock";
  let usePropertiesForAttributes = 1;
}
#endif
