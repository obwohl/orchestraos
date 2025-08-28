include "mlir/IR/DialectBase.td"

def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::mlir::orchestra";
}
