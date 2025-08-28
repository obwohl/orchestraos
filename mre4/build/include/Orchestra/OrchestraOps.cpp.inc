/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OrchestraOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::orchestra::DummyOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace orchestra {
} // namespace orchestra
} // namespace mlir
namespace mlir {
namespace orchestra {

//===----------------------------------------------------------------------===//
// ::mlir::orchestra::DummyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
DummyOpAdaptor::DummyOpAdaptor(DummyOp op) : DummyOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult DummyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void DummyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void DummyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DummyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult DummyOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult DummyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace orchestra
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::orchestra::DummyOp)


#endif  // GET_OP_CLASSES
