/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OrchestraDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::orchestra::OrchestraDialect)
namespace mlir {
namespace orchestra {

OrchestraDialect::OrchestraDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<OrchestraDialect>())

     {

  initialize();
}

OrchestraDialect::~OrchestraDialect() = default;

} // namespace orchestra
} // namespace mlir
