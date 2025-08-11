/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OrchestraOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::orchestra::OrchestraDialect)
namespace orchestra {

OrchestraDialect::OrchestraDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<OrchestraDialect>()) {

  initialize();
}

OrchestraDialect::~OrchestraDialect() = default;

} // namespace orchestra
