/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OrchestraOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace orchestra {

class OrchestraDialect : public ::mlir::Dialect {
  explicit OrchestraDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~OrchestraDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("orchestra");
  }
};
} // namespace orchestra
MLIR_DECLARE_EXPLICIT_TYPE_ID(::orchestra::OrchestraDialect)
