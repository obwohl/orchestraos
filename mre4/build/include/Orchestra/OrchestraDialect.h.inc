/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OrchestraDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::orchestra::OrchestraDialect)
