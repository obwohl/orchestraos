#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "Orchestra/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {

class TransferOpLowering : public OpConversionPattern<TransferOp> {
public:
  using OpConversionPattern<TransferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransferOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};

struct LowerOrchestraToXeGPUPass
    : public PassWrapper<LowerOrchestraToXeGPUPass,
                         OperationPass<gpu::GPUFuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToXeGPUPass)

  StringRef getArgument() const final {
    return "lower-orchestra-to-xegpu";
  }
  StringRef getDescription() const final {
    return "Lower Orchestra to XeGPU dialect";
  }

  void runOnOperation() override {
    // Do nothing.
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect,
                    xegpu::XeGPUDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();
  }
};

}  // namespace
