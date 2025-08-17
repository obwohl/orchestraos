/*===- LowerOrchestraToGPU.cpp - Orchestra to GPU lowering passes -----*- C++ -*-===//
 *
 * This file implements a pass to lower the Orchestra dialect to GPU dialects.
 *
 *===----------------------------------------------------------------------===*/

#include "Orchestra/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"

namespace {

class TransferOpLowering
    : public mlir::OpConversionPattern<mlir::orchestra::TransferOp> {
public:
  using OpConversionPattern<mlir::orchestra::TransferOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::orchestra::TransferOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For now, we ignore the 'from' and 'to' attributes. A more advanced
    // lowering would use these to determine memory spaces.

    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(adaptor.getSource().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "requires MemRef type");
    }

    // 1. Allocate the destination buffer.
    auto destBuffer =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), sourceType);

    // 2. Create the memcpy operation.
    rewriter.create<mlir::gpu::MemcpyOp>(
        op.getLoc(), mlir::TypeRange{},
        mlir::ValueRange{destBuffer.getResult(), adaptor.getSource()});

    // 3. Replace the transfer op with the destination buffer.
    rewriter.replaceOp(op, {destBuffer.getResult()});

    return mlir::success();
  }
};

class LowerOrchestraToGPUPass
    : public mlir::PassWrapper<LowerOrchestraToGPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToGPUPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::gpu::GPUDialect, mlir::arith::ArithDialect,
                           mlir::memref::MemRefDialect,
                           mlir::orchestra::OrchestraDialect>();

    // Mark the transfer op as illegal since we are lowering it.
    target.addIllegalOp<mlir::orchestra::TransferOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TransferOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  mlir::StringRef getArgument() const final { return "lower-orchestra-to-gpu"; }
  mlir::StringRef getDescription() const final {
    return "Lower Orchestra dialect to GPU dialects";
  }
};

} // namespace

void mlir::orchestra::registerLoweringToGPUPasses() {
  mlir::PassRegistration<LowerOrchestraToGPUPass>();
}
