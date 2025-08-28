#include "Orchestra/Transforms/LowerRockToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"

namespace {
class GemmOpLowering : public mlir::ConversionPattern {
public:
  explicit GemmOpLowering(mlir::MLIRContext *context)
      : ConversionPattern(mlir::rock::GemmOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto gemmOp = mlir::cast<mlir::rock::GemmOp>(op);
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();

    auto matrixAType = gemmOp.getMatrixA().getType().cast<mlir::RankedTensorType>();
    auto matrixBType = gemmOp.getMatrixB().getType().cast<mlir::RankedTensorType>();
    auto matrixCType = gemmOp.getMatrixC().getType().cast<mlir::RankedTensorType>();

    auto memrefAType = mlir::MemRefType::get(matrixAType.getShape(), matrixAType.getElementType());
    auto memrefBType = mlir::MemRefType::get(matrixBType.getShape(), matrixBType.getElementType());
    auto memrefCType = mlir::MemRefType::get(matrixCType.getShape(), matrixCType.getElementType());

    auto memrefA = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, memrefAType, gemmOp.getMatrixA());
    auto memrefB = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, memrefBType, gemmOp.getMatrixB());
    auto memrefC = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, memrefCType, gemmOp.getMatrixC());

    auto vectorType = mlir::VectorType::get({16}, matrixAType.getElementType());
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value vectorA = rewriter.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefA, mlir::ValueRange{zero, zero});
    mlir::Value vectorB = rewriter.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefB, mlir::ValueRange{zero, zero});
    mlir::Value vectorC = rewriter.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefC, mlir::ValueRange{zero, zero});

    auto mfmaOp = rewriter.create<mlir::amdgpu::MFMAOp>(
        loc, vectorType,
        rewriter.getI32IntegerAttr(32), rewriter.getI32IntegerAttr(32),
        rewriter.getI32IntegerAttr(16), rewriter.getI32IntegerAttr(1),
        vectorA, vectorB, vectorC,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0),
        mlir::amdgpu::MFMAPermBAttr::get(getContext(),
                                          mlir::amdgpu::MFMAPermB::none),
        rewriter.getUnitAttr(), rewriter.getUnitAttr(),
        rewriter.getUnitAttr(), rewriter.getUnitAttr());

    rewriter.create<mlir::vector::TransferWriteOp>(loc, mfmaOp.getDestD(), memrefC,
                                                   mlir::ValueRange{zero, zero});

    auto resultTensor = rewriter.create<mlir::bufferization::ToTensorOp>(loc, memrefC);

    rewriter.replaceOp(op, resultTensor);
    return mlir::success();
  }
};

class LowerRockToAMDGPU
    : public mlir::PassWrapper<LowerRockToAMDGPU,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::amdgpu::AMDGPUDialect,
                           mlir::ROCDL::ROCDLDialect,
                           mlir::vector::VectorDialect,
                           mlir::memref::MemRefDialect,
                           mlir::arith::ArithDialect,
                           mlir::bufferization::BufferizationDialect>();
    target.addIllegalDialect<mlir::rock::RockDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GemmOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const final { return "lower-rock-to-amdgpu"; }

  mlir::StringRef getDescription() const final {
    return "Lower Rock operations to AMDGPU and ROCDL operations.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<mlir::rock::RockDialect, mlir::amdgpu::AMDGPUDialect,
                mlir::ROCDL::ROCDLDialect, mlir::vector::VectorDialect,
                mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                mlir::bufferization::BufferizationDialect>();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::orchestra::createLowerRockToAMDGPUPass() {
  return std::make_unique<LowerRockToAMDGPU>();
}
