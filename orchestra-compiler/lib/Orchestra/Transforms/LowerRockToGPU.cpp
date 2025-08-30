#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h" // Include the Ops header
#include "Orchestra/Transforms/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // Needed for tensor.empty
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-rock-to-gpu"

namespace {

// The conversion pattern for rock.gemm
class GemmLoweringPattern : public mlir::OpConversionPattern<mlir::rock::GemmOp> {
public:
  using OpConversionPattern<mlir::rock::GemmOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::rock::GemmOp gemmOp,
                  mlir::rock::GemmOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = gemmOp.getLoc();

    //===------------------------------------------------------------------===//
    // 1. Define Tiling and Vectorization Parameters
    //===------------------------------------------------------------------===//
    constexpr int64_t mTileSize = 32;
    constexpr int64_t nTileSize = 32;
    constexpr int64_t kTileSize = 2;

    //===------------------------------------------------------------------===//
    // 2. Get Operands and Tensor Shapes
    //===------------------------------------------------------------------===//
    mlir::Value matrixA = gemmOp.getMatrixA();
    mlir::Value matrixB = gemmOp.getMatrixB();
    auto resultType = gemmOp.getMatrixC().getType().cast<mlir::RankedTensorType>();

    int64_t dimM = resultType.getShape()[0];
    int64_t dimK = matrixA.getType().cast<mlir::RankedTensorType>().getShape()[1];
    int64_t dimN = resultType.getShape()[1];

    // Create an empty tensor for the output. This will be the initial value
    // for the loop-carried dependency.
    mlir::Value initialOutput = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    //===------------------------------------------------------------------===//
    // 3. Generate the Loop Nest for Tiling
    //===------------------------------------------------------------------===//
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value mBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dimM);
    mlir::Value nBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dimN);
    mlir::Value kBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dimK);
    mlir::Value mStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, mTileSize);
    mlir::Value nStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, nTileSize);
    mlir::Value kStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kTileSize);

    auto outerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, mBound, mStep, mlir::ValueRange{initialOutput},
        [&](mlir::OpBuilder &builder, mlir::Location loopLoc, mlir::Value m_iv,
            mlir::ValueRange mIterArgs) {
          mlir::Value cIntermediate = mIterArgs[0];
          auto middleLoop = builder.create<mlir::scf::ForOp>(
              loopLoc, c0, nBound, nStep, mlir::ValueRange{cIntermediate},
              [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
                  mlir::Value n_iv, mlir::ValueRange nIterArgs) {
                mlir::Value cInnerIntermediate = nIterArgs[0];

                auto vectorAccumulatorType = mlir::VectorType::get(
                    {mTileSize, nTileSize}, resultType.getElementType());
                mlir::Value initialAccumulator =
                    builder.create<mlir::arith::ConstantOp>(
                        loc, vectorAccumulatorType,
                        builder.getZeroAttr(vectorAccumulatorType));

                auto innerLoop = builder.create<mlir::scf::ForOp>(
                    loopLoc, c0, kBound, kStep,
                    mlir::ValueRange{initialAccumulator},
                    [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
                        mlir::Value k_iv, mlir::ValueRange kIterArgs) {
                      builder.create<mlir::scf::YieldOp>(loopLoc, kIterArgs[0]);
                    });
                mlir::Value finalAcc = innerLoop.getResult(0);

                // Placeholder for vector.store
                builder.create<mlir::scf::YieldOp>(loopLoc, cInnerIntermediate);
              });
          builder.create<mlir::scf::YieldOp>(loopLoc, middleLoop.getResult(0));
        });

    rewriter.replaceOp(gemmOp, outerLoop.getResult(0));
    return mlir::success();
  }
};

// The main pass definition
struct LowerRockToGPUPass
    : public mlir::PassWrapper<LowerRockToGPUPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRockToGPUPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::amdgpu::AMDGPUDialect, mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::amdgpu::AMDGPUDialect, mlir::scf::SCFDialect,
                           mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                           mlir::tensor::TensorDialect>();

    target.addIllegalOp<mlir::rock::GemmOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GemmLoweringPattern>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const final {
    return "lower-rock-to-gpu";
  }

  mlir::StringRef getDescription() const final {
    return "Lowers the Rock dialect to AMDGPU and vector intrinsics.";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::orchestra::createLowerRockToGPUPass() {
  return std::make_unique<LowerRockToGPUPass>();
}
