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
#include "mlir/Support/LLVM.h"
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

    // Define the types based on the mfma_f32_32x32x2f32 specification.
    auto f32Type = rewriter.getF32Type();
    auto vectorTypeA = mlir::VectorType::get({mTileSize, kTileSize}, f32Type);
    auto vectorTypeB = mlir::VectorType::get({kTileSize, nTileSize}, f32Type);
    auto vectorAccumulatorType =
        mlir::VectorType::get({mTileSize, nTileSize}, f32Type);

    //===------------------------------------------------------------------===//
    // 2. Get Operands and Tensor Shapes
    //===------------------------------------------------------------------===//
    mlir::Value matrixA = adaptor.getMatrixA();
    mlir::Value matrixB = adaptor.getMatrixB();
    auto resultType = gemmOp.getResult().getType().cast<mlir::RankedTensorType>();

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

                // Load the initial accumulator tile from the C tensor.
                mlir::Value c0_f32 = builder.create<mlir::arith::ConstantOp>(
                    loc, f32Type, builder.getF32FloatAttr(0.0));
                auto identityMap =
                    mlir::AffineMap::getMultiDimIdentityMap(2, builder.getContext());
                mlir::ArrayAttr inBoundsAttr = builder.getBoolArrayAttr({true, true});

                mlir::Value initialAccumulator =
                    builder.create<mlir::vector::TransferReadOp>(
                        loc, vectorAccumulatorType, cInnerIntermediate,
                        mlir::ValueRange{m_iv, n_iv},
                        mlir::AffineMapAttr::get(identityMap), c0_f32,
                        /*mask=*/nullptr, inBoundsAttr);

                auto innerLoop = builder.create<mlir::scf::ForOp>(
                    loopLoc, c0, kBound, kStep,
                    mlir::ValueRange{initialAccumulator},
                    [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
                        mlir::Value k_iv, mlir::ValueRange kIterArgs) {
                      mlir::Value currentAcc = kIterArgs[0];

                      auto vecA = builder.create<mlir::vector::TransferReadOp>(
                          loc, vectorTypeA, matrixA, mlir::ValueRange{m_iv, k_iv},
                          mlir::AffineMapAttr::get(identityMap), c0_f32,
                          /*mask=*/nullptr, inBoundsAttr);

                      auto vecB = builder.create<mlir::vector::TransferReadOp>(
                          loc, vectorTypeB, matrixB, mlir::ValueRange{k_iv, n_iv},
                          mlir::AffineMapAttr::get(identityMap), c0_f32,
                          /*mask=*/nullptr, inBoundsAttr);

                      // Perform the matrix multiplication on the tiles using vector.contract.
                      mlir::AffineMap mapA = mlir::AffineMap::get(3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(2)}, builder.getContext());
                      mlir::AffineMap mapB = mlir::AffineMap::get(3, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(1)}, builder.getContext());
                      mlir::AffineMap mapC = mlir::AffineMap::get(3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)}, builder.getContext());

                      auto iteratorTypes = builder.getArrayAttr({
                          mlir::vector::IteratorTypeAttr::get(builder.getContext(), mlir::vector::IteratorType::parallel),
                          mlir::vector::IteratorTypeAttr::get(builder.getContext(), mlir::vector::IteratorType::parallel),
                          mlir::vector::IteratorTypeAttr::get(builder.getContext(), mlir::vector::IteratorType::reduction)
                      });
                      auto contractResult = builder.create<mlir::vector::ContractionOp>(
                          loopLoc, vecA, vecB, currentAcc,
                          builder.getAffineMapArrayAttr({mapA, mapB, mapC}),
                          iteratorTypes);

                      builder.create<mlir::scf::YieldOp>(loopLoc,
                                                        contractResult.getResult());
                    });
                mlir::Value finalAcc = innerLoop.getResult(0);

                // Store the final accumulated vector back into the C tensor.
                auto writeOp =
                    builder.create<mlir::vector::TransferWriteOp>(
                        loc, finalAcc, cInnerIntermediate,
                        mlir::ValueRange{m_iv, n_iv},
                        mlir::AffineMapAttr::get(identityMap),
                        /*mask=*/nullptr, inBoundsAttr);
                mlir::Value updatedC = writeOp.getResult();

                builder.create<mlir::scf::YieldOp>(loopLoc, updatedC);
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
    registry.insert<mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::scf::SCFDialect,
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
