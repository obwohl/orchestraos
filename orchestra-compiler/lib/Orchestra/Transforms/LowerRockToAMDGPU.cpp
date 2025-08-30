#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"
#include "Orchestra/Transforms/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {
/// This OpConversionPattern provides the complete lowering logic for
/// converting a `rock::GemmOp` into a tiled loop nest that uses the
/// `amdgpu::MFMAOp` intrinsic. It correctly handles immutable tensor semantics.
class GemmLoweringPattern : public mlir::OpConversionPattern<mlir::rock::GemmOp> {
public:
  using OpConversionPattern<mlir::rock::GemmOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::rock::GemmOp gemmOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = gemmOp.getLoc();
    mlir::MLIRContext *context = getContext();

    //===------------------------------------------------------------------===//
    // 1. Define Tiling and Vectorization Parameters
    //===------------------------------------------------------------------===//
    // These parameters are chosen for a 32x32x2 f32 MFMA variant, common on
    // CDNA-class hardware.
    constexpr int64_t mTileSize = 32;
    constexpr int64_t nTileSize = 32;
    constexpr int64_t kTileSize = 2;

    // Define the vector types required by the target amdgpu.mfma intrinsic.
    // The accumulator is a 2D vector, while the inputs are 1D vectors that
    // the hardware interprets as matrices. This is a common pattern.
    auto f32Type = rewriter.getF32Type();
    auto vectorAType = mlir::VectorType::get({mTileSize * kTileSize}, f32Type);
    auto vectorBType = mlir::VectorType::get({kTileSize * nTileSize}, f32Type);
    auto vectorAccumulatorType =
        mlir::VectorType::get({mTileSize, nTileSize}, f32Type);

    //===------------------------------------------------------------------===//
    // 2. Get Operands and Tensor Shapes
    //===------------------------------------------------------------------===//
    mlir::Value matrixA = adaptor.getMatrixA();
    mlir::Value matrixB = adaptor.getMatrixB();
    mlir::Value matrixC = adaptor.getMatrixC(); // Initial output tensor

    auto tensorAType = matrixA.getType().cast<mlir::RankedTensorType>();
    auto tensorBType = matrixB.getType().cast<mlir::RankedTensorType>();
    int64_t dimM = tensorAType.getShape()[0];
    int64_t dimK = tensorAType.getShape()[1];
    int64_t dimN = tensorBType.getShape()[1];

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

    // Create a 2D identity map for simple 2D tile loads.
    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(2, context);

    // Create a zero-padding value for vector.transfer_read. This is essential
    // for handling boundary conditions in a production implementation.
    mlir::Value padding = rewriter.create<mlir::arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0.0));

    // The outer M loop iterates over the rows of the output tensor.
    // It carries the progressively updated output tensor `matrixC` as a
    // loop-carried variable (`iter_args`).
    auto outerLoopM = rewriter.create<mlir::scf::ForOp>(
        loc, c0, mBound, mStep, /*iterArgs=*/mlir::ValueRange{matrixC},
        [&](mlir::OpBuilder &builder, mlir::Location loopLoc, mlir::Value m_iv,
            mlir::ValueRange mIterArgs) {
          mlir::Value cTensorM = mIterArgs[0];

          // The middle N loop iterates over the columns of the output tensor.
          // It also carries the updated output tensor.
          auto middleLoopN = builder.create<mlir::scf::ForOp>(
              loopLoc, c0, nBound, nStep, /*iterArgs=*/mlir::ValueRange{cTensorM},
              [&](mlir::OpBuilder &builder, mlir::Location loopLoc, mlir::Value n_iv,
                  mlir::ValueRange nIterArgs) {
                mlir::Value cTensorN = nIterArgs[0];

                // Load the initial accumulator tile from the C tensor.
                // This read must happen *inside* the M/N loops.
                mlir::Value accVector = builder.create<mlir::vector::TransferReadOp>(
                    loopLoc, vectorAccumulatorType, cTensorN,
                    mlir::ValueRange{m_iv, n_iv},
                    mlir::AffineMapAttr::get(identityMap), padding,
                    /*mask=*/mlir::Value(),
                    /*inBounds=*/mlir::ArrayAttr());

                // The innermost K loop performs the reduction.
                // It carries the vector accumulator `accVector`.
                auto innerLoopK = builder.create<mlir::scf::ForOp>(
                    loopLoc, c0, kBound, kStep, /*iterArgs=*/mlir::ValueRange{accVector},
                    [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
                        mlir::Value k_iv, mlir::ValueRange kIterArgs) {
                      mlir::Value currentAcc = kIterArgs[0];

                      // Load a tile from matrix A.
                      mlir::Value vecA = builder.create<mlir::vector::TransferReadOp>(
                          loopLoc, vectorAType, matrixA,
                          mlir::ValueRange{m_iv, k_iv},
                          mlir::AffineMapAttr::get(identityMap), padding);

                      // Load a tile from matrix B.
                      mlir::Value vecB = builder.create<mlir::vector::TransferReadOp>(
                          loopLoc, vectorBType, matrixB,
                          mlir::ValueRange{k_iv, n_iv},
                          mlir::AffineMapAttr::get(identityMap), padding);

                      // Reshape the 1D loaded vectors into 2D vectors suitable for MFMA.
                      // This is often required as hardware expects specific register layouts.
                      auto reshapedVecAType = mlir::VectorType::get({mTileSize, kTileSize}, f32Type);
                      auto reshapedVecBType = mlir::VectorType::get({kTileSize, nTileSize}, f32Type);
                      mlir::Value reshapedVecA = builder.create<mlir::vector::ShapeCastOp>(loopLoc, reshapedVecAType, vecA);
                      mlir::Value reshapedVecB = builder.create<mlir::vector::ShapeCastOp>(loopLoc, reshapedVecBType, vecB);

                      // Invoke the amdgpu.mfma intrinsic.
                      auto mfmaOp = builder.create<mlir::amdgpu::MFMAOp>(
                          loopLoc, vectorAccumulatorType, reshapedVecA, reshapedVecB,
                          currentAcc, builder.getI32IntegerAttr(mTileSize),
                          builder.getI32IntegerAttr(nTileSize),
                          builder.getI32IntegerAttr(kTileSize),
                          /*blocks=*/builder.getI32IntegerAttr(1));

                      builder.create<mlir::scf::YieldOp>(loopLoc, mfmaOp.getDestC());
                    });

                mlir::Value finalAcc = innerLoopK.getResult(0);

                // "Update" the output tensor by creating a new tensor with the
                // accumulated tile inserted at the correct position.
                mlir::Value updatedCTensor =
                    builder.create<mlir::tensor::InsertSliceOp>(
                        loopLoc, finalAcc, cTensorN,
                        /*offsets=*/mlir::ValueRange{m_iv, n_iv},
                        /*sizes=*/mlir::ValueRange{},
                        /*strides=*/mlir::ValueRange{},
                        /*static_offsets=*/
                        rewriter.getDenseI64ArrayAttr({mlir::ShapedType::kDynamic,
                                                       mlir::ShapedType::kDynamic}),
                        /*static_sizes=*/
                        rewriter.getDenseI64ArrayAttr({mTileSize, nTileSize}),
                        /*static_strides=*/
                        rewriter.getDenseI64ArrayAttr({1, 1}));

                builder.create<mlir::scf::YieldOp>(loopLoc, updatedCTensor);
              });
          builder.create<mlir::scf::YieldOp>(loopLoc, middleLoopN.getResult(0));
        });

    //===------------------------------------------------------------------===//
    // 4. Finalize the Lowering
    //===------------------------------------------------------------------===//
    // Replace the original rock.gemm operation with the final result of the
    // loop nest.
    rewriter.replaceOp(gemmOp, outerLoopM.getResult(0));
    return mlir::success();
  }
};

struct RockToAMDGPUConversionPass
    : public mlir::PassWrapper<RockToAMDGPUConversionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  mlir::StringRef getArgument() const final { return "lower-rock-to-amdgpu"; }
  mlir::StringRef getDescription() const final {
    return "Lower Rock dialect to AMDGPU dialect.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::amdgpu::AMDGPUDialect, mlir::arith::ArithDialect,
                    mlir::vector::VectorDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<mlir::rock::RockDialect>();
    target.addLegalDialect<mlir::amdgpu::AMDGPUDialect,
                           mlir::arith::ArithDialect,
                           mlir::vector::VectorDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GemmLoweringPattern>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // anonymous namespace

namespace mlir {
namespace rock {
std::unique_ptr<mlir::Pass> createLowerRockToAMDGPUConversionPass() {
  return std::make_unique<RockToAMDGPUConversionPass>();
}

void registerLowerRockToAMDGPU() {
  mlir::PassRegistration<RockToAMDGPUConversionPass>();
}
} // namespace rock
} // namespace mlir
