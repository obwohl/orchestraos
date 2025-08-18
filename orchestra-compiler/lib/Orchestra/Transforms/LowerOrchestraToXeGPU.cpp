#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/OrchestraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Orchestra/OrchestraDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"

namespace {

class TransferOpLowering
    : public mlir::OpConversionPattern<mlir::orchestra::TransferOp> {
public:
  using OpConversionPattern<mlir::orchestra::TransferOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::orchestra::TransferOp op,
                  typename mlir::orchestra::TransferOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct LowerOrchestraToXeGPUPass
    : public mlir::PassWrapper<LowerOrchestraToXeGPUPass,
                               mlir::OperationPass<mlir::gpu::GPUFuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToXeGPUPass)

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::xegpu::XeGPUDialect,
                    mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
  }
};

} // namespace

mlir::LogicalResult TransferOpLowering::matchAndRewrite(
    mlir::orchestra::TransferOp op,
    typename mlir::orchestra::TransferOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = adaptor.getSource();
  auto dst = op.getResult();

  auto srcType = src.getType().cast<mlir::MemRefType>();
  auto dstType = dst.getType().cast<mlir::MemRefType>();

  if (srcType.getRank() != 2) {
    return op.emitError("only 2D transfers are currently supported");
  }

  auto elementType = srcType.getElementType();
  if (!elementType.isF32() && !elementType.isF16() && !elementType.isBF16()) {
    return op.emitError("unsupported element type for XeGPU lowering. "
                        "Only f32, f16, and bf16 are supported.");
  }

  int64_t tileHeight = 32;
  int64_t tileWidth = 32;

  auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto outerDimSize = rewriter.create<mlir::memref::DimOp>(loc, src, 0);
  auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tileHeight);

  auto forOp = rewriter.create<mlir::scf::ForOp>(
      loc, zero, outerDimSize, step, std::nullopt);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());

  mlir::Value iv = forOp.getInductionVar();
  auto tileShape = llvm::ArrayRef<int64_t>{tileHeight, tileWidth};

  auto createTdesc = [&](mlir::Value memref, mlir::MemRefType type) {
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    offsets.push_back(iv);
    offsets.push_back(rewriter.getIndexAttr(0));

    mlir::MemRefLayoutAttrInterface layout;
    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (failed(getStridesAndOffset(type, strides, offset))) {
      op.emitError("failed to get strides and offset");
      return (mlir::xegpu::CreateNdDescOp)nullptr;
    }

    llvm::SmallVector<mlir::OpFoldResult> stride_results;
    for (int64_t stride : strides) {
      stride_results.push_back(rewriter.getIndexAttr(stride));
    }

    auto tdescType = mlir::xegpu::TensorDescType::get(
        rewriter.getContext(), tileShape, type.getElementType(),
        type.getMemorySpace(), mlir::Attribute());

    return rewriter.create<mlir::xegpu::CreateNdDescOp>(
        loc, tdescType, memref, offsets, stride_results,
        /*bound_check_mode=*/nullptr, /*layout=*/nullptr);
  };

  auto srcTdesc = createTdesc(src, srcType);
  if (!srcTdesc)
    return mlir::failure();
  auto dstTdesc = createTdesc(dst, dstType);
  if (!dstTdesc)
    return mlir::failure();

  auto vectorType =
      mlir::VectorType::get(tileShape, srcType.getElementType());

  auto loadedVector =
      rewriter.create<mlir::xegpu::LoadNdOp>(loc, vectorType, srcTdesc->getResult(0));
  rewriter.create<mlir::xegpu::StoreNdOp>(loc, loadedVector, dstTdesc->getResult(0));

  rewriter.setInsertionPointAfter(forOp);
  rewriter.create<mlir::xegpu::FenceOp>(loc, mlir::xegpu::FenceScopeAttr::get(rewriter.getContext(), mlir::xegpu::FenceScope::Workgroup));

  rewriter.eraseOp(op);

  return mlir::success();
}

void LowerOrchestraToXeGPUPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<mlir::orchestra::TransferOp>();
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect, mlir::xegpu::XeGPUDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<TransferOpLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir::orchestra {
std::unique_ptr<mlir::Pass> createLowerOrchestraToXeGPUPass() {
  return std::make_unique<LowerOrchestraToXeGPUPass>();
}

void registerLoweringToXeGPUPasses() {
  ::mlir::PassRegistration<LowerOrchestraToXeGPUPass>();
}
} // namespace mlir::orchestra
