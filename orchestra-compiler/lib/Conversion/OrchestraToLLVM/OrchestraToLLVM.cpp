#include "OrchestraToLLVM.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVM/IR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "orchestra/include/Orchestra/OrchestraDialect.h"

namespace {

class OrchestraTransferOpLowering
    : public mlir::ConvertOpToLLVMPattern<orchestra::TransferOp> {
public:
  using mlir::ConvertOpToLLVMPattern<
      orchestra::TransferOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(orchestra::TransferOp op,
                  orchestra::TransferOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    auto source = adaptor.getSource();
    auto resultType = op.getResult().getType().cast<mlir::MemRefType>();

    // Allocate the destination buffer.
    auto dest = rewriter.create<mlir::memref::AllocOp>(loc, resultType);

    // Get the element type and size.
    auto elementType = resultType.getElementType();
    int64_t elementSize = 1; // Default to 1 byte
    if (elementType.isIntOrFloat()) {
        elementSize = elementType.getIntOrFloatBitWidth() / 8;
    }

    // Get the total size in bytes.
    int64_t totalSize = resultType.getNumElements() * elementSize;
    auto size = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(totalSize));

    // Get the name of the memcpy function.
    auto memcpyName = "memcpy";

    // Check if the memcpy function is already declared.
    auto memcpyFunc = module.lookupSymbol<mlir::func::FuncOp>(memcpyName);
    if (!memcpyFunc) {
      // If not, declare it.
      auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      auto funcType = mlir::FunctionType::get(
          rewriter.getContext(), {ptrType, ptrType, rewriter.getI64Type()},
          {});
      memcpyFunc = rewriter.create<mlir::func::FuncOp>(
          loc, memcpyName, funcType);
      memcpyFunc.setPrivate();
    }

    // Cast the source and destination memrefs to void*.
    auto srcPtr = rewriter.create<mlir::memref::CastOp>(
        loc, source, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
    auto dstPtr = rewriter.create<mlir::memref::CastOp>(
        loc, dest, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

    // Call memcpy.
    rewriter.create<mlir::func::CallOp>(
        loc, memcpyFunc, mlir::TypeRange{},
        mlir::ValueRange{dstPtr, srcPtr, size});

    // Replace the original op with the destination memref.
    rewriter.replaceOp(op, {dest});

    return mlir::success();
  }
};

struct OrchestraToLLVMConversionPass
    : public mlir::PassWrapper<OrchestraToLLVMConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrchestraToLLVMConversionPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::func::FuncOp,
                    mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<OrchestraTransferOpLowering>(typeConverter);

    auto module = getOperation();
    if (failed(
            applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  OrchestraToLLVMConversionPass() = default;
};

} // end anonymous namespace

namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<OrchestraToLLVMConversionPass>();
}

void registerOrchestraToLLVMPass() {
  mlir::PassRegistration<OrchestraToLLVMConversionPass>(
      "lower-orchestra-to-llvm", "Lowers the Orchestra dialect to LLVM.");
}
} // namespace orchestra
} // namespace mlir
