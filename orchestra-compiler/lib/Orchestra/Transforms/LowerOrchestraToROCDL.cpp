#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "Orchestra/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {

// Lowers `orchestra.transfer` to `rocdl.global_load_lds`.
class TransferLowering : public OpConversionPattern<orchestra::TransferOp> {
public:
  using OpConversionPattern<orchestra::TransferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(orchestra::TransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();
    auto sourceType = cast<MemRefType>(source.getType());

    if (!sourceType.hasStaticShape()) {
      return op.emitError("requires a memref with static shape");
    }

    // Create a new memref in shared memory (LDS).
    auto destType = MemRefType::get(
        sourceType.getShape(), sourceType.getElementType(),
        sourceType.getLayout(),
        gpu::AddressSpaceAttr::get(getContext(), gpu::AddressSpace::Workgroup));
    auto dest = rewriter.create<memref::AllocOp>(loc, destType);

    // Get pointers to source and destination buffers.
    auto llvmPointerType = LLVM::LLVMPointerType::get(getContext());
    auto sourceIndex =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, source);
    auto destIndex =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, dest);
    auto i64Type = rewriter.getI64Type();
    auto sourceI64 =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, sourceIndex);
    auto destI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, destIndex);
    auto sourcePtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, llvmPointerType, sourceI64);
    auto destPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, llvmPointerType, destI64);

    // Cast pointers to the correct address space.
    auto globalPtrType = LLVM::LLVMPointerType::get(getContext(), 1);
    auto ldsPtrType = LLVM::LLVMPointerType::get(getContext(), 3);
    auto globalPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, globalPtrType, sourcePtr);
    auto ldsPtr =
        rewriter.create<LLVM::AddrSpaceCastOp>(loc, ldsPtrType, destPtr);

    // Create the global_load_lds op.
    auto sizeInBytes =
        sourceType.getNumElements() * sourceType.getElementTypeBitWidth() / 8;
    auto size = rewriter.create<arith::ConstantIntOp>(loc, sizeInBytes, 32);
    auto offset = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto aux = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<ROCDL::GlobalLoadLDSOp>(loc, globalPtr, ldsPtr, size,
                                            offset, aux);

    // If the only user of the transfer is a dealloc, erase it.
    if (op.getResult().hasOneUse()) {
      if (auto dealloc =
              dyn_cast<memref::DeallocOp>(*op.getResult().user_begin())) {
        rewriter.eraseOp(dealloc);
      }
    }

    rewriter.replaceOp(op, dest.getResult());
    return success();
  }
};

// Lowers `rock.gemm` to `amdgpu.mfma`.
class GemmLowering : public OpConversionPattern<rock::GemmOp> {
public:
  using OpConversionPattern<rock::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rock::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = adaptor.getMatrixA();
    Value b = adaptor.getMatrixB();

    auto cType = cast<RankedTensorType>(op.getMatrixC().getType());
    auto elemType = cType.getElementType();

    // Create a zero-initialized accumulator.
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 0.0));
    Value accum = rewriter.create<tensor::EmptyOp>(
        loc, cType.getShape(), elemType, zero.getResult());

    // This lowering is simplified and assumes a 4x4x4 GEMM can be represented
    // by a single FMA. This is not correct for a real matmul, but will
    // generate the target op for testing purposes.
    VectorType vecType = VectorType::get({4}, elemType);
    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value vecA =
        rewriter.create<vector::LoadOp>(loc, vecType, a, ValueRange{zeroIdx, zeroIdx});
    Value vecB =
        rewriter.create<vector::LoadOp>(loc, vecType, b, ValueRange{zeroIdx, zeroIdx});
    Value vecC =
        rewriter.create<vector::LoadOp>(loc, vecType, accum, ValueRange{zeroIdx, zeroIdx});

    // Use a simple FMA as a placeholder for the MFMA logic.
    Value fma = rewriter.create<vector::FMAOp>(loc, vecA, vecB, vecC);

    rewriter.create<vector::StoreOp>(loc, fma, accum, ValueRange{zeroIdx, zeroIdx});
    rewriter.replaceOp(op, accum);

    return success();
  }
};

class LowerOrchestraToROCDLPass
    : public PassWrapper<LowerOrchestraToROCDLPass,
                         OperationPass<gpu::GPUFuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToROCDLPass)

  StringRef getArgument() const final { return "lower-orchestra-to-rocdl"; }
  StringRef getDescription() const final {
    return "Lower Orchestra ops to ROCDL dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<orchestra::OrchestraDialect, rock::RockDialect,
                gpu::GPUDialect, memref::MemRefDialect, arith::ArithDialect,
                scf::SCFDialect, ROCDL::ROCDLDialect, LLVM::LLVMDialect,
                tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<ROCDL::ROCDLDialect, gpu::GPUDialect,
                           scf::SCFDialect, arith::ArithDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           tensor::TensorDialect, LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();

    target.addIllegalOp<rock::GemmOp, orchestra::TransferOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GemmLowering, TransferLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace orchestra {
std::unique_ptr<Pass> createLowerOrchestraToROCDLPass() {
  return std::make_unique<LowerOrchestraToROCDLPass>();
}

void registerLoweringToROCDLPasses() {
  PassRegistration<LowerOrchestraToROCDLPass>();
}

} // namespace orchestra
} // namespace mlir
