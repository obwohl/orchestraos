#include "Orchestra/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {

// This is the original logic for lowering to ROCDL, extracted into its own pass.
class LowerOrchestraToROCDLPass
    : public mlir::PassWrapper<LowerOrchestraToROCDLPass,
                               mlir::OperationPass<mlir::gpu::GPUFuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToROCDLPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<mlir::orchestra::OrchestraDialect, mlir::gpu::GPUDialect,
                mlir::memref::MemRefDialect,
                mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::ROCDL::ROCDLDialect, mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::gpu::GPUFuncOp funcOp = getOperation();
    auto gpuModule = funcOp->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!gpuModule) {
      funcOp.emitError("must be nested inside a gpu.module");
      signalPassFailure();
      return;
    }

    // Collect all transfer ops to be rewritten.
    llvm::SmallVector<TransferOp, 4> transferOps;
    funcOp.walk(
        [&](TransferOp op) { transferOps.push_back(op); });

    mlir::OpBuilder builder(funcOp.getContext());
    for (auto op : transferOps) {
      builder.setInsertionPoint(op);
      auto loc = op.getLoc();
      auto source = op.getSource();
      auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());

      if (!sourceType || !sourceType.hasStaticShape()) {
        op.emitError("requires a memref with static shape");
        signalPassFailure();
        return;
      }

      // Create a new memref on the GPU in shared memory.
      auto destType = mlir::MemRefType::get(
          sourceType.getShape(), sourceType.getElementType(),
          sourceType.getLayout(),
          mlir::gpu::AddressSpaceAttr::get(op.getContext(),
                                           mlir::gpu::AddressSpace::Workgroup));
      auto dest = builder.create<mlir::memref::AllocOp>(loc, destType);

      // Get a pointer to the source and destination buffers.
      auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

      // Get the base addresses of the source and destination memrefs as integers.
      auto sourceIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, source);
      auto destIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, dest.getResult());

      // Cast the index type to a standard integer type (i64) for IntToPtrOp.
      auto i64Type = builder.getI64Type();
      auto sourceI64 = builder.create<arith::IndexCastOp>(loc, i64Type, sourceIndex.getResult());
      auto destI64 = builder.create<arith::IndexCastOp>(loc, i64Type, destIndex.getResult());

      auto sourcePtr = builder.create<LLVM::IntToPtrOp>(loc, llvmPointerType, sourceI64.getResult());
      auto destPtr = builder.create<LLVM::IntToPtrOp>(loc, llvmPointerType, destI64.getResult());

      auto sizeInBytes = sourceType.getNumElements() * sourceType.getElementTypeBitWidth() / 8;
      auto size = builder.create<arith::ConstantIntOp>(loc, sizeInBytes, 32);
      auto offset = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      auto aux = builder.create<arith::ConstantIntOp>(loc, 0, 32);

      auto globalPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 1);
      auto ldsPtrType = LLVM::LLVMPointerType::get(builder.getContext(), 3);

      auto globalPtr = builder.create<LLVM::AddrSpaceCastOp>(loc, globalPtrType, sourcePtr);
      auto ldsPtr = builder.create<LLVM::AddrSpaceCastOp>(loc, ldsPtrType, destPtr);

      // Create a rocdl.global_load_lds op.
      builder.create<mlir::ROCDL::GlobalLoadLDSOp>(
          loc, globalPtr, ldsPtr, size, offset, aux);

      op.getResult().replaceAllUsesWith(dest.getResult());
      op.erase();
    }
  }
};
} // namespace

namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerOrchestraToROCDLPass() {
  return std::make_unique<LowerOrchestraToROCDLPass>();
}
} // namespace orchestra
} // namespace mlir
