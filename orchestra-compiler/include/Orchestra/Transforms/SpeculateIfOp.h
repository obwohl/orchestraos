#ifndef ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H
#define ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace orchestra {

struct SpeculateIfOpPattern : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern<mlir::scf::IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H
