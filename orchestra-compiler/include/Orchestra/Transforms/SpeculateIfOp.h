#ifndef ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H
#define ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace orchestra {

struct SpeculateIfOpPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_SPECULATEIFOP_H
