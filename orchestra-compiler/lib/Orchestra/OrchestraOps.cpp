#include "Orchestra/OrchestraOps.h"
#include "Orchestra/OrchestraDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace orchestra;

void CommitOp::print(OpAsmPrinter &p) {
  p << " " << getCondition();
  p << " true(" << getTrueValues() << ")";
  p << " false(" << getFalseValues() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"num_true"});
  p << " : " << FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

ParseResult CommitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand condition;
  if (parser.parseOperand(condition))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> trueValues;
  if (parser.parseKeyword("true") || parser.parseLParen() ||
      parser.parseOperandList(trueValues) || parser.parseRParen())
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> falseValues;
  if (parser.parseKeyword("false") || parser.parseLParen() ||
      parser.parseOperandList(falseValues) || parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  FunctionType type;
  if (parser.parseColonType(type))
    return failure();

  if (parser.resolveOperand(condition, type.getInput(0), result.operands))
    return failure();

  auto valueTypes = type.getInputs().drop_front();
  for (size_t i = 0; i < trueValues.size(); ++i) {
    if (parser.resolveOperand(trueValues[i], valueTypes[i], result.operands))
      return failure();
  }
  for (size_t i = 0; i < falseValues.size(); ++i) {
    if (parser.resolveOperand(falseValues[i], valueTypes[i + trueValues.size()], result.operands))
      return failure();
  }

  result.addTypes(type.getResults());
  result.addAttribute("num_true", parser.getBuilder().getI32IntegerAttr(trueValues.size()));
  return success();
}

mlir::LogicalResult TaskOp::verify() {
  if (!getTargetArch()) {
    return emitOpError("requires attribute 'target_arch'");
  }
  auto arch = getTargetArch().get("arch");
  if (!arch) {
    return emitOpError("requires a non-empty 'arch' in 'target_arch'");
  }
  if (!arch.isa<StringAttr>()) {
    return emitOpError("requires a string 'arch' in 'target_arch'");
  }
  if (arch.cast<StringAttr>().getValue().empty()) {
    return emitOpError("requires a non-empty 'arch' in 'target_arch'");
  }
  return mlir::success();
}

mlir::LogicalResult ScheduleOp::verify() {
  for (auto &op : getBody().front()) {
    if (!isa<TaskOp, ReturnOp>(op)) {
      return op.emitOpError("only 'orchestra.task' operations are allowed inside a 'orchestra.schedule'");
    }
  }
  return mlir::success();
}

mlir::LogicalResult CommitOp::verify() {
  if (getTrueValues().size() != getFalseValues().size()) {
    return emitOpError("has mismatched number of true and false operands");
  }

  for (size_t i = 0; i < getTrueValues().size(); ++i) {
    if (getTrueValues()[i].getType() != getFalseValues()[i].getType()) {
      return emitOpError("requires 'true' and 'false' value types to match");
    }
  }

  if (getNumResults() != getTrueValues().size()) {
    return emitOpError("requires number of results to match number of values in each branch");
  }

  for (size_t i = 0; i < getNumResults(); ++i) {
    if (getResult(i).getType() != getTrueValues()[i].getType()) {
      return emitOpError("requires result types to match operand types");
    }
  }

  return mlir::success();
}

namespace {
struct CommitOpCanonicalization : public OpRewritePattern<CommitOp> {
  using OpRewritePattern<CommitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommitOp op,
                              PatternRewriter &rewriter) const override {
    IntegerAttr condAttr;
    if (!matchPattern(op.getCondition(), m_Constant(&condAttr))) {
      return failure();
    }

    if (condAttr.getValue().isOne()) {
      rewriter.replaceOp(op, op.getTrueValues());
    } else {
      rewriter.replaceOp(op, op.getFalseValues());
    }
    return success();
  }
};
} // namespace

void CommitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<CommitOpCanonicalization>(context);
}

// Stubs for other ops to link.
namespace {
struct TransferOpCanonicalization : public OpRewritePattern<TransferOp> {
  using OpRewritePattern<TransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op,
                              PatternRewriter &rewriter) const override {
    auto definingOp = op.getSource().getDefiningOp<TransferOp>();
    if (!definingOp) {
      return failure();
    }

    if (!definingOp->hasOneUse()) {
      return failure();
    }

    if (op.getFrom() != definingOp.getTo()) {
      return failure();
    }

    // Choose the higher priority.
    IntegerAttr newPriority = op.getPriorityAttr();
    if (definingOp.getPriorityAttr() &&
        (!newPriority || definingOp.getPriorityAttr().getValue().getSExtValue() > newPriority.getValue().getSExtValue())) {
      newPriority = definingOp.getPriorityAttr();
    }

    rewriter.replaceOpWithNewOp<TransferOp>(op, op.getType(), definingOp.getSource(),
                                           definingOp.getFrom(), op.getTo(),
                                           newPriority);
    return success();
  }
};
} // namespace

void TransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<TransferOpCanonicalization>(context);
}
mlir::LogicalResult TransferOp::verify() { return mlir::success(); }

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
