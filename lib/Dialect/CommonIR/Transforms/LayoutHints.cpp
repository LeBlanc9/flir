#include "mlir-ext/Dialect/CommonIR/Transforms/LayoutHints.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace mlir::triton::tile {

static LogicalResult mergeExplicitMemoryEncoding(Operation *op, Value value,
                                                 Attribute candidate,
                                                 Attribute &encoding) {
  if (!candidate)
    return success();

  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.getEncoding() != candidate)
      return op->emitOpError("has explicit CommonIR encoding that does not match "
                             "the tensor type encoding");
  }

  if (!encoding) {
    encoding = candidate;
    return success();
  }

  if (encoding == candidate)
    return success();

  op->emitOpError("has conflicting explicit CommonIR memory encodings:\n  ")
      << encoding << "\nand\n  " << candidate;
  return failure();
}

LogicalResult inferExplicitMemoryEncoding(Operation *op, Attribute &encoding) {
  encoding = getExplicitMemoryEncoding(op);

  for (OpResult result : op->getResults()) {
    if (failed(mergeExplicitMemoryEncoding(
            op, result,
            getExplicitResultEncoding(op, result.getResultNumber()), encoding)))
      return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (failed(mergeExplicitMemoryEncoding(
            op, operand.get(), getExplicitValueEncoding(operand.get()), encoding)))
      return failure();
  }

  return success();
}

namespace {

// Propagate explicit tensor-layout requirements from tile.set_layout.
struct LayoutHintInfo {
  Attribute encoding;
  bool mayVary = true;

  explicit operator bool() const { return bool(encoding); }
};

static LogicalResult mergeLayoutHintInfo(LayoutHintInfo oldInfo,
                                        LayoutHintInfo newInfo, Operation *op,
                                        LayoutHintInfo &merged) {
  if (!oldInfo) {
    merged = newInfo;
    return success();
  }
  if (!newInfo) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.encoding == newInfo.encoding) {
    merged = oldInfo;
    merged.mayVary = oldInfo.mayVary && newInfo.mayVary;
    return success();
  }
  if (oldInfo.mayVary && !newInfo.mayVary) {
    merged = newInfo;
    return success();
  }
  if (!oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }

  op->emitOpError("found conflicting CommonIR layout hints for value:\n  ")
      << oldInfo.encoding << "\nand\n  " << newInfo.encoding;
  return failure();
}

static LogicalResult
updateLayoutHint(ArrayRef<Value> values, LayoutHintInfo info, FuncOp func,
                 llvm::MapVector<Value, LayoutHintInfo> &valueToEncoding,
                 llvm::PriorityWorklist<Value> &worklist,
                 const LayoutHintCallbacks &callbacks) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    if (!callbacks.canPropagateTo(value, info.encoding)) {
      if (!info.mayVary)
        return func.emitError("unsupported explicit layout for value ") << value;
      continue;
    }

    auto [it, inserted] = valueToEncoding.insert({value, info});
    if (!inserted) {
      Operation *defOp = value.getDefiningOp();
      Operation *diagOp = defOp ? defOp : func.getOperation();
      LayoutHintInfo merged;
      if (failed(mergeLayoutHintInfo(it->second, info, diagOp, merged)))
        return failure();
      if (merged.encoding == it->second.encoding &&
          merged.mayVary == it->second.mayVary)
        continue;
      it->second = merged;
    }
    worklist.insert(value);
  }
  return success();
}

static LogicalResult
propagateFunctionLayoutHints(FuncOp func, const LayoutHintCallbacks &callbacks) {
  llvm::SmallVector<std::pair<Value, LayoutHintInfo>> seedEncodings;
  func.walk([&](tile::SetLayoutOp op) {
    seedEncodings.push_back(
        {op.getSource(), LayoutHintInfo{op.getLayout(), true}});
    seedEncodings.push_back(
        {op.getResult(), LayoutHintInfo{op.getLayout(), false}});
  });
  if (seedEncodings.empty())
    return success();

  llvm::MapVector<Value, LayoutHintInfo> valueToEncoding;
  llvm::PriorityWorklist<Value> worklist;
  auto canPropagateTile = [&](Operation *op, Attribute encoding) {
    return callbacks.canPropagateThroughTensorTile &&
           callbacks.canPropagateThroughTensorTile(op, encoding);
  };
  for (auto &[value, info] : seedEncodings) {
    if (failed(updateLayoutHint({value}, info, func, valueToEncoding, worklist,
                                callbacks)))
      return failure();
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    LayoutHintInfo info = valueToEncoding[value];
    assert(info && "worklist value must have an encoding");
    // Splat constants can be rematerialized in each user's layout. Do not
    // broadcast a consumer's inferred encoding through a shared constant to
    // unrelated users (for example, loads sharing the same masked fill value).
    if (auto constant = value.getDefiningOp<arith::ConstantOp>();
        info.mayVary && constant &&
        isa<SplatElementsAttr>(constant.getValue()))
      continue;

    // Only the set_layout result itself is fixed. Every outgoing requirement
    // is inferred, including those crossing tied control-flow values.
    info.mayVary = true;

    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        int offset = 3 * isa<scf::ForOp>(op);
        auto tiedArgs = callbacks.getTiedArgs(op, use.getOperandNumber() - offset);
        if (failed(updateLayoutHint(tiedArgs, info, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
        continue;
      }
      if (isa<scf::YieldOp>(op)) {
        auto tiedArgs =
            callbacks.getTiedArgs(op->getParentOp(), use.getOperandNumber());
        if (failed(updateLayoutHint(tiedArgs, info, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
        continue;
      }
      if (auto dot = dyn_cast<triton::DotOpInterface>(op)) {
        // A/B use dot-operand encodings, while C and D use the parent MMA
        // encoding. Only C and D are layout-equivalent across a dot.
        if (use.getOperandNumber() == 2 &&
            failed(updateLayoutHint({dot.getD()}, info, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
        continue;
      }
      if (isa<tile::SetLayoutOp>(op))
        continue;

      if (isa<ExtractTileOp, InsertTileOp>(op)) {
        // Insert's base and result have the same type. Across a size-changing
        // tile boundary, sharing a layout is only a target-checked preference.
        bool tiedBase = isa<InsertTileOp>(op) && use.getOperandNumber() == 0;
        if ((tiedBase || canPropagateTile(op, info.encoding)) &&
            failed(updateLayoutHint({op->getResult(0)}, info, func,
                                    valueToEncoding, worklist, callbacks)))
          return failure();
        continue;
      }

      Attribute dstEncoding = callbacks.inferDstEncoding(op, info.encoding);
      if (!dstEncoding)
        continue;
      LayoutHintInfo dstInfo{dstEncoding};
      if (failed(updateLayoutHint(llvm::to_vector_of<Value>(op->getResults()),
                                  dstInfo, func, valueToEncoding, worklist,
                                  callbacks)))
        return failure();
    }

    if (auto opResult = dyn_cast<OpResult>(value)) {
      Operation *definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs =
            callbacks.getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateLayoutHint(tiedArgs, info, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
      } else if (isa<triton::DotOpInterface>(definingOp)) {
        // A/B have independent anchors. Only propagate D's layout to C;
        // an incompatible accumulator can be converted at the dot boundary.
        if (failed(updateLayoutHint({definingOp->getOperand(2)},
                                     {info.encoding, true}, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
      } else if (auto extract = dyn_cast<ExtractTileOp>(definingOp)) {
        if (canPropagateTile(extract, info.encoding) &&
            failed(updateLayoutHint({extract.getSource()}, {info.encoding, true},
                                    func, valueToEncoding, worklist, callbacks)))
          return failure();
      } else if (auto insert = dyn_cast<InsertTileOp>(definingOp)) {
        if (failed(updateLayoutHint({insert.getBase()}, info, func,
                                    valueToEncoding, worklist, callbacks)))
          return failure();
        if (canPropagateTile(insert, info.encoding) &&
            failed(updateLayoutHint({insert.getUpdate()}, {info.encoding, true},
                                    func, valueToEncoding, worklist, callbacks)))
          return failure();
      } else if (!isa<tile::SetLayoutOp>(definingOp)) {
        Attribute srcEncoding =
            callbacks.inferSrcEncoding(definingOp, info.encoding);
        if (srcEncoding) {
          LayoutHintInfo srcInfo{srcEncoding};
          if (failed(updateLayoutHint(
                  llvm::to_vector_of<Value>(definingOp->getOperands()), srcInfo,
                  func, valueToEncoding, worklist, callbacks)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        int offset = isa<scf::ForOp>(parentOp);
        auto tiedArgs =
            callbacks.getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        if (failed(updateLayoutHint(tiedArgs, info, func, valueToEncoding,
                                     worklist, callbacks)))
          return failure();
      }
    }
  }

  for (auto &[value, info] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(value.getType());
    if (existingTy.getEncoding() != info.encoding) {
      auto newTy = existingTy.cloneWithEncoding(info.encoding);
      value.setType(newTy);

      if (auto opResult = dyn_cast<OpResult>(value)) {
        if (auto constant = dyn_cast<arith::ConstantOp>(opResult.getOwner())) {
          if (auto elements =
                  dyn_cast<DenseElementsAttr>(constant.getValueAttr()))
            constant.setValueAttr(elements.reshape(newTy));
        }
      }
    }
  }

  // A fixed producer (for example a global load) may stop propagation. Align
  // typed boundaries without treating the inferred layout as an explicit anchor.
  auto alignOperand = [&](OpOperand &operand, Type requiredType) {
    auto type = dyn_cast<RankedTensorType>(requiredType);
    if (!type || !type.getEncoding() || operand.get().getType() == type)
      return;
    OpBuilder builder(operand.getOwner());
    operand.set(callbacks.convertLayout(builder, operand.getOwner()->getLoc(),
                                       operand.get(), type));
  };
  auto alignOperands = [&](auto operands, TypeRange types) {
    for (auto [operand, requiredType] : llvm::zip(operands, types))
      alignOperand(operand, requiredType);
  };
  func.walk([&](Operation *op) {
    if (auto insert = dyn_cast<InsertTileOp>(op)) {
      alignOperand(insert.getBaseMutable(), insert.getType());
    } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      alignOperands(loop.getInitsMutable(), loop->getResultTypes());
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      alignOperands(loop.getInitsMutable(),
                    ValueRange(loop.getBeforeArguments()).getTypes());
    } else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      Operation *parent = op->getParentOp();
      if (auto loop = dyn_cast<scf::WhileOp>(parent))
        alignOperands(yield.getResultsMutable(),
                      ValueRange(loop.getBeforeArguments()).getTypes());
      else if (isa<scf::ForOp, scf::IfOp>(parent))
        alignOperands(yield.getResultsMutable(), parent->getResultTypes());
    } else if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
      alignOperands(condition.getArgsMutable(),
                    op->getParentOp()->getResultTypes());
    }
  });

  return success();
}

} // namespace

LogicalResult propagateLayoutHints(ModuleOp module,
                                  const LayoutHintCallbacks &callbacks) {
  for (FuncOp func : module.getOps<FuncOp>())
    if (failed(propagateFunctionLayoutHints(func, callbacks)))
      return failure();
  return success();
}

void clearLayoutHints(ModuleOp module) {
  module.walk([](Operation *op) {
    op->removeAttr(getExplicitMemoryEncodingAttrName());
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      op->removeAttr(getExplicitEncodingAttrName(i));
  });
}

} // namespace mlir::triton::tile
