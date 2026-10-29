#ifndef COMMONIR_TRANSFORMS_LAYOUTHINTS_H
#define COMMONIR_TRANSFORMS_LAYOUTHINTS_H

#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::tile {

// Reuse the caller's operation rules without depending on a GPU dialect or
// transform library. A null encoding stops propagation through that operation.
struct LayoutHintCallbacks {
  llvm::function_ref<Attribute(Operation *, Attribute)> inferSrcEncoding;
  llvm::function_ref<Attribute(Operation *, Attribute)> inferDstEncoding;
  llvm::function_ref<SmallVector<Value>(Operation *, int)> getTiedArgs;
  // A target may keep a producer in its native layout and convert at its use.
  llvm::function_ref<bool(Value, Attribute)> canPropagateTo;
  // Align typed boundaries without creating another explicit layout constraint.
  llvm::function_ref<Value(OpBuilder &, Location, Value, RankedTensorType)>
      convertLayout;
  // Whether a logical tile and its full tensor can directly share an encoding.
  // Without a target rule, stop propagation across the tile boundary.
  llvm::function_ref<bool(Operation *, Attribute)> canPropagateThroughTensorTile =
      nullptr;
};

// Propagate tile.set_layout requirements into tensor encodings. Inferred values
// are not explicit constraints: the original set_layout ops remain in the IR
// and their lowering preserves the anchors. The caller validates target layouts
// beforehand and supplies conversions for typed boundaries.
LogicalResult propagateLayoutHints(ModuleOp module,
                                  const LayoutHintCallbacks &callbacks);

// Check agreement between result/operand constraints on a memory operation.
// The caller decides which operations require a common memory encoding.
LogicalResult inferExplicitMemoryEncoding(Operation *op, Attribute &encoding);

// Drop planning markers after layout-sensitive optimization, retaining types.
void clearLayoutHints(ModuleOp module);

} // namespace mlir::triton::tile

#endif
