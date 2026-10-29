#ifndef COMMONIR_TRANSFORMS_LAYOUTHINTS_H
#define COMMONIR_TRANSFORMS_LAYOUTHINTS_H

#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::tile {

// Caller-supplied rules; a null encoding stops propagation through an operation.
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

// Infer tensor encodings without making them explicit constraints; set_layout
// ops remain as anchors. The caller must validate target layouts beforehand.
LogicalResult propagateLayoutHints(ModuleOp module,
                                  const LayoutHintCallbacks &callbacks);

} // namespace mlir::triton::tile

#endif
