#ifndef COMMONIR_IR_LAYOUTHINTATTRS_H
#define COMMONIR_IR_LAYOUTHINTATTRS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace mlir::triton::tile {

static constexpr const char *kExplicitEncodingAttrPrefix =
    "tile.explicit_encoding.";
static constexpr const char *kExplicitMemoryEncodingAttrName =
    "tile.explicit_memory_encoding";

inline std::string getExplicitEncodingAttrName(unsigned resultNumber) {
  return (llvm::Twine(kExplicitEncodingAttrPrefix) + llvm::Twine(resultNumber))
      .str();
}

inline const char *getExplicitMemoryEncodingAttrName() {
  return kExplicitMemoryEncodingAttrName;
}

inline Attribute getExplicitResultEncoding(Operation *op, unsigned resultNumber) {
  return op->getAttr(getExplicitEncodingAttrName(resultNumber));
}

inline void setExplicitResultEncoding(Operation *op, unsigned resultNumber,
                                      Attribute encoding) {
  op->setAttr(getExplicitEncodingAttrName(resultNumber), encoding);
}

inline void setExplicitResultEncoding(OpResult result, Attribute encoding) {
  setExplicitResultEncoding(result.getOwner(), result.getResultNumber(),
                            encoding);
}

inline Attribute getExplicitMemoryEncoding(Operation *op) {
  return op->getAttr(kExplicitMemoryEncodingAttrName);
}

inline void setExplicitMemoryEncoding(Operation *op, Attribute encoding) {
  op->setAttr(kExplicitMemoryEncodingAttrName, encoding);
}

inline Attribute getExplicitValueEncoding(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return getExplicitResultEncoding(result.getOwner(),
                                     result.getResultNumber());
  return nullptr;
}

// Block arguments have no defining operation. Region results also cannot carry
// an indexed hint reliably across generic control-flow canonicalization. Keep a
// convert_layout as the carrier at these boundaries instead.
inline LogicalResult trySetExplicitValueEncoding(Value value,
                                                        Attribute encoding) {
  auto result = dyn_cast<OpResult>(value);
  if (!result || result.getOwner()->getNumRegions())
    return failure();
  if (Attribute existing = getExplicitValueEncoding(value))
    if (existing != encoding)
      return failure();
  setExplicitResultEncoding(result, encoding);
  return success();
}

} // namespace mlir::triton::tile

#endif
