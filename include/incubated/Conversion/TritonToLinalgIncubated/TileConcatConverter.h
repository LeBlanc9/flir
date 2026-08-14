#ifndef TRITON_INCUBATED_CONVERSION_TILE_CONCAT_CONVERTER_H_
#define TRITON_INCUBATED_CONVERSION_TILE_CONCAT_CONVERTER_H_

#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace TileConcatConverter {

using namespace mlir;

// Lowers tile.concat into standard tensor ops:
//   - 1D tensors: tensor.empty + a sequence of tensor.insert_slice
//   - 2D+ tensors: tensor.concat
class ConcatConverter
    : public OpConversionPattern<triton::tile::ConcatOp> {
public:
  using OpConversionPattern<triton::tile::ConcatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::tile::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace TileConcatConverter

namespace mlir::triton::tile {
void populateTileConcatOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                            mlir::RewritePatternSet &patterns);
} // namespace mlir::triton::tile

#endif // TRITON_INCUBATED_CONVERSION_TILE_CONCAT_CONVERTER_H_
