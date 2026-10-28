#include "mlir-ext/Conversion/CommonIRToTritonGPU/Patterns.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::tile {
namespace {

// Preserve the logical ops while TTIR -> TTGIR assigns their external layouts.
// Physical register selection is handled by backend lowering.
template <typename Op> struct TensorTileTypePattern : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fullTy = cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    // Preserve inferred extract layouts; insert results must match the base.
    if (isa<InsertTileOp>(op) || !resultTy.getEncoding())
      resultTy = resultTy.cloneWithEncoding(fullTy.getEncoding());
    rewriter.template replaceOpWithNewOp<Op>(
        op, TypeRange{resultTy}, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

} // namespace

void populateTensorTileTypePatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   ConversionTarget &target) {
  patterns.add<TensorTileTypePattern<ExtractTileOp>,
               TensorTileTypePattern<InsertTileOp>>(typeConverter,
                                                   patterns.getContext());
  target.addDynamicallyLegalOp<ExtractTileOp, InsertTileOp>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
}

} // namespace mlir::triton::tile
