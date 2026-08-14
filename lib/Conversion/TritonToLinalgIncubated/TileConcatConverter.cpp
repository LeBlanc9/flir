#include "incubated/Conversion/TritonToLinalgIncubated/TileConcatConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

namespace TileConcatConverter {

LogicalResult
ConcatConverter::matchAndRewrite(triton::tile::ConcatOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  int64_t dim = op.getDim();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!resType || !lhsType || !rhsType) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
  }

  int64_t rank = resType.getRank();
  if (dim < 0 || dim >= rank) {
    return rewriter.notifyMatchFailure(op, "concat dim out of range");
  }

  // 2D+ tensors: lower directly to tensor.concat, which already supports
  // concatenation along an arbitrary static dimension for any rank.
  if (rank >= 2) {
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(
        op, resType, dim, ValueRange{lhs, rhs});
    return success();
  }

  // 1D tensors: build the result via tensor.empty + a sequence of
  // tensor.insert_slice, placing lhs then rhs back-to-back along `dim`.
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                  resType.getElementType());

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  SmallVector<OpFoldResult> lhsSizes =
      llvm::map_to_vector(lhsType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getIndexAttr(t));
      });
  SmallVector<OpFoldResult> rhsSizes =
      llvm::map_to_vector(rhsType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getIndexAttr(t));
      });

  auto insertLhs = rewriter.create<tensor::InsertSliceOp>(
      loc, lhs, emptyOp, offsets, lhsSizes, strides);

  offsets[dim] = rewriter.getIndexAttr(lhsType.getShape()[dim]);
  auto insertRhs = rewriter.create<tensor::InsertSliceOp>(
      loc, rhs, insertLhs, offsets, rhsSizes, strides);

  rewriter.replaceOp(op, insertRhs.getResult());
  return success();
}

} // namespace TileConcatConverter

namespace mlir::triton::tile {
void populateTileConcatOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<TileConcatConverter::ConcatConverter>(patterns.getContext());
}
} // namespace mlir::triton::tile
