//===-VectorToVectorConversions.cpp - Conversions within Vector -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains conversions and rewrites to the Vector dialect to make
// it compatible with the available vector instructions in AIE architectures
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "AIEVecUtils.h"
#include "Passes.h"
#include "Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aievec-canonicalization"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace mlir::iree_compiler::aievec;

//============================================================================//
//================== Common AIE canonicalization analysis ====================//
//============================================================================//

static bool isGemmBTransposedContractionOp(vector::ContractionOp op) {
  if (op.getKind() != vector::CombiningKind::ADD) return false;

  // Get and check shape of operands
  auto lhsShape = op.getLhsType().getShape();
  auto rhsShape = op.getRhsType().getShape();
  auto accShape = cast<ShapedType>(op.getAccType()).getShape();
  if (lhsShape.size() < 2 || rhsShape.size() < 2 || accShape.size() < 2)
    return false;

  // Check that the innermost iterators match gemm-like iterators
  SmallVector<vector::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.size() < 3) return false;
  auto innerMostIterators =
      SmallVector<vector::IteratorType>(iterators.end() - 3, iterators.end());
  if (vector::IteratorType::parallel != innerMostIterators[0] ||
      vector::IteratorType::parallel != innerMostIterators[1] ||
      vector::IteratorType::reduction != innerMostIterators[2])
    return false;

  // Get indexing maps of iterators for operands
  SmallVector<AffineMap, 4> indexingMaps(op.getIndexingMapsArray());
  SmallVector<int64_t> outerMostResults;
  for (int64_t i = 0; i < indexingMaps[0].getNumResults() - 2; i++)
    outerMostResults.push_back(i);

  auto innerLhsMap = indexingMaps[0].dropResults(outerMostResults);
  auto innerRhsMap = indexingMaps[1].dropResults(outerMostResults);
  auto innerAccMap = indexingMaps[2].dropResults(outerMostResults);

  // Check whether they conform to a "transposed B" gemm
  auto ctx = op.getContext();
  auto mmAidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0, 2}, ctx)
          .dropResults(0);
  auto mmBidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 1, 2}, ctx)
          .dropResults(0);
  auto mmCidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{2, 0, 1}, ctx)
          .dropResults(0);
  int64_t numOuterMostDims = indexingMaps[0].getNumDims() - 3;
  return innerLhsMap == mmAidxMap.shiftDims(numOuterMostDims) &&
         innerRhsMap == mmBidxMap.shiftDims(numOuterMostDims) &&
         innerAccMap == mmCidxMap.shiftDims(numOuterMostDims);
}

//============================================================================//
//============ Common AIE canonicalization conversion patterns ===============//
//============================================================================//

// This pattern converts a `vector.transfer_read` with an unaligned access
// into an aligned `vector.transfer_read` twice as long, followed by a
// `vector.extract_strided_slice` selecting the subvector matching the
// original `vector.transfer_read`.
struct SplitUnalignedTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  SplitUnalignedTransferReadPattern(MLIRContext *context, int64_t maxVectorSize,
                                    int64_t alignment)
      : OpConversionPattern<vector::TransferReadOp>(context),
        maxVectorSize(maxVectorSize),
        vectorAlignment(alignment) {}

  LogicalResult matchAndRewrite(
      vector::TransferReadOp readOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Check that it's not a splat transfer read.
    if (adaptor.getPermutationMap().isConstant()) return failure();

    // Check if the transfer is unaligned.
    auto vType = readOp.getVectorType();
    int64_t offset =
        getTransferReadAlignmentOffset(adaptor, vType, vectorAlignment)
            .value_or(0);
    if (offset == 0) return failure();

    // Verify that we can load a vector 2x as long as the original
    auto vLen = vType.getShape().back();
    auto longVecTy = VectorType::get(2 * vLen, vType.getElementType());
    auto longVecSize = getElementSizeInBits(vType) * 2 * vLen;
    if (longVecSize > maxVectorSize) return failure();

    // Calculate the aligned indices for the lower and higher parts.
    // TODO: Add support for cases where the offset is greater than the
    // TODO: vector length.
    auto loc = readOp.getLoc();
    Value oldInnerMostIdx = adaptor.getIndices().back();
    auto offsetCorrectionMap =
        AffineMap::get(1, 0, getAffineDimExpr(0, readOp.getContext()) - offset);
    Value newInnerMostIdx = rewriter
                                .create<affine::AffineApplyOp>(
                                    readOp.getLoc(), offsetCorrectionMap,
                                    SmallVector<Value, 1>({oldInnerMostIdx}))
                                .getResult();
    SmallVector<Value, 8> alignedIdx;
    alignedIdx.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    alignedIdx[alignedIdx.size() - 1] = newInnerMostIdx;

    // Create the aligned transfer read for a vector 2x as long that covers the
    // elements of the unaligned vector.
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        loc, longVecTy, adaptor.getSource(), alignedIdx, adaptor.getPadding());

    // Create a `vector.extract_strided_slice` to extract the unaligned vector.
    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newReadOp.getResult(), offset, vLen, 1);

    return success();
  }

  int64_t maxVectorSize;
  int64_t vectorAlignment;
};

// This pattern converts a `vector.transfer_read` with a splat permutation map
// into a contiguous `vector.transfer_read` followed by a `vector.extract` to
// obtain the splat value and a `vector.broadcast` to broadcast it into a
// vector of the right size.
struct ConvertSplatTransferReadToBroadcastPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  ConvertSplatTransferReadToBroadcastPattern(MLIRContext *context)
      : OpConversionPattern<vector::TransferReadOp>(context) {}

  LogicalResult matchAndRewrite(
      vector::TransferReadOp readOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    AffineMap map = readOp.getPermutationMap();
    if (!map.isConstant()) return failure();

    Value srcMemRef = adaptor.getSource();
    SmallVector<Value, 8> indices;
    Value newIdx;
    int64_t offset = 0;
    // If it's a zero-rank memory access
    if (cast<MemRefType>(srcMemRef.getType()).getRank() == 0) {
      srcMemRef = rewriter
                      .create<memref::ExpandShapeOp>(
                          readOp.getLoc(), SmallVector<int64_t, 1>({1}),
                          srcMemRef, SmallVector<ReassociationIndices, 1>({}))
                      .getResult();
      newIdx = rewriter.create<arith::ConstantOp>(readOp.getLoc(),
                                                  rewriter.getIndexAttr(0L));
      indices.push_back(newIdx);
    } else {
      indices.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
      newIdx = indices[indices.size() - 1];
      // If the innermost index comes from an `affine.apply` op, take the base
      // as the new innermost index for the new `vector.transfer_read`, and the
      // offset as the index for the `aievec.broadcast` op.
      if (auto applyOp = newIdx.getDefiningOp<affine::AffineApplyOp>())
        if (applyOp.getAffineMap().getNumDims() == 1) {
          newIdx = applyOp.getMapOperands()[0];
          offset = applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0];
        }
    }
    // XXX: We assume we are reading 1D vectors
    int64_t vlen = readOp.getVector().getType().getShape()[0];
    if (offset >= vlen) {
      // If the splat element is beyond the first vector, we calculate the
      // address of the vector containing the element.
      int64_t numElemsToSkip = vlen * (offset / vlen);
      offset = offset % vlen;
      auto newAddrMap = AffineMap::get(
          1, 0, getAffineDimExpr(0, readOp.getContext()) + numElemsToSkip);
      newIdx =
          rewriter
              .create<affine::AffineApplyOp>(readOp.getLoc(), newAddrMap,
                                             SmallVector<Value, 1>({newIdx}))
              .getResult();
    }
    indices[indices.size() - 1] = newIdx;
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVector().getType(), srcMemRef, indices,
        adaptor.getPadding());
    auto extractOp = rewriter.create<vector::ExtractOp>(
        readOp.getLoc(), newReadOp.getResult(), ArrayRef<int64_t>{offset});
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        readOp, newReadOp.getVector().getType(), extractOp.getResult());
    return success();
  }
};

static SmallVector<Value> collapseInnerMostDimIndices(PatternRewriter &b,
                                                      Location loc, int numDims,
                                                      ValueRange indices,
                                                      ArrayRef<int64_t> shape,
                                                      AffineMap layout) {
  // TODO: Don't assume trivial layout
  assert(layout.isMinorIdentity() &&
         "dimension collapse in non-identity layout is not implemented");
  auto newIdxExpr = b.getAffineDimExpr(numDims - 1);
  int64_t stride = 1;
  for (int64_t dim = numDims - 2; dim >= 0; dim--) {
    stride *= shape[shape.size() - (numDims - dim - 1)];
    newIdxExpr = newIdxExpr + b.getAffineDimExpr(dim) * stride;
  }
  auto newIndexMap = AffineMap::get(numDims, 0, newIdxExpr);
  Value newInnerMostIdxValue =
      b.create<affine::AffineApplyOp>(loc, newIndexMap,
                                      indices.take_back(numDims))
          .getResult();
  SmallVector<Value> newIdxRange;
  for (auto idx : indices.drop_back(numDims)) newIdxRange.push_back(idx);
  newIdxRange.push_back(newInnerMostIdxValue);
  return newIdxRange;
}

static Value collapseInnerMostShapeDims(PatternRewriter &b, Location loc,
                                        int numDims, Value val) {
  auto memRefTy = cast<MemRefType>(val.getType());
  auto shape = memRefTy.getShape();
  int64_t newInnerMostDim = std::accumulate(shape.end() - numDims, shape.end(),
                                            1, std::multiplies<>());
  SmallVector<int64_t, 4> newShape{shape.begin(), shape.end() - numDims + 1};
  newShape[shape.size() - numDims] = newInnerMostDim;
  auto newNumDims = newShape.size();
  auto ctx = b.getContext();
  auto newMemRefTy = MemRefType::get(
      newShape, memRefTy.getElementType(),
      AffineMap::getMinorIdentityMap(newNumDims, newNumDims, ctx),
      memRefTy.getMemorySpace());
  auto reassocIndices =
      getReassociationIndicesForCollapse(shape, newShape).value();
  return b
      .create<memref::CollapseShapeOp>(loc, newMemRefTy, val, reassocIndices)
      .getResult();
}

// This pattern flatten multidimensional `vector.transfer_read` operations
// replacing them with a `memref.collapse_shape`, a 1D `vector.transfer_read`,
// and a `vector.shape_cast`.
struct FlattenMultDimTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp readOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!adaptor.getPermutationMap().isMinorIdentity() || adaptor.getMask())
      return failure();
    VectorType vectorTy = readOp.getVector().getType();
    if (vectorTy.getRank() < 2) return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(adaptor.getSource().getType());
    if (!memRefTy) return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, readOp.getLoc(), vecShape.size(),
                                    adaptor.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, readOp.getLoc(), vecShape.size(), adaptor.getSource());
    auto newVector = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), newVectorTy, newSource, newIndices);

    auto inBoundsArrayAttrOpt = adaptor.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newVector.getProperties().setInBounds(
          rewriter.getBoolArrayAttr(newInBounds));
    }

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, vectorTy,
                                                     newVector);

    return success();
  }
};

// This pattern flatten multidimensional `vector.transfer_write` operations
// replacing them with a `memref.collapse_shape`, a `vector.shape_cast`, and a
// 1D `vector.transfer_write`,
struct FlattenMultDimTransferWritePattern
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp writeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!adaptor.getPermutationMap().isMinorIdentity() || adaptor.getMask())
      return failure();
    VectorType vectorTy = cast<VectorType>(adaptor.getVector().getType());
    if (vectorTy.getRank() < 2) return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(adaptor.getSource().getType());
    if (!memRefTy) return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newVector = rewriter
                         .create<vector::ShapeCastOp>(
                             writeOp.getLoc(), newVectorTy, adaptor.getVector())
                         .getResult();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, writeOp.getLoc(), vecShape.size(),
                                    adaptor.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, writeOp.getLoc(), vecShape.size(), adaptor.getSource());

    auto newOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        writeOp, newVector, newSource, newIndices);

    auto inBoundsArrayAttrOpt = adaptor.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newOp.getProperties().setInBounds(rewriter.getBoolArrayAttr(newInBounds));
    }

    return success();
  }
};

// This pattern extracts an implicit transposition of the 2 innermost
// dimensions of `rhs` in a gemm-like contraction op, making it an explicit
// `vector.transpose` op.
// If `rhs` is coming from a widening op (`extf`/`extsi`/`extui`), the
// transposition will be hoisted above the widening op.
struct ExtractTransposeFromContractionOp
    : public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern<vector::ContractionOp>::OpConversionPattern;

  static VectorType getTransposedVectorType(VectorType vecTy) {
    SmallVector<int64_t> shape{vecTy.getShape()};
    auto nDim = shape.size();
    int64_t dimNm1 = shape[nDim - 1];
    shape[nDim - 1] = shape[nDim - 2];
    shape[nDim - 2] = dimNm1;
    auto elemTy = vecTy.getElementType();
    return VectorType::get(shape, elemTy);
  }

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isGemmBTransposedContractionOp(contractOp)) return failure();

    Location loc = contractOp.getLoc();
    auto ctx = rewriter.getContext();

    Value rhsVal = adaptor.getRhs();
    VectorType rhsVecTy = contractOp.getRhsType();
    Type rhsElemTy = rhsVecTy.getElementType();

    bool doExtF = false, doExtSI = false, doExtUI = false;
    if (auto extfRhsOp = rhsVal.getDefiningOp<arith::ExtFOp>()) {
      rhsVal = extfRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtF = true;
    } else if (auto extsiRhsOp = rhsVal.getDefiningOp<arith::ExtSIOp>()) {
      rhsVal = extsiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtSI = true;
    } else if (auto extuiRhsOp = rhsVal.getDefiningOp<arith::ExtUIOp>()) {
      rhsVal = extuiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtUI = true;
    }

    int64_t nDim = rhsVecTy.getShape().size();
    SmallVector<int64_t> rhsPermutation;
    for (int64_t i = 0; i < nDim - 2; i++) rhsPermutation.push_back(i);
    rhsPermutation.push_back(nDim - 1);
    rhsPermutation.push_back(nDim - 2);
    auto transpRhsVecTy = getTransposedVectorType(rhsVecTy);
    rhsVal = rewriter
                 .create<vector::TransposeOp>(loc, transpRhsVecTy, rhsVal,
                                              rhsPermutation)
                 .getResult();

    if (doExtF)
      rhsVal =
          rewriter
              .create<arith::ExtFOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtSI)
      rhsVal =
          rewriter
              .create<arith::ExtSIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtUI)
      rhsVal =
          rewriter
              .create<arith::ExtUIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();

    SmallVector<AffineMap, 4> oldIdxMaps(contractOp.getIndexingMapsArray());

    nDim = oldIdxMaps[1].getNumDims();
    SmallVector<int64_t> innerDimPerm;
    for (int64_t i = 0; i < nDim - 2; i++) innerDimPerm.push_back(i);
    innerDimPerm.push_back(nDim - 1);
    innerDimPerm.push_back(nDim - 2);
    auto transpPermMap = AffineMap::getPermutationMap(innerDimPerm, ctx);

    auto newIdxMaps = rewriter.getAffineMapArrayAttr(
        {oldIdxMaps[0], oldIdxMaps[1].compose(transpPermMap), oldIdxMaps[2]});

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, contractOp.getResult().getType(), adaptor.getLhs(), rhsVal,
        adaptor.getAcc(), newIdxMaps, contractOp.getIteratorTypes());

    return success();
  }
};

//============================================================================//
//============ AIE2 canonicalization conversion patterns ===============//
//============================================================================//

//============================================================================//
//================ Common AIE canonicalization configuration =================//
//============================================================================//
static void configureCommonAIECanonicalizeLegalizations(
    ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
}

static void populateCommonAIECanonicalizeConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

//============================================================================//
//============== AIE2-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIE2CanonicalizeLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType(), 256)
                       .value_or(0) == 0 &&
               op.getVector().getType().getRank() < 2;
      });
  target.addDynamicallyLegalOp<vector::TransferWriteOp>(
      [](vector::TransferWriteOp op) {
        return cast<VectorType>(op.getVector().getType()).getRank() < 2;
      });
  target.addDynamicallyLegalOp<vector::ContractionOp>(
      [](vector::ContractionOp op) {
        return !isGemmBTransposedContractionOp(op);
      });
}

static void populateAIE2CanonicalizeConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext(), 1024,
                                                  256);
  patterns
      .add<ExtractTransposeFromContractionOp, FlattenMultDimTransferReadPattern,
           FlattenMultDimTransferWritePattern>(patterns.getContext());
}

//============================================================================//
//=================== Common AIE Canonicalization Passes =====================//
//============================================================================//

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// two steps:
//    1) Replace splat transfer reads with contiguous transfer reads followed
//       by `extract` + `broadcast` operations.
//    2) Split unaligned transfer reads into a wider aligned transfer read
//       followed by a `vector.extract_strided_slice` operation.
struct CanonicalizeVectorForAIEVecPass
    : public PassWrapper<CanonicalizeVectorForAIEVecPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeVectorForAIEVecPass)
  StringRef getArgument() const final {
    return "canonicalize-vector-for-aievec";
  }

  StringRef getDescription() const final {
    return "Canonicalize vector operations for AIEVec conversion";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateCommonAIECanonicalizeConversionPatterns(patterns);
    configureCommonAIECanonicalizeLegalizations(target);
    populateAIE2CanonicalizeConversionPatterns(patterns);
    configureAIE2CanonicalizeLegalizations(target);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static std::unique_ptr<::mlir::Pass> createCanonicalizeVectorForAIEVecPass() {
  return std::make_unique<CanonicalizeVectorForAIEVecPass>();
}

//============================================================================//
//=============== Main Vector2Vector Pipeline Configuration ==================//
//============================================================================//

void mlir::iree_compiler::aievec::buildCanonicalizeVectorForAIEVec(
    OpPassManager &pm) {
  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(createCanonicalizeVectorForAIEVecPass());
}
