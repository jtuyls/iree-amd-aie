// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-dma-utils"

namespace mlir::iree_compiler::AMDAIE {

namespace {


/// Recognize linear accesses across multiple dimensions and fold them.
class FoldDimsInDmaAddressing : public OpRewritePattern<AMDAIE::DmaCpyNdOp> {
  using OpRewritePattern<AMDAIE::DmaCpyNdOp>::OpRewritePattern;

  LogicalResult foldLinearAddressing(AMDAIE::DmaCpyNdOp op,
                                     SmallVector<OpFoldResult> &offsets,
                                     SmallVector<OpFoldResult> &sizes,
                                     SmallVector<OpFoldResult> &strides,
                                     SmallVector<OpFoldResult> &newOffsets,
                                     SmallVector<OpFoldResult> &newSizes,
                                     SmallVector<OpFoldResult> &newStrides) const {
    llvm::outs() << "foldLinearAddressing\n";  
    MLIRContext *ctx = op.getContext();
    if (offsets.size() == 0) {
      return failure();
    }
    // Make a single dimension linear access implicit. This requires offset 0 and stride 1.
    // For example: offsets: [%c0], sizes: [%c1024], strides: [%c1]
    // becomes offsets: [], sizes: [], strides: []
    if (offsets.size() == 1 && 
        getConstantIntValue(offsets[0]) && getConstantIntValue(offsets[0]).value() == 0 &&
        getConstantIntValue(strides[0]) && getConstantIntValue(strides[0]).value() == 1) {
      return success();
    }

    // Iterate backwards through the offsets, sizes and strides and try to fold dimensions.
    size_t size = offsets.size();
    OpFoldResult curOffset = offsets[size - 1];
    OpFoldResult curSize = sizes[size - 1];
    OpFoldResult curStride = strides[size - 1];
    int64_t runningSize = 1;
    auto pushIntoNew = [&]() -> void {
      newOffsets.push_back(curOffset);
      newStrides.push_back(curStride);
      if (runningSize != 1) {
        int64_t curSizeInt = getConstantIntValue(curSize).value();
        runningSize *= curSizeInt;
        newSizes.push_back(getAsIndexOpFoldResult(ctx, runningSize));
        runningSize = 1;
      } else {
        newSizes.push_back(curSize);
      }
    };
    for (int i = size - 2; i >= 0; i--) {
      OpFoldResult nextOffset = offsets[i];
      OpFoldResult nextSize = sizes[i];
      OpFoldResult nextStride = strides[i];
      llvm::outs() << "FOR: " << i << "\n";
      llvm::outs() << "--nextOffset: " << nextOffset << "\n";
      llvm::outs() << "--nextSize: " << nextSize << "\n";
      llvm::outs() << "--nextStride: " << nextStride << "\n";
      if (!getConstantIntValue(curOffset) || !getConstantIntValue(curSize) || !getConstantIntValue(curStride) ||
          !getConstantIntValue(nextOffset) || !getConstantIntValue(nextSize) || !getConstantIntValue(nextStride)) {
        // Only handle static values for now
        pushIntoNew();
      } else {
        int64_t curOffsetInt = getConstantIntValue(curOffset).value();
        int64_t curSizeInt = getConstantIntValue(curSize).value();
        int64_t curStrideInt = getConstantIntValue(curStride).value();
        int64_t nextOffsetInt = getConstantIntValue(nextOffset).value();
        int64_t nextStrideInt = getConstantIntValue(nextStride).value();
        if (curOffsetInt == 0 && nextOffsetInt == 0 &&
            nextStrideInt == runningSize * curSizeInt * curStrideInt) {
          runningSize *= curSizeInt;
          curSize = nextSize;
          continue;
        } else {
          pushIntoNew();
        }
      }
      curOffset = nextOffset;
      curSize = nextSize;
      curStride = nextStride;
    }
    // Push last remaining dimension
    pushIntoNew();
    // Because we iterated backwards and pushed into the back earlier,
    // we now need to reverse to get to the format standard.
    std::reverse(newOffsets.begin(), newOffsets.end());
    std::reverse(newSizes.begin(), newSizes.end());
    std::reverse(newStrides.begin(), newStrides.end());
    if (llvm::equal(offsets, newOffsets) || llvm::equal(sizes, newSizes) || llvm::equal(strides, newStrides)) {
      // No change
      return failure();
    }
    return success();
  }

  LogicalResult matchAndRewrite(AMDAIE::DmaCpyNdOp op,
                                PatternRewriter &rewriter) const override {
    llvm::outs() << "FoldDimsInDmaAddressing: " << op << "\n";  
    SmallVector<OpFoldResult> srcOffsets = op.getSrcOffsets();
    SmallVector<OpFoldResult> srcSizes = op.getSrcSizes();
    SmallVector<OpFoldResult> srcStrides = op.getSrcStrides();
    SmallVector<OpFoldResult> dstOffsets = op.getDstOffsets();
    SmallVector<OpFoldResult> dstSizes = op.getDstSizes();
    SmallVector<OpFoldResult> dstStrides = op.getDstStrides();
    SmallVector<OpFoldResult> newSrcOffsets;
    SmallVector<OpFoldResult> newSrcSizes;
    SmallVector<OpFoldResult> newSrcStrides;
    SmallVector<OpFoldResult> newDstOffsets;
    SmallVector<OpFoldResult> newDstSizes;
    SmallVector<OpFoldResult> newDstStrides;
    auto srcRes = foldLinearAddressing(op, srcOffsets, srcSizes, srcStrides, newSrcOffsets, newSrcSizes, newSrcStrides);
    auto dstRes = foldLinearAddressing(op, dstOffsets, dstSizes, dstStrides, newDstOffsets, newDstSizes, newDstStrides);
    if (failed(srcRes) && failed(dstRes)) {
      return failure();
    }

    Location loc = op->getLoc();
    auto newDmaOp = rewriter.replaceOpWithNewOp<AMDAIE::DmaCpyNdOp>(
      op,
      rewriter.getIndexType(),
      op.getDst(),
      getValueOrCreateConstantIndexOp(rewriter, loc, newDstOffsets),
      getValueOrCreateConstantIndexOp(rewriter, loc, newDstSizes),
      getValueOrCreateConstantIndexOp(rewriter, loc, newDstStrides),
      op.getSrc(),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSrcOffsets),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSrcSizes),
      getValueOrCreateConstantIndexOp(rewriter, loc, newSrcStrides)
    );
    llvm::outs() << "new dma op: " << newDmaOp << "\n";
    return success();
  }
};


/// TODO(jornt): refactor into separate pass + can we use an interface? + does this logic already exist somewhere?
/// For now, simplify by:
///   - Recognizing linear accesses across multiple dimensions and folding them
class AMDAIESimplifyDmaAddressingPass
    : public impl::AMDAIESimplifyDmaAddressingBase<AMDAIESimplifyDmaAddressingPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIESimplifyDmaAddressingPass() = default;
  AMDAIESimplifyDmaAddressingPass(const AMDAIESimplifyDmaAddressingPass &pass){};
  void runOnOperation() override;
};

void AMDAIESimplifyDmaAddressingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldDimsInDmaAddressing>(context);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}


}  // namespace

std::unique_ptr<Pass> createAMDAIESimplifyDmaAddressingPass() {
  return std::make_unique<AMDAIESimplifyDmaAddressingPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
