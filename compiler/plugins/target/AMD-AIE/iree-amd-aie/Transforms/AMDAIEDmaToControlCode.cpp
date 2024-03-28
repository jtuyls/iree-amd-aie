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

#define DEBUG_TYPE "iree-amdaie-dma-to-control-code"

using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult dmaToControlCode(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](AMDAIE::AIERegionOp regionOp) {
    // Find control code op within region
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });
    Block *controlCodeBlock = &controlCodeOp.getRegion().front();
    Block *newMainBlock = rewriter.createBlock(controlCodeBlock);
    Block *newEndBlock = rewriter.createBlock(controlCodeBlock);

    regionOp.walk([&](AMDAIE::DmaCpyNdOp dmaOp) {
      auto srcMemSpace = dmaOp.getSrcObjectFifo().getMemrefType().getMemorySpace();
      auto dstMemSpace = dmaOp.getDstObjectFifo().getMemrefType().getMemorySpace();

      if (!srcMemSpace || !dstMemSpace) {
        // L3 -> L2 or L2 -> L3
        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPointToEnd(newMainBlock);
        // auto ipuDmaCpy = 
        auto ipuDmaCpy = rewriter.create<AMDAIE::IpuDmaCpyNdOp>(
          loc,
          rewriter.getIndexType(), // SmallVector<Type, 1>{},
          dmaOp.getResult(),
          // dmaOp.getDst(),
          dmaOp.getDstOffsets(),
          dmaOp.getDstSizes(),
          dmaOp.getDstStrides(),
          // dmaOp.getSrc(),
          dmaOp.getSrcOffsets(),
          dmaOp.getSrcSizes(),
          dmaOp.getSrcStrides()
        );
        rewriter.setInsertionPointToEnd(newEndBlock);
        rewriter.create<AMDAIE::IpuDmaWaitOp>(
          rewriter.getUnknownLoc(),
          SmallVector<Type, 1>{},
          ipuDmaCpy.getResult(),
          // ipuDmaCpy.getDst()
          dmaOp.getDst()
        );
        // if (!dstMemSpace) {
        //   // L2 -> L3
        //   rewriter.setInsertionPointToEnd(newEndBlock);
        //   // rewriter.create<AMDAIE::LogicalObjectFifoWait>(
        //   //   rewriter.getUnknownLoc(),
        //   //   SmallVector<Type, 1>{},
        //   //   // ipuDmaCpy.getDst()
        //   //   dmaOp.getDst()
        //   // );
        // }

        rewriter.setInsertionPoint(dmaOp);
        SmallVector<OpFoldResult> empty;
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          rewriter.getUnknownLoc(),
          rewriter.getIndexType(),
          dmaOp.getDst(),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty),
          dmaOp.getSrc(),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty),
          getValueOrCreateConstantIndexOp(rewriter, loc, empty)
        );
        rewriter.replaceAllUsesWith(dmaOp.getResult(), newDmaOp.getResult());
        rewriter.eraseOp(dmaOp);
      }
      return WalkResult::advance();
    });

    rewriter.inlineBlockBefore(newMainBlock, controlCodeBlock->getTerminator());
    rewriter.inlineBlockBefore(newEndBlock, controlCodeBlock->getTerminator());
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

class AMDAIEDmaToControlCodePass
    : public impl::AMDAIEDmaToControlCodeBase<AMDAIEDmaToControlCodePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaToControlCodePass() = default;
  AMDAIEDmaToControlCodePass(const AMDAIEDmaToControlCodePass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaToControlCodePass::runOnOperation() {
  // MLIRContext *context = &getContext();
  // RewritePatternSet patterns(context);
  // patterns.insert<ConsumeToAcquireRelease>(context);

  // if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
  //   return signalPassFailure();
  // }
  if (failed(dmaToControlCode(getOperation()))) {
    return signalPassFailure();
  }
}


}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaToControlCodePass() {
  return std::make_unique<AMDAIEDmaToControlCodePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
