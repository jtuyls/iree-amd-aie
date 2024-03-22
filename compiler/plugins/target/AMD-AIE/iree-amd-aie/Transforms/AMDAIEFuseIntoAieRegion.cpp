// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fuse-fill-into-forall"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class DmaMemcpyNdIntoSubsequentAieRegion: public OpRewritePattern<AMDAIE::DmaCpyNdOp> {
  using OpRewritePattern<AMDAIE::DmaCpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DmaCpyNdOp dmaOp,
                                PatternRewriter &rewriter) const override {
    if (isa<AMDAIE::AIERegionOp>(dmaOp->getParentOp())) {
      return failure();
    }
    llvm::outs() << "DMA OP: " << dmaOp << "\n";
    auto dstLogicalObjectFifo = dyn_cast<LogicalObjectFifoFromMemref>(dmaOp.getDst().getDefiningOp());
    auto dstType = dmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
    llvm::outs() << dstType << "\n";
    // For now, we only fuse dma ops with destination into subsequent region
    if (!dstType.getMemorySpace() || dyn_cast<IntegerAttr>(dstType.getMemorySpace()).getInt() != 1) {
      return failure();
    }
    // llvm::outs() << dstMemref << "\n";

    AMDAIE::AIERegionOp firstRegionOp;
    Operation *firstDmaUserOp = nullptr;
    SmallVector<AMDAIE::DmaCpyNdOp> dmaUserOps;
    for (Operation *userOp : dstLogicalObjectFifo->getUsers()) {
      if (auto parentRegion = dyn_cast<AMDAIE::AIERegionOp>(userOp->getParentOp())) { // userOp->getParentOfType<AMDAIE::AIERegionOp>()) {
        if (parentRegion->getBlock() != dmaOp->getBlock() || parentRegion->isBeforeInBlock(dmaOp)) {
          // Skip aie regions which are not in the same context
          continue;
        } else if (!firstDmaUserOp || ((parentRegion != firstRegionOp) && parentRegion->isBeforeInBlock(firstRegionOp))) {
          firstDmaUserOp = userOp;
          firstRegionOp = parentRegion;
        } else if (parentRegion == firstRegionOp && userOp->isBeforeInBlock(firstDmaUserOp)) {
          firstDmaUserOp = userOp;
        }
        // Keep track of all dma user ops
        if (auto userDmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp)) {
          dmaUserOps.push_back(userDmaOp);
        }
      }
    }

    // Remove produce users of all dma users as dma ops are now linked through the objectfifo
    // TODO(jornt): Is this too implicit/unclear? Can we improve consume/produce outside cores?
    for (auto dmaUserOp : dmaUserOps) {
      for (Operation *userOp : dmaUserOp->getUsers()) {
        if (auto produceUserOp = dyn_cast<LogicalObjectFifoProduce>(userOp)) {
          // llvm::outs() << "delete produce: " << produceUserOp << "\n";
          rewriter.eraseOp(produceUserOp);
        }
      }
    }

    // auto firstDmaUserOp = dmaUserOps[0];
    llvm::outs() << "First DMA op: " << firstRegionOp << "\n";
    llvm::outs() << "First DMA op: " << firstDmaUserOp << "\n";
    auto regionOp = firstDmaUserOp->getParentOfType<AMDAIE::AIERegionOp>();
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });
    llvm::outs() << "Control code: " << controlCodeOp << "\n";
    // TODO: last load core op always the correct location to insert??
    AMDAIE::LoadCoreOp lastLoadCoreOp;
    controlCodeOp->walk([&](AMDAIE::LoadCoreOp op) {
      lastLoadCoreOp = op;
      return WalkResult::advance();
    });
    llvm::outs() << "lastLoadCoreOp: " << lastLoadCoreOp << "\n";

    // Insert into AIE objectfifo connections without addressing
    rewriter.setInsertionPoint(firstDmaUserOp);
    auto loc = rewriter.getUnknownLoc();
    SmallVector<OpFoldResult> empty;
    rewriter.create<AMDAIE::DmaCpyNdOp>(
      rewriter.getUnknownLoc(),
      rewriter.getIndexType(), // SmallVector<Type, 1>{}, // rewriter.getIndexType(),
      dmaOp.getDst(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstOffsets(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstSizes(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstStrides(),
      dmaOp.getSrc(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getSrcOffsets(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getSrcSizes(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty) // dmaOp.getSrcStrides()
    );

    // IPU instructions take care of addressing
    rewriter.setInsertionPointAfter(lastLoadCoreOp);
    rewriter.create<AMDAIE::IpuDmaCpyNdOp>(
      loc,
      SmallVector<Type, 1>{},
      dmaOp.getDst(),
      dmaOp.getDstOffsets(),
      dmaOp.getDstSizes(),
      dmaOp.getDstStrides(),
      dmaOp.getSrc(),
      dmaOp.getSrcOffsets(),
      dmaOp.getSrcSizes(),
      dmaOp.getSrcStrides()
    );

    rewriter.eraseOp(dmaOp);
    return success();
  }
};

class DmaMemcpyNdIntoPrecedingAieRegion: public OpRewritePattern<AMDAIE::DmaCpyNdOp> {
  using OpRewritePattern<AMDAIE::DmaCpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DmaCpyNdOp dmaOp,
                                PatternRewriter &rewriter) const override {
    if (isa<AMDAIE::AIERegionOp>(dmaOp->getParentOp())) {
      return failure();
    }
    llvm::outs() << dmaOp << "\n";
    llvm::outs() << dmaOp->getParentOp() << "\n";
    // auto srcMemref = dyn_cast<LogicalObjectFifoFromMemref>(dmaOp.getSrc().getDefiningOp()).getMemref();
    auto srcLogicalObjectFifo = dyn_cast<LogicalObjectFifoFromMemref>(dmaOp.getSrc().getDefiningOp());
    auto srcType = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
    llvm::outs() << srcType << "\n";
    // For now, we only fuse dma ops with source on L2
    if (!srcType.getMemorySpace() || dyn_cast<IntegerAttr>(srcType.getMemorySpace()).getInt() != 1) {
      return failure();
    }
    llvm::outs() << "SRC: " << srcLogicalObjectFifo << "\n";

    AMDAIE::AIERegionOp lastRegionOp;
    Operation *lastUserOp = nullptr;
    SmallVector<AMDAIE::DmaCpyNdOp> dmaUserOps;
    for (OpOperand &opOperand : srcLogicalObjectFifo->getUses()) {
      auto op = opOperand.getOwner();
      auto regionOp = dyn_cast<AMDAIE::AIERegionOp>(op->getParentOp());
      if (regionOp && regionOp->getBlock() == dmaOp->getBlock() && regionOp->isBeforeInBlock(dmaOp)) {
        if (auto userDmaOp = dyn_cast<DmaCpyNdOp>(op)) {
          if (userDmaOp.getDst() == srcLogicalObjectFifo.getOutput()) {
            if (!lastUserOp || (regionOp != lastRegionOp && lastRegionOp->isBeforeInBlock(regionOp))) {
              lastUserOp = userDmaOp;
              lastRegionOp = regionOp;
            } else if (lastUserOp->isBeforeInBlock(userDmaOp)) {
              lastUserOp = userDmaOp;
            }
            dmaUserOps.push_back(userDmaOp);
          }
        } else if (auto consumeOp = dyn_cast<LogicalObjectFifoConsume>(op)) {
          // TODO(jornt): this block can be removed, no?
          // llvm::outs() << "Consume op: " << consumeOp << "\n";
          // if (lastUserOp)
          //   llvm::outs() << "lastUserOp: " << lastUserOp << "\n";
          if (consumeOp.getObjectfifo() == srcLogicalObjectFifo.getOutput()) {
            if (!lastUserOp || (regionOp != lastRegionOp && lastRegionOp->isBeforeInBlock(regionOp))) {
              lastUserOp = consumeOp;
              lastRegionOp = regionOp;
            } else if (lastUserOp->isBeforeInBlock(consumeOp)) {
              lastUserOp = consumeOp;
            }
          }
        }
      }
    }

    // Remove consume users of all dma users as dma ops are now linked through the objectfifo
    // TODO(jornt): Is this too implicit/unclear? Can we improve consume/produce outside cores?
    for (auto dmaUserOp : dmaUserOps) {
      for (Operation *userOp : dmaUserOp->getUsers()) {
        if (auto consumeUserOp = dyn_cast<LogicalObjectFifoConsume>(userOp)) {
          // llvm::outs() << "delete consume: " << consumeUserOp << "\n";
          rewriter.eraseOp(consumeUserOp);
        }
      }
    }

    if (!lastUserOp) {
      LLVM_DEBUG(llvm::dbgs() << "DMA not used.\n");
      llvm::outs() << "No lastUserOp" << "\n";
      return failure();
    }
    llvm::outs() << "Last DMA op: " << lastUserOp << "\n";

    auto regionOp = lastUserOp->getParentOfType<AMDAIE::AIERegionOp>();
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });
    // TODO: last load core op always the correct location to insert??
    AMDAIE::EndOp endOp;
    controlCodeOp->walk([&](AMDAIE::EndOp op) {
      endOp = op;
      return WalkResult::interrupt();
    });
    llvm::outs() << "endOp: " << endOp << "\n";

    // Insert into AIE objectfifo connections without addressing
    rewriter.setInsertionPointAfter(lastUserOp);
    auto loc = rewriter.getUnknownLoc();
    SmallVector<OpFoldResult> empty;
    rewriter.create<AMDAIE::DmaCpyNdOp>(
      rewriter.getUnknownLoc(),
      rewriter.getIndexType(), // SmallVector<Type, 1>{}, // rewriter.getIndexType(),
      dmaOp.getDst(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstOffsets(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstSizes(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getDstStrides(),
      dmaOp.getSrc(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getSrcOffsets(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty), // dmaOp.getSrcSizes(),
      getValueOrCreateConstantIndexOp(rewriter, loc, empty) // dmaOp.getSrcStrides()
    );

    // IPU instructions take care of addressing
    rewriter.setInsertionPoint(endOp);
    auto ipuDmaCpy = rewriter.create<AMDAIE::IpuDmaCpyNdOp>(
      loc,
      SmallVector<Type, 1>{},
      dmaOp.getDst(),
      dmaOp.getDstOffsets(),
      dmaOp.getDstSizes(),
      dmaOp.getDstStrides(),
      dmaOp.getSrc(),
      dmaOp.getSrcOffsets(),
      dmaOp.getSrcSizes(),
      dmaOp.getSrcStrides()
    );
    rewriter.create<AMDAIE::LogicalObjectFifoWait>(
      rewriter.getUnknownLoc(),
      SmallVector<Type, 1>{},
      ipuDmaCpy.getDst()
    );

    rewriter.eraseOp(dmaOp);
    return success();
  }
};

class AMDAIEFuseDmaCpyIntoAieRegionPass
    : public impl::AMDAIEFuseDmaCpyIntoAieRegionBase<AMDAIEFuseDmaCpyIntoAieRegionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFuseDmaCpyIntoAieRegionPass() = default;
  AMDAIEFuseDmaCpyIntoAieRegionPass(const AMDAIEFuseDmaCpyIntoAieRegionPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseDmaCpyIntoAieRegionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<DmaMemcpyNdIntoSubsequentAieRegion>(context);
  patterns.insert<DmaMemcpyNdIntoPrecedingAieRegion>(context);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

class LogicalObjectfifoFromMemrefIntoAieRegion: public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemref> {
  using OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemref>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::LogicalObjectFifoFromMemref op,
                                PatternRewriter &rewriter) const override {
    if (isa<AMDAIE::AIERegionOp>(op->getParentOp())) {
      return failure();
    }

    AMDAIE::AIERegionOp regionOp;
    for (OpOperand &opOperand : op->getUses()) {
      auto parentOp = opOperand.getOwner()->getParentOfType<AMDAIE::AIERegionOp>();
      if (!parentOp) {
        // llvm::outs() << "Found use of LogicalObjectfifoFromMemref in non-AieRegion" << "\n";
        op.emitError("Found use of LogicalObjectfifoFromMemref in non-AieRegion");
        return failure();
      }
      if (!regionOp) {
        regionOp = parentOp;
      } else if (regionOp != parentOp) {
        // llvm::outs() << "Found use of LogicalObjectfifoFromMemref multiple different AieRegion ops" << "\n";
        op.emitError("Found use of LogicalObjectfifoFromMemref multiple different AieRegion ops");
        return failure();
      }
    }
    // llvm::outs() << "regionOp: " << regionOp << "\n";
    auto &block = regionOp.getRegion().front();
    rewriter.moveOpBefore(op, &block, block.begin());
    // llvm::outs() << "AFTER MOVE AFTER\n";
    // rewriter.eraseOp(dmaOp);
    return success();
  }
};

class AMDAIEFuseFromMemrefIntoAieRegionPass
    : public impl::AMDAIEFuseFromMemrefIntoAieRegionBase<AMDAIEFuseFromMemrefIntoAieRegionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFuseFromMemrefIntoAieRegionPass() = default;
  AMDAIEFuseFromMemrefIntoAieRegionPass(const AMDAIEFuseFromMemrefIntoAieRegionPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseFromMemrefIntoAieRegionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<LogicalObjectfifoFromMemrefIntoAieRegion>(context);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

class SCFForIntoAieRegion: public OpRewritePattern<AMDAIE::AIERegionOp> {
  using OpRewritePattern<AMDAIE::AIERegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::AIERegionOp regionOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<scf::ForOp>(regionOp->getParentOp())) {
      return failure();
    }
    auto forOp = dyn_cast<scf::ForOp>(regionOp->getParentOp());
    auto iv = forOp.getInductionVar();
    llvm::outs() << "forOp: " << forOp << "\n";

    // Find control code op
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });

    // rewriter.setInsertionPointToStart(&(controlCodeOp.getRegion().front()));
    rewriter.setInsertionPoint(controlCodeOp);
    auto newForOp = rewriter.create<scf::ForOp>(
      rewriter.getUnknownLoc(), forOp.getLowerBound(),
      forOp.getUpperBound(), forOp.getStep());
    // Block* splitBlock = rewriter.splitBlock(&(controlCodeOp.getRegion().front()), ++controlCodeOp.getRegion().front().begin());
    // llvm::outs() << "splitBlock: " << splitBlock << "\n";
    // rewriter.setInsertionPointToStart(newForOp.getBody());
    // auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    // auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
    // for (auto &op : controlCodeOp.getRegion().front().getOperations()) {
    //   llvm::outs() << "op: " << op << "\n";
    //   if (!(isa<scf::ForOp>(op) && dyn_cast<scf::ForOp>(op) == newForOp) && !isa<AMDAIE::EndOp>(op)) {
    //     llvm::outs() << "moveOpBefore: " << op << "\n";
    //     // rewriter.moveOpBefore(&op, newForOp.getBody(), newForOp.getBody()->end());
    //     rewriter.moveOpBefore(&op, yieldOp);
    //   }
    // }
    auto begin = controlCodeOp.getRegion().front().begin();
    auto end = --controlCodeOp.getRegion().front().end();
    newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->begin(), controlCodeOp.getRegion().front().getOperations(), begin, end);
    llvm::outs() << "newForOp: " << newForOp << "\n";
    auto newIvs = newForOp.getInductionVar();
    rewriter.replaceAllUsesWith(iv, newIvs);

    rewriter.moveOpBefore(newForOp, &(controlCodeOp.getRegion().front()), controlCodeOp.getRegion().front().begin());

    rewriter.moveOpBefore(regionOp, forOp);
    rewriter.eraseOp(forOp);

    // return failure();
    // llvm::outs() << "regionOp: " << regionOp << "\n";
    // auto &block = regionOp.getRegion().front();
    // rewriter.moveOpBefore(op, &block, block.begin());
    // llvm::outs() << "AFTER MOVE AFTER\n";
    // rewriter.eraseOp(dmaOp);
    return success();
  }
};

class AMDAIEFuseSCFForIntoAieRegionPass
    : public impl::AMDAIEFuseSCFForIntoAieRegionBase<AMDAIEFuseSCFForIntoAieRegionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFuseSCFForIntoAieRegionPass() = default;
  AMDAIEFuseSCFForIntoAieRegionPass(const AMDAIEFuseSCFForIntoAieRegionPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseSCFForIntoAieRegionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SCFForIntoAieRegion>(context);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseDmaCpyIntoAieRegionPass() {
  return std::make_unique<AMDAIEFuseDmaCpyIntoAieRegionPass>();
}

std::unique_ptr<Pass> createAMDAIEFuseFromMemrefIntoAieRegionPass() {
  return std::make_unique<AMDAIEFuseFromMemrefIntoAieRegionPass>();
}

std::unique_ptr<Pass> createAMDAIEFuseSCFForIntoAieRegionPass() {
  return std::make_unique<AMDAIEFuseSCFForIntoAieRegionPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
