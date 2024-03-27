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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fuse-aie-regions"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class FuseAieRegions: public OpRewritePattern<AMDAIE::AIERegionOp> {
  using OpRewritePattern<AMDAIE::AIERegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::AIERegionOp regionOp,
                                PatternRewriter &rewriter) const override {
    llvm::outs() << "FuseAieRegions\n";
    auto parentOp = regionOp->getParentOp();
    AMDAIE::AIERegionOp nextRegionOp;
    parentOp->walk([&](AMDAIE::AIERegionOp op) {
      if (op != regionOp && regionOp->isBeforeInBlock(op)) {
        nextRegionOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!nextRegionOp) {
      return failure();
    }
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });
    AMDAIE::ControlCodeRegionOp nextControlCodeOp;
    nextRegionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      nextControlCodeOp = op;
      return WalkResult::interrupt();
    });
    // llvm::outs() << "controlCodeOp: " << controlCodeOp << "\n";
    // llvm::outs() << "nextControlCodeOp: " << nextControlCodeOp << "\n";

    auto insertIt = controlCodeOp->getIterator();
    auto nextRegionBegin = nextRegionOp.getRegion().front().begin();
    auto nextRegionEnd = nextControlCodeOp->getIterator();
    regionOp.getRegion().front().getOperations().splice(
      insertIt, nextRegionOp.getRegion().front().getOperations(), nextRegionBegin, nextRegionEnd
    );

    // llvm::outs() << "After moving ops" << "\n";
    auto endIt = --controlCodeOp.getRegion().front().end();
    auto nextBegin = nextControlCodeOp.getRegion().front().begin();
    auto nextEnd = --nextControlCodeOp.getRegion().front().end();
    controlCodeOp.getRegion().front().getOperations().splice(
      endIt, nextControlCodeOp.getRegion
      ().front().getOperations(), nextBegin, nextEnd
    );
    // llvm::outs() << "After control code" << "\n";
    rewriter.eraseOp(nextControlCodeOp);
    rewriter.eraseOp(nextRegionOp);
    return success();
  }
};

LogicalResult mergeTiles(mlir::ModuleOp moduleOp) {
  llvm::outs() << "mergeTiles\n";
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp.walk([&](AMDAIE::AIERegionOp regionOp) {
    DenseMap<std::tuple<int,int>, xilinx::AIE::TileOp> tileMap;
    regionOp->walk([&](xilinx::AIE::TileOp op) {
      auto loc = std::make_tuple(op.colIndex(), op.rowIndex());
      if (!tileMap.contains(loc)) {
        tileMap[loc] = op;
      } else {
        rewriter.replaceAllUsesWith(op.getResult(), tileMap[loc].getResult());
        rewriter.eraseOp(op);
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
  return success();
}

LogicalResult mergeCores(mlir::ModuleOp moduleOp) {
  llvm::outs() << "Before mergeCores: " << moduleOp << "\n";
  llvm::outs() << "mergeCores\n";
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](AMDAIE::AIERegionOp regionOp) {
    // 1) Merge the core ops and update uses
    DenseMap<xilinx::AIE::TileOp, xilinx::AIE::CoreOp> coreMap;
    auto res = regionOp->walk([&](xilinx::AIE::CoreOp op) {
      if (!coreMap.contains(op.getTileOp())) {
        coreMap[op.getTileOp()] = op;
      } else {
        auto &coreBlock = op.getBody().front();
        auto &prevCoreOp = coreMap[op.getTileOp()];
        auto &prevCoreBlock = prevCoreOp.getBody().front();
        rewriter.eraseOp(prevCoreBlock.getTerminator());
        rewriter.mergeBlocks(&coreBlock, &prevCoreBlock);
        rewriter.replaceAllUsesWith(op.getResult(), prevCoreOp.getResult());
        rewriter.eraseOp(op);
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      regionOp.emitError("Could not merge cores in this AIE region");
      return WalkResult::interrupt();
    }

    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });

    // Move the preceding core to make sure all values are defined 
    for (auto &elem : coreMap)
      rewriter.moveOpBefore(elem.second, controlCodeOp);

    // 2) The LoadCoreOps in the control code block should be merged
    DenseMap<xilinx::AIE::CoreOp, AMDAIE::LoadCoreOp> coreLoadMap;
    auto loadCoreWalkResult = controlCodeOp->walk([&](AMDAIE::LoadCoreOp op) {
      auto coreOp = dyn_cast<xilinx::AIE::CoreOp>(op.getCore().getDefiningOp());
      if (!coreLoadMap.contains(coreOp)) {
        coreLoadMap[coreOp] = op;
      } else {
        auto firstCoreLoadOp = coreLoadMap[coreOp];
        // All merged load core ops should be in the same block for the core merge to be valid
        if (op->getBlock() != firstCoreLoadOp->getBlock()) {
          op.emitError("Trying to merge cores which are not loaded in the same context");
          return WalkResult::interrupt();
        }
        rewriter.eraseOp(op);
      }
      return WalkResult::advance();
    });
    if (loadCoreWalkResult.wasInterrupted())
      return WalkResult::interrupt();
      
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

SmallVector<AMDAIE::DmaCpyNdOp> getSrcDmaCopies(AMDAIE::LogicalObjectFifoFromMemref op) {
  // llvm::outs() << "getSrcDmaCopies" << "\n";
  SmallVector<AMDAIE::DmaCpyNdOp> res;
  for (auto userOp : op->getUsers()) {
    // llvm::outs() << "userOp: " << userOp << "\n";
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp); dmaOp && dmaOp.getSrcObjectFifo() == op) {
      // llvm::outs() << "--: " << dmaOp << "\n";
      res.push_back(dmaOp);
    }
  }
  return res;
}

SmallVector<AMDAIE::DmaCpyNdOp> getDstDmaCopies(AMDAIE::LogicalObjectFifoFromMemref op) {
  SmallVector<AMDAIE::DmaCpyNdOp> res;
  for (auto userOp : op->getUsers())
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp); dmaOp && dmaOp.getDstObjectFifo() == op)
      res.push_back(dmaOp);
  return res;
}

/// TODO
bool sameAddressing(AMDAIE::DmaCpyNdOp a, AMDAIE::DmaCpyNdOp b) {
  return llvm::equal(a.getDstOffsets(), b.getDstOffsets()) &&
   llvm::equal(a.getDstSizes(), b.getDstSizes()) &&
   llvm::equal(a.getDstStrides(),  b.getDstStrides()) &&
   llvm::equal(a.getSrcOffsets(),  b.getSrcOffsets()) &&
   llvm::equal(a.getSrcSizes(),  b.getSrcSizes()) &&
   llvm::equal(a.getSrcStrides(),  b.getSrcStrides());
}

/// TODO
bool sameExceptSrc(AMDAIE::DmaCpyNdOp a, AMDAIE::DmaCpyNdOp b) {
  return sameAddressing(a, b) && a.getDst() == b.getDst();
}

/// TODO
bool sameExceptDst(AMDAIE::DmaCpyNdOp a, AMDAIE::DmaCpyNdOp b) {
  return sameAddressing(a, b) && a.getSrc() == b.getSrc();
}

bool containSameOpsExceptDst(SmallVector<AMDAIE::DmaCpyNdOp> a, SmallVector<AMDAIE::DmaCpyNdOp> b) {
  for (auto &dmaA : a) {
    // llvm::outs() << "dmaA: " << dmaA << "\n";
    if (!llvm::any_of(b, [&](AMDAIE::DmaCpyNdOp &dmaB) {
          // llvm::outs() << "dmaB: " << dmaB << "\n";
          return sameExceptDst(dmaA, dmaB);
        })) {
      return false;
    }
  }
  return true;
}

DenseSet<xilinx::AIE::CoreOp> getCoreUsers(AMDAIE::DmaCpyNdOp &dmaOp) {
  DenseSet<xilinx::AIE::CoreOp> users;
  for (auto userOp : dmaOp->getUsers())
    if (auto coreOp = userOp->getParentOfType<xilinx::AIE::CoreOp>())
      users.insert(coreOp);
  return users;
}

/// TODO(jornt): the need for this function is quite bad :(. Once we distribute the L1 memrefs (no reuse), this
/// should not be needed anymore.
bool containSameOpsExceptSrcAndCoreUsage(SmallVector<AMDAIE::DmaCpyNdOp> a, SmallVector<AMDAIE::DmaCpyNdOp> b) {
  for (auto &dmaA : a) {
    // llvm::outs() << "dmaA: " << dmaA << "\n";
    DenseSet<xilinx::AIE::CoreOp> coreUsageA = getCoreUsers(dmaA);
    // If no core usage, this check is not valid as destinations are not on core side
    if (coreUsageA.empty())
      return false;
    if (!llvm::any_of(b, [&](AMDAIE::DmaCpyNdOp &dmaB) {
          // llvm::outs() << "dmaB: " << dmaB << "\n";
          DenseSet<xilinx::AIE::CoreOp> coreUsageB = getCoreUsers(dmaB);
          return sameExceptSrc(dmaB, dmaA) && coreUsageA == coreUsageB;
        })) {
      return false;
    }
  }
  return true;
}

LogicalResult mergeLogicalObjectFifoFromMemref(mlir::ModuleOp moduleOp) {
  llvm::outs() << "Before mergeLogicalObjectFifoFromMemref: " << moduleOp << "\n";
  IRRewriter rewriter(moduleOp.getContext());
  auto res = moduleOp.walk([&](AMDAIE::AIERegionOp regionOp) {
    // 1) 
    SmallVector<AMDAIE::LogicalObjectFifoFromMemref> uniqueLogicalObjectFifos;
    regionOp->walk([&](AMDAIE::LogicalObjectFifoFromMemref op) {
      // Just consider L2
      auto memSpace = op.getMemrefType().getMemorySpace();
      if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
        return WalkResult::advance();
      }
      llvm::outs() << "OP: " << op << "\n";
      for (auto src : getSrcDmaCopies(op))
        llvm::outs() << "--src dma: " << src << "\n";
      bool replaced = false;
      for (auto &other : uniqueLogicalObjectFifos) {
        llvm::outs() << "OTHER: " << other << "\n";
        for (auto src : getSrcDmaCopies(other))
          llvm::outs() << "--src dma: " << src << "\n";
        llvm::outs() << "--src bool: " << containSameOpsExceptDst(getDstDmaCopies(op), getDstDmaCopies(other)) << "\n";
        llvm::outs() << "--dst bool: " << containSameOpsExceptSrcAndCoreUsage(getSrcDmaCopies(op), getSrcDmaCopies(other)) << "\n";
        // TODO(jornt): refactor + simplify checks
        // TODO(jornt): checks to handle output side (L1 -> L2 -> L3) still need to be added
        if (containSameOpsExceptDst(getDstDmaCopies(op), getDstDmaCopies(other)) &&
            containSameOpsExceptSrcAndCoreUsage(getSrcDmaCopies(op), getSrcDmaCopies(other))) {
          llvm::outs() << "SAME: " << op << " AND " << other << "\n";
          rewriter.replaceAllUsesWith(op, other);
          rewriter.eraseOp(op);
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        uniqueLogicalObjectFifos.push_back(op);
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return failure();
  return success();
}

class MergeDmaMemCpyNds: public OpRewritePattern<AMDAIE::DmaCpyNdOp> {
  using OpRewritePattern<AMDAIE::DmaCpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DmaCpyNdOp dmaOp,
                                PatternRewriter &rewriter) const override {
    auto srcObjectFifo = dyn_cast<LogicalObjectFifoFromMemref>(dmaOp.getSrc().getDefiningOp());
    bool rewriteHappened = false;
    for (Operation *userOp : srcObjectFifo->getUsers()) {
      if (auto dmaUserOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp)) {
        if (dmaUserOp == dmaOp)
          continue;
        if (llvm::equal(dmaOp.getOperands(), dmaUserOp.getOperands())) {
          // llvm::outs() << "equal: " << dmaOp << " && " << dmaUserOp << "\n";
          rewriteHappened = true;
          rewriter.replaceAllUsesWith(dmaUserOp.getResult(), dmaOp.getResult());
          rewriter.eraseOp(dmaUserOp);
        }
      }
    }
    if (!rewriteHappened)
      return failure();
    return success();
  }
};

LogicalResult removeUnusedCores(mlir::ModuleOp moduleOp) {
  llvm::outs() << "removeUnusedCores\n";
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp.walk([&](xilinx::AIE::CoreOp coreOp) {
    if (coreOp->use_empty()) {
      llvm::outs() << "usused coreOp: " << coreOp << "\n";
      // TODO(jornt): next line can be dropped, no?
      coreOp->dropAllUses();
      rewriter.eraseOp(coreOp);
      llvm::outs() << "after erase op \n";
      return WalkResult::advance();
    }
      
    return WalkResult::advance();
  });
  llvm::outs() << "removeUnusedCores: " << moduleOp << "\n";
  return success();
}



class AMDAIEFuseAieRegionsPass
    : public impl::AMDAIEFuseAieRegionsBase<AMDAIEFuseAieRegionsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect>();
  }

  AMDAIEFuseAieRegionsPass() = default;
  AMDAIEFuseAieRegionsPass(const AMDAIEFuseAieRegionsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseAieRegionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FuseAieRegions>(context);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
  // if (failed(mergeTiles(getOperation()))) {
  //   return signalPassFailure();
  // }
  // if (failed(mergeCores(getOperation()))) {
  //   return signalPassFailure();
  // }
  // RewritePatternSet patterns2(context);
  // patterns2.insert<MergeDmaMemCpyNds>(context);
  // if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns2)))) {
  //   return signalPassFailure();
  // }
  // if (failed(removeUnusedCores(getOperation()))) {
  //   return signalPassFailure();
  // }
}


class AMDAIESimplifyAieRegionsPass
    : public impl::AMDAIESimplifyAieRegionsBase<AMDAIESimplifyAieRegionsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect>();
  }

  AMDAIESimplifyAieRegionsPass() = default;
  AMDAIESimplifyAieRegionsPass(const AMDAIESimplifyAieRegionsPass &pass){};
  void runOnOperation() override;
};

void AMDAIESimplifyAieRegionsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  if (failed(mergeTiles(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(mergeCores(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(mergeLogicalObjectFifoFromMemref(getOperation()))) {
    return signalPassFailure();
  }
  llvm::outs() << "Before MergeDmaMemCpyNds: " << getOperation() << "\n";
  RewritePatternSet patterns(context);
  patterns.insert<MergeDmaMemCpyNds>(context);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  // if (failed(mergeCores(getOperation()))) {
  //   return signalPassFailure();
  // }
  // if (failed(removeUnusedCores(getOperation()))) {
  //   return signalPassFailure();
  // }
}




}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseAieRegionsPass() {
  return std::make_unique<AMDAIEFuseAieRegionsPass>();
}

std::unique_ptr<Pass> createAMDAIESimplifyAieRegionsPass() {
  return std::make_unique<AMDAIESimplifyAieRegionsPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
