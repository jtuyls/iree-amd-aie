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
    regionOp->walk([&](xilinx::AIE::CoreOp op) {
      if (!coreMap.contains(op.getTileOp())) {
        // llvm::outs() << "NOT CONTAINS: " << op.getTileOp() << "\n";
        coreMap[op.getTileOp()] = op;
      } else {
        // llvm::outs() << "CONTAINS: " << op.getTileOp() << "\n";
        // auto &coreBlock = op.getBody().front();
        // auto prevCoreOp = coreMap[op.getTileOp()];
        // auto &prevCoreBlock = prevCoreOp.getBody().front();
        // auto insertIt = --prevCoreBlock.end();
        // auto begin = coreBlock.begin();
        // auto end = --coreBlock.end();
        // coreBlock.getOperations().splice(insertIt, coreBlock.getOperations(), begin, end);
        // rewriter.replaceAllUsesWith(op.getResult(), prevCoreOp.getResult());
        // // Move the preceding core to make sure all values are defined 
        // // rewriter.moveOpBefore(prevCoreOp, op);
        // // rewriter.eraseOp(op);
        // // 
        // // opsToBeErased.insert(op);
        // // return WalkResult::interrupt();

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
  // llvm::outs() << "Before removeUnusedCores: " << moduleOp << "\n";
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
