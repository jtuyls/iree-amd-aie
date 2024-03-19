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

#define DEBUG_TYPE "iree-amdaie-fuse-into-aie-core"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class SCFForIntoAieCore: public OpRewritePattern<AMDAIE::LoadCoreOp> {
  using OpRewritePattern<AMDAIE::LoadCoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::LoadCoreOp loadCoreOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<scf::ForOp>(loadCoreOp->getParentOp())) {
      return failure();
    }
    auto forOp = dyn_cast<scf::ForOp>(loadCoreOp->getParentOp());
    // auto iv = forOp.getInductionVar();
    // llvm::outs() << "forOp: " << forOp << "\n";
    auto coreOp = dyn_cast<xilinx::AIE::CoreOp>(loadCoreOp.getCore().getDefiningOp());
    // llvm::outs() << "core: " << coreOp << "\n";

    rewriter.setInsertionPoint(coreOp);
    auto newForOp = rewriter.create<scf::ForOp>(
      rewriter.getUnknownLoc(), forOp.getLowerBound(),
      forOp.getUpperBound(), forOp.getStep());

    auto begin = coreOp.getRegion().front().begin();
    auto end = --coreOp.getRegion().front().end(); // Skip aie.end
    newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->begin(), coreOp.getRegion().front().getOperations(), begin, end);
    
    rewriter.moveOpBefore(newForOp, &(coreOp.getRegion().front()), coreOp.getRegion().front().begin());
    rewriter.moveOpBefore(loadCoreOp, forOp);
    return success();
  }
};

class AMDAIEFuseSCFForIntoAieCorePass
    : public impl::AMDAIEFuseSCFForIntoAieCoreBase<AMDAIEFuseSCFForIntoAieCorePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect>();
  }

  AMDAIEFuseSCFForIntoAieCorePass() = default;
  AMDAIEFuseSCFForIntoAieCorePass(const AMDAIEFuseSCFForIntoAieCorePass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseSCFForIntoAieCorePass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SCFForIntoAieCore>(context);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseSCFForIntoAieCorePass() {
  return std::make_unique<AMDAIEFuseSCFForIntoAieCorePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
