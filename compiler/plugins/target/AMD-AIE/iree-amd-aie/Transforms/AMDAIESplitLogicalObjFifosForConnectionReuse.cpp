// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIELogicalObjFifoSplittingUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos-for-connection-reuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESplitLogicalObjFifosForConnectionReusePass
    : public impl::AMDAIESplitLogicalObjFifosForConnectionReuseBase<
          AMDAIESplitLogicalObjFifosForConnectionReusePass> {
 public:
  using AMDAIESplitLogicalObjFifosForConnectionReuseBase::
      AMDAIESplitLogicalObjFifosForConnectionReuseBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosForConnectionReusePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // // Walk through CoreOps gathering 3rd input DmaOps (if applicable) which will
  // // be used to split L2 objectFifos of elementwise input for connection reuse.
  // SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  // WalkResult res = moduleOp->walk([&](AMDAIE::CoreOp coreOp) {
  //   SmallVector<Value> inputDmas = coreOp.getInputDmas();
  //   if (inputDmas.size() != 3) return WalkResult::skip();
  //   auto dmaCpyNdOp = inputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>();
  //   if (!dmaCpyNdOp) {
  //     coreOp->emitOpError() << "failed to get a DmaCpyNdOp from the input";
  //     return WalkResult::interrupt();
  //   }
  //   l2ToL1DmaOps.push_back(dmaCpyNdOp);
  //   return WalkResult::advance();
  // });
  // if (res.wasInterrupted()) return signalPassFailure();

  // if (failed(splitLogicalObjectFifoForElementwiseOp(rewriter, l2ToL1DmaOps,
  //                                                   context))) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "Failed to perform splitting of logicalobjectfifos");
  //   return signalPassFailure();

  SmallVector<std::tuple<AMDAIE::DmaCpyNdOp, size_t, size_t>> dmaOps;
  moduleOp->walk([&](AMDAIE::DmaCpyNdOp op) {
    std::optional<uint8_t> sourceMemSpace = op.getSourceMemorySpaceAsUInt();
    std::optional<uint8_t> targetMemSpace = op.getTargetMemorySpaceAsUInt();
    LogicalObjectFifoFromMemrefOp tgtObjectFifo = op.getTargetObjectFifo();
    ArrayRef<int64_t> memrefShape = tgtObjectFifo.getMemrefType().getShape();
    if (sourceMemSpace && sourceMemSpace.value() == 1 && targetMemSpace &&
        targetMemSpace.value() == 0) {
      dmaOps.push_back({op, 0, 0});
    } else if (sourceMemSpace && sourceMemSpace.value() == 0 &&
               targetMemSpace && targetMemSpace.value() == 1 &&
               memrefShape.size() >= 2 && memrefShape[0] == 4 &&
               memrefShape[1] == 1) {
      // A
      dmaOps.push_back({op, 0, 0});
    } 
    else if (sourceMemSpace && sourceMemSpace.value() == 0 &&
               targetMemSpace && targetMemSpace.value() == 1 &&
               memrefShape.size() >= 2 && memrefShape[0] == 1 &&
               memrefShape[1] == 4) {
      // B
      dmaOps.push_back({op, 1, 2});
    }
    return WalkResult::advance();
  });

  for (auto &&[dmaOp, sourceSplitDim, targetSplitDim] : dmaOps) {
    auto stridedOp =
        cast<AMDAIE::DoublyStridedOpInterface>(dmaOp.getOperation());
    if (failed(splitDoublyStridedOp(rewriter, stridedOp, sourceSplitDim,
                                    targetSplitDim))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of logicalobjectfifos");
      return signalPassFailure();
    }
  }

  SmallVector<std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp, size_t>> objFifoOps;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    ArrayRef<int64_t> memrefShape = op.getMemrefType().getShape();
    if (op.getMemorySpaceAsUInt() == 1 && memrefShape.size() > 2 &&
        memrefShape[0] == 4 && memrefShape[1] == 4) {
      llvm::outs() << "push objFifo: " << op << "\n";
      objFifoOps.push_back({op, 0});
    } else if (op.getMemorySpaceAsUInt() == 1 && memrefShape.size() > 2 &&
               memrefShape[0] == 4 && memrefShape[1] == 1) {
      llvm::outs() << "push objFifo A: " << op << "\n";
      objFifoOps.push_back({op, 0});
    } else if (op.getMemorySpaceAsUInt() == 1 && memrefShape.size() > 2 &&
               memrefShape[0] == 1 && memrefShape[1] == 4) {
      llvm::outs() << "push objFifo B: " << op << "\n";
      objFifoOps.push_back({op, 1});
    }
    return WalkResult::advance();
  });
  for (auto &&[op, splitDim] : objFifoOps) {
    if (failed(splitObjFifo(rewriter, op, splitDim))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosForConnectionReusePass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosForConnectionReusePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
