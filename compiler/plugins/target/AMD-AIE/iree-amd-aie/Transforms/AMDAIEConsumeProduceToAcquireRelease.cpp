// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-produce-consume-to-acquire-release"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to find the parent operation of the provided operation within the
/// same block as the provided one if it exists.
Operation *getParentOpInBlock(Block *block, Operation *op) {
  if (!op || op->getBlock() == block) return op;
  auto parentOp = op->getParentOp();
  return getParentOpInBlock(block, parentOp);
}

/// Walk all consume/produce operations within the core operations and insert
/// semaphore operations.
// template <typename OpTy>
LogicalResult readAccessToAcquireRelease(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  // Map from DMA source/target logical objectFifos to those respective DMA
  // operations.
  DenseMap<Value, AMDAIE::CircularDmaCpyNdOp> logicalObjectFifoToDma;
  parentOp->walk([&](AMDAIE::CircularDmaCpyNdOp dmaOp) {
    logicalObjectFifoToDma[dmaOp.getSource()] = dmaOp;
    logicalObjectFifoToDma[dmaOp.getTarget()] = dmaOp;
  });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, AMDAIE::LogicalObjectFifoAccessOp>
        logicalObjectFifoToLastAccess;
    WalkResult res =
        coreOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
          if (accessOp.getAccessType() != AMDAIE::MemoryAccess::Read)
            return WalkResult::advance();

          if (logicalObjectFifoToLastAccess.contains(accessOp.getInput())) {
            rewriter.setInsertionPoint(accessOp);
            rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
                rewriter.getUnknownLoc(),
                logicalObjectFifoToDma[accessOp.getInput()].getResult(),
                LogicalObjectFifoPort::Consume);
          }

          if (!logicalObjectFifoToDma.contains(accessOp.getInput())) {
            accessOp.emitOpError() << "not found as source of DMA operation";
            return WalkResult::interrupt();
          }
          rewriter.setInsertionPoint(accessOp);
          auto acquireOp = rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
              rewriter.getUnknownLoc(),
              llvm::cast<LogicalObjectFifoType>(accessOp.getInput().getType()),
              logicalObjectFifoToDma[accessOp.getInput()].getResult(),
              LogicalObjectFifoPort::Consume);
          auto newAccessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
              rewriter.getUnknownLoc(), acquireOp.getResult(),
              AMDAIE::MemoryAccess::Read);
          rewriter.replaceAllUsesWith(accessOp.getResult(),
                                      newAccessOp.getResult());

          logicalObjectFifoToLastAccess[accessOp.getInput()] = accessOp;
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return failure();

    // Insert release for remaining access operations at end of block.
    for (auto &&[value, accessOp] : logicalObjectFifoToLastAccess) {
      Block *parentBlock = accessOp->getBlock();
      if (!parentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(parentBlock);
      } else {
        rewriter.setInsertionPoint(parentBlock->getTerminator());
      }
      if (!logicalObjectFifoToDma.contains(accessOp.getInput())) {
        accessOp.emitOpError() << "not found as source of DMA operation";
        return failure();
      }
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
          rewriter.getUnknownLoc(), logicalObjectFifoToDma[accessOp.getInput()],
          LogicalObjectFifoPort::Consume);
    }
  }
  return success();
}

/// TODO
LogicalResult writeAccessToAcquireRelease(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  // Map from DMA source/target logical objectFifos to those respective DMA
  // operations.
  DenseMap<Value, AMDAIE::CircularDmaCpyNdOp> logicalObjectFifoToDma;
  parentOp->walk([&](AMDAIE::CircularDmaCpyNdOp dmaOp) {
    logicalObjectFifoToDma[dmaOp.getSource()] = dmaOp;
    logicalObjectFifoToDma[dmaOp.getTarget()] = dmaOp;
  });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, SmallVector<AMDAIE::LogicalObjectFifoAccessOp>>
        logicalObjectFifoToAccesses;
    DenseMap<Value, AMDAIE::LogicalObjectFifoAccessOp>
        logicalObjectFifoLastWriteAccesses;
    WalkResult res = coreOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp
                                          accessOp) {
      // llvm::outs() << "Access op: " << accessOp << "\n";
      // If there is another write op on the same logical objectFifo,
      // release it before the acquire.
      if (logicalObjectFifoLastWriteAccesses.contains(accessOp.getInput())) {
        llvm::outs() << "logicalObjectFifoLastWriteAccesses.contains\n";
        AMDAIE::LogicalObjectFifoAccessOp prevAccess =
            logicalObjectFifoLastWriteAccesses[accessOp.getInput()];
        if (!logicalObjectFifoToDma.contains(prevAccess.getInput())) {
          prevAccess.emitOpError() << "not found as source of DMA operation";
          return WalkResult::interrupt();
        }
        rewriter.setInsertionPoint(accessOp);
        rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
            rewriter.getUnknownLoc(),
            logicalObjectFifoToDma[prevAccess.getInput()],
            LogicalObjectFifoPort::Produce);
        // Remove from last access as settled.
        logicalObjectFifoLastWriteAccesses.erase(prevAccess.getInput());
      }
      // Insert acquire for write access at first `Any` access or at start
      // of block.
      if (accessOp.getAccessType() == AMDAIE::MemoryAccess::Write) {
        if (!logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
          rewriter.setInsertionPointToStart(accessOp->getBlock());
        } else {
          AMDAIE::LogicalObjectFifoAccessOp firstAccess =
              logicalObjectFifoToAccesses[accessOp.getInput()][0];
          rewriter.setInsertionPoint(firstAccess);
        }
        if (!logicalObjectFifoToDma.contains(accessOp.getInput())) {
          accessOp.emitOpError() << "not found as source of DMA operation";
          llvm::outs() << "not found as source of DMA operation\n";
          return WalkResult::interrupt();
        }
        auto acquireOp = rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
            rewriter.getUnknownLoc(),
            llvm::cast<LogicalObjectFifoType>(accessOp.getInput().getType()),
            logicalObjectFifoToDma[accessOp.getInput()].getResult(),
            LogicalObjectFifoPort::Produce);
        auto newAccessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), acquireOp.getResult(),
            AMDAIE::MemoryAccess::Write);

        // Update uses of this access operation and the preceding ones.
        rewriter.replaceAllUsesWith(accessOp.getResult(),
                                    newAccessOp.getResult());
        if (logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
          for (AMDAIE::LogicalObjectFifoAccessOp precedingAccessOp :
               logicalObjectFifoToAccesses[accessOp.getInput()]) {
            rewriter.replaceAllUsesWith(precedingAccessOp.getResult(),
                                        newAccessOp.getResult());
          }
        }

        // Insert into last access map
        logicalObjectFifoLastWriteAccesses[accessOp.getInput()] = accessOp;
      }
      // Insert any access operation into first access map.
      if (!logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
        logicalObjectFifoToAccesses[accessOp.getInput()] = {accessOp};
      } else {
        logicalObjectFifoToAccesses[accessOp.getInput()].push_back(accessOp);
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    // Insert release for remaining access operations at end of block.
    for (auto &&[value, writeAccessOp] : logicalObjectFifoLastWriteAccesses) {
      Block *parentBlock = writeAccessOp->getBlock();
      if (!parentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(parentBlock);
      } else {
        rewriter.setInsertionPoint(parentBlock->getTerminator());
      }
      if (!logicalObjectFifoToDma.contains(writeAccessOp.getInput())) {
        writeAccessOp.emitOpError() << "not found as source of DMA operation";
        return failure();
      }
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
          rewriter.getUnknownLoc(),
          logicalObjectFifoToDma[writeAccessOp.getInput()],
          LogicalObjectFifoPort::Produce);
    }
  }
  return success();
}

class AMDAIEConsumeProduceToAcquireReleasePass
    : public impl::AMDAIEConsumeProduceToAcquireReleaseBase<
          AMDAIEConsumeProduceToAcquireReleasePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEConsumeProduceToAcquireReleasePass() = default;
  AMDAIEConsumeProduceToAcquireReleasePass(
      const AMDAIEConsumeProduceToAcquireReleasePass &pass){};
  void runOnOperation() override;
};

void AMDAIEConsumeProduceToAcquireReleasePass::runOnOperation() {
  Operation *parentOp = getOperation();
  if (failed(readAccessToAcquireRelease(parentOp))) {
    return signalPassFailure();
  }
  if (failed(writeAccessToAcquireRelease(parentOp))) {
    return signalPassFailure();
  }
  // Clean up
  IRRewriter rewriter(parentOp->getContext());
  parentOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
    if (!isa<AMDAIE::LogicalObjectFifoAcquire>(accessOp.getInput().getDefiningOp())) {
      rewriter.eraseOp(accessOp);
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConsumeProduceToAcquireReleasePass() {
  return std::make_unique<AMDAIEConsumeProduceToAcquireReleasePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
