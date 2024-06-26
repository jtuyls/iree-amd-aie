// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-distribute-cores-and-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

// Used to annotate loops that should be unrolled.
static const llvm::StringLiteral kAMDAIELoopUnroll = "amdaie.unroll";

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Comparator for a pair of `amdaie.dma_cpy_nd` on the first tile operation's
/// column index.
bool dmaColComparator(
    std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>> &a,
    std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>> &b) {
  return TileOp::tileColumnComparator(a.second[0], b.second[0]);
};

/// Utility to use tuple coordinates as key of a `DenseMap`.
struct LocationMapInfo {
  static SmallVector<std::pair<int64_t, int64_t>> getEmptyKey() {
    return {std::make_pair(int64_t(-1), int64_t(-1))};
  }

  static SmallVector<std::pair<int64_t, int64_t>> getTombstoneKey() {
    return {std::make_pair(int64_t(-2), int64_t(-2))};
  }

  static unsigned getHashValue(
      const SmallVector<std::pair<int64_t, int64_t>> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
  }

  static bool isEqual(const SmallVector<std::pair<int64_t, int64_t>> &lhs,
                      const SmallVector<std::pair<int64_t, int64_t>> &rhs) {
    return lhs == rhs;
  }
};

//===----------------------------------------------------------------------===//
// AMDAIEDistributeCoresAndObjectFifosPass
//===----------------------------------------------------------------------===//

/// Distribute local memory accesses through subviews by allocating a single
/// smaller memory. This is needed because cores can't operate on one larger L1
/// memory.
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  SmallVector<Operation *> toBeErased;
  // Map from alloc operations to a new alloc operations to be used.
  DenseMap<memref::AllocOp, memref::AllocOp> memrefToNew;

  moduleOp->walk([&](memref::AllocOp allocOp) {
    // Only consider local memory (L1).
    Attribute memSpace =
        cast<MemRefType>(allocOp.getResult().getType()).getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs()
               << "DistributeLocalMemory for: " << allocOp << "\n");

    SmallVector<AMDAIE::DmaCpyNdOp> dmaUsers;
    for (Operation *userOp : allocOp->getUsers()) {
      if (auto logicalObjectFifo =
              dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
        for (Operation *objFifoUserOp : logicalObjectFifo->getUsers()) {
          if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
              dmaOp.getSourceObjectFifo() == logicalObjectFifo) {
            dmaUsers.push_back(dmaOp);
          }
        }
      }
    }
    if (dmaUsers.empty()) return WalkResult::advance();
    LLVM_DEBUG(llvm::dbgs() << "DMA users: " << dmaUsers.size() << "\n");

    for (Operation *userOp : allocOp->getUsers()) {
      auto subviewOp = dyn_cast<memref::SubViewOp>(userOp);
      if (!subviewOp) continue;

      if (!memrefToNew.contains(allocOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Create new allocate\n");
        rewriter.setInsertionPoint(allocOp);
        auto memRefType = cast<MemRefType>(subviewOp.getResult().getType());
        MemRefType allocType = MemRefType::get(
            memRefType.getShape(), memRefType.getElementType(),
            MemRefLayoutAttrInterface{}, memRefType.getMemorySpace());
        auto newAllocOp = rewriter.create<memref::AllocOp>(
            rewriter.getUnknownLoc(), allocType);
        auto newDeallocOp = rewriter.create<memref::DeallocOp>(
            rewriter.getUnknownLoc(), newAllocOp);
        newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
        memrefToNew[allocOp] = newAllocOp;
      }
      auto newAlloc = memrefToNew[allocOp];
      rewriter.replaceAllUsesWith(subviewOp, newAlloc);
      toBeErased.push_back(subviewOp);
    }

    // Update the alloc's DMA users.
    if (memrefToNew.contains(allocOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Update allocate DMA users: " << dmaUsers.size() << "\n");
      auto newAlloc = memrefToNew[allocOp];
      auto type = cast<MemRefType>(newAlloc.getType());
      for (AMDAIE::DmaCpyNdOp dmaOp : dmaUsers) {
        SmallVector<Value> empty;
        rewriter.setInsertionPoint(dmaOp.getSourceObjectFifo());
        auto source = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
            newAlloc.getResult());
        rewriter.replaceOp(dmaOp.getSourceObjectFifo(), source);
        rewriter.setInsertionPoint(dmaOp);
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
            dmaOp.getLoc(), dmaOp.getTarget(), dmaOp.getTargetOffsets(),
            dmaOp.getTargetSizes(), dmaOp.getTargetStrides(), source,
            dmaOp.getSourceOffsets(), dmaOp.getSourceSizes(),
            dmaOp.getSourceStrides());
        rewriter.replaceOp(dmaOp, newDmaOp);
      }

      // Insert dealloc
      memref::DeallocOp deallocOp;
      for (Operation *userOp : allocOp->getUsers()) {
        if (auto deallocUser = dyn_cast<memref::DeallocOp>(userOp)) {
          deallocOp = deallocUser;
        }
      }
      if (deallocOp) {
        toBeErased.push_back(deallocOp);
      }
      toBeErased.push_back(allocOp);
    }
    return WalkResult::advance();
  });

  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  return success();
}

/// Convert inner scf.forall ops chosen for parallel distribution to scf.for
/// ops.
LogicalResult localForallToFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    // We index on thread mapping for core and dma unrolling and buffer
    // distribution.
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    SmallVector<Operation *> loopResults;
    if (failed(scf::forallToForLoop(rewriter, forallOp, &loopResults))) {
      forallOp.emitOpError() << "failed to transform scf.forall to scf.for";
      return WalkResult::interrupt();
    }
    // Set attribute to unroll this loop later in this pass.
    for (Operation *loopRes : loopResults) {
      scf::ForOp forOp = dyn_cast<scf::ForOp>(loopRes);
      if (!forOp) {
        forallOp.emitOpError() << "failed to retrieve generated scf.for from "
                                  "scf::forallToForLoop conversion";
        return WalkResult::interrupt();
      }
      forOp->setAttr(kAMDAIELoopUnroll,
                     mlir::BoolAttr::get(forOp->getContext(), true));
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Hoist an affine apply op on a scf.for op's induction variable into that
/// scf.for block.
LogicalResult hoistAffineApplyDependingOnFor(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp->walk([&](affine::AffineApplyOp applyOp) {
    (void)hoistForAffineApplyOp(rewriter, applyOp);
  });
  return success();
}

/// Unroll the scf.for loops selected for parallel execution on the AIE
/// array. Try to hoist dma ops that don't depend on the loops' induction
/// variables to avoid duplicated copies.
///
/// This rewriter consists of a sequence of transformations:
///   1) First, try to promote the for loop if possible.
///   2) Try hoisting dma ops outside the scf.for operation.
///   3) Unroll and distribute the logical objectfifos remaining in the scf.for
///   loop.
class AMDAIEUnrollLocalLoops : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  /// Hoist dma ops outside the scf.for operation if there are no dependencies
  /// on the scf.for loop, or on other dmas producing or consuming data from the
  /// one considered.
  ///
  /// NOTE: This method makes an assumption that the operations
  /// happen on the most local memory and therefore dmas moving data into more
  /// local memory can be hoisted before the scf.for loop. And on the other
  /// hand, dmas moving data away from local memory can be hoisted behind the
  /// scf.for. This assumption could be removed in the future.
  template <typename Iterator>
  LogicalResult hoistDmaOps(PatternRewriter &rewriter, scf::ForOp forOp) const {
    // Keep track of whether a hoist happened.
    bool hoistHappened{false};

    // Create utility function to check whether an operand depends on a scf.for
    // induction variable or a value within the scf.for's scope
    auto dependsOnLoop = [&](OpOperand &operand) -> bool {
      Operation *op = operand.get().getDefiningOp();
      if (!op) return operand.get() == forOp.getInductionVar();

      // Check for an induction var and whether the parent scf.for is the same
      // as the one we're using for the hoist
      auto parentForOp = op->getParentOfType<scf::ForOp>();
      return (operand.get() == forOp.getInductionVar()) ||
             (parentForOp == forOp);
    };

    // Logical objectfifo dependencies introduced in loop body walk.
    DenseSet<AMDAIE::LogicalObjectFifoFromMemrefOp> dependencies;

    // Utility to add logical objectfifos to the dependencies set.
    auto addDependencies = [&](AMDAIE::DmaCpyNdOp dmaOp) {
      if (std::is_same<Iterator, ForwardIterator>::value) {
        dependencies.insert(dmaOp.getTargetObjectFifo());
      } else if (std::is_same<Iterator, ReverseIterator>::value) {
        dependencies.insert(dmaOp.getSourceObjectFifo());
      }
    };

    // Walk all dma ops and try to hoist them if:
    //   1) There are no dependencies on the loop's induction variable.
    //   2) There are no dependencies on other dmas producing into a logical
    //   objectfifo which this dma is consuming.
    //   3) There are no dmas waiting for this dma to produce data.
    //
    // The last two conditions are partially checked through comparing source
    // and target memory spaces (Global < L2 < L1):
    //   1) In the forward sweep, only hoist dmas for which
    //   (source.memory < target.memory), i.e. moving data to AIE cores.
    //   2) In the backward sweep, only hoist dmas for which
    //   (source.memory > target.memory), i.e. moving data away from AIE cores.
    //
    // These last checks in theory limit the hoisting detection capability, but
    // should be valid.
    forOp.walk<WalkOrder::PostOrder, Iterator>([&](AMDAIE::DmaCpyNdOp dmaOp) {
      if (llvm::any_of(dmaOp->getOpOperands(), [&](OpOperand &operand) {
            return dependsOnLoop(operand);
          })) {
        addDependencies(dmaOp);
        return WalkResult::advance();
      }

      uint64_t sourceMemspace =
          dmaOp.getSourceObjectFifo().getMemorySpaceAsUInt();
      uint64_t targetMemspace =
          dmaOp.getTargetObjectFifo().getMemorySpaceAsUInt();
      if (std::is_same<Iterator, ForwardIterator>::value &&
          !dependencies.contains(dmaOp.getSourceObjectFifo()) &&
          sourceMemspace < targetMemspace) {
        rewriter.moveOpBefore(dmaOp, forOp);
        hoistHappened = true;
        return WalkResult::advance();
      } else if (std::is_same<Iterator, ReverseIterator>::value &&
                 !dependencies.contains(dmaOp.getTargetObjectFifo()) &&
                 sourceMemspace > targetMemspace) {
        rewriter.moveOpAfter(dmaOp, forOp);
        hoistHappened = true;
        return WalkResult::advance();
      }

      // If this dma op can't be hoisted due to dependencies, keep adding new
      // dependencies.
      addDependencies(dmaOp);
      return WalkResult::advance();
    });
    if (hoistHappened) return success();
    return failure();
  }

  /// Unroll the for loop, skipping the first iteration. While unrolling, create
  /// new clones of `LogicalObjectFifoFromMemrefOp` so they can be distributed
  /// onto multiple physical locations later.
  LogicalResult loopUnrollAndDistributeLogicalObjectFifos(
      RewriterBase &rewriter, scf::ForOp forOp) const {
    Block *loopBodyBlock = forOp.getBody();
    OpBuilder builder = OpBuilder::atBlockTerminator(loopBodyBlock);

    // Keep a pointer to the last non-terminator operation in the original block
    // so that we know what to clone (since we are doing this in-place).
    Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

    // Update loop bounds
    int64_t lbInt = getConstantIntValue(forOp.getLowerBound()).value();
    int64_t ubInt = getConstantIntValue(forOp.getUpperBound()).value();
    int64_t stepInt = getConstantIntValue(forOp.getStep()).value();
    if (lbInt != 0 || stepInt != 1) return failure();
    if (stepInt == (ubInt - lbInt)) return failure();
    Value forOpIV = forOp.getInductionVar();
    forOp.setUpperBound(
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1));

    // Iterate through the loop and create body
    for (auto i = lbInt + stepInt; i < ubInt; i += stepInt) {
      IRMapping operandMap;
      Value ivUnroll =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), i);
      if (!forOpIV.use_empty()) {
        operandMap.map(forOpIV, ivUnroll);
      }

      // Iterate through body and map internal logical objectfifos to new ones
      // and fill operand map.
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(*it)) {
          AMDAIE::LogicalObjectFifoFromMemrefOp source =
              dmaOp.getSourceObjectFifo();
          uint64_t sourceMemSpaceInt = source.getMemorySpaceAsUInt();
          AMDAIE::LogicalObjectFifoFromMemrefOp target =
              dmaOp.getTargetObjectFifo();
          uint64_t targetMemSpaceInt = target.getMemorySpaceAsUInt();
          if (targetMemSpaceInt > sourceMemSpaceInt) {
            rewriter.setInsertionPoint(target);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getTarget().getDefiningOp()));
            operandMap.map(target.getOutput(), cloneOp.getOutput());
          } else if (sourceMemSpaceInt > targetMemSpaceInt) {
            rewriter.setInsertionPoint(source);
            auto cloneOp = dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.clone(*dmaOp.getSource().getDefiningOp()));
            operandMap.map(source.getOutput(), cloneOp.getOutput());
          }
        }
      }

      // Iterate through body and clone ops
      for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd);
           it++) {
        builder.clone(*it, operandMap);
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Unroll loops only if annotated to be unrolled earlier in the pass.
    if (!forOp->hasAttr(kAMDAIELoopUnroll) ||
        !cast<BoolAttr>(forOp->getAttr(kAMDAIELoopUnroll)).getValue())
      return failure();

    // Skip for ops with nested for ops. Wait until nested ones get resolved
    // first.
    auto nestedForOps = forOp.getOps<scf::ForOp>();
    if (!nestedForOps.empty()) return failure();

    // First, try to promote the for loop
    if (succeeded(forOp.promoteIfSingleIteration(rewriter))) {
      return success();
    }

    // Hoist non-dma loop invariant operations (like constants, affine apply,
    // etc) out of the loop like operation to allow more DMA operations to be
    // hoisted.
    moveLoopInvariantCode(dyn_cast<LoopLikeOpInterface>(forOp.getOperation()));

    // Try hoisting dma ops outside the scf.for operation by sweeping once
    // forward and once backward to hoist to before, respectively after the
    // scf.for.
    (void)hoistDmaOps<ForwardIterator>(rewriter, forOp);
    (void)hoistDmaOps<ReverseIterator>(rewriter, forOp);

    // Unroll and distribute the logical objectfifos
    if (failed(loopUnrollAndDistributeLogicalObjectFifos(rewriter, forOp))) {
      return failure();
    }
    return success();
  }
};

/// Return the tiles of the sources respectively targets of the users of this
/// logical objectfifo, depending on whether the OperateOn template parameter is
/// set to `OperateOn::Source` respectively `OperateOn::Target`.
template <CopyOpOperateOn OperateOn>
LogicalResult getUserTiles(
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    SmallVectorImpl<AMDAIE::TileOp> &tiles) {
  llvm::SmallSetVector<AMDAIE::TileOp, 16> tileSet;
  for (Operation *user : logicalObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(user)) {
      ValueRange tileIndices;
      if constexpr (OperateOn == CopyOpOperateOn::Source) {
        if (dmaOp.getTargetObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getSourceObjectFifo().getTiles();
      } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
        if (dmaOp.getSourceObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getTargetObjectFifo().getTiles();
      }

      // Only fill in tiles when all sources have tiles.
      if (tileIndices.empty()) return failure();
      for (Value index : tileIndices)
        tileSet.insert(dyn_cast<AMDAIE::TileOp>(index.getDefiningOp()));
    }
  }
  tiles = tileSet.takeVector();
  return success();
}

/// Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
/// memrefs from logical objectfifos and update the computational operations to
/// operate on these local memrefs.
LogicalResult insertLogicalObjectFifoAccess(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  moduleOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                               AMDAIE::MemoryAccess>>
        memrefToLogicalObjectFifo;
    // First walk to collect consume/produce DMA accesses and map respective
    // memrefs to logical objectifos.
    coreOp->walk([&](Operation *op) {
      // TODO(jornt): can we avoid produce/consume?
      if (auto consumeOp = dyn_cast<AMDAIE::LogicalObjectFifoConsume>(op)) {
        Value targetMemref =
            consumeOp.getDmaCpyNdOp().getTargetObjectFifo().getMemref();
        memrefToLogicalObjectFifo[targetMemref] =
            std::make_pair(consumeOp.getDmaCpyNdOp().getTargetObjectFifo(),
                           AMDAIE::MemoryAccess::Read);
      } else if (auto produceOp =
                     dyn_cast<AMDAIE::LogicalObjectFifoProduce>(op)) {
        Value sourceMemref =
            produceOp.getDmaCpyNdOp().getSourceObjectFifo().getMemref();
        memrefToLogicalObjectFifo[sourceMemref] =
            std::make_pair(produceOp.getDmaCpyNdOp().getSourceObjectFifo(),
                           AMDAIE::MemoryAccess::Write);
      }
    });

    WalkResult res = coreOp->walk([&](Operation *op) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        for (auto &&[idx, operand] :
             llvm::enumerate(linalgOp->getOpOperands())) {
          if (memrefToLogicalObjectFifo.contains(operand.get())) {
            rewriter.setInsertionPointToStart(coreOp.getBody());
            std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                       AMDAIE::MemoryAccess>
                value = memrefToLogicalObjectFifo[operand.get()];
            rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), std::get<0>(value),
                std::get<1>(value));
            // TODO(jornt): Temporary, enable after access operations are used
            // for inserting synchronization stubs instead of consume/produce.
            // linalgOp->setOperand(idx, accessOp);
          } else if (auto type =
                         llvm::dyn_cast<MemRefType>(operand.get().getType())) {
            Value memref = operand.get();
            rewriter.setInsertionPoint(coreOp);
            auto logicalObjectFifo =
                rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                    rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
                    memref);
            rewriter.setInsertionPointToStart(coreOp.getBody());
            rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), logicalObjectFifo,
                AMDAIE::MemoryAccess::None);
            // TODO(jornt): Temporary, enable after access operations are used
            // for inserting synchronization stubs instead of consume/produce.
            // linalgOp->setOperand(idx, accessOp);
          }
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }
  return success();
}

/// Utility to recursively find users of the provided logical objectFifo inside
/// `amdaie.core` operations and return the tile coordinates.
LogicalResult findUsersInCoreAndAddTiles(
    Operation *op, AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> &tiles) {
  for (Operation *userOp : op->getUsers()) {
    if (auto coreOp = userOp->getParentOfType<AMDAIE::CoreOp>()) {
      AMDAIE::TileOp tileOp = coreOp.getTileOp();
      std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
      std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
      if (!column || !row) {
        return coreOp.emitOpError() << "has non-constant tile location";
      }
      tiles.insert(std::make_pair(column.value(), row.value()));
    }
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
      return findUsersInCoreAndAddTiles(subviewOp, logicalObjectFifo, tiles);
    } else if (auto userLogicalObjectFifo =
                   dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(userOp)) {
      return findUsersInCoreAndAddTiles(userLogicalObjectFifo,
                                        logicalObjectFifo, tiles);
    }
  }
  return success();
}

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalAieTiles(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  WalkResult res = moduleOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
        Attribute memSpace = logicalObjectFifo.getMemorySpace();
        if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
          return WalkResult::advance();

        llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
        if (failed(findUsersInCoreAndAddTiles(
                logicalObjectFifo, logicalObjectFifo, tileLocations))) {
          return WalkResult::interrupt();
        }
        // Handle subviews.
        for (Operation *userOp :
             logicalObjectFifo.getMemref().getDefiningOp()->getUsers()) {
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(userOp)) {
            if (failed(findUsersInCoreAndAddTiles(subviewOp, logicalObjectFifo,
                                                  tileLocations))) {
              return WalkResult::interrupt();
            }
          }
        }

        SmallVector<Value> tiles;
        tiles.reserve(tileLocations.size());
        rewriter.setInsertionPoint(logicalObjectFifo);
        for (auto [column, row] : tileLocations) {
          auto colIndex = rewriter.create<arith::ConstantIndexOp>(
              rewriter.getUnknownLoc(), column);
          auto rowIndex = rewriter.create<arith::ConstantIndexOp>(
              rewriter.getUnknownLoc(), row);
          auto tileOp = rewriter.create<AMDAIE::TileOp>(
              rewriter.getUnknownLoc(), colIndex, rowIndex);
          tiles.push_back(tileOp.getResult());
        }
        // Sort for deterministic output IR.
        llvm::sort(tiles.begin(), tiles.end(),
                   AMDAIE::TileOp::tileValueColumnAndRowComparator);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            logicalObjectFifo,
            cast<LogicalObjectFifoType>(
                logicalObjectFifo.getOutput().getType()),
            logicalObjectFifo.getMemref(), tiles);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Assign a set of potential physical AIE tiles to logical objectFifos. This
/// rewrite takes an iterative approach by matching logical objectfifos and only
/// assigning tiles when linked through dma ops with other logical objectfifos
/// which already have tiles assigned. If the linked logical objectfifos don't
/// have tiles assigned yet, we will return a failure and give the linked
/// logical objectfifos a chance to assign tiles before returning to this one.
///
/// TODO(jornt): There are decisions being made in this pass on which tiles to
/// assign to a logical objectfifo. This logic is very simple for now and tries
/// to use the tiles in the same columns as targets and sources. At some point,
/// we probably need some AIE device model to guide the assignement here for
/// performance and to avoid hardware resource issues later on.
class FillAieTiles
    : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemrefOp> {
  using OpRewritePattern<
      AMDAIE::LogicalObjectFifoFromMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "FillAieTiles: " << logicalObjectFifo << "\n");
    if (!logicalObjectFifo.getTiles().empty()) {
      return failure();
    }

    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    // Skip logical objectfifos within local memory as they should already be
    // assigned.
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() == 2) {
      if (logicalObjectFifo.getTiles().empty()) {
        logicalObjectFifo.emitOpError()
            << "found logical objectfifo on local memory space with no tiles "
               "assigned.";
      }
      return failure();
    }
    // HandLe both L3/shim and L2/Memtiles.
    // Skip logical objectfifos within non-global and non-shared memory.
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
      return logicalObjectFifo.emitOpError()
             << "found logical objectfifo with unknown memory space";
    }

    SmallVector<AMDAIE::TileOp, 16> targetTiles;
    SmallVector<AMDAIE::TileOp, 16> sourceTiles;
    LogicalResult dstRes =
        getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    LogicalResult srcRes =
        getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);

    // If no source and target tiles found, skip.
    if (failed(dstRes) && failed(srcRes)) {
      return failure();
    }

    // TODO(jornt): avoid row hardcoding. Will need to update the mlir-aie
    // target model for this.
    int64_t rowInt = memSpace ? 1 : 0;
    llvm::SmallSetVector<std::pair<int64_t, int64_t>, 16> tileLocations;
    auto createTileLocations =
        [&](SmallVector<AMDAIE::TileOp, 16> &tiles) -> LogicalResult {
      // TODO(jornt): For now, for deterministic behaviour, sort on column
      // index and use first one. This needs to be generalized to assign
      // tiles based on a resource model.
      std::sort(tiles.begin(), tiles.end(),
                AMDAIE::TileOp::tileColumnComparator);
      // Erase duplicates.
      tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
      for (AMDAIE::TileOp tile : tiles) {
        std::optional<int64_t> column = getConstantIntValue(tile.getCol());
        if (!column) return tile.emitOpError() << "found non-constant column";
        tileLocations.insert(std::make_pair(column.value(), rowInt));
      }
      return success();
    };

    if (!targetTiles.empty() && !sourceTiles.empty()) {
      return logicalObjectFifo.emitOpError()
             << "found logical objectfifo with both source and target tiles, "
                "which is not supported yet";
    } else if (!targetTiles.empty()) {
      // Create tile locations for this logical objectfifo based on target
      // tiles.
      if (failed(createTileLocations(targetTiles))) {
        return failure();
      }
    } else if (!sourceTiles.empty()) {
      // Create tile locations for this logical objectfifo based on source
      // tiles.
      if (failed(createTileLocations(sourceTiles))) {
        return failure();
      }
    } else {
      // Don't assign this logicalObjectFifo to a physical tile (yet!). Wait
      // for other logical objectfifos to be assigned first.
      return failure();
    }

    // If no tile results, skip, and maybe in a next iteration another tile will
    // be found.
    if (tileLocations.empty()) {
      return failure();
    }

    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo, logicalObjectFifo.getMemref(),
        tileLocations.takeVector());
    return success();
  }
};

/// Return the user DMA operations and corresponding assigned tiles in the
/// specified direction (source or target).
template <CopyOpOperateOn OperateOn>
SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
getUserDmasAndTiles(AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
  SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
      dmaOps;
  for (Operation *user : logicalObjectFifo->getUsers()) {
    if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(user)) {
      ValueRange tileIndices;
      if constexpr (OperateOn == CopyOpOperateOn::Source) {
        if (dmaOp.getTargetObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getSourceObjectFifo().getTiles();
      } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
        if (dmaOp.getSourceObjectFifo() != logicalObjectFifo) continue;
        tileIndices = dmaOp.getTargetObjectFifo().getTiles();
      }
      SmallVector<AMDAIE::TileOp> tiles;
      for (Value index : tileIndices)
        tiles.push_back(dyn_cast<AMDAIE::TileOp>(index.getDefiningOp()));
      dmaOps.push_back(std::make_pair(dmaOp, tiles));
    }
  }
  return dmaOps;
}

/// Assign specific tile locations to objectFifos, starting from the set of
/// potential tile locations filled in earlier.
LogicalResult assignAieTilesAndDistributeLogicalObjectFifos(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles = llvm::map_to_vector(
        logicalObjectFifo.getTiles(),
        [](Value tile) { return dyn_cast<TileOp>(tile.getDefiningOp()); });
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileColumnComparator);
    SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
        sourceDmaOps =
            getUserDmasAndTiles<CopyOpOperateOn::Source>(logicalObjectFifo);
    SmallVector<std::pair<AMDAIE::DmaCpyNdOp, SmallVector<AMDAIE::TileOp>>>
        targetDmaOps =
            getUserDmasAndTiles<CopyOpOperateOn::Target>(logicalObjectFifo);

    // Assign tiles for following cases:
    // 1) No source DMA operations (e.g. L3 -> L2): distribute onto multiple
    // tiles to potentially use multiple shim DMAs for reading from global
    // memory in different columns.
    // 2) No target DMA operations (e.g. L2 -> L3):
    // distribute onto multiple tiles to potentially use multiple shim DMAs for
    // writing to global memory in different columns.
    // 3) Default: assign first tile from the sorted sequence of potential
    // tiles.
    if (sourceDmaOps.empty() && targetDmaOps.size() == tiles.size()) {
      llvm::sort(targetDmaOps.begin(), targetDmaOps.end(), dmaColComparator);
      for (auto &&[tile, dmaOpElem] : llvm::zip(tiles, targetDmaOps)) {
        rewriter.setInsertionPoint(logicalObjectFifo);
        SmallVector<Value> tileResults = {cast<Value>(tile.getResult())};
        auto newLogicalObjectFifo =
            rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.getUnknownLoc(),
                cast<LogicalObjectFifoType>(
                    logicalObjectFifo.getOutput().getType()),
                logicalObjectFifo.getMemref(), tileResults);
        dmaOpElem.first->replaceUsesOfWith(logicalObjectFifo.getResult(),
                                           newLogicalObjectFifo.getResult());
      }
    } else if (targetDmaOps.empty() && sourceDmaOps.size() == tiles.size()) {
      llvm::sort(sourceDmaOps.begin(), sourceDmaOps.end(), dmaColComparator);
      for (auto &&[tile, dmaOpElem] : llvm::zip(tiles, sourceDmaOps)) {
        rewriter.setInsertionPoint(logicalObjectFifo);
        SmallVector<Value> tileResults = {cast<Value>(tile.getResult())};
        auto newLogicalObjectFifo =
            rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                rewriter.getUnknownLoc(),
                cast<LogicalObjectFifoType>(
                    logicalObjectFifo.getOutput().getType()),
                logicalObjectFifo.getMemref(), tileResults);
        dmaOpElem.first->replaceUsesOfWith(logicalObjectFifo.getResult(),
                                           newLogicalObjectFifo.getResult());
      }
    } else {
      // For now, use first tile in sorted list. This will need to become more
      // complex in the future to account for potential hardware limitations and
      // constraints.
      SmallVector<Value> tileResults = {cast<Value>(tiles[0].getResult())};
      rewriter.setInsertionPoint(logicalObjectFifo);
      rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          logicalObjectFifo,
          cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
          logicalObjectFifo.getMemref(), tileResults);
    }
    return WalkResult::advance();
  });
  return success();
}

/// Allocate different memories for logical objectFifos on the same shared
/// memory tile to ensure different buffers will be used for them.
LogicalResult distributeSharedMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Map from local objectfifos found to the tiles where they are used
  DenseMap<SmallVector<std::pair<int64_t, int64_t>>, Value, LocationMapInfo>
      locationsToMemref;

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
          return dyn_cast<AMDAIE::TileOp>(tile.getDefiningOp());
        });
    llvm::sort(tiles.begin(), tiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);

    SmallVector<AMDAIE::TileOp> targetTiles;
    (void)getUserTiles<CopyOpOperateOn::Target>(logicalObjectFifo, targetTiles);
    llvm::sort(targetTiles.begin(), targetTiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);
    tiles.insert(tiles.end(), std::make_move_iterator(targetTiles.begin()),
                 std::make_move_iterator(targetTiles.end()));

    SmallVector<AMDAIE::TileOp> sourceTiles;
    (void)getUserTiles<CopyOpOperateOn::Source>(logicalObjectFifo, sourceTiles);
    llvm::sort(sourceTiles.begin(), sourceTiles.end(),
               AMDAIE::TileOp::tileValueColumnAndRowComparator);
    tiles.insert(tiles.end(), std::make_move_iterator(sourceTiles.begin()),
                 std::make_move_iterator(sourceTiles.end()));
    LLVM_DEBUG(llvm::dbgs() << "Op: " << logicalObjectFifo
                            << ", number of tiles: " << tiles.size() << "\n");

    SmallVector<std::pair<int64_t, int64_t>> locations =
        llvm::map_to_vector(tiles, [](AMDAIE::TileOp tile) {
          return std::make_pair(
              (int64_t)getConstantIntValue(tile.getCol()).value(),
              (int64_t)getConstantIntValue(tile.getRow()).value());
        });
    if (!locationsToMemref.contains(locations)) {
      auto allocOp = dyn_cast<memref::AllocOp>(
          logicalObjectFifo.getMemref().getDefiningOp());
      rewriter.setInsertionPoint(allocOp);
      auto newAllocOp =
          dyn_cast<memref::AllocOp>(rewriter.clone(*allocOp.getOperation()));
      auto newDeallocOp = rewriter.create<memref::DeallocOp>(
          rewriter.getUnknownLoc(), newAllocOp);
      newDeallocOp->moveBefore(&newAllocOp->getBlock()->back());
      locationsToMemref[locations] = newAllocOp.getResult();
    }
    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        locationsToMemref[locations], logicalObjectFifo.getTiles());
    return WalkResult::advance();
  });
  return success();
}

class AMDAIEDistributeCoresAndObjectFifosPass
    : public impl::AMDAIEDistributeCoresAndObjectFifosBase<
          AMDAIEDistributeCoresAndObjectFifosPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeCoresAndObjectFifosPass() = default;
  AMDAIEDistributeCoresAndObjectFifosPass(
      const AMDAIEDistributeCoresAndObjectFifosPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeCoresAndObjectFifosPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  // Convert local scf.forall operations selected for parallel distribution to
  // nested scf.for operations.
  if (failed(localForallToFor(moduleOp))) {
    moduleOp.emitOpError()
        << "local `scf.forall` to `scf.for` conversion failed";
    return signalPassFailure();
  }

  // Hoist the affine apply ops on scf.for induction variables to the
  // corresponding scf.for's body.
  if (failed(hoistAffineApplyDependingOnFor(moduleOp))) {
    moduleOp.emitOpError() << "`affine.apply` hoisting failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after localForallToFor: \n"
                          << moduleOp << "\n");

  if (failed(distributeLocalMemory(moduleOp))) {
    moduleOp.emitOpError() << "local memory distribution failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after distributeLocalMemory: \n"
                          << moduleOp << "\n");

  // Unroll local parallel loops and try hoisting dma operations if
  // possible.
  RewritePatternSet unrollLocalLoopsPatterns(context);
  unrollLocalLoopsPatterns.insert<AMDAIEUnrollLocalLoops>(context);
  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(unrollLocalLoopsPatterns)))) {
    moduleOp.emitOpError()
        << "loop unrolling of loops selected for parallel execution failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after AMDAIEUnrollLocalLoops: \n"
                          << moduleOp << "\n");

  // Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
  // memrefs from logical objectfifos and update the computational operations to
  // operate on these local memrefs. These access operations will be used to
  // assign local AIE tiles to local logical objectFifos later.
  if (failed(insertLogicalObjectFifoAccess(moduleOp))) {
    moduleOp.emitOpError()
        << "insertion of `amdaie.logicalobjectfif.access` operations failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after insertLogicalObjectFifoAccess: \n"
                          << moduleOp << "\n");

  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalAieTiles(moduleOp))) {
    moduleOp.emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after assignLocalAieTiles: \n"
                          << moduleOp << "\n");

  // Assign a set of potential tile locations to the remaining logical
  // objectFifos.
  RewritePatternSet assignAieTilePatters(context);
  assignAieTilePatters.insert<FillAieTiles>(context);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(assignAieTilePatters)))) {
    moduleOp.emitOpError()
        << "collection of tile candidates for logical objectFifos failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after FillAieTiles: \n"
                          << moduleOp << "\n");

  // Assign specific tile locations to objectFifos, starting from the set of
  // potential tile locations filled in earlier.
  if (failed(assignAieTilesAndDistributeLogicalObjectFifos(moduleOp))) {
    moduleOp.emitOpError()
        << "tile assignment and logical objectFifo distribution failed";
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs()
             << "Module after assignAieTilesAndDistributeLogicalObjectFifos: \n"
             << moduleOp << "\n");

  // Allocate different memories for logical objectFifos on the same shared
  // memory tile to ensure different buffers will be used for them.
  if (failed(distributeSharedMemory(moduleOp))) {
    moduleOp.emitOpError() << "distribution of shared memory failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeCoresAndObjectFifosPass() {
  return std::make_unique<AMDAIEDistributeCoresAndObjectFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
