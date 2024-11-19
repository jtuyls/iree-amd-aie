// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-assign-tiles"

namespace mlir::iree_compiler::AMDAIE {

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
      for (Value index : tileIndices) {
        tileSet.insert(
            dyn_cast_if_present<AMDAIE::TileOp>(index.getDefiningOp()));
      }
    }
  }
  tiles = tileSet.takeVector();
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

/// Utility to clear non-local tile assignments.
LogicalResult clearNonLocalTiles(RewriterBase &rewriter, Operation *op) {
  op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp objFifo) {
    if (objFifo.getMemorySpaceAsUInt() != 2) {
      rewriter.setInsertionPoint(objFifo);
      SmallVector<Value> tiles;
      rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          objFifo, cast<LogicalObjectFifoType>(objFifo.getOutput().getType()),
          objFifo.getMemref(), tiles);
    }
  });
  return success();
}

/// TODO(jornt): too hardcoded?
LogicalResult duplicateGlobalObjFifos(RewriterBase &rewriter, Operation *op) {
  op->walk([&](AMDAIE::DoublyStridedCopyOpInterface copyOp) {
    auto source = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        copyOp.getSource().getDefiningOp());
    auto target = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        copyOp.getTarget().getDefiningOp());
    if (source && source.getMemorySpaceAsUInt() == 0) {
      rewriter.setInsertionPoint(copyOp);
      auto newSource = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(),
          cast<LogicalObjectFifoType>(source.getOutput().getType()),
          source.getMemref());
      rewriter.replaceUsesWithIf(
          source.getOutput(), newSource.getOutput(), [&](OpOperand &use) {
            return use.getOwner() == copyOp.getOperation();
          });
    }
    if (target && target.getMemorySpaceAsUInt() == 0) {
      rewriter.setInsertionPoint(copyOp);
      auto newTarget = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          rewriter.getUnknownLoc(),
          cast<LogicalObjectFifoType>(target.getOutput().getType()),
          target.getMemref());
      rewriter.replaceUsesWithIf(
          target.getOutput(), newTarget.getOutput(), [&](OpOperand &use) {
            return use.getOwner() == copyOp.getOperation();
          });
    }
  });
  return success();
}

/// Assign tiles to the logical objectfifos with local memory space (L1).
/// The tiles are derived from the usage of the logical objectfifos within
/// core operations, which are already assigned a tile location.
LogicalResult assignLocalTiles(RewriterBase &rewriter, Operation *op) {
  WalkResult res =
      op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
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

/// Assign a set of candidate physical AIE tiles to logical objectFifos. This
/// rewrite takes an iterative approach by matching logical objectfifos and only
/// assigning tiles when linked through dma ops with other logical objectfifos
/// which already have tiles assigned. If the linked logical objectfifos don't
/// have tiles assigned yet, we will return a failure and give the linked
/// logical objectfifos a chance to assign tiles before returning to this one.
class FillTiles
    : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemrefOp> {
  using OpRewritePattern<
      AMDAIE::LogicalObjectFifoFromMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "FillTiles: " << logicalObjectFifo << "\n");
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
      return rewriter.notifyMatchFailure(
          logicalObjectFifo,
          "No tile locations found for this logical objFifo. Maybe in a next "
          "iteration, with more information, a tile location can be found.");
    }

    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo, logicalObjectFifo.getMemref(),
        tileLocations.takeVector());
    return success();
  }
};

///
// struct TileLocAndUsage {
//   TileLoc loc;
//   size_t usage{0};
//   TileLocAndUsage(TileLoc loc, size_t usage)
//       : loc(std::move(loc)), usage(usage) {}
// }

/// Compares TileLocAndUsage by usage, column and row in the respective order
/// for deterministic and easily predictible behaviour. The ordering happens on
/// these parameters in the following way:
/// - Tiles with lower usage have priority over tiles with larger usage.
/// - If usage is the same, tiles with a lower column index have priority over
/// tiles with a larger column index.
/// - If both usage and column index are the same, tiles with a lower row index
/// have priority over tiles with a larger row index.
// struct TileLocAndUsageCmp {
//   bool operator()(const TileLocAndUsage &tileLocAndUsage1,
//                   const TileLocAndUsage &tileLocAndUsage2) const {
//     if (tileLocAndUsage1.usage > tileLocAndUsage2.usage) return true;
//     if (tileLocAndUsage1.usage < tileLocAndUsage2.usage) return false;
//     if (tileLocAndUsage1.column > tileLocAndUsage2.column) return true;
//     if (tileLocAndUsage1.column < tileLocAndUsage2.column) return false;
//     if (tileLocAndUsage1.row > tileLocAndUsage2.row) return true;
//     if (tileLocAndUsage1.row < tileLocAndUsage2.row) return false;
//     assert(false && "same tiles should never be compared");
//   }
// };

// bool tileLocAndUsageCmp(const TileLocAndUsage &a, const TileLocAndUsage &b) {
//   if (a.usage > b.usage) return true;
//   if (a.usage < b.usage) return false;
//   if (a.column > b.column) return true;
//   if (a.column < b.column) return false;
//   if (a.row > b.row) return true;
//   if (a.row < b.row) return false;
//   assert(false && "same tiles should never be compared");
// }

/// Assign specific tile locations to objectFifos, starting from the set of
/// potential tile locations filled in earlier.
LogicalResult assignNonLocalTiles(RewriterBase &rewriter, Operation *op) {
  MLIRContext *context = rewriter.getContext();

  if (failed(clearNonLocalTiles(rewriter, op)))
    return op->emitOpError() << "failed to clear non-local tile assignemts";

  // Find tile candidates
  RewritePatternSet fillTilePatterns(context);
  fillTilePatterns.insert<FillTiles>(context);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(fillTilePatterns)))) {
    return op->emitOpError()
           << "collection of tile candidates for logical objectFifos failed";
  }
  if (failed(verify(op, true))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After fillTiles: \n" << *op << "\n");

  DenseMap<TileLoc, size_t> tileLocToUsage;
  auto tileLocAndUsageCmp = [&](AMDAIE::TileOp a, AMDAIE::TileOp b) -> bool {
    // llvm::outs() << "A: " << a << "\n";
    // llvm::outs() << "B: " << b << "\n";
    int64_t colA = getConstantIndexOrAssert(a.getCol());
    int64_t rowA = getConstantIndexOrAssert(a.getRow());
    int64_t colB = getConstantIndexOrAssert(b.getCol());
    int64_t rowB = getConstantIndexOrAssert(b.getRow());
    size_t usageA = tileLocToUsage[TileLoc(colA, rowA)];
    size_t usageB = tileLocToUsage[TileLoc(colB, rowB)];
    // llvm::outs() << "usageA: " << usageA << "\n";
    // llvm::outs() << "usageB: " << usageB << "\n";
    if (usageA < usageB) return true;
    if (usageA > usageB) return false;
    if (colA < colB) return true;
    if (colA > colB) return false;
    if (rowA < rowB) return true;
    if (rowA > rowB) return false;
    assert(false && "same tiles should never be compared");
  };

  // After filling candidates, choose a specific one.
  DenseMap<MemRefType, int64_t> logicalObjFifoToTileId;
  op->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
    // llvm::outs() << logicalObjectFifo << "\n";
    Attribute memSpace = logicalObjectFifo.getMemorySpace();
    if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() != 1)
      return WalkResult::advance();

    SmallVector<AMDAIE::TileOp> tiles =
        llvm::map_to_vector(logicalObjectFifo.getTiles(), [](Value tile) {
          return dyn_cast_if_present<TileOp>(tile.getDefiningOp());
        });
    llvm::sort(tiles.begin(), tiles.end(), tileLocAndUsageCmp);
    // AMDAIE::TileOp::tileColumnComparator);

    int64_t memTileId = 0;
    assert(memTileId < tiles.size());
    // llvm::outs() << "choose tile: " << tiles[memTileId] << "\n";
    SmallVector<Value> tileResults = {
        cast<Value>(tiles[memTileId].getResult())};
    // Increase usage of the chosen tile.
    int64_t col = getConstantIndexOrAssert(tiles[memTileId].getCol());
    int64_t row = getConstantIndexOrAssert(tiles[memTileId].getRow());
    tileLocToUsage[TileLoc(col, row)] += 1;
    // llvm::outs() << 

    rewriter.setInsertionPoint(logicalObjectFifo);
    rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        logicalObjectFifo,
        cast<LogicalObjectFifoType>(logicalObjectFifo.getOutput().getType()),
        logicalObjectFifo.getMemref(), tileResults);
    return WalkResult::advance();
  });
  return success();
}

namespace {

class AMDAIEAssignTilesPass
    : public impl::AMDAIEAssignTilesBase<AMDAIEAssignTilesPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignTilesPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(&getContext());

  // TODO

  // Assign tile locations to logical objectfifos on local (L1) memory.
  if (failed(assignLocalTiles(rewriter, parentOp))) {
    parentOp->emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After assignLocalTiles: \n" << *parentOp << "\n");

  if (failed(duplicateGlobalObjFifos(rewriter, parentOp))) {
    parentOp->emitOpError() << "failed duplicating global object fifos";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After duplicateGlobalObjFifos: \n"
                          << *parentOp << "\n");

  // Assign tile locations to logical objectfifos on non-local (not L1) memory.
  if (failed(assignNonLocalTiles(rewriter, parentOp))) {
    parentOp->emitOpError() << "local tile assignment failed";
    return signalPassFailure();
  }
  if (failed(verify(parentOp, true))) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After assignNonLocalTiles: \n"
                          << *parentOp << "\n");
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignTilesPass() {
  return std::make_unique<AMDAIEAssignTilesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
