// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>

#define DEBUG_TYPE "iree-amdaie-to-objectfifo"

namespace mlir::iree_compiler::AMDAIE {

namespace {


//// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class DmaMemcpyNdToLogicalObjectfifo : public OpRewritePattern<xilinx::air::DmaMemcpyNdOp> {
  using OpRewritePattern<xilinx::air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {

    auto srcType = op.getSrc().getType().cast<MemRefType>();
    auto dstType = op.getDst().getType().cast<MemRefType>();
    rewriter.setInsertionPointToStart(op->getBlock());
    AMDAIE::LogicalObjectFifoFromMemref src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
      rewriter.getUnknownLoc(), AMDAIEObjectFifoType::get(srcType), op.getSrc()
    );
    AMDAIE::LogicalObjectFifoFromMemref dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
      rewriter.getUnknownLoc(), AMDAIEObjectFifoType::get(dstType), op.getDst()
    );

    rewriter.setInsertionPoint(op);
    // AMDAIE::DmaCpyNdOp dmaCopy = 
    rewriter.create<AMDAIE::DmaCpyNdOp>(
      op.getLoc(),
      rewriter.getIndexType(), // SmallVector<Type, 1>{}, // rewriter.getIndexType(),
      dst,
      op.getDstOffsets(),
      op.getDstSizes(),
      op.getDstStrides(),
      src,
      op.getSrcOffsets(),
      op.getSrcSizes(),
      op.getSrcStrides()
    );
    // rewriter.replaceOp(op, dmaCopy);
    rewriter.eraseOp(op);
    return success();
  }
};


/// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class SCFForAllToLogicalObjectfifo : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    std::cout << "SCFForAllToLogicalObjectfifo" << std::endl;
    if (forallOp.getNumResults() != 0) {
      return failure();
    }
    if (!forallOp.getMapping().has_value()) {
      return failure();
    }
    SmallVector<Attribute> threadMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());
    if (llvm::any_of(threadMapping, [](Attribute map) {
        return !llvm::isa<mlir::gpu::GPUThreadMappingAttr>(map);
      })) {
      std::cout << "Not a thread mapping attr" << std::endl;
      return failure();
    }
    // forallOp.emitWarning() << "HERE: ";
    // llvm::outs() << "TEST TEST: \n" << forallOp << "\n";

    forallOp->walk([&](xilinx::air::DmaMemcpyNdOp op) {
      llvm::outs() << "DmaMemcpyNdOp: " << op << "\n";
      llvm::outs() << op.getSrc() << "\n";

      auto srcType = op.getSrc().getType().cast<MemRefType>();
      auto dstType = op.getDst().getType().cast<MemRefType>();
      rewriter.setInsertionPointToStart(forallOp->getBlock());
      AMDAIE::LogicalObjectFifoFromMemref src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
        forallOp.getLoc(), AMDAIEObjectFifoType::get(srcType), op.getSrc()
      );
      AMDAIE::LogicalObjectFifoFromMemref dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
        forallOp.getLoc(), AMDAIEObjectFifoType::get(dstType), op.getDst()
      );
      // rewriter.setInsertionPoint(forallOp);

      rewriter.setInsertionPoint(op);
      // AMDAIE::LogicalObjectFifoFromMemref src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
      //   forallOp.getLoc(), AMDAIEObjectFifoType::get(srcType), op.getSrc()
      // );
      // AMDAIE::LogicalObjectFifoFromMemref dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
      //   forallOp.getLoc(), AMDAIEObjectFifoType::get(dstType), op.getDst()
      // );
      // AMDAIE::DmaCpyNdOp dmaCopy = 
      rewriter.create<AMDAIE::DmaCpyNdOp>(
        op.getLoc(),
        rewriter.getIndexType(), // rewriter.getIndexType(), // SmallVector<Type, 1>{}, // rewriter.getIndexType(),
        dst,
        op.getDstOffsets(),
        op.getDstSizes(),
        op.getDstStrides(),
        src,
        op.getSrcOffsets(),
        op.getSrcSizes(),
        op.getSrcStrides()
      );
      // rewriter.replaceOp(op, dmaCopy);
      rewriter.eraseOp(op);
      return WalkResult::advance();
    });
    return failure();
    return success();
  }
};


/// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class SCFForAllToStructural : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult resolveAffineApply(PatternRewriter &rewriter,
                                   scf::ForallOp forallOp,
                                   DenseMap<Value, OpFoldResult> &symbolTable) const {
    forallOp->walk([&](affine::AffineApplyOp op) {
      SmallVector<Attribute> constOperands(op->getNumOperands());
      for (unsigned i = 0; i < constOperands.size(); ++i)
        constOperands[i] = symbolTable[op->getOperand(i)].get<Attribute>();
      SmallVector<OpFoldResult> foldResults;
      if (failed(op->fold(constOperands, foldResults)) || foldResults.size() != 1) {
        return WalkResult::interrupt();
      }
      symbolTable[op->getResult(0)] = foldResults[0];
      llvm::outs() << op->getResult(0) << "\n";
      std::cout << "AffineApplyOp" << std::endl;
      return WalkResult::advance();
    });
    return success();
  }

  void eraseDmaCpyNds(PatternRewriter &rewriter,
                      scf::ForallOp forallOp) const {
    forallOp->walk([&](AMDAIE::DmaCpyNdOp op) {
      rewriter.eraseOp(op);
      return WalkResult::advance();
    });
  }

  LogicalResult resolveWorkGroup(int workGroupId,
                                 PatternRewriter &rewriter,
                                 scf::ForallOp forallOp,
                                 Block *newBlock,
                                 Block *controlCodeBlock,
                                 Block *controlCodeEndBlock,
                                 xilinx::AIE::TileOp& tileOp,
                                 DenseMap<Value, OpFoldResult> &symbolTable,
                                 DenseMap<Operation *, SmallVector<std::tuple<SmallVector<int64_t>, AMDAIE::DmaCpyNdOp>>>& constructedOps) const {
    std::cout << "resolveWorkGroup" << std::endl;
    rewriter.setInsertionPointToEnd(newBlock);
    // Create AIE core op + block for inserting L1 ops
    auto core = rewriter.create<xilinx::AIE::CoreOp>(rewriter.getUnknownLoc(), tileOp);
    Region &coreRegion = core.getBody();
    auto coreBlock = rewriter.createBlock(&coreRegion);

    // Add core to control code block
    rewriter.setInsertionPointToEnd(controlCodeBlock);
    rewriter.create<AMDAIE::LoadCoreOp>(rewriter.getUnknownLoc(), core);
    rewriter.setInsertionPointToEnd(newBlock);

    // TODO(jornt): we're now using both `constructedOps` and `updatedMemrefs` to figure out broadcasting.
    // We should be able to somplify this logic.
    DenseSet<Operation *> updatedMemrefs;
    Location loc = forallOp.getLoc();
    forallOp->walk([&](Operation *op) {
      if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op)) {
        rewriter.setInsertionPointToEnd(newBlock);
        if (!constructedOps.contains(op)) {
          SmallVector<std::tuple<SmallVector<int64_t>, AMDAIE::DmaCpyNdOp>> values;
          constructedOps[op] = values;
        }
        SmallVector<OpFoldResult> emptyDst(dmaOp.getDstOffsets().size(), rewriter.getI64IntegerAttr(0));
        SmallVector<OpFoldResult> emptySrc(dmaOp.getSrcOffsets().size(), rewriter.getI64IntegerAttr(0));
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          loc,
          rewriter.getIndexType(),
          dmaOp.getDst(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst),
          dmaOp.getSrc(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc)
        );
        llvm::outs() << "AMDAIE::DmaCpyNdOp: " << dmaOp << "\n";
        // Check wether dmaOp operands depend on the induction variables and whether the evaluated values
        // haven't been seen before.
        SmallVector<int64_t> valueResults;
        for (OpOperand &opOperand : dmaOp->getOpOperands()) {
          Value operand = opOperand.get();
          if (symbolTable.contains(operand)) {
            newDmaOp->setOperand(opOperand.getOperandNumber(),
                                 getValueOrCreateConstantIndexOp(rewriter, loc, symbolTable[operand]));
            valueResults.push_back(getConstantIntValue(symbolTable[operand]).value());
          } else {
            newDmaOp->setOperand(opOperand.getOperandNumber(), operand);
          }
        }
        llvm::outs() << "AMDAIE::DmaCpyNdOp after resolve\n";

        // Outputs can't be broadcasted on source. TODO(jornt): can we improve this check?
        auto srcMemSpace = dmaOp.getSrcObjectFifo().getMemrefType().getMemorySpace();
        auto it = std::find_if(
          constructedOps[op].begin(), constructedOps[op].end(), 
          [&](std::tuple<SmallVector<int64_t>, AMDAIE::DmaCpyNdOp> &elem) { return std::get<0>(elem) == valueResults; }
        );
        if (it != constructedOps[op].end() && 
            !updatedMemrefs.contains(dmaOp.getSrcObjectFifo().getMemref().getDefiningOp()) &&
            (!srcMemSpace || dyn_cast<IntegerAttr>(srcMemSpace).getInt() != 2)) {
          // DMA op with broadcast capability, but we still need to add the consumer/producer to the AIE core
          AMDAIE::DmaCpyNdOp broadcastDmaOp = std::get<1>(*it);
          llvm::outs() << "BROADCAST DMA OP: " << broadcastDmaOp << "\n";
          auto srcType = broadcastDmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
          rewriter.setInsertionPointToEnd(coreBlock);
          auto memSpace = srcType.getMemorySpace();
          if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() == 1) {
            // To core
            rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              broadcastDmaOp // newDmaOp.getDst()
            );
          } else if (memSpace && dyn_cast<IntegerAttr>(memSpace).getInt() == 2) {
            // From core
            rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              broadcastDmaOp // newDmaOp.getSrc()
            );
          }
          rewriter.moveOpAfter(core, broadcastDmaOp);
          llvm::outs() << "erase op\n";
          rewriter.eraseOp(newDmaOp);
        } else {
          llvm::outs() << "UNICAST DMA OP: " << newDmaOp << "\n";
          constructedOps[op].push_back(std::make_tuple(valueResults, newDmaOp));
          // In case of unicast, the dst memref is updated and we keep track of this in case another
          // dma op uses it.
          updatedMemrefs.insert(newDmaOp.getDstObjectFifo().getMemref().getDefiningOp());

          auto srcType = newDmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
          auto srcMemSpace = srcType.getMemorySpace();
          auto dstType = newDmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
          auto dstMemSpace = dstType.getMemorySpace();
          if (srcMemSpace && dyn_cast<IntegerAttr>(srcMemSpace).getInt() == 1 &&
              dstMemSpace && dyn_cast<IntegerAttr>(dstMemSpace).getInt() == 2) {
            // rewriter.setInsertionPointToEnd(newBlock);
            // auto produceOp = rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
            //   rewriter.getUnknownLoc(),
            //   SmallVector<Type, 1>{},
            //   newDmaOp // newDmaOp.getSrc()
            // );
            rewriter.setInsertionPointToEnd(coreBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp // newDmaOp.getDst()
            );
            // rewriter.moveOpAfter(core, produceOp);
            // rewriter.moveOpAfter(core, newDmaOp);
          } else if (srcMemSpace && dyn_cast<IntegerAttr>(srcMemSpace).getInt() == 2 &&
                     dstMemSpace && dyn_cast<IntegerAttr>(dstMemSpace).getInt() == 1) {
            // rewriter.setInsertionPointToEnd(newBlock);
            // auto consumeOp = rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
            //   rewriter.getUnknownLoc(),
            //   SmallVector<Type, 1>{},
            //   newDmaOp // newDmaOp.getDst()
            // );
            // rewriter.moveOpAfter(core, consumeOp);
            rewriter.setInsertionPointToEnd(coreBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp // newDmaOp.getSrc()
            );
            
            // TODO(jornt): this feels too hardcoded, we assume here that we're always waiting on
            // L2 consumption. It might be better to wait on every produce/consume op and then later
            // optimize unnecessary ones out.
            // TODO(jornt): commented for now as waits are assumed on L3 side. In general, this is not a good assumption.
            // rewriter.setInsertionPointToEnd(controlCodeEndBlock);
            // rewriter.create<AMDAIE::LogicalObjectFifoWait>(
            //   rewriter.getUnknownLoc(),
            //   SmallVector<Type, 1>{},
            //   newDmaOp.getDst()
            // );
          }
          rewriter.moveOpAfter(core, newDmaOp);
        }
      } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        OpBuilder builder = OpBuilder::atBlockEnd(coreBlock);
        mlir::clone(builder, linalgOp, linalgOp->getResultTypes(), linalgOp->getOperands());
      }
      return WalkResult::advance();
    });

    rewriter.setInsertionPointToEnd(coreBlock);
    rewriter.create<xilinx::AIE::EndOp>(rewriter.getUnknownLoc());
    return success();
  }

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    std::cout << "SCFForAllToStructural" << std::endl;
    if (forallOp.getNumResults() != 0) {
      return failure();
    }
    if (!forallOp.getMapping().has_value()) {
      return failure();
    }
    SmallVector<Attribute> threadMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());
    if (llvm::any_of(threadMapping, [](Attribute map) {
        return !llvm::isa<mlir::gpu::GPUThreadMappingAttr>(map);
      })) {
      std::cout << "Not a thread mapping attr" << std::endl;
      return failure();
    }
    if (threadMapping.size() != 2) {
      return failure();
    }

    // Create AIE region block, AIE func block
    auto regionOp = rewriter.create<AMDAIE::AIERegionOp>(rewriter.getUnknownLoc());
    Region& region = regionOp.getRegion();
    Block *newBlock = rewriter.createBlock(&region);
    // Block *newBlock = rewriter.createBlock(forallOp->getParentRegion());
    rewriter.setInsertionPointToStart(newBlock);
    auto controlCodeOp = rewriter.create<AMDAIE::ControlCodeRegionOp>(rewriter.getUnknownLoc());
    Region& controlCodeRegion = controlCodeOp.getRegion();
    Block *controlCodeBlock = rewriter.createBlock(&controlCodeRegion);
    Block *controlCodeEndBlock = rewriter.createBlock(&controlCodeRegion);

    auto ivs = forallOp.getInductionVars();
    auto lowerBounds = getConstantIntValues(forallOp.getMixedLowerBound()).value();
    auto upperBounds = getConstantIntValues(forallOp.getMixedUpperBound()).value();
    auto steps = getConstantIntValues(forallOp.getMixedStep()).value();

    DenseMap<Value, OpFoldResult> symbolTable;
    DenseMap<Operation *, SmallVector<std::tuple<SmallVector<int64_t>, AMDAIE::DmaCpyNdOp>>> constructedOps;
    int workGroupId = 0;
    for (auto y = lowerBounds[0]; y < upperBounds[0]; y+=steps[0]) {
      for (auto x = lowerBounds[1]; x < upperBounds[1]; x+=steps[1]) {
        std::cout << "y: " << y << std::endl;
        std::cout << "x: " << x << std::endl;
        symbolTable[ivs[0]] = rewriter.getI64IntegerAttr(y);
        symbolTable[ivs[1]] = rewriter.getI64IntegerAttr(x);
        // TODO(jornt): 2 -> AIE core rows start from 2, avoid hardcoding here.
        rewriter.setInsertionPointToStart(newBlock);
        auto tileOp = rewriter.create<xilinx::AIE::TileOp>(rewriter.getUnknownLoc(), x, 2 + y);
        if (failed(resolveAffineApply(rewriter, forallOp, symbolTable))) {
          return failure();
        }
        if (failed(resolveWorkGroup(workGroupId++, rewriter, forallOp, newBlock, controlCodeBlock, controlCodeEndBlock, tileOp, symbolTable, constructedOps))) {
          return failure();
        }
      }
    }
    // rewriter.inlineBlockBefore(newBlock, forallOp); // before aie region was added
    eraseDmaCpyNds(rewriter, forallOp);
    rewriter.eraseOp(forallOp);
    rewriter.setInsertionPointToEnd(newBlock);
    auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
    rewriter.setInsertionPointToEnd(controlCodeEndBlock);
    rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
    rewriter.mergeBlocks(controlCodeEndBlock, controlCodeBlock);
    rewriter.moveOpBefore(controlCodeOp, endOp);

    llvm::outs() << newBlock << "\n";
    return success();
  }
};

LogicalResult cleanupLogicalObjectFifoFromMemref(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  DenseSet<AMDAIE::LogicalObjectFifoFromMemref> toBeErased;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemref op) {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return WalkResult::advance();
    }
    if (toBeErased.contains(op)) {
      // llvm::outs() << "To be erased: " << op << "\n";
      rewriter.eraseOp(op);
      return WalkResult::advance();
    }
    auto memref = op.getMemref();
    llvm::outs() << "---------------------\n";
    llvm::outs() << "Memref: " << memref << "\n";
    for (auto user : memref.getUsers()) {
      if (auto userOp = dyn_cast<LogicalObjectFifoFromMemref>(user)) {
        if (userOp == op || userOp->getParentRegion() != op->getParentRegion()) continue;
        llvm::outs() << "userOp.getOutput(): " << userOp.getOutput() << "\n";
        llvm::outs() << "op.getOutput(): " << op.getOutput() << "\n";
        rewriter.replaceAllUsesWith(userOp.getOutput(), op.getOutput());
        toBeErased.insert(userOp);
        // rewriter.eraseOp(userOp);
      }
    }
    return WalkResult::advance();
  });
  // TODO some still used?? Something wrong here. To be debugged, but ok for now.
  // for (auto op : toBeErased) {
  //   if (op->use_empty())
  //     rewriter.eraseOp(op);
  //   else
  //     llvm::outs() << "Still used: " << op << "\n";
  // }
  return success();
}

LogicalResult distributeLogicalObjectFifos(mlir::ModuleOp moduleOp) {
  auto isBeforeInBlock = [](Operation *a, Operation *b) -> bool {
    return a->isBeforeInBlock(b);
  };
  IRRewriter rewriter(moduleOp.getContext());
  auto res = moduleOp.walk([&](AMDAIE::LogicalObjectFifoFromMemref logicalObjectFifo) {
    // Focus just on L2 for now
    auto memSpace = logicalObjectFifo.getMemrefType().getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
      return WalkResult::advance();
    }
    llvm::outs() << "Walk: " << logicalObjectFifo << "\n";
    // Find regions in which this logicalObjectFifo is used
    DenseSet<AMDAIE::AIERegionOp> regions;
    for (auto userOp : logicalObjectFifo->getUsers()) {
      auto regionOp = userOp->getParentOfType<AMDAIE::AIERegionOp>();
      if (!regionOp) {
        logicalObjectFifo.emitError("used in non-AIERegion op");
      }
      regions.insert(regionOp);
    }
    // Check for parallel channels/dma_cpy_nd to L1 and distribute onto multiple
    // logical objectfifos. 
    // TODO(jornt): Right now, only the case where `dma size == logicalObjectFifo size`,
    // is handled and this will need to be extended.
    // TODO(jornt): simplify this nested logic
    for (auto regionOp : regions) {
      llvm::outs() << "REGION: " << &regionOp << "\n";
      // Sort dma users
      SmallVector<AMDAIE::DmaCpyNdOp> users;
      for (auto userOp : logicalObjectFifo->getUsers()) {
        if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp);
            dmaOp->getParentOfType<AMDAIE::AIERegionOp>() == regionOp) {
          users.push_back(dmaOp);
        }
      }
      llvm::sort(users, isBeforeInBlock);
      // Keep track of producers into this logicalObjectFifo until a consumer is found. TODO(jornt): robust?
      SmallVector<AMDAIE::DmaCpyNdOp> producers;
      for (auto dmaOp : users) {
        // if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp);
        //     dmaOp->getParentOfType<AMDAIE::AIERegionOp>() == regionOp) {
        if (logicalObjectFifo == dmaOp.getDstObjectFifo()) {
          producers.push_back(dmaOp);
        } else if (logicalObjectFifo == dmaOp.getSrcObjectFifo() &&
            logicalObjectFifo.getType().cast<AMDAIEObjectFifoType>().getStaticSize() ==
            dmaOp.getDstObjectFifo().getType().cast<AMDAIEObjectFifoType>().getStaticSize()) {
          llvm::outs() << "--parallel op: " << dmaOp << "\n";
          
          rewriter.setInsertionPoint(logicalObjectFifo);
          auto newAlloc = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(), logicalObjectFifo.getMemrefType());
          llvm::outs() << "--new alloc op: " << newAlloc << "\n";
          auto *parentBlock = newAlloc->getBlock();
          newAlloc->moveBefore(&parentBlock->front());
          auto newLogicalObjectFifo = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
            rewriter.getUnknownLoc(), logicalObjectFifo.getType().cast<AMDAIEObjectFifoType>(), newAlloc.getResult()
          );
          rewriter.setInsertionPoint(dmaOp);
          // TODO(jornt): use setOperand instead?
          auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
            rewriter.getUnknownLoc(),
            rewriter.getIndexType(),
            dmaOp.getDst(),
            dmaOp.getDstOffsets(),
            dmaOp.getDstSizes(),
            dmaOp.getDstStrides(),
            newLogicalObjectFifo,
            dmaOp.getSrcOffsets(),
            dmaOp.getSrcSizes(),
            dmaOp.getSrcStrides()
          );
          rewriter.replaceAllUsesWith(dmaOp, newDmaOp);
          rewriter.eraseOp(dmaOp);

          for (auto &producerDmaOp : producers) {
            rewriter.setInsertionPoint(producerDmaOp);
            // TODO(jornt): use setOperand instead?
            auto newProducerDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
              rewriter.getUnknownLoc(),
              rewriter.getIndexType(),
              newLogicalObjectFifo,
              producerDmaOp.getDstOffsets(),
              producerDmaOp.getDstSizes(),
              producerDmaOp.getDstStrides(),
              producerDmaOp.getSrc(),
              producerDmaOp.getSrcOffsets(),
              producerDmaOp.getSrcSizes(),
              producerDmaOp.getSrcStrides()
            );
            rewriter.replaceAllUsesWith(producerDmaOp, newProducerDmaOp);
            rewriter.eraseOp(producerDmaOp);
          }
          producers.clear();

          rewriter.setInsertionPoint(logicalObjectFifo);
          auto dealloc = rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(), newAlloc);
          dealloc->moveBefore(&parentBlock->back());
        }
        // }   
      }
    }
    // TODO(jornt): not sure why this is necessary?
    // logicalObjectFifo->dropAllUses();
    // llvm::outs() << "--uses: " << logicalObjectFifo->use_empty() << "\n";
    // if (!logicalObjectFifo->use_empty()) {
    //   for (auto user : logicalObjectFifo->getUsers())
    //     llvm::outs() << "--user: " << user << "\n";
    // }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return failure();
  return success();
}

LogicalResult hoistStaticallyBoundAllocationsInModule(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    hoistStaticallyBoundAllocationsInFunc<memref::AllocaOp>(rewriter, funcOp);
    hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(rewriter, funcOp);
  }
  return success();
}

class AMDAIEToObjectfifoPass
    : public impl::AMDAIEToObjectfifoBase<AMDAIEToObjectfifoPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::air::airDialect, xilinx::AIE::AIEDialect, AMDAIEDialect>();
  }

  AMDAIEToObjectfifoPass() = default;
  AMDAIEToObjectfifoPass(const AMDAIEToObjectfifoPass &pass){};
  void runOnOperation() override;
};

void AMDAIEToObjectfifoPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SCFForAllToLogicalObjectfifo>(context);
  patterns.insert<SCFForAllToStructural>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
  if (failed(cleanupLogicalObjectFifoFromMemref(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(distributeLogicalObjectFifos(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(cleanupLogicalObjectFifoFromMemref(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(hoistStaticallyBoundAllocationsInModule(getOperation()))) {
    return signalPassFailure();
  }
}

class AMDAIEDmaToObjectfifoPass
    : public impl::AMDAIEDmaToObjectfifoBase<AMDAIEDmaToObjectfifoPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::air::airDialect, xilinx::AIE::AIEDialect, AMDAIEDialect>();
  }

  AMDAIEDmaToObjectfifoPass() = default;
  AMDAIEDmaToObjectfifoPass(const AMDAIEDmaToObjectfifoPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaToObjectfifoPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<DmaMemcpyNdToLogicalObjectfifo>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}


//// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class LogicalObjectfifoFromMemrefCleanup : public OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemref> {
  using OpRewritePattern<AMDAIE::LogicalObjectFifoFromMemref>::OpRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::LogicalObjectFifoFromMemref op,
                                PatternRewriter &rewriter) const override {
    auto memref = op.getMemref();
    for (Operation *user : memref.getUsers()) {
      if (auto userOp = dyn_cast<LogicalObjectFifoFromMemref>(user)) {
        if (userOp == op || userOp->getParentRegion() != op->getParentRegion()) break;
        llvm::outs() << "userOp.getOutput(): " << userOp.getOutput() << "\n";
        llvm::outs() << "op.getOutput(): " << op.getOutput() << "\n";
        rewriter.replaceAllUsesWith(userOp.getOutput(), op.getOutput());
        rewriter.eraseOp(userOp);
      }
    }
    return success();
  }
};

class AMDAIELogicalObjectfifoFromMemrefCleanupPass
    : public impl::AMDAIELogicalObjectfifoFromMemrefCleanupBase<AMDAIELogicalObjectfifoFromMemrefCleanupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIELogicalObjectfifoFromMemrefCleanupPass() = default;
  AMDAIELogicalObjectfifoFromMemrefCleanupPass(const AMDAIELogicalObjectfifoFromMemrefCleanupPass &pass){};
  void runOnOperation() override;
};

void AMDAIELogicalObjectfifoFromMemrefCleanupPass::runOnOperation() {
  if (failed(cleanupLogicalObjectFifoFromMemref(getOperation()))) {
    return signalPassFailure();
  }
}


}  // namespace

std::unique_ptr<Pass> createAMDAIEToObjectfifoPass() {
  return std::make_unique<AMDAIEToObjectfifoPass>();
}

std::unique_ptr<Pass> createAMDAIEDmaToObjectfifoPass() {
  return std::make_unique<AMDAIEDmaToObjectfifoPass>();
}

std::unique_ptr<Pass> createAMDAIELogicalObjectfifoFromMemrefCleanupPass() {
  return std::make_unique<AMDAIELogicalObjectfifoFromMemrefCleanupPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
