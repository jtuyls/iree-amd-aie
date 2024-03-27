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

#define DEBUG_TYPE "iree-amdaie-fuse-aie-regions"

using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult assignAieTiles(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](AMDAIE::LogicalObjectFifoFromMemref logicalObjectFifo) {
    Block *aieRegionBlock = &logicalObjectFifo->getParentOfType<AMDAIE::AIERegionOp>().getRegion().front();
    rewriter.setInsertionPointToStart(aieRegionBlock);
    auto memSpace = logicalObjectFifo.getMemrefType().getMemorySpace();
    // Very hardcoded for now to (0, 0) and (0, 1). Needs generalization.
    // xilinx::AIE::TileOp tileOp;
    SmallVector<OpFoldResult> tileResults;
    if (!memSpace) {
      tileResults.push_back(rewriter.create<xilinx::AIE::TileOp>(rewriter.getUnknownLoc(), 0, 0).getResult());
    } else if (dyn_cast<IntegerAttr>(memSpace).getInt() == 1) {
      tileResults.push_back(rewriter.create<xilinx::AIE::TileOp>(rewriter.getUnknownLoc(), 0, 1).getResult());
    } else if (dyn_cast<IntegerAttr>(memSpace).getInt() == 2) {
      // Add core tiles that use this logical objectfifo
      for (auto userOp : logicalObjectFifo->getUsers()) {
        if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp)) {
          for (auto dmaUserOp : dmaOp->getUsers()) {
            if (auto coreParentOp = dmaUserOp->getParentOfType<xilinx::AIE::CoreOp>()) {
              if (llvm::find(tileResults, (OpFoldResult) coreParentOp.getTileOp().getResult()) == tileResults.end()) {
                // llvm::outs() << "coreParentOp: " << coreParentOp << "\n";
                tileResults.push_back(coreParentOp.getTileOp().getResult());
                rewriter.moveOpAfter(coreParentOp.getTileOp(), aieRegionBlock, aieRegionBlock->begin());
              } 
            }
          }
        }
      }
    } else {
      logicalObjectFifo.emitError("found logical objectfifo with unknown memory space");
      return WalkResult::interrupt();
    }
    if (!tileResults.empty()) {
      // TODO(jornt): can we just update the current LogicalObjectFifoFromMemref?
      // SmallVector<OpFoldResult> empty = {tileOp.getResult()};
      rewriter.setInsertionPoint(logicalObjectFifo);
      auto newLogicalObjectFifo = rewriter.create<AMDAIE::LogicalObjectFifoFromMemref>(
        rewriter.getUnknownLoc(),
        logicalObjectFifo.getOutput().getType().cast<AMDAIEObjectFifoType>(),
        logicalObjectFifo.getMemref(),
        getValueOrCreateConstantIndexOp(rewriter, rewriter.getUnknownLoc(), tileResults)
      );
      rewriter.replaceAllUsesWith(logicalObjectFifo, newLogicalObjectFifo);
      rewriter.eraseOp(logicalObjectFifo);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

Operation *getParentOpInBlock(Block* block, Operation *op) {
  if (!op || op->getBlock() == block)
    return op;
  auto parentOp = op->getParentOp();
  return getParentOpInBlock(block, parentOp);
}

LogicalResult consumeToAcquireRelease(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](xilinx::AIE::CoreOp coreOp) {
    DenseMap<AMDAIE::DmaCpyNdOp, AMDAIE::LogicalObjectFifoConsume> dmaToConsumerMap;
    coreOp->walk([&](AMDAIE::LogicalObjectFifoConsume op) {
      llvm::outs() << "LogicalObjectFifoConsume: " << op << "\n";
      rewriter.setInsertionPoint(op);
      rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op.getDma().getDefiningOp());
      if (!dmaToConsumerMap.contains(dmaOp)) {
        dmaToConsumerMap[dmaOp] = op;
        return WalkResult::advance();
      }
      auto parentOpInBlock = getParentOpInBlock(dmaToConsumerMap[dmaOp]->getBlock(), op);
      if (parentOpInBlock) {
        rewriter.setInsertionPoint(parentOpInBlock);
      } else {
        rewriter.setInsertionPoint(dmaToConsumerMap[dmaOp]->getBlock()->getTerminator());
      }
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      rewriter.eraseOp(dmaToConsumerMap[dmaOp]);
      dmaToConsumerMap[dmaOp] = op;
      return WalkResult::advance();
    });
    // Add release for remaining LogicalObjectFifoConsume ops
    for (auto &elem : dmaToConsumerMap) {
      auto op = elem.second;
      rewriter.setInsertionPoint(op->getBlock()->getTerminator());
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      rewriter.eraseOp(op);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult produceToAcquireRelease(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](xilinx::AIE::CoreOp coreOp) {
    DenseMap<AMDAIE::DmaCpyNdOp, AMDAIE::LogicalObjectFifoProduce> dmaToProducerMap;
    coreOp->walk<WalkOrder::PostOrder, mlir::ReverseIterator>([&](AMDAIE::LogicalObjectFifoProduce op) {
      llvm::outs() << "LogicalObjectFifoProduce: " << op << "\n";
      rewriter.setInsertionPoint(op);
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op.getDma().getDefiningOp());
      if (!dmaToProducerMap.contains(dmaOp)) {
        dmaToProducerMap[dmaOp] = op;
        return WalkResult::advance();
      }
      auto parentOpInBlock = getParentOpInBlock(dmaToProducerMap[dmaOp]->getBlock(), op);
      if (parentOpInBlock) {
        rewriter.setInsertionPointAfter(parentOpInBlock);
      } else {
        rewriter.setInsertionPointToStart(dmaToProducerMap[dmaOp]->getBlock());
      }
      rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      rewriter.eraseOp(dmaToProducerMap[dmaOp]);
      dmaToProducerMap[dmaOp] = op;
      return WalkResult::advance();
    });
    // Add release for remaining LogicalObjectFifoConsume ops
    for (auto &elem : dmaToProducerMap) {
      auto op = elem.second;
      rewriter.setInsertionPointToStart(op->getBlock());
      rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
        rewriter.getUnknownLoc(),
        SmallVector<Type, 1>{},
        op.getDma(),
        op.getPort()
      );
      rewriter.eraseOp(op);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult addExplicitLogicalObjectfifoLinks(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto walkResult = moduleOp.walk([&](AMDAIE::LogicalObjectFifoFromMemref logicalObjectFifo) {
    // Only consider L2/MT for links as L1/L3 don't need this linking through AIE objectfifos.
    auto memSpace = logicalObjectFifo.getOutput().getType().cast<AMDAIEObjectFifoType>().getElementType().getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
      llvm::outs() << "not memspace: " << memSpace << "\n";
      return WalkResult::advance();
    }
    if (!isa<AMDAIE::AIERegionOp>(logicalObjectFifo->getParentOp())) {
      logicalObjectFifo->emitError("does not have an AIE region as parent op, which is needed for this pass to behave correctly");
      return WalkResult::interrupt();
    }
    SmallVector<OpFoldResult> ins;
    SmallVector<OpFoldResult> outs;
    AMDAIE::DmaCpyNdOp lastUserOp;
    for (auto userOp : logicalObjectFifo->getUsers()) {
      if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(userOp)) {
        if (!isa<AMDAIE::AIERegionOp>(userOp->getParentOp())) {
          dmaOp->emitError("does not have an AIE region as parent op");
          return WalkResult::interrupt();
        }
        if (!lastUserOp || lastUserOp->isBeforeInBlock(dmaOp)) {
          lastUserOp = dmaOp;
        }
        if (logicalObjectFifo == dmaOp.getSrcObjectFifo()) {
          outs.push_back(dmaOp->getResult(0));
        } else {
          ins.push_back(dmaOp->getResult(0));
        }
      }
    }
    // If used in DmaCpyNd ops
    if (lastUserOp) {
      rewriter.setInsertionPointAfter(lastUserOp);
      rewriter.create<AMDAIE::LogicalObjectFifoLink>(
        rewriter.getUnknownLoc(),
        getValueOrCreateConstantIndexOp(rewriter, rewriter.getUnknownLoc(), ins),
        getValueOrCreateConstantIndexOp(rewriter, rewriter.getUnknownLoc(), outs)
      );
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

AIE::ObjectFifoCreateOp createObjectFifo(IRRewriter& rewriter,
                                         ValueRange srcTiles,
                                         ValueRange dstTiles,
                                         AMDAIE::DmaCpyNdOp& dmaOp,
                                         StringAttr &symName) {
  auto srcTile = srcTiles[0];
  // TODO: offets
  SmallVector<AIE::BDDimLayoutAttr> srcBDDimLayoutAttrs;
  for (auto [size, stride] : llvm::zip(dmaOp.getSrcSizes(), dmaOp.getSrcStrides())) {
    srcBDDimLayoutAttrs.push_back(
      AIE::BDDimLayoutAttr::get(
        rewriter.getContext(),
        getConstantIntValue(size).value(),
        getConstantIntValue(stride).value()
      )
    );
  }
  SmallVector<AIE::BDDimLayoutAttr> dstBDDimLayoutAttrs;
  for (auto [size, stride] : llvm::zip(dmaOp.getDstSizes(), dmaOp.getDstStrides())) {
    dstBDDimLayoutAttrs.push_back(
      AIE::BDDimLayoutAttr::get(
        rewriter.getContext(),
        getConstantIntValue(size).value(),
        getConstantIntValue(stride).value()
      )
    );
  }
  AIE::BDDimLayoutArrayAttr srcDims =
    AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(), ArrayRef(srcBDDimLayoutAttrs));
  SmallVector<AIE::BDDimLayoutArrayAttr> dstDimsVec;
  dstDimsVec.push_back(AIE::BDDimLayoutArrayAttr::get(rewriter.getContext(), ArrayRef(dstBDDimLayoutAttrs)));
  AIE::BDDimLayoutArrayArrayAttr dstDims = 
    AIE::BDDimLayoutArrayArrayAttr::get(rewriter.getContext(), ArrayRef(dstDimsVec));
  llvm::outs() << "srcDims: " << srcDims << "\n";
  llvm::outs() << "dstDims: " << dstDims << "\n";
  // For now, set datatype to the one with the lowest rank
  // TODO(jornt): I think objectfifos should support source type != dest type
  auto srcType = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
  auto dstType = dmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
  AIE::AIEObjectFifoType dtype = dstType.getRank() <= srcType.getRank() ?
    AIE::AIEObjectFifoType::get(dstType) : AIE::AIEObjectFifoType::get(srcType);
  llvm::outs() << "dtype: " << dtype << "\n";
  auto depth = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType().getElementTypeBitWidth() / 8;
  AIE::ObjectFifoCreateOp fifo = rewriter.create<AIE::ObjectFifoCreateOp>(
    rewriter.getUnknownLoc(),
    symName,
    srcTile,
    dstTiles,
    rewriter.getIntegerAttr(rewriter.getI32Type(), depth),
    dtype,
    srcDims,
    dstDims
  );
  return fifo;
}

LogicalResult coreToObjectFifo(IRRewriter &rewriter,
                               AIE::CoreOp &coreOp,
                               DenseMap<AMDAIE::DmaCpyNdOp, AIE::ObjectFifoCreateOp> &dmaObjFifoMap) {
  DenseMap<Operation *, Operation *> memrefMap;
  auto walkResult = coreOp.walk([&](Operation * op) {
    rewriter.setInsertionPoint(op);
    if (auto acquireOp = dyn_cast<AMDAIE::LogicalObjectFifoAcquire>(op)) {
      llvm::outs() << "acquireOp: " << acquireOp << "\n";
      auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(acquireOp.getDma().getDefiningOp());
      auto srcType = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
      srcType = MemRefType::Builder(srcType).setMemorySpace(rewriter.getI64IntegerAttr(1));
      auto dstType = dmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
      dstType = MemRefType::Builder(dstType).setMemorySpace(rewriter.getI64IntegerAttr(1));
      auto objFifo = dmaObjFifoMap[dmaOp];
      AIE::AIEObjectFifoType ofTy =
        cast<AIE::AIEObjectFifoType>(objFifo.getElemType());
      auto elementType = ofTy.getElementType();
      // elementType.setMemorySpace(2);
      // auto memrefType = 
      elementType = MemRefType::Builder(elementType).setMemorySpace(rewriter.getI64IntegerAttr(1));
      llvm::outs() << "Element type: " << elementType << "\n";
      auto subviewType = AIE::AIEObjectFifoSubviewType::get(elementType);
      AIE::ObjectFifoPort port = acquireOp.getPort() == ObjectFifoPort::Produce
        ? AIE::ObjectFifoPort::Produce : AIE::ObjectFifoPort::Consume;
      auto objFifoAquireOp = rewriter.create<AIE::ObjectFifoAcquireOp>(
        rewriter.getUnknownLoc(), subviewType, port, objFifo.getName(), 1);
      auto subview = rewriter.create<AIE::ObjectFifoSubviewAccessOp>(
        rewriter.getUnknownLoc(), elementType, objFifoAquireOp.getSubview(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
      
      auto memSpace = dstType.getMemorySpace();
      if (!memSpace) {
        dmaOp.emitError("no memspace for dma op used in CoreOp is not supported");
        return WalkResult::interrupt();
      }
      if (dyn_cast<IntegerAttr>(memSpace).getInt() == 2) {
        auto dstMemref = dmaOp.getDstObjectFifo().getMemref().getDefiningOp();
        auto sizes = dstType.getShape();
        auto [strides, baseOffset] = getStridesAndOffset(dstType);
        auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
          rewriter.getUnknownLoc(), dstType, subview.getOutput(), baseOffset, sizes, strides
        );
        memrefMap[dstMemref] = reinterpretOp;
      } else if (dyn_cast<IntegerAttr>(memSpace).getInt() == 1) {
        auto srcMemref = dmaOp.getSrcObjectFifo().getMemref().getDefiningOp();
        auto sizes = srcType.getShape();
        auto [strides, baseOffset] = getStridesAndOffset(srcType);
        auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
          rewriter.getUnknownLoc(), srcType, subview.getOutput(), baseOffset, sizes, strides
        );
        memrefMap[srcMemref] = reinterpretOp;
      }
      rewriter.eraseOp(acquireOp);
    } else if (auto releaseOp = dyn_cast<AMDAIE::LogicalObjectFifoRelease>(op)) {
      llvm::outs() << "releaseOp: " << releaseOp << "\n";
      auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(releaseOp.getDma().getDefiningOp());
      auto objFifo = dmaObjFifoMap[dmaOp];
      AIE::ObjectFifoPort port = releaseOp.getPort() == ObjectFifoPort::Produce
        ? AIE::ObjectFifoPort::Produce : AIE::ObjectFifoPort::Consume;
      rewriter.create<AIE::ObjectFifoReleaseOp>(
        rewriter.getUnknownLoc(), port, objFifo.getName(), 1);
      rewriter.eraseOp(releaseOp);
    } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // llvm::outs() << "Linalg op: \n";
      for (int i = 0; i < linalgOp->getNumOperands(); ++i) {
        auto operand = linalgOp->getOperand(i);
        // llvm::outs() << "--operand: " << operand << "\n";
        // llvm::outs() << "--contains: " << memrefMap.contains(operand.getDefiningOp()) << "\n";
        if (memrefMap.contains(operand.getDefiningOp())) {
          linalgOp->setOperand(i, memrefMap[operand.getDefiningOp()]->getResult(0));
        }
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult toObjectFifo(mlir::ModuleOp moduleOp) {
  llvm::outs() << "ModuleOp: " << moduleOp << "\n";
  IRRewriter rewriter(moduleOp.getContext());
  Block *moduleBlock = &moduleOp->getRegion(0).front();

  auto walkResult = moduleOp.walk([&](AMDAIE::AIERegionOp regionOp) {
    // Block *regionBlock = &regionOp.getRegion().front();
    // Find control code op within region
    AMDAIE::ControlCodeRegionOp controlCodeOp;
    regionOp->walk([&](AMDAIE::ControlCodeRegionOp op) {
      controlCodeOp = op;
      return WalkResult::interrupt();
    });

    // Insert AIE DeviceOp 
    rewriter.setInsertionPoint(moduleBlock, moduleBlock->begin());
    auto deviceOp = rewriter.create<xilinx::AIE::DeviceOp>(
      rewriter.getUnknownLoc(),
      xilinx::AIE::AIEDeviceAttr::get(rewriter.getContext(), xilinx::AIE::AIEDevice::ipu)
    );
    deviceOp.getRegion().emplaceBlock();
    Block *deviceBlock = &deviceOp.getRegion().front();
    rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

    DenseMap<AMDAIE::DmaCpyNdOp, AIE::ObjectFifoCreateOp> dmaSymMap;
    int dmaId = 0;
    regionOp.walk([&](Operation *op) {
      if (auto tileOp = dyn_cast<AIE::TileOp>(op)) {
        rewriter.moveOpBefore(tileOp, deviceBlock, deviceBlock->begin());
      } else if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op)) {
        // TODO(jornt): refactor
        SmallVector<Value> srcTiles;
        auto srcMemSpace = dmaOp.getSrcObjectFifo().getMemrefType().getMemorySpace();
        if (!srcMemSpace || dyn_cast<IntegerAttr>(srcMemSpace).getInt() != 2) {
          srcTiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getSrc().getDefiningOp()).getTiles();
        } else {
          for (auto userOp : dmaOp->getUsers()) {
            if (auto coreOp = userOp->getParentOfType<xilinx::AIE::CoreOp>()) {
              srcTiles.push_back(coreOp.getTileOp().getResult());
            }
          }
        }
        SmallVector<Value> dstTiles;
        auto dstMemSpace = dmaOp.getDstObjectFifo().getMemrefType().getMemorySpace();
        if (!dstMemSpace || dyn_cast<IntegerAttr>(dstMemSpace).getInt() != 2) {
          dstTiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getDst().getDefiningOp()).getTiles();
        } else {
          for (auto userOp : dmaOp->getUsers()) {
            if (auto coreOp = userOp->getParentOfType<xilinx::AIE::CoreOp>()) {
              dstTiles.push_back(coreOp.getTileOp().getResult());
            }
          }
        }
        // auto dstTiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getDst().getDefiningOp()).getTiles();
        auto symName = "in" + std::to_string(dmaId++);
        auto symAttr = rewriter.getStringAttr(symName);
        auto fifo = createObjectFifo(rewriter, srcTiles, dstTiles, dmaOp, symAttr);
        dmaSymMap[dmaOp] = fifo;
        llvm::outs() << "fifo: " << fifo << "\n";
        // rewriter.eraseOp(dmaOp);
      } else if (auto linkOp = dyn_cast<AMDAIE::LogicalObjectFifoLink>(op)) {
        SmallVector<Attribute> inSyms;
        for (auto in : linkOp.getIns()) {
          auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(in.getDefiningOp());
          inSyms.push_back(SymbolRefAttr::get(rewriter.getContext(), dmaSymMap[dmaOp].getSymName()));
          // inSyms.push_back(dmaSymMap[dmaOp].getSymName());
        }
        SmallVector<Attribute> outSyms;
        for (auto out : linkOp.getOuts()) {
          auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(out.getDefiningOp());
          outSyms.push_back(SymbolRefAttr::get(rewriter.getContext(), dmaSymMap[dmaOp].getSymName()));
          // outSyms.push_back(dmaSymMap[dmaOp].getSymName());
        }
        auto objectFifoLinkOp = rewriter.create<AIE::ObjectFifoLinkOp>(
          rewriter.getUnknownLoc(),
          rewriter.getArrayAttr(inSyms),
          rewriter.getArrayAttr(outSyms)
        );
        llvm::outs() << "fifo link op: " << objectFifoLinkOp << "\n";
        rewriter.eraseOp(linkOp);
      } else if (auto coreOp = dyn_cast<AIE::CoreOp>(op)) {
        rewriter.moveOpBefore(coreOp, deviceBlock, deviceBlock->end());
        if (failed(coreToObjectFifo(rewriter, coreOp, dmaSymMap))) {
          coreOp.emitError("could not convert to AIE ops");
          return WalkResult::interrupt();
        }
        
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}


class AMDAIEToAIEPass
    : public impl::AMDAIEToAIEBase<AMDAIEToAIEPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect>();
  }

  AMDAIEToAIEPass() = default;
  AMDAIEToAIEPass(const AMDAIEToAIEPass &pass){};
  void runOnOperation() override;
};

void AMDAIEToAIEPass::runOnOperation() {
  // MLIRContext *context = &getContext();
  // RewritePatternSet patterns(context);
  // patterns.insert<ConsumeToAcquireRelease>(context);

  // if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
  //   return signalPassFailure();
  // }
  if (failed(consumeToAcquireRelease(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(produceToAcquireRelease(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(addExplicitLogicalObjectfifoLinks(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(assignAieTiles(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(toObjectFifo(getOperation()))) {
    return signalPassFailure();
  }
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






}  // namespace

std::unique_ptr<Pass> createAMDAIEToAIEPass() {
  return std::make_unique<AMDAIEToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
