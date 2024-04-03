// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MathExtras.h"
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
  // llvm::outs() << "srcDims: " << srcDims << "\n";
  // llvm::outs() << "dstDims: " << dstDims << "\n";
  // For now, set datatype to the one with the lowest rank
  // TODO(jornt): I think objectfifos should support source type != dest type
  auto srcType = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
  auto dstType = dmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
  AIE::AIEObjectFifoType dtype = dstType.getRank() <= srcType.getRank() ?
    AIE::AIEObjectFifoType::get(dstType) : AIE::AIEObjectFifoType::get(srcType);
  // llvm::outs() << "dtype: " << dtype << "\n";
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
                               AIE::DeviceOp &deviceOp,
                               DenseMap<AMDAIE::DmaCpyNdOp, AIE::ObjectFifoCreateOp> &dmaObjFifoMap) {
  // Block *deviceBlock = &deviceOp.getRegion().front();
  Block *coreBlock = &coreOp.getBody().front();
  auto addConstants = [&](Operation *op) -> void {
    for (int i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      if (!operand || !operand.getDefiningOp()) {
        continue;
      }
      if (auto constantOp = dyn_cast<arith::ConstantOp>(operand.getDefiningOp())) {
        if (!constantOp->getParentOfType<AIE::CoreOp>()) {
          llvm::outs() << "constantOp: " << constantOp << "\n";
          rewriter.setInsertionPointToStart(coreBlock);
          auto newOp = mlir::clone(rewriter, constantOp, constantOp->getResultTypes(), constantOp->getOperands());
          op->setOperand(i, newOp->getResult(0));
        } 
      }
    }
  };
  DenseMap<Operation *, Operation *> memrefMap;
  auto walkResult = coreOp.walk([&](Operation * op) {
    rewriter.setInsertionPoint(op);
    if (auto acquireOp = dyn_cast<AMDAIE::LogicalObjectFifoAcquire>(op)) {
      llvm::outs() << "acquireOp: " << acquireOp << "\n";
      auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(acquireOp.getDma().getDefiningOp());
      auto srcType = dmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
      srcType = MemRefType::Builder(srcType).setMemorySpace(rewriter.getI64IntegerAttr(1));
      auto dstType = dmaOp.getDstType().cast<AMDAIEObjectFifoType>().getElementType();
      // TODO refactor to avoid memory space overwrite issues
      auto memSpace = dstType.getMemorySpace();
      dstType = MemRefType::Builder(dstType).setMemorySpace(rewriter.getI64IntegerAttr(1));
      auto objFifo = dmaObjFifoMap[dmaOp];
      AIE::AIEObjectFifoType ofTy =
        cast<AIE::AIEObjectFifoType>(objFifo.getElemType());
      auto elementType = ofTy.getElementType();
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
      for (int i = 0; i < linalgOp->getNumOperands(); ++i) {
        auto operand = linalgOp->getOperand(i);
        // llvm::outs() << "--operand: " << operand << "\n";
        // llvm::outs() << "--contains: " << memrefMap.contains(operand.getDefiningOp()) << "\n";
        if (memrefMap.contains(operand.getDefiningOp())) {
          linalgOp->setOperand(i, memrefMap[operand.getDefiningOp()]->getResult(0));
        }
      }
      addConstants(linalgOp);
    } else {
      addConstants(op);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

// void toStaticAddressing(SmallVector<Value> offsets,
//                         SmallVector<Value> sizes,
//                         SmallVector<Value> strides,
//                         SmallVector<int64_t> &staticOffsets,
//                         SmallVector<int64_t> &staticSizes,
//                         SmallVector<int64_t> &staticStrides) {
         
// }

LogicalResult controlCodeToAie(IRRewriter &rewriter,
                               AMDAIE::ControlCodeRegionOp &controlCodeOp,
                               func::FuncOp &funcOp,
                               DenseMap<Value, Value> &argMap,
                               DenseMap<AMDAIE::DmaCpyNdOp, AIE::ObjectFifoCreateOp> &dmaObjFifoMap) {
  Block *funcBlock = &funcOp.getBody().front();
  Block *controlCodeBlock = &controlCodeOp.getRegion().front();

  // Utility to add constants needed to the function
  auto addConstants = [&](Operation *op) -> void {
    for (int i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      if (!operand || !operand.getDefiningOp()) {
        continue;
      }
      if (auto constantOp = dyn_cast<arith::ConstantOp>(operand.getDefiningOp())) {
        if (!constantOp->getParentOfType<AMDAIE::ControlCodeRegionOp>()) {
          llvm::outs() << "constantOp: " << constantOp << "\n";
          // llvm::outs() << "hasOperandStorage: " << constantOp->hasOperandStorage() << "\n";
          rewriter.setInsertionPointToStart(controlCodeBlock);
          // llvm::outs() << "Before clone\n";
          auto newOp = mlir::clone(rewriter, constantOp, constantOp->getResultTypes(), constantOp->getOperands());
          // llvm::outs() << "Before setOperand\n";
          op->setOperand(i, newOp->getResult(0));
        } 
      }
    }
  };

  SmallVector<Operation *> toBeErased;
  auto res = controlCodeOp->walk([&](Operation *op) {
    if (auto loadCoreOp = dyn_cast<AMDAIE::LoadCoreOp>(op)) {
      // In MLIR-AIE, the cores are initialized implicitly before ipu instruction execution
      // starts, so we don't need to convert this into anything for now.
      // TODO(@jornt): We should however add checks that all load ops are at the start of the
      // control code region to make sure we don't convert IR that is invalid right now. More 
      // longer term, we should maybe add support for load core instructions in the backend.
      rewriter.eraseOp(loadCoreOp);
      return WalkResult::advance();
    } else if (auto dmaOp = dyn_cast<AMDAIE::IpuDmaCpyNdOp>(op)) {
      // Convert bidirectional dma copy nd op into two halves
      rewriter.setInsertionPoint(dmaOp);
      if (dmaOp.hasSrcAddressing()) {
        SmallVector<Value> empty;
        SmallVector<Value> srcOffsets = dmaOp.getSrcOffsets();
        SmallVector<Value> srcSizes = dmaOp.getSrcSizes();
        SmallVector<Value> srcStrides = dmaOp.getSrcStrides();
        // Current IpuDmaMemcpyNdOp assumption/requirement
        if (getConstantIntValue(srcStrides[srcStrides.size() - 1]).value() != 1) {
          dmaOp.emitError("invalid last stride, should be 1");
          return WalkResult::interrupt();
        }                 
        // For constants, see IpuDmaMemcpyNdOp size requirements
        SmallVector<int64_t, 4> staticOffsets(4, 1);
        SmallVector<int64_t, 4> staticSizes(4, 1);
        SmallVector<int64_t, 3> staticStrides(3, 1);
        for (int i = 0; i < srcOffsets.size(); ++i)
          staticOffsets[4 - srcOffsets.size() + i] = getConstantIntValue(srcOffsets[i]).value();
        for (int i = 0; i < srcSizes.size(); ++i)
          staticSizes[4 - srcSizes.size() + i] = getConstantIntValue(srcSizes[i]).value();
        for (int i = 0; i < srcStrides.size() - 1; ++i)
          staticStrides[3 - srcStrides.size() + i] = getConstantIntValue(srcStrides[i]).value();
        auto dmaCpyNd = dmaOp.getDmaCpyNdOp();
        // TODO, not always there
        auto memref = argMap[dmaCpyNd.getSrcObjectFifo().getMemref()];
        auto symbol = dmaObjFifoMap[dmaCpyNd].getName();
        // TODO(jornt): bd_id != 0
        rewriter.create<AIEX::IpuDmaMemcpyNdOp>(
          rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, 0, 0, memref,
          empty, empty, empty, staticOffsets, staticSizes, staticStrides, symbol, 0);
      }
      if (dmaOp.hasDstAddressing()) {
        llvm::outs() << "dmaOp.hasDstAddressing(): " << dmaOp << "\n";
        SmallVector<Value> empty;
        SmallVector<Value> dstOffsets = dmaOp.getDstOffsets();
        SmallVector<Value> dstSizes = dmaOp.getDstSizes();
        SmallVector<Value> dstStrides = dmaOp.getDstStrides();
        // Current IpuDmaMemcpyNdOp assumption/requirement
        if (getConstantIntValue(dstStrides[dstStrides.size() - 1]).value() != 1) {
          dmaOp.emitError("invalid last stride, should be 1");
          return WalkResult::interrupt();
        }                 
        // For constants, see IpuDmaMemcpyNdOp size requirements
        SmallVector<int64_t, 4> staticOffsets(4, 1);
        SmallVector<int64_t, 4> staticSizes(4, 1);
        SmallVector<int64_t, 3> staticStrides(3, 1);
        for (int i = 0; i < dstOffsets.size(); ++i)
          staticOffsets[4 - dstOffsets.size() + i] = getConstantIntValue(dstOffsets[i]).value();
        for (int i = 0; i < dstSizes.size(); ++i)
          staticSizes[4 - dstSizes.size() + i] = getConstantIntValue(dstSizes[i]).value();
        for (int i = 0; i < dstStrides.size() - 1; ++i)
          staticStrides[3 - dstStrides.size() + i] = getConstantIntValue(dstStrides[i]).value();
        auto dmaCpyNd = dmaOp.getDmaCpyNdOp();
        // TODO, not always there
        auto memref = argMap[dmaCpyNd.getDstObjectFifo().getMemref()];
        auto symbol = dmaObjFifoMap[dmaCpyNd].getName();
        llvm::outs() << "Memref key: " << dmaCpyNd.getDstObjectFifo().getMemref() << "\n";
        llvm::outs() << "Memref: " << memref << "\n";
        // TODO(jornt): bd_id != 0
        rewriter.create<AIEX::IpuDmaMemcpyNdOp>(
          rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, 0, 0, memref,
          empty, empty, empty, staticOffsets, staticSizes, staticStrides, symbol, 0);
      }
      // TODO(jornt): can we avoid this inplace rewrites with issues if directly erasing and still used
      toBeErased.push_back(dmaOp);
      // dmaOp->dropAllUses();
      // rewriter.eraseOp(dmaOp);
    } else if (auto waitOp = dyn_cast<AMDAIE::IpuDmaWaitOp>(op)) {
      rewriter.setInsertionPoint(waitOp);
      auto objFifo = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(waitOp.getObjectfifo().getDefiningOp());
      SmallVector<Value> tileResults = objFifo.getTiles();
      if (tileResults.size() != 1) {
        waitOp.emitError("expected 1 tile op for wait");
        return WalkResult::interrupt();
      }
      auto tile = dyn_cast<AIE::TileOp>(tileResults[0].getDefiningOp());
      auto col = rewriter.getI32IntegerAttr(tile.getCol());
      auto row = rewriter.getI32IntegerAttr(tile.getRow());
      // auto dir = rewriter.getI32IntegerAttr(0); // TODO derive from source/destination
      // auto dir = dyn_cast<IntegerAttr>(waitOp.getDirection());
      llvm::outs() << "BEFORE" << "\n";
      llvm::outs() << waitOp.getDmaOp();
      llvm::outs() << waitOp.getDmaOp().getDmaCpyNdOp();
      llvm::outs() << waitOp.getDirection() << "\n";
      auto dir = rewriter.getI32IntegerAttr((int32_t) waitOp.getDirection());
      llvm::outs() << "AFTER" << "\n";
      auto channel = rewriter.getI32IntegerAttr(0); // Used??
      auto col_num = rewriter.getI32IntegerAttr(1); // Used??
      auto row_num = rewriter.getI32IntegerAttr(1); // Used??
      rewriter.create<AIEX::IpuSyncOp>(
        rewriter.getUnknownLoc(), col, row, dir, channel, col_num, row_num
      );
      rewriter.eraseOp(waitOp);
    } else if (auto endOp = dyn_cast<AMDAIE::EndOp>(op)) {
      rewriter.eraseOp(endOp);
    } else {
      addConstants(op);
    }
    return WalkResult::advance();
  });
  for (auto *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
  if (res.wasInterrupted())
    return failure();
  // rewriter.mergeBlocks(controlCodeBlock, funcBlock);
  rewriter.inlineBlockBefore(controlCodeBlock, funcBlock->getTerminator());
  // rewriter.
  return success();
}

LogicalResult toObjectFifo(mlir::ModuleOp moduleOp) {
  llvm::outs() << "ModuleOp: " << moduleOp << "\n";
  IRRewriter rewriter(moduleOp.getContext());
  Block *moduleBlock = &moduleOp->getRegion(0).front();
  auto funcRes = moduleOp.walk([&](func::FuncOp funcOp) {
    Block *funcBlock = &funcOp.getBody().front();
    // Insert AIE DeviceOp 
    rewriter.setInsertionPoint(moduleBlock, moduleBlock->begin());
    auto deviceOp = rewriter.create<xilinx::AIE::DeviceOp>(
      rewriter.getUnknownLoc(),
      xilinx::AIE::AIEDeviceAttr::get(rewriter.getContext(), xilinx::AIE::AIEDevice::ipu)
    );
    deviceOp.getRegion().emplaceBlock();
    Block *deviceBlock = &deviceOp.getRegion().front();

    // Create the signature of the IPU instruction sequence function. The HAL interface bindings are
    // used to order the function parameters correctly.
    DenseMap<Value, Value> ipuFuncArgMap;
    auto subspanRange = funcBlock->getOps<IREE::HAL::InterfaceBindingSubspanOp>();
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps(subspanRange.begin(), subspanRange.end());
    llvm::sort(subspanOps, [](IREE::HAL::InterfaceBindingSubspanOp a, IREE::HAL::InterfaceBindingSubspanOp b) {
      return a.getBinding().getZExtValue() < b.getBinding().getZExtValue();
    });
    llvm::outs() << "subspanOps: \n";
    for (auto op : subspanOps)
      llvm::outs() << "--: " << op << "\n";
    SmallVector<Type> inputTypes;
    for (auto op : subspanOps)
      inputTypes.push_back(op.getType());
    FunctionType funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
    rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());
    auto ipuFuncOp = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), rewriter.getStringAttr("sequence"), funcType);
    ipuFuncOp.setPublic();
    rewriter.setInsertionPointToStart(ipuFuncOp.addEntryBlock());
    rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc());
    for (int i = 0; i < ipuFuncOp.getNumArguments(); ++i) {
      llvm::outs() << "arg: " << ipuFuncOp.getArgument(i) << "\n";
      ipuFuncArgMap[subspanOps[i].getResult()] = ipuFuncOp.getArgument(i);
    }

    // Walk the AIE regions ops and convert ops into pure AIEDialect ops
    auto regionRes = funcOp.walk([&](AMDAIE::AIERegionOp regionOp) {
      // Block *regionBlock = &regionOp.getRegion().front();
      rewriter.setInsertionPoint(deviceBlock, deviceBlock->begin());

      // Walk all operations in the AIE region and convert to AIE ops
      DenseMap<AMDAIE::DmaCpyNdOp, AIE::ObjectFifoCreateOp> dmaSymMap;
      int dmaId = 0;
      regionOp.walk([&](Operation *op) {
        if (auto tileOp = dyn_cast<AIE::TileOp>(op)) {
          rewriter.moveOpBefore(tileOp, deviceBlock, deviceBlock->begin());
        } else if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op)) {
          // TODO(jornt): refactor
          llvm::SmallSetVector<Value, 4> srcTiles;
          auto srcMemSpace = dmaOp.getSrcObjectFifo().getMemrefType().getMemorySpace();
          if (!srcMemSpace || dyn_cast<IntegerAttr>(srcMemSpace).getInt() != 2) {
            auto tiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getSrc().getDefiningOp()).getTiles();
            srcTiles.insert(tiles.begin(), tiles.end());
          } else {
            for (auto userOp : dmaOp->getUsers()) {
              if (auto coreOp = userOp->getParentOfType<xilinx::AIE::CoreOp>()) {
                srcTiles.insert(coreOp.getTileOp().getResult());
              }
            }
          }
          llvm::SmallSetVector<Value, 4> dstTiles;
          auto dstMemSpace = dmaOp.getDstObjectFifo().getMemrefType().getMemorySpace();
          if (!dstMemSpace || dyn_cast<IntegerAttr>(dstMemSpace).getInt() != 2) {
            auto tiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getDst().getDefiningOp()).getTiles();
            dstTiles.insert(tiles.begin(), tiles.end());
          } else {
            for (auto userOp : dmaOp->getUsers()) {
              if (auto coreOp = userOp->getParentOfType<xilinx::AIE::CoreOp>()) {
                dstTiles.insert(coreOp.getTileOp().getResult());
              }
            }
          }
          // auto dstTiles = dyn_cast<AMDAIE::LogicalObjectFifoFromMemref>(dmaOp.getDst().getDefiningOp()).getTiles();
          auto symName = "in" + std::to_string(dmaId++);
          auto symAttr = rewriter.getStringAttr(symName);
          auto fifo = createObjectFifo(rewriter, srcTiles.getArrayRef(), dstTiles.getArrayRef(), dmaOp, symAttr);
          dmaSymMap[dmaOp] = fifo;
          llvm::outs() << "fifo: " << fifo << "\n";
          // rewriter.eraseOp(dmaOp);
        } else if (auto linkOp = dyn_cast<AMDAIE::LogicalObjectFifoLink>(op)) {
          SmallVector<Attribute> inSyms;
          for (auto in : linkOp.getIns()) {
            auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(in.getDefiningOp());
            inSyms.push_back(SymbolRefAttr::get(rewriter.getContext(), dmaSymMap[dmaOp].getSymName()));
          }
          SmallVector<Attribute> outSyms;
          for (auto out : linkOp.getOuts()) {
            auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(out.getDefiningOp());
            outSyms.push_back(SymbolRefAttr::get(rewriter.getContext(), dmaSymMap[dmaOp].getSymName()));
          }
          rewriter.create<AIE::ObjectFifoLinkOp>(
            rewriter.getUnknownLoc(),
            rewriter.getArrayAttr(inSyms),
            rewriter.getArrayAttr(outSyms)
          );
          rewriter.eraseOp(linkOp);
        } else if (auto coreOp = dyn_cast<AIE::CoreOp>(op)) {
          rewriter.moveOpBefore(coreOp, deviceBlock, deviceBlock->end());
          if (failed(coreToObjectFifo(rewriter, coreOp, deviceOp, dmaSymMap))) {
            coreOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
        } else if (auto controlCodeOp = dyn_cast<AMDAIE::ControlCodeRegionOp>(op)) {
          if (failed(controlCodeToAie(rewriter, controlCodeOp, ipuFuncOp, ipuFuncArgMap, dmaSymMap))) {
            controlCodeOp.emitError("could not convert to AIEDialect ops");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      return WalkResult::advance();
    });
    if (regionRes.wasInterrupted())
      return WalkResult::interrupt();
    
    rewriter.moveOpBefore(ipuFuncOp, deviceBlock, deviceBlock->end());
    // After walking the FuncOp, it should be converted completely into 
    // an AIE::DeviceOp and can be erased safely.
    rewriter.eraseOp(funcOp);
    return WalkResult::advance();
  });
  if (funcRes.wasInterrupted())
      return failure();
  return success();
}


LogicalResult controlCodeLoopUnroll(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  auto res = moduleOp.walk([&](AMDAIE::ControlCodeRegionOp regionOp) {
    auto forRes = regionOp.walk([&](scf::ForOp forOp) {
      auto lbInt = getConstantIntValue(forOp.getLowerBound()).value();
      auto ubInt = getConstantIntValue(forOp.getUpperBound()).value();
      auto stepInt = getConstantIntValue(forOp.getStep()).value();
      int64_t tripCount = mlir::ceilDiv(ubInt - lbInt, stepInt);
      if (failed(loopUnrollByFactor(forOp, tripCount))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (forRes.wasInterrupted())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
      return failure();
  return success();
}


// TODO(jornt): duplicate, add to cleanup pass
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


class AMDAIEPrepareForAIEPass
    : public impl::AMDAIEPrepareForAIEBase<AMDAIEPrepareForAIEPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect>();
  }

  AMDAIEPrepareForAIEPass() = default;
  AMDAIEPrepareForAIEPass(const AMDAIEPrepareForAIEPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPrepareForAIEPass::runOnOperation() {
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
  if (failed(controlCodeLoopUnroll(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(mergeTiles(getOperation()))) {
    return signalPassFailure();
  }
}


class AMDAIEToAIEPass
    : public impl::AMDAIEToAIEBase<AMDAIEToAIEPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect>();
  }

  AMDAIEToAIEPass() = default;
  AMDAIEToAIEPass(const AMDAIEToAIEPass &pass){};
  void runOnOperation() override;
};

void AMDAIEToAIEPass::runOnOperation() {
  if (failed(toObjectFifo(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPrepareForAIEPass() {
  return std::make_unique<AMDAIEPrepareForAIEPass>();
}

std::unique_ptr<Pass> createAMDAIEToAIEPass() {
  return std::make_unique<AMDAIEToAIEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
