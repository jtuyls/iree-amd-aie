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
      SmallVector<Type, 1>{}, // rewriter.getIndexType(),
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

      rewriter.setInsertionPoint(forallOp);

      rewriter.setInsertionPoint(op);
      // AMDAIE::DmaCpyNdOp dmaCopy = 
      rewriter.create<AMDAIE::DmaCpyNdOp>(
        op.getLoc(),
        SmallVector<Type, 1>{}, // rewriter.getIndexType(),
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

  LogicalResult resolveWorkGroup(PatternRewriter &rewriter,
                                 scf::ForallOp forallOp,
                                 Block *newBlock,
                                 Block *controlCodeBlock,
                                 Block *controlCodeEndBlock,
                                 xilinx::AIE::TileOp& tileOp,
                                 DenseMap<Value, OpFoldResult> &symbolTable,
                                 DenseMap<Operation *, SmallVector<SmallVector<int64_t>>>& constructedOps) const {
    std::cout << "resolveWorkGroup" << std::endl;
    rewriter.setInsertionPointToEnd(newBlock);
    // Block::iterator it = newBlock->begin();
    // Create AIE core op + block for inserting L1 ops
    auto core = rewriter.create<xilinx::AIE::CoreOp>(rewriter.getUnknownLoc(), tileOp);
    Region &coreRegion = core.getBody();
    auto coreBlock = rewriter.createBlock(&coreRegion);

    // Add core to func block
    rewriter.setInsertionPointToEnd(controlCodeBlock);
    rewriter.create<AMDAIE::LoadCoreOp>(rewriter.getUnknownLoc(), core);
    rewriter.setInsertionPointToEnd(newBlock);

    Location loc = forallOp.getLoc();
    forallOp->walk([&](Operation *op) {
      if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op)) {
        // rewriter.setInsertionPoint(dmaOp);
        rewriter.setInsertionPointToEnd(newBlock);
        if (!constructedOps.contains(op)) {
          SmallVector<SmallVector<int64_t>> values;
          constructedOps[op] = values;
        }
        SmallVector<OpFoldResult> emptyDst(dmaOp.getDstOffsets().size(), rewriter.getI64IntegerAttr(0));
        SmallVector<OpFoldResult> emptySrc(dmaOp.getSrcOffsets().size(), rewriter.getI64IntegerAttr(0));
        auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
          loc,
          SmallVector<Type, 1>{}, // rewriter.getIndexType(),
          dmaOp.getDst(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst), // dmaOp.getDstOffsets(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst), // dmaOp.getDstSizes(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptyDst), // dmaOp.getDstStrides(),
          dmaOp.getSrc(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc), // dmaOp.getSrcOffsets(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc), // dmaOp.getSrcSizes(),
          getValueOrCreateConstantIndexOp(rewriter, loc, emptySrc) // dmaOp.getSrcStrides()
        );
        std::cout << "AMDAIE::DmaCpyNdOp" << std::endl;
        SmallVector<int64_t> valueResults;
        for (OpOperand &opOperand : dmaOp->getOpOperands()) {
          Value operand = opOperand.get();
          // llvm::outs() << "Operand: " << operand << "\n";
          if (symbolTable.contains(operand)) {
            // llvm::outs() << "Found operand in symbol table: " << operand << ": " << getConstantIntValue(symbolTable[operand]) << "\n";
            newDmaOp->setOperand(opOperand.getOperandNumber(),
                                 getValueOrCreateConstantIndexOp(rewriter, loc, symbolTable[operand]));
            valueResults.push_back(getConstantIntValue(symbolTable[operand]).value());
          } else {
            newDmaOp->setOperand(opOperand.getOperandNumber(), operand);
          }
        }
        if (std::find(constructedOps[op].begin(), constructedOps[op].end(), valueResults) != constructedOps[op].end()) {
          // DMA op with broadcast capability, but we still need to add the consumer/producer to the AIE core
          auto srcType = newDmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
          rewriter.setInsertionPointToEnd(coreBlock);
          if (dyn_cast<IntegerAttr>(srcType.getMemorySpace()).getInt() == 1) {
            // To core
            rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getDst()
            );
          } else if (dyn_cast<IntegerAttr>(srcType.getMemorySpace()).getInt() == 2) {
            // From core
            rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getSrc()
            );
          } else {
            dmaOp->emitError("found unsupported source memory space");
            return WalkResult::interrupt();
          }
          rewriter.eraseOp(newDmaOp);
        } else {
          constructedOps[op].push_back(valueResults);

          auto srcType = newDmaOp.getSrcType().cast<AMDAIEObjectFifoType>().getElementType();
          // auto dstType = newDmaOp.getDst().getType().cast<MemRefType>();
          if (dyn_cast<IntegerAttr>(srcType.getMemorySpace()).getInt() == 1) {
            rewriter.setInsertionPointToEnd(newBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getSrc()
            );
            rewriter.setInsertionPointToEnd(coreBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getDst()
            );
          } else if (dyn_cast<IntegerAttr>(srcType.getMemorySpace()).getInt() == 2) {
            rewriter.setInsertionPointToEnd(newBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getDst()
            );
            rewriter.setInsertionPointToEnd(coreBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getSrc()
            );
            // TODO(jornt): this feels too hardcoded, we assume here that we're always waiting on
            // L2 consumption. It might be better to wait on every produce/consume op and then later
            // optimize unnecessary ones out.
            rewriter.setInsertionPointToEnd(controlCodeEndBlock);
            rewriter.create<AMDAIE::LogicalObjectFifoWait>(
              rewriter.getUnknownLoc(),
              SmallVector<Type, 1>{},
              newDmaOp.getDst()
            );
          } else {
            dmaOp->emitError("found unsupported source memory space");
            return WalkResult::interrupt();
          }
          // llvm::outs() << "Constructed ops push: ";
          // for (auto value : valueResults) {
          //   llvm::outs() << value << ", ";
          // }
          // llvm::outs() << "\n";
          
        }
      } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        // rewriter.setInsertionPointToEnd(coreBlock);
        std::cout << "Linalg op" << std::endl;
        OpBuilder builder = OpBuilder::atBlockEnd(coreBlock);
        auto clonedOp = mlir::clone(builder, linalgOp, linalgOp->getResultTypes(), linalgOp->getOperands());
        llvm::outs() << "ClonedOp: " << clonedOp << "\n";
        // rewriter.moveOpBefore(clonedOp, coreBlock, coreBlock->end());
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
    DenseMap<Operation *, SmallVector<SmallVector<int64_t>>> constructedOps;
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
        if (failed(resolveWorkGroup(rewriter, forallOp, newBlock, controlCodeBlock, controlCodeEndBlock, tileOp, symbolTable, constructedOps))) {
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
  //patterns.insert<IREE::LinalgExt::ForallOpToAsyncRewriter>(context);
  patterns.insert<SCFForAllToLogicalObjectfifo>(context);
  patterns.insert<SCFForAllToStructural>(context);
  // patterns.insert<DmaMemcpyNdToLogicalObjectfifo>(context);
  
  // patterns.insert<SCFForAllUnroll>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
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
  // patterns.insert<LogicalObjectfifoFromMemrefCleanup>(context);
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
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(context);
  DenseSet<AMDAIE::LogicalObjectFifoFromMemref> toBeErased;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemref op) {
    if (toBeErased.contains(op)) {
      WalkResult::advance();
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
      }
    }
    return WalkResult::advance();
  });
  // TODO some still used?? Something wrong here. To be debugged, but ok for now.
  for (auto op : toBeErased) {
    if (op->use_empty())
      rewriter.eraseOp(op);
    else
      llvm::outs() << "Still used: " << op << "\n";
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
