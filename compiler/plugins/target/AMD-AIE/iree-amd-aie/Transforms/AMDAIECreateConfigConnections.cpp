// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the transformation that TODO
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-create-config-connections"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult insertConfigRoutes(AMDAIE::WorkgroupOp workgroupOp,
                                 const AMDAIE::AMDAIEDeviceModel &deviceModel) {
  IRRewriter rewriter(workgroupOp->getContext());
  WalkResult res = workgroupOp->walk([&](AMDAIE::TileOp tileOp) {
    int64_t col = getConstantIndexOrAssert(tileOp.getCol());
    int64_t row = getConstantIndexOrAssert(tileOp.getRow());
    // TODO(jornt): only insert configuration routes for shim right now.
    if (deviceModel.isShimNOCTile(col, row)) {
      rewriter.setInsertionPointAfter(tileOp);
      auto sourceChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tileOp, 0, StrmSwPortType::DMA);
      auto targetChannelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tileOp, 0, StrmSwPortType::CTRL);
      SmallVector<Value> sourceChannels = {sourceChannelOp.getResult()};
      SmallVector<Value> targetChannels = {targetChannelOp.getResult()};
      rewriter.create<AMDAIE::FlowOp>(
          rewriter.getUnknownLoc(), sourceChannels, targetChannels, true, nullptr);
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIECreateConfigConnectionsPass
    : public impl::AMDAIECreateConfigConnectionsBase<
          AMDAIECreateConfigConnectionsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIECreateConfigConnectionsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to determine where to "
           "insert AIE configuration routes";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  // IRRewriter rewriter(parentOp->getContext());
  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(insertConfigRoutes(workgroupOp, deviceModel))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateConfigConnectionsPass() {
  return std::make_unique<AMDAIECreateConfigConnectionsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
