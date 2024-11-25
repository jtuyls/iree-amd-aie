// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOGICALOBJFIFOSPLITTINGUTILS_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to split logicalobjectfifos given a vector of L2->L1 dma ops.
LogicalResult splitLogicalObjectFifoForElementwiseOp(
    IRRewriter &rewriter, SmallVector<AMDAIE::DmaCpyNdOp> &l2ToL1DmaOps,
    MLIRContext *context);

/// Utility to get the split dimension and factor given a L2 objectFifo op.
LogicalResult getSplitDimAndFactorFromObjFifo(
    AMDAIE::LogicalObjectFifoFromMemrefOp op, int64_t &splitDim,
    int64_t &splitFactor);

/// Utility to get the split dimension and factor from a L3->L2 dma op.
LogicalResult getSplitDimAndFactorFromDma(AMDAIE::DmaCpyNdOp op,
                                          int64_t &splitDim,
                                          int64_t &splitFactor,
                                          int64_t &splitDimInL2Dma);

// /// Split L2 space input and output logical objectFifos.
// LogicalResult splitLogicalObjectFifo(IRRewriter &rewriter,
//                                      AMDAIE::LogicalObjectFifoFromMemrefOp
//                                      op, int64_t splitDim, int64_t
//                                      splitFactor);

// /// Split DmaCpyNd ops between L2 and L3 memory spaces.
// LogicalResult splitDoublyStridedOp(IRRewriter &rewriter, AMDAIE::DmaCpyNdOp
// op,
//                                    int64_t splitDim, int64_t splitFactor,
//                                    int64_t splitDimInL2Dma);

/// Split a logical objectFifo on the provided split dimension with the
/// specified splitting factor.
LogicalResult splitLogicalObjectFifo(IRRewriter &rewriter,
                                     AMDAIE::LogicalObjectFifoFromMemrefOp op,
                                     size_t splitDim = 0,
                                     int64_t splitFactor = -1);

/// Split doubly strided operations on a source and target split dimension with
/// the provided split factor.
LogicalResult splitDoublyStridedOp(IRRewriter &rewriter,
                                   AMDAIE::DoublyStridedOpInterface op,
                                   size_t sourceSplitDim = 0,
                                   size_t targetSplitDim = 0,
                                   int64_t splitFactor = -1);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
