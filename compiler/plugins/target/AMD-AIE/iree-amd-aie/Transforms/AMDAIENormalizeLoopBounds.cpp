// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-normalize-loop-bounds"

namespace mlir::iree_compiler::AMDAIE {

/// NOTE: Copy of existing utility struct in llvm-project:
/// https://github.com/llvm/llvm-project/blob/172759492a162592da7ae9e03888661c108b1be4/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L33C1-L40C15.
/// Can be replaced with that utility struct once it is exposed.
///
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};
} // namespace

/// NOTE: Copy of existing utility function in llvm-project:
/// https://github.com/llvm/llvm-project/blob/172759492a162592da7ae9e03888661c108b1be4/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L485.
/// Can be replaced with that utility function once it is exposed.
///
/// Transform a loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1 {
///     affine applu
/// Insert the induction variable remapping in the body of `inner`, which is
/// expected to be either `loop` or another loop perfectly nested under `loop`.
/// Insert the definition of new bounds immediate before `outer`, which is
/// expected to be either `loop` or its parent in the loop nest.
static LoopParams emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc,
                                           Value lb, Value ub, Value step) {
  // For non-index types, generate `arith` instructions
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto lbCst = getConstantIntValue(lb)) isZeroBased = lbCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = getConstantIntValue(step))
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne) return {lb, ub, step};

  Value diff = isZeroBased ? ub : rewriter.create<arith::SubIOp>(loc, ub, lb);
  Value newUpperBound =
      isStepOne ? diff : rewriter.create<arith::CeilDivSIOp>(loc, diff, step);

  Value newLowerBound = isZeroBased
                            ? lb
                            : rewriter.create<arith::ConstantOp>(
                                  loc, rewriter.getZeroAttr(lb.getType()));
  Value newStep = isStepOne
                      ? step
                      : rewriter.create<arith::ConstantOp>(
                            loc, rewriter.getIntegerAttr(step.getType(), 1));

  return {newLowerBound, newUpperBound, newStep};
}

LogicalResult normalizeLoopBounds(RewriterBase &rewriter, scf::ForOp forOp) {

}

namespace {
struct AMDAIENormalizeLoopBounds
    : public impl::AMDAIEHoistForLoopAffineApplyBase<
          AMDAIEHoistForLoopAffineApply> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](affine::AffineApplyOp applyOp) {
      (void)hoistForAffineApplyOp(rewriter, applyOp);
    });
  }
};
}  // namespace

// std::unique_ptr<Pass> createAMDAIEHoistForLoopAffineApplyPass() {
//   return std::make_unique<AMDAIEHoistForLoopAffineApply>();
// }

}  // namespace mlir::iree_compiler::AMDAIE
