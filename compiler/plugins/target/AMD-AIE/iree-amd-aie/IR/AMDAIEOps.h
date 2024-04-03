// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_OPS_H_
#define IREE_COMPILER_AMDAIE_OPS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"

// clang-format off
#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#define GET_OP_CLASSES
#include "iree-amd-aie/IR/AMDAIEOps.h.inc"
// clang-format on

namespace mlir::iree_compiler::AMDAIE {

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_AMDAIE_OPS_H_