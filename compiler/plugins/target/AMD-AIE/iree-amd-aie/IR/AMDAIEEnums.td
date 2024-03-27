// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEENUMS
#define IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEENUMS

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// Duplicate of xilinx::AIE::ObjectFifoPort 
def ObjectFifoPort: I32EnumAttr<"ObjectFifoPort",
  "Ports of an object FIFO",
  [
    I32EnumAttrCase<"Produce", 0>,
    I32EnumAttrCase<"Consume", 1>
  ]
  > {
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

#endif // IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS
