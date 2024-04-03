// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_DMAOPINTERFACE
#define IREE_AMDAIE_DIALECT_DMAOPINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Defines the interface for dma-like operations.
//===----------------------------------------------------------------------===//

// TODO(jornt): use source and target to work with CopyOpInterface
def DoublyStridedOpInterface : OpInterface<"DoublyStridedInterface"> {
  let description = [{
    TODO
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getDstOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getDstSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getDstStrides"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSrcOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSrcSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSrcStrides"
    >
  ];
}

// TODO(jornt): use CopyOpInterface instead with source and target
def DmaOpInterface : OpInterface<"DmaOpInterface"> {
  let description = [{
    A dma-like operation is one that copies from source value to target value with 
    specified source and target addressing.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the source value for this copy operation",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSrc" // /*methodName=*/"getSource" TODO
    >,
    InterfaceMethod<
      /*desc=*/"Returns the target value for this copy operation",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getDst"
      // /*methodName=*/"getTarget" TODO
    >
  ];
}

#endif // IREE_AMDAIE_DIALECT_DMAOPINTERFACE
