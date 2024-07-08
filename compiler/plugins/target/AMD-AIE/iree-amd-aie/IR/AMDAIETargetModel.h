// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In absence of a complete hardware model interface, this file contains some
// constants to describe hardware-related parameters used in transformations.
// This is meant to be temporary.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_AMDAIE_TARGET_MODEL_H_
#define IREE_COMPILER_AMDAIE_TARGET_MODEL_H_

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
//
// DMA iteration dimensions
//
// DMAs support multi-dimensional addressing through buffer descriptors in two
// ways:
// 1. Intra-iteration access pattern. Specified via 'strides' ('steps' in buffer
// descriptor lingo), 'sizes' ('wraps' in buffer descriptro lingo) and
// 'padding'. When a DMA executes a buffer descriptor, it will access the data
// (read/write) as specified by the intra-iteration access pattern.
// 2. Inter-iteration access pattern. Specified via an iteration 'stride',
// 'size' and 'current_iteration' ('stride' is the same as 'stepsize' and 'size'
// is the same as 'wrap' in buffer descriptor lingo). Here, 'current_iteration'
// keeps track of the current execution iteration of the buffer descriptor and
// is incremented after buffer descriptor execution. the 'stride' is the offset
// to be used for each execution of the buffer descriptor, relative to the
// previous one. When 'iteration_current' is equal to 'size', the
// 'iteration_current' is reset to zero.
//
// Although all DMAs use the same buffer descriptor format to describe the
// execution configuration, the intra-iteration and inter-dimensions are
// typically used for different purposes on different DMAs (see below).
//
//===----------------------------------------------------------------------===//

/// Shim DMAs support 3 intra-iteration dimensions + 1 inter-iteration
/// dimension. As the shim DMA typically isn't synchronized with other DMAs
/// (through semaphore locks), the inter-iteration access pattern is typically
/// used as an additional intra-iteration access pattern, resulting in 4 DMA
/// dimensions which can be used to address global memory data.
static const int64_t kAMDAIEShimDmaNbDims = 4;

/// MemTile DMAs support 4 intra-iteration dimensions + 1 inter-iteration
/// dimension. However, as the MemTile DMAs are typically synchronized with
/// other DMAs for stream-through, double-buffering purposes, the
/// inter-iteration can't typically be used in the same way as the
/// intra-iteration dimensions. Therefore, for now, only the intra-iteration
/// dimensions can be used for DMA access patterns.
static const int64_t kAMDAIEMemTileDmaNbDims = 4;

/// Core DMAs support 3 intra-iteration dimensions + 1 inter-iteration
/// dimension. However, as the core DMAs are typically synchronized with
/// with the core processor for data access purposes (read/write), the
/// inter-iteration can't typically be used in the same way as the
/// intra-iteration dimensions. Therefore, for now, only the intra-iteration
/// dimensions can be used for DMA access patterns.
static const int64_t kAMDAIECoreDmaNbDims = 3;

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_TARGET_MODEL_H_
