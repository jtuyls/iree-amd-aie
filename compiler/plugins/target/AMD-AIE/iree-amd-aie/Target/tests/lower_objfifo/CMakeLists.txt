# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    buffers_xclbin.mlir
    basic_dma_transpose.mlir
    basic_matrix_multiplication_single_core.mlir
  TOOLS
    iree-opt
    iree-compile
    FileCheck
    not
)
