# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_c_embed_data(
  NAME
    AMDAIEUkernels
  SRCS
    "uKernels/npu1/chess/chess_intrinsic_wrapper.cpp"
    "uKernels/npu1/chess/matmul.cc"
    "uKernels/npu1/chess/softmax.cc"
    "uKernels/npu1/chess/zero_fill.cc"
    "uKernels/npu4/chess/chess_intrinsic_wrapper.cpp"
    "uKernels/npu4/chess/matmul.cc"
    "uKernels/npu4/chess/softmax.cc"
    "uKernels/npu4/chess/trunci.cc"
    "uKernels/npu4/chess/zero_fill.cc"
    "uKernels/npu4/peano/matmul.cc"
    "uKernels/npu4/peano/trunci.cc"
    "uKernels/npu4/peano/zero_fill.cc"
  C_FILE_OUTPUT
    "ukernels.c"
  H_FILE_OUTPUT
    "ukernels.h"
  IDENTIFIER
    "iree_amd_aie_ukernels"
  STRIP_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/uKernels/"
  PUBLIC
)

iree_cc_library(
  NAME
    AMDAIERT
  HDRS
    "AMDAIERT.h"
  SRCS
    "AMDAIERT.cpp"
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

set(_amd_aie_target_deps
  ::AMDAIERT
  ::AMDAIEUkernels
  iree-amd-aie::aie_runtime::iree_aie_runtime_static
  iree::target::amd-aie::Transforms
)

if(IREE_AMD_AIE_BOOTGEN_EXECUTABLE)
  file(TO_CMAKE_PATH "${IREE_AMD_AIE_BOOTGEN_EXECUTABLE}" _amd_aie_bootgen_executable)
elseif(IREE_AMD_AIE_ENABLE_BOOTGEN)
  list(APPEND _amd_aie_target_deps iree-aie-bootgen)
else()
  message(FATAL_ERROR "The AMD-AIE compiler target requires either bundled bootgen or IREE_AMD_AIE_BOOTGEN_EXECUTABLE")
endif()

iree_cc_library(
  NAME
    AIETargets
  SRCS
    "AMDAIETargetBCF.cpp"
    "AMDAIETargetCDODirect.cpp"
    "AMDAIETargetLdScript.cpp"
    "XCLBinGen.cpp"
  DEPS
    ${_amd_aie_target_deps}
  INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(IREE_AMD_AIE_BOOTGEN_EXECUTABLE)
  target_compile_definitions(
    iree_target_amd-aie_Target_AIETargets
    PRIVATE
      "IREE_AMD_AIE_BOOTGEN_EXECUTABLE=\"${_amd_aie_bootgen_executable}\""
  )
endif()

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  add_dependencies(iree_target_amd-aie_Target_AIETargets iree-aie-xclbinutil)
endif()

iree_cc_library(
  NAME
    Target
  HDRS
    "AIETarget.h"
  SRCS
    "AIETarget.cpp"
  DEPS
    ::AIETargets
    ::AMDAIERT
    iree-amd-aie::schemas::xrt_executable_def_c_fbs
    iree-amd-aie::schemas::pdi_executable_def_c_fbs
    iree::base::internal::flatcc::building
    iree::base::internal::flatcc::parsing
    iree::compiler::Dialect::HAL::Target
    iree::target::amd-aie::IR::AMDAIEDialect
    iree::target::amd-aie::Transforms
    iree::target::amd-aie::air::AIRDialectIR
    MLIRLLVMDialect
  PUBLIC
)
