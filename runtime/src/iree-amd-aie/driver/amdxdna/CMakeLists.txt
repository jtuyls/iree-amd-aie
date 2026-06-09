# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32))
  message(STATUS "amdxdna HAL backend is only implemented for Linux KMQ and Windows MCDM; skipping")
  return()
endif()

iree_add_all_subdirs()

iree_register_external_hal_driver(
  NAME
    amdxdna
  DRIVER_TARGET
    iree-amd-aie::driver::amdxdna::registration
  REGISTER_FN
    iree_hal_amdxdna_driver_module_register
)

set(IREE_HAL_AMDXDNA_NATIVE_SRCS
  native.h
)
if(WIN32)
  set(IREE_HAL_AMDXDNA_NATIVE_DEPS
    iree-amd-aie::driver::amdxdna::shim::windows::mcdm::kmt_api
  )
  list(APPEND IREE_HAL_AMDXDNA_NATIVE_SRCS
    native_windows_mcdm.cc
  )
else()
  set(IREE_HAL_AMDXDNA_NATIVE_DEPS
    iree-amd-aie::driver::amdxdna::shim::linux::kmq::shim-xdna
  )
  list(APPEND IREE_HAL_AMDXDNA_NATIVE_SRCS
    native_linux_kmq.cc
  )
endif()

iree_cc_library(
  NAME
    amdxdna
  SRCS
    allocator.cc
    allocator.h
    api.h
    async_queue.cc
    async_queue.h
    buffer.cc
    buffer.h
    direct_command_buffer.cc
    direct_command_buffer.h
    device.cc
    device_internal.h
    driver.cc
    executable.cc
    executable.h
    executable_internal.h
    ${IREE_HAL_AMDXDNA_NATIVE_SRCS}
    nop_event.cc
    nop_event.h
    nop_executable_cache.cc
    nop_executable_cache.h
    nop_semaphore.cc
    nop_semaphore.h
    util.h
    xclbin_util.cc
    xclbin_util.h
  DEPS
    iree::async
    iree::async::util::proactor_pool
    iree::base
    iree::base::core_headers
    iree::hal::memory::cpu_slab_provider
    iree::hal::memory::passthrough_pool
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    iree::hal::utils::resource_set
    iree::base::internal::flatcc::parsing
    iree-amd-aie::schemas::amdxdna_xclbin_executable_def_c_fbs
    iree-amd-aie::schemas::pdi_executable_def_c_fbs
    ${IREE_HAL_AMDXDNA_NATIVE_DEPS}
  COPTS
    $<$<CXX_COMPILER_ID:MSVC>:/std:c++20>
  PUBLIC
)
