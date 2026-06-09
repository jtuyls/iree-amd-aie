// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_DIRECT_COMMAND_BUFFER_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_DIRECT_COMMAND_BUFFER_H_

#include <cstdint>

#include "iree-amd-aie/driver/amdxdna/device.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

// `out_command_buffer` must be released by the caller (see
// iree_hal_command_buffer_release).
iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

#if defined(_WIN32)
// Returns true when a cached chain command's already staged control-code words
// must be rewritten to match a freshly recorded command. Generic allocator
// caching may legally change HAL buffer wrapper/native BO identity between
// otherwise equivalent submissions; the expensive Windows MCDM aperture-code
// dirty path must key on the effective patched control-code bytes, not object
// identity.
bool iree_hal_amdxdna_direct_command_buffer_control_words_changed(
    const uint32_t* cached_words, iree_host_size_t cached_word_count,
    const uint32_t* fresh_words, iree_host_size_t fresh_word_count);
#endif  // defined(_WIN32)

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_DIRECT_COMMAND_BUFFER_H_
