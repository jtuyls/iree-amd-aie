// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/allocator.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/caching_allocator.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AllocatorTest, CreateAndRelease) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
      iree_allocator_system(), /*native_device=*/nullptr, &allocator));
  ASSERT_NE(allocator, nullptr);
  iree_hal_allocator_release(allocator);
}

TEST(AllocatorTest, QueryMemoryHeapsSupportsGenericCachingAllocator) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
      iree_allocator_system(), /*native_device=*/nullptr, &allocator));

  iree_hal_allocator_memory_heap_t heap = {};
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      allocator, /*capacity=*/1, &heap, &heap_count));
  EXPECT_EQ(heap_count, 1u);
  EXPECT_TRUE(iree_all_bits_set(heap.type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_TRUE(iree_all_bits_set(heap.allowed_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER |
                                    IREE_HAL_BUFFER_USAGE_DISPATCH |
                                    IREE_HAL_BUFFER_USAGE_MAPPING));

  iree_hal_allocator_t* caching_allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_caching_allocator_create_unbounded(
      allocator, iree_allocator_system(), &caching_allocator));
  iree_hal_allocator_release(caching_allocator);
  iree_hal_allocator_release(allocator);
}

}  // namespace
