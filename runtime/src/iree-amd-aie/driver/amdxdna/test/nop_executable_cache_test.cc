// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/nop_executable_cache.h"

#include <cstring>
#include <limits>
#include <vector>

#include "iree-amd-aie/schemas/amdxdna_xclbin_executable_def_builder.h"
#include "iree/base/api.h"
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void WriteU32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU64(std::vector<uint8_t>* data, size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

std::vector<uint8_t> MakeXclbinWithOnePdi() {
  constexpr size_t kSectionTableOffset = 0x1C8;
  constexpr size_t kAiePartitionOffset = 0x220;
  constexpr size_t kAiePartitionSize = 0x200;
  constexpr size_t kPdiTableOffset = 0xC8;
  constexpr size_t kPdiOffset = 0x1A0;

  std::vector<uint8_t> xclbin(kAiePartitionOffset + kAiePartitionSize, 0);
  std::memcpy(xclbin.data(), "xclbin2", 7);

  WriteU32(&xclbin, 0x1C0, 1);
  WriteU32(&xclbin, kSectionTableOffset, 32);
  WriteU64(&xclbin, kSectionTableOffset + 24, kAiePartitionOffset);
  WriteU64(&xclbin, kSectionTableOffset + 32, kAiePartitionSize);

  const size_t aie = kAiePartitionOffset;
  const std::vector<uint8_t> pdi = {0x10, 0x20, 0x30, 0x40};
  WriteU32(&xclbin, aie + 120, 1);
  WriteU32(&xclbin, aie + 124, kPdiTableOffset);
  WriteU32(&xclbin, aie + kPdiTableOffset + 16, pdi.size());
  WriteU32(&xclbin, aie + kPdiTableOffset + 20, kPdiOffset);
  std::memcpy(xclbin.data() + aie + kPdiOffset, pdi.data(), pdi.size());
  return xclbin;
}

std::vector<uint8_t> MakeXadxExecutable() {
  flatcc_builder_t builder;
  flatcc_builder_init(&builder);

  std::vector<uint8_t> xclbin = MakeXclbinWithOnePdi();
  flatbuffers_string_ref_t xclbin_string = flatcc_builder_create_string(
      &builder, reinterpret_cast<const char*>(xclbin.data()), xclbin.size());
  iree_amd_aie_hal_amdxdna_xclbin_XclbinDef_ref_t xclbin_ref =
      iree_amd_aie_hal_amdxdna_xclbin_XclbinDef_create(&builder,
                                                        xclbin_string);
  flatcc_builder_ref_t xclbin_refs[] = {xclbin_ref};
  iree_amd_aie_hal_amdxdna_xclbin_XclbinDef_vec_ref_t xclbins_ref =
      flatcc_builder_create_offset_vector(&builder, xclbin_refs, 1);

  const uint32_t control_code[] = {0x14};
  flatbuffers_uint32_vec_ref_t control_code_ref =
      flatcc_builder_create_vector(&builder, control_code, 1,
                                   sizeof(control_code[0]),
                                   alignof(uint32_t),
                                   std::numeric_limits<size_t>::max());
  iree_amd_aie_hal_amdxdna_xclbin_RunDef_start(&builder);
  iree_amd_aie_hal_amdxdna_xclbin_RunDef_control_code_add(&builder,
                                                           control_code_ref);
  iree_amd_aie_hal_amdxdna_xclbin_RunDef_ref_t run_ref =
      iree_amd_aie_hal_amdxdna_xclbin_RunDef_end(&builder);
  flatcc_builder_ref_t run_refs[] = {run_ref};
  iree_amd_aie_hal_amdxdna_xclbin_RunDef_vec_ref_t runs_ref =
      flatcc_builder_create_offset_vector(&builder, run_refs, 1);

  flatbuffers_string_ref_t name_ref =
      flatcc_builder_create_string_str(&builder, "DPU_PDI_0");
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_start(&builder);
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_name_add(&builder, name_ref);
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_pdi_index_add(&builder, 0);
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_add(&builder, 0);
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_runs_add(&builder, runs_ref);
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_ref_t entry_ref =
      iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_end(&builder);
  flatcc_builder_ref_t entry_refs[] = {entry_ref};
  iree_amd_aie_hal_amdxdna_xclbin_EntryPointDef_vec_ref_t entry_points_ref =
      flatcc_builder_create_offset_vector(&builder, entry_refs, 1);

  iree_amd_aie_hal_amdxdna_xclbin_ExecutableDef_create_as_root(
      &builder, xclbins_ref, entry_points_ref);

  size_t flatbuffer_size = 0;
  void* flatbuffer =
      flatcc_builder_finalize_aligned_buffer(&builder, &flatbuffer_size);
  std::vector<uint8_t> bytes(static_cast<uint8_t*>(flatbuffer),
                             static_cast<uint8_t*>(flatbuffer) +
                                 flatbuffer_size);
  flatcc_builder_aligned_free(flatbuffer);
  flatcc_builder_clear(&builder);
  return bytes;
}

TEST(NopExecutableCacheTest, CanPreparePublicAmdxdnaFormat) {
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_nop_executable_cache_create(
      /*native_device=*/nullptr, iree_make_cstring_view("default"),
      iree_allocator_system(), &executable_cache));

  EXPECT_TRUE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0,
      iree_make_cstring_view("amdaie-pdi-fb")));
  EXPECT_TRUE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0,
      iree_make_cstring_view("amdaie-amdxdna-xclbin-fb")));
  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("PDIR")));
  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("FOO?")));

  iree_hal_executable_cache_release(executable_cache);
}

TEST(NopExecutableCacheTest, PrepareXadxExtractsPdiFromXclbin) {
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_nop_executable_cache_create(
      /*native_device=*/nullptr, iree_make_cstring_view("default"),
      iree_allocator_system(), &executable_cache));

  std::vector<uint8_t> flatbuffer = MakeXadxExecutable();
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format =
      iree_make_cstring_view("amdaie-amdxdna-xclbin-fb");
  executable_params.executable_data =
      iree_make_const_byte_span(flatbuffer.data(), flatbuffer.size());

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST(NopExecutableCacheTest, PrepareRejectsUnknownFormatBeforeParsing) {
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_nop_executable_cache_create(
      /*native_device=*/nullptr, iree_make_cstring_view("default"),
      iree_allocator_system(), &executable_cache));

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = iree_make_cstring_view("PDIR");
  executable_params.executable_data = iree_const_byte_span_empty();

  iree_hal_executable_t* executable = nullptr;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);

  iree_hal_executable_cache_release(executable_cache);
}

}  // namespace
