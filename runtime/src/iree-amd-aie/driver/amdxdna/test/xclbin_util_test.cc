// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/xclbin_util.h"

#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void WriteU32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU64(std::vector<uint8_t>* data, size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

std::vector<uint8_t> MakeXclbinWithTwoPdis() {
  constexpr size_t kSectionTableOffset = 0x1C8;
  constexpr size_t kAiePartitionOffset = 0x220;
  constexpr size_t kAiePartitionSize = 0x200;
  constexpr size_t kPdiTableOffset = 0xC8;
  constexpr size_t kPdiRecordSize = 0x60;
  constexpr size_t kPdi0Offset = 0x1A0;
  constexpr size_t kPdi1Offset = 0x1B0;

  std::vector<uint8_t> xclbin(kAiePartitionOffset + kAiePartitionSize, 0);
  std::memcpy(xclbin.data(), "xclbin2", 7);

  WriteU32(&xclbin, 0x1C0, 1);
  WriteU32(&xclbin, kSectionTableOffset, 32);
  WriteU64(&xclbin, kSectionTableOffset + 24, kAiePartitionOffset);
  WriteU64(&xclbin, kSectionTableOffset + 32, kAiePartitionSize);

  const size_t aie = kAiePartitionOffset;
  WriteU32(&xclbin, aie + 120, 2);
  WriteU32(&xclbin, aie + 124, kPdiTableOffset);

  const std::vector<uint8_t> pdi0 = {0x10, 0x20, 0x30, 0x40, 0x50};
  const std::vector<uint8_t> pdi1 = {0xAA, 0xBB, 0xCC, 0xDD};
  WriteU32(&xclbin, aie + kPdiTableOffset + 16, pdi0.size());
  WriteU32(&xclbin, aie + kPdiTableOffset + 20, kPdi0Offset);
  WriteU32(&xclbin, aie + kPdiTableOffset + kPdiRecordSize + 16,
           pdi1.size());
  WriteU32(&xclbin, aie + kPdiTableOffset + kPdiRecordSize + 20, kPdi1Offset);
  std::memcpy(xclbin.data() + aie + kPdi0Offset, pdi0.data(), pdi0.size());
  std::memcpy(xclbin.data() + aie + kPdi1Offset, pdi1.data(), pdi1.size());
  return xclbin;
}

TEST(XclbinUtilTest, ExtractsSelectedPdiFromAiePartition) {
  std::vector<uint8_t> xclbin = MakeXclbinWithTwoPdis();

  std::vector<uint8_t> pdi;
  IREE_ASSERT_OK(iree_hal_amdxdna_xclbin_extract_pdi(
      iree_make_const_byte_span(xclbin.data(), xclbin.size()), 0, &pdi));
  EXPECT_EQ(pdi, std::vector<uint8_t>({0x10, 0x20, 0x30, 0x40, 0x50}));

  IREE_ASSERT_OK(iree_hal_amdxdna_xclbin_extract_pdi(
      iree_make_const_byte_span(xclbin.data(), xclbin.size()), 1, &pdi));
  EXPECT_EQ(pdi, std::vector<uint8_t>({0xAA, 0xBB, 0xCC, 0xDD}));
}

TEST(XclbinUtilTest, RejectsOutOfRangePdiIndex) {
  std::vector<uint8_t> xclbin = MakeXclbinWithTwoPdis();
  std::vector<uint8_t> pdi;

  iree_status_t status = iree_hal_amdxdna_xclbin_extract_pdi(
      iree_make_const_byte_span(xclbin.data(), xclbin.size()), 2, &pdi);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_free(status);
  EXPECT_TRUE(pdi.empty());
}

TEST(XclbinUtilTest, RejectsMissingAiePartition) {
  std::vector<uint8_t> xclbin(0x220, 0);
  std::memcpy(xclbin.data(), "xclbin2", 7);
  WriteU32(&xclbin, 0x1C0, 0);

  std::vector<uint8_t> pdi;
  iree_status_t status = iree_hal_amdxdna_xclbin_extract_pdi(
      iree_make_const_byte_span(xclbin.data(), xclbin.size()), 0, &pdi);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  EXPECT_TRUE(pdi.empty());
}

}  // namespace
