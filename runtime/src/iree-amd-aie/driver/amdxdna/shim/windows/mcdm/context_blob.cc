// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context_blob.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr size_t kAxlfBaseOffset = 0xE8;
constexpr size_t kTailSizeOnePdi = 0x530;
constexpr uint64_t kCommandApertureBase = 0x04000000;
constexpr uint64_t kContextCommandBoSize = 0x1000;
constexpr uint32_t kBuildMetadataSection = 14;
constexpr uint32_t kAiePartitionSection = 32;

struct AxlfSection {
  uint32_t kind = 0;
  std::string name;
  uint64_t offset = 0;
  uint64_t size = 0;
};

bool Fail(const std::string& message, std::string* out_error) {
  if (out_error) *out_error = message;
  return false;
}

bool CheckRange(size_t data_size, uint64_t offset, uint64_t size,
                const char* what, std::string* out_error) {
  if (offset > data_size || size > data_size - offset) {
    std::ostringstream os;
    os << what << " is out of bounds";
    return Fail(os.str(), out_error);
  }
  return true;
}

uint16_t ReadU16(const uint8_t* data, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint32_t ReadU32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint64_t ReadU64(const uint8_t* data, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

void WriteU32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU64(std::vector<uint8_t>* data, size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

std::string CString(const uint8_t* data, size_t size) {
  size_t length = 0;
  while (length < size && data[length] != 0) ++length;
  return std::string(reinterpret_cast<const char*>(data), length);
}

bool WriteCString(std::vector<uint8_t>* data, size_t offset, size_t size,
                  const std::string& value, std::string* out_error) {
  if (value.size() >= size) {
    std::ostringstream os;
    os << "string '" << value << "' does not fit in " << size << " bytes";
    return Fail(os.str(), out_error);
  }
  std::fill(data->begin() + offset, data->begin() + offset + size, uint8_t{0});
  std::memcpy(data->data() + offset, value.data(), value.size());
  return true;
}

bool ParseSections(const uint8_t* xclbin, size_t xclbin_size,
                   std::vector<AxlfSection>* out_sections,
                   std::string* out_error) {
  if (xclbin_size < 0x1C8 || std::memcmp(xclbin, "xclbin2\0", 8) != 0) {
    return Fail("input is not an AXLF/xclbin2 file", out_error);
  }
  uint32_t section_count = ReadU32(xclbin, 0x1C0);
  if (!CheckRange(xclbin_size, 0x1C8, uint64_t{40} * section_count,
                  "AXLF section table", out_error)) {
    return false;
  }

  out_sections->clear();
  out_sections->reserve(section_count);
  for (uint32_t i = 0; i < section_count; ++i) {
    size_t record = 0x1C8 + size_t{i} * 40;
    AxlfSection section;
    section.kind = ReadU32(xclbin, record);
    section.name = CString(xclbin + record + 4, 16);
    section.offset = ReadU64(xclbin, record + 24);
    section.size = ReadU64(xclbin, record + 32);
    if (!CheckRange(xclbin_size, section.offset, section.size,
                    "AXLF section", out_error)) {
      return false;
    }
    out_sections->push_back(section);
  }
  return true;
}

const AxlfSection* FindFirstSection(const std::vector<AxlfSection>& sections,
                                    uint32_t kind) {
  for (const AxlfSection& section : sections) {
    if (section.kind == kind) return &section;
  }
  return nullptr;
}

bool ExtractJsonStringAfter(const std::string& json, size_t start,
                            const char* key, std::string* out_value) {
  size_t key_pos = json.find(key, start);
  if (key_pos == std::string::npos) return false;
  size_t colon = json.find(':', key_pos + std::strlen(key));
  if (colon == std::string::npos) return false;
  size_t quote = json.find('"', colon + 1);
  if (quote == std::string::npos) return false;
  std::string value;
  for (size_t i = quote + 1; i < json.size(); ++i) {
    char c = json[i];
    if (c == '"') {
      *out_value = value;
      return true;
    }
    if (c == '\\' && i + 1 < json.size()) {
      value.push_back(json[++i]);
    } else {
      value.push_back(c);
    }
  }
  return false;
}

std::string DeriveKernelName(const uint8_t* xclbin,
                             const std::vector<AxlfSection>& sections) {
  const AxlfSection* section = FindFirstSection(sections, kBuildMetadataSection);
  if (!section) return "kernel";
  std::string json(reinterpret_cast<const char*>(xclbin + section->offset),
                   static_cast<size_t>(section->size));

  size_t kernels_pos = json.find("\"kernels\"");
  std::string name;
  if (kernels_pos != std::string::npos &&
      ExtractJsonStringAfter(json, kernels_pos, "\"name\"", &name) &&
      !name.empty()) {
    return name;
  }

  if (ExtractJsonStringAfter(json, 0, "\"xclbin_name\"", &name) &&
      !name.empty()) {
    constexpr char kLinkSuffix[] = ".link";
    if (name.size() > std::strlen(kLinkSuffix) &&
        name.compare(name.size() - std::strlen(kLinkSuffix),
                     std::strlen(kLinkSuffix), kLinkSuffix) == 0) {
      name.resize(name.size() - std::strlen(kLinkSuffix));
    }
    return name;
  }

  return "kernel";
}

bool ParseAiePartition(const uint8_t* xclbin, size_t xclbin_size,
                       const std::vector<AxlfSection>& sections,
                       ContextBlobInfo* info, std::string* out_error) {
  const AxlfSection* section = FindFirstSection(sections, kAiePartitionSection);
  if (!section) return Fail("AIE_PARTITION AXLF section is missing", out_error);
  if (!CheckRange(xclbin_size, section->offset, section->size,
                  "AIE_PARTITION section", out_error)) {
    return false;
  }
  const uint8_t* data = xclbin + section->offset;
  size_t size = static_cast<size_t>(section->size);
  if (size < 0xC8) {
    return Fail("AIE_PARTITION section is too small", out_error);
  }

  info->column_width = ReadU16(data, 32);
  uint32_t start_columns_count = ReadU32(data, 40);
  uint32_t start_columns_offset = ReadU32(data, 44);
  if (start_columns_count == 0) {
    info->start_column = 0;
  } else {
    if (!CheckRange(size, start_columns_offset, 2, "start_columns",
                    out_error)) {
      return false;
    }
    info->start_column = ReadU16(data, start_columns_offset);
  }

  uint32_t pdi_count = ReadU32(data, 120);
  uint32_t pdi_offset = ReadU32(data, 124);
  if (pdi_count != 1) {
    std::ostringstream os;
    os << "prototype context builder only supports one-PDI xclbins; got "
       << pdi_count;
    return Fail(os.str(), out_error);
  }
  if (!CheckRange(size, pdi_offset, 0x60, "AIE_PARTITION PDI table",
                  out_error)) {
    return false;
  }

  uint32_t cdo_count = ReadU32(data, pdi_offset + 24);
  uint32_t cdo_offset = ReadU32(data, pdi_offset + 28);
  if (cdo_count == 0 ||
      !CheckRange(size, cdo_offset, 0x70, "AIE_PARTITION CDO table",
                  out_error)) {
    return Fail("AIE_PARTITION CDO table is missing", out_error);
  }
  uint32_t pdi_name_offset = ReadU32(data, cdo_offset);
  if (!CheckRange(size, pdi_name_offset, 64, "AIE_PARTITION CDO name",
                  out_error)) {
    return false;
  }
  info->pdi_name = CString(data + pdi_name_offset, 64);

  uint32_t kernel_count = ReadU32(data, cdo_offset + 16);
  uint32_t kernel_offset = ReadU32(data, cdo_offset + 20);
  if (kernel_count == 0 ||
      !CheckRange(size, kernel_offset, 8, "AIE_PARTITION kernel id table",
                  out_error)) {
    return Fail("AIE_PARTITION kernel id table is missing", out_error);
  }
  info->dpu_kernel_id = ReadU64(data, kernel_offset);
  return true;
}

}  // namespace

bool BuildContextPrivateDataFromXclbin(const uint8_t* xclbin,
                                       size_t xclbin_size, uint32_t process_id,
                                       std::vector<uint8_t>* out_blob,
                                       ContextBlobInfo* out_info,
                                       std::string* out_error) {
  if (!xclbin || !out_blob) return Fail("invalid output/context arguments", out_error);
  if (xclbin_size < 0x1B0) return Fail("xclbin is too small", out_error);

  std::vector<AxlfSection> sections;
  if (!ParseSections(xclbin, xclbin_size, &sections, out_error)) return false;

  ContextBlobInfo info;
  info.kernel_name = DeriveKernelName(xclbin, sections);
  if (!ParseAiePartition(xclbin, xclbin_size, sections, &info, out_error)) {
    return false;
  }

  size_t total_size = kAxlfBaseOffset + xclbin_size + kTailSizeOnePdi;
  std::vector<uint8_t> blob(total_size, 0);

  std::memcpy(blob.data(), xclbin + 0x1A0, 16);
  WriteU64(&blob, 0x48, kCommandApertureBase);
  WriteU64(&blob, 0x50, 0x48);
  WriteU64(&blob, 0x58, total_size - 0x80);
  WriteU64(&blob, 0x60, process_id);
  WriteU64(&blob, 0x80, 1);
  WriteU64(&blob, 0xC8, kContextCommandBoSize);
  WriteU64(&blob, 0xD0, xclbin_size);
  WriteU64(&blob, 0xD8, total_size - 0x138);
  WriteU64(&blob, 0xE0, total_size - 0xE8);
  std::memcpy(blob.data() + kAxlfBaseOffset, xclbin, xclbin_size);

  size_t tail = kAxlfBaseOffset + xclbin_size;
  if (!WriteCString(&blob, tail + 0x00, 64, info.kernel_name, out_error)) {
    return false;
  }
  blob[tail + 0x3F] = '0';
  WriteU64(&blob, tail + 0x40, kContextCommandBoSize);
  WriteU64(&blob, tail + 0x48, info.column_width);

  if (!WriteCString(&blob, tail + 0x1C0, 64, info.pdi_name, out_error)) {
    return false;
  }
  blob[tail + 0x1FF] = '0';
  WriteU64(&blob, tail + 0x200, info.dpu_kernel_id << 8);
  WriteU64(&blob, tail + 0x208, 8);
  WriteU64(&blob, tail + 0x218, info.dpu_kernel_id);

  WriteU32(&blob, tail + 0x520, 0x4000);
  WriteU32(&blob, tail + 0x524, info.start_column + 1);
  WriteU32(&blob, tail + 0x528, info.column_width);

  if (out_info) *out_info = info;
  *out_blob = std::move(blob);
  return true;
}

}  // namespace iree::hal::amdxdna::mcdm
