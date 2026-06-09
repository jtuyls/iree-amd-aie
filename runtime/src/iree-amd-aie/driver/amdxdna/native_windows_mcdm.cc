// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/native.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/ert.h"
#include "iree-amd-aie/driver/amdxdna/shim/windows/mcdm/context_blob.h"
#include "iree-amd-aie/driver/amdxdna/shim/windows/mcdm/kmt_api.h"

namespace mcdm = iree::hal::amdxdna::mcdm;

namespace {

constexpr uint64_t kMaxExecBoSize = 4096;
constexpr uint32_t kXgqCmdOpStartCuIdx = 0x100;
constexpr uint32_t kXgqSqCmdNew = 1;
constexpr uint32_t kXgqCuDomainPl = 0;
constexpr size_t kXgqHeaderWords = 2;
// The working Windows/XRT IREE matmul capture submits START_CU/type CU with
// payload count 0x12: one CU mask plus 17 data words. The XML ABI names fewer
// arguments, but XRT pads the tail with zeros and the driver receives 0x4c
// packet bytes.
constexpr uint32_t kWindowsDpuRegmapWords = 17;
constexpr uint32_t kWindowsDpuInstructionRegWord = 2;
constexpr uint64_t kWindowsDpuInstructionApertureOffset = 0x8000;
constexpr uint64_t kWindowsDpuAperturePacketOffset = 0;
// XRT's module-runlist path presents each START_NPU child instruction stream at
// a 0x8000-spaced aperture VA (0x04008000, 0x04010000, ...). The path-B parent
// chain descriptor is accepted for one packed child, but multi-child START_NPU
// chains fail unless we preserve this slot cadence.
constexpr size_t kWindowsDpuChainCodeAlignment = 0x8000;
constexpr uint64_t kWindowsDpuChainDescriptorApertureOffset = 0x10000;
constexpr size_t kWindowsDpuChainDescriptorHeaderSize = 0x34;
constexpr size_t kWindowsDpuStartNpuChainDescriptorSize = 0x3c;
// XRT runlists are logically unbounded but internally submitted in fixed-size
// ERT_CMD_CHAIN chunks. XRT 2.19 hardwires that native submit chunk size to 24,
// so the Windows MCDM shim uses the same value for now. The recovered path-B
// descriptor envelope has observed headroom up to 34 children, but 34 is not the
// default until we can prove it is compatible with XRT's intended contract.
// This is a per-native-submit chunk size, not a logical command-chain limit;
// larger logical chains are split by direct_command_buffer.cc before reaching
// this layer.
constexpr size_t kWindowsDpuRunlistSubmitSize = 24;
constexpr uint64_t kWindowsDpuPathBExecBoSize = 0xe0;

struct SubmitTimingStats {
  std::atomic<uint64_t> count{0};
  std::atomic<uint64_t> total_ns{0};
  std::atomic<uint64_t> min_ns{std::numeric_limits<uint64_t>::max()};
  std::atomic<uint64_t> max_ns{0};
};

SubmitTimingStats& submit_timing_stats() {
  static SubmitTimingStats stats;
  return stats;
}

uint64_t submit_timing_skip_count() {
  return 0;
}

void record_submit_timing(uint64_t ns) {
  static std::atomic<uint64_t> seen{0};
  const uint64_t seen_index = seen.fetch_add(1, std::memory_order_relaxed);
  if (seen_index < submit_timing_skip_count()) return;
  auto& stats = submit_timing_stats();
  stats.count.fetch_add(1, std::memory_order_relaxed);
  stats.total_ns.fetch_add(ns, std::memory_order_relaxed);
  uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
  while (ns < min_ns &&
         !stats.min_ns.compare_exchange_weak(min_ns, ns,
                                             std::memory_order_relaxed)) {
  }
  uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);
  while (ns > max_ns &&
         !stats.max_ns.compare_exchange_weak(max_ns, ns,
                                             std::memory_order_relaxed)) {
  }
}

struct SubmitTimingReporter {
  ~SubmitTimingReporter() {
    auto& stats = submit_timing_stats();
    uint64_t count = stats.count.load(std::memory_order_relaxed);
    if (!count) return;
    uint64_t total_ns = stats.total_ns.load(std::memory_order_relaxed);
    uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
    uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[amdxdna:mcdm-submit-timing] section=pathb_submit_wait "
                 "count=%llu mean_us=%.3f min_us=%.3f max_us=%.3f "
                 "total_us=%.3f\n",
                 static_cast<unsigned long long>(count),
                 static_cast<double>(total_ns) / count / 1000.0,
                 static_cast<double>(min_ns) / 1000.0,
                 static_cast<double>(max_ns) / 1000.0,
                 static_cast<double>(total_ns) / 1000.0);
  }
};

bool submit_timing_enabled() {
  return false;
}

enum class NativeSubmitTimingSection : uint8_t {
  total = 0,
  finalize_regmap,
  non_pathb_exec_materialize,
  bound_residency,
  pathb_pre_bound_sync,
  pathb_pre_sync,
  pathb_stage_code,
  pathb_ensure_dummy,
  pathb_exec_materialize,
  pathb_bo_table,
  pathb_exec_sync,
  pathb_submit,
  pathb_post_sync,
  pathb_post_bound_sync,
  final_exec_sync,
  count,
};

const char* native_submit_timing_section_name(
    NativeSubmitTimingSection section) {
  switch (section) {
    case NativeSubmitTimingSection::total:
      return "total";
    case NativeSubmitTimingSection::finalize_regmap:
      return "finalize_regmap";
    case NativeSubmitTimingSection::non_pathb_exec_materialize:
      return "non_pathb_exec_materialize";
    case NativeSubmitTimingSection::bound_residency:
      return "bound_residency";
    case NativeSubmitTimingSection::pathb_pre_bound_sync:
      return "pathb_pre_bound_sync";
    case NativeSubmitTimingSection::pathb_pre_sync:
      return "pathb_pre_sync";
    case NativeSubmitTimingSection::pathb_stage_code:
      return "pathb_stage_code";
    case NativeSubmitTimingSection::pathb_ensure_dummy:
      return "pathb_ensure_dummy";
    case NativeSubmitTimingSection::pathb_exec_materialize:
      return "pathb_exec_materialize";
    case NativeSubmitTimingSection::pathb_bo_table:
      return "pathb_bo_table";
    case NativeSubmitTimingSection::pathb_exec_sync:
      return "pathb_exec_sync";
    case NativeSubmitTimingSection::pathb_submit:
      return "pathb_submit";
    case NativeSubmitTimingSection::pathb_post_sync:
      return "pathb_post_sync";
    case NativeSubmitTimingSection::pathb_post_bound_sync:
      return "pathb_post_bound_sync";
    case NativeSubmitTimingSection::final_exec_sync:
      return "final_exec_sync";
    case NativeSubmitTimingSection::count:
      break;
  }
  return "unknown";
}

SubmitTimingStats* native_submit_timing_stats() {
  static SubmitTimingStats stats[static_cast<size_t>(
      NativeSubmitTimingSection::count)];
  return stats;
}

void record_native_submit_timing(NativeSubmitTimingSection section,
                                 uint64_t ns) {
  auto& stats = native_submit_timing_stats()[static_cast<size_t>(section)];
  stats.count.fetch_add(1, std::memory_order_relaxed);
  stats.total_ns.fetch_add(ns, std::memory_order_relaxed);
  uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
  while (ns < min_ns &&
         !stats.min_ns.compare_exchange_weak(min_ns, ns,
                                             std::memory_order_relaxed)) {
  }
  uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);
  while (ns > max_ns &&
         !stats.max_ns.compare_exchange_weak(max_ns, ns,
                                             std::memory_order_relaxed)) {
  }
}

struct NativeSubmitTimingReporter {
  ~NativeSubmitTimingReporter() {
    SubmitTimingStats* stats = native_submit_timing_stats();
    for (size_t i = 0;
         i < static_cast<size_t>(NativeSubmitTimingSection::count); ++i) {
      uint64_t count = stats[i].count.load(std::memory_order_relaxed);
      if (!count) continue;
      uint64_t total_ns = stats[i].total_ns.load(std::memory_order_relaxed);
      uint64_t min_ns = stats[i].min_ns.load(std::memory_order_relaxed);
      uint64_t max_ns = stats[i].max_ns.load(std::memory_order_relaxed);
      auto section = static_cast<NativeSubmitTimingSection>(i);
      std::fprintf(stderr,
                   "[amdxdna:native-submit-timing] phase=%s count=%llu "
                   "mean_us=%.3f min_us=%.3f max_us=%.3f total_us=%.3f\n",
                   native_submit_timing_section_name(section),
                   static_cast<unsigned long long>(count),
                   static_cast<double>(total_ns) / count / 1000.0,
                   static_cast<double>(min_ns) / 1000.0,
                   static_cast<double>(max_ns) / 1000.0,
                   static_cast<double>(total_ns) / 1000.0);
    }
  }
};

bool native_submit_timing_enabled() {
  return false;
}

uint64_t now_ns();

struct NativeSubmitTimingScope {
  explicit NativeSubmitTimingScope(NativeSubmitTimingSection section)
      : section(section), enabled(native_submit_timing_enabled()) {
    if (enabled) start_ns = now_ns();
  }

  ~NativeSubmitTimingScope() {
    if (!enabled) return;
    record_native_submit_timing(section, now_ns() - start_ns);
  }

  NativeSubmitTimingSection section;
  bool enabled = false;
  uint64_t start_ns = 0;
};

uint64_t now_ns() {
  return 0;
}

enum class DiagnosticStage : uint8_t {
  none = 0,
  load_api,
  find_adapter,
  create_device,
  alloc_buffer,
  context_blob,
  create_context,
  open_cu,
  create_command,
  sync_buffer,
  ready_submit,
  stage_aperture,
  submit,
  trace,
};

enum class SubmitMode : uint8_t {
  qhdl,
  direct,
  aperture,
  pathb,
};

struct BoundBuffer {
  size_t position = 0;
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  iree_device_size_t offset = 0;
  iree_device_size_t size = 0;
};

std::string string_view_to_string(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string normalize_cu_name(std::string name) {
  size_t instance_separator = name.find(':');
  if (instance_separator != std::string::npos) {
    name.resize(instance_separator);
  }
  return name;
}

bool env_flag_enabled(const char* name) {
  (void)name;
  return false;
}

bool env_flag_enabled_by_default(const char* name) {
  (void)name;
  return true;
}

uint64_t windows_dpu_pathb_chain_exec_bo_size() {
  return sizeof(ert_packet) + sizeof(ert_cmd_chain_data) +
         kWindowsDpuRunlistSubmitSize * sizeof(uint64_t);
}

uint32_t chain_slot_capacity(size_t exec_bo_size) {
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  return exec_bo_size > header
             ? static_cast<uint32_t>((exec_bo_size - header) / sizeof(uint64_t))
             : 1;
}

bool zero_instruction_size_enabled() {
  return false;
}

bool xrt_host_sfence_enabled() {
  return false;
}

bool xrt_code_stage_readback_enabled() {
  return false;
}

bool xrt_bound_readback_enabled() {
  return false;
}

bool xrt_bound_readback_include_outputs_enabled() {
  return false;
}

bool xrt_output_readback_enabled() {
  return false;
}

bool trace_qhdl_enabled() {
  return false;
}

bool skip_pathb_bound_sync_enabled() {
  return false;
}

bool skip_pathb_exec_sync_enabled() {
  return false;
}

bool pathb_chain_sync9_enabled() {
  return false;
}

bool skip_clean_chain_sync_enabled() {
  return true;
}

bool partial_elf_dummy_bos_enabled() {
  return true;
}

bool partial_elf_bo_table_enabled() {
  return true;
}

bool compact_execbuf_enabled() {
  return true;
}

uint32_t env_u32(const char* name, uint32_t default_value = 0) {
  (void)name;
  return default_value;
}

void flush_host_writes_to_mcdm() {
#if defined(_MSC_VER)
  if (xrt_host_sfence_enabled()) {
    _mm_sfence();
  }
#endif
  std::atomic_thread_fence(std::memory_order_seq_cst);
  FlushProcessWriteBuffers();
}

iree_status_t status_from_mcdm_error(const char* label,
                                     const std::string& error) {
  return iree_make_status(IREE_STATUS_INTERNAL, "%s: %s", label,
                          error.c_str());
}

const char* diagnostic_stage_name(DiagnosticStage stage) {
  switch (stage) {
    case DiagnosticStage::none:
      return "none";
    case DiagnosticStage::load_api:
      return "load-api";
    case DiagnosticStage::find_adapter:
      return "find-adapter";
    case DiagnosticStage::create_device:
      return "create-device";
    case DiagnosticStage::alloc_buffer:
      return "alloc-buffer";
    case DiagnosticStage::context_blob:
      return "context-blob";
    case DiagnosticStage::create_context:
      return "create-context";
    case DiagnosticStage::open_cu:
      return "open-cu";
    case DiagnosticStage::create_command:
      return "create-command";
    case DiagnosticStage::sync_buffer:
      return "sync-buffer";
    case DiagnosticStage::ready_submit:
      return "ready-submit";
    case DiagnosticStage::stage_aperture:
      return "stage-aperture";
    case DiagnosticStage::submit:
      return "submit";
    case DiagnosticStage::trace:
      return "trace";
  }
  return "unknown";
}

bool parse_diagnostic_stage(iree_string_view_t value,
                            DiagnosticStage* out_stage) {
  *out_stage = DiagnosticStage::none;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("load-api"))) {
    *out_stage = DiagnosticStage::load_api;
  } else if (iree_string_view_equal(value, IREE_SV("find-adapter"))) {
    *out_stage = DiagnosticStage::find_adapter;
  } else if (iree_string_view_equal(value, IREE_SV("create-device"))) {
    *out_stage = DiagnosticStage::create_device;
  } else if (iree_string_view_equal(value, IREE_SV("alloc-buffer"))) {
    *out_stage = DiagnosticStage::alloc_buffer;
  } else if (iree_string_view_equal(value, IREE_SV("context-blob"))) {
    *out_stage = DiagnosticStage::context_blob;
  } else if (iree_string_view_equal(value, IREE_SV("create-context"))) {
    *out_stage = DiagnosticStage::create_context;
  } else if (iree_string_view_equal(value, IREE_SV("open-cu"))) {
    *out_stage = DiagnosticStage::open_cu;
  } else if (iree_string_view_equal(value, IREE_SV("create-command"))) {
    *out_stage = DiagnosticStage::create_command;
  } else if (iree_string_view_equal(value, IREE_SV("sync-buffer"))) {
    *out_stage = DiagnosticStage::sync_buffer;
  } else if (iree_string_view_equal(value, IREE_SV("ready-submit"))) {
    *out_stage = DiagnosticStage::ready_submit;
  } else if (iree_string_view_equal(value, IREE_SV("stage-aperture"))) {
    *out_stage = DiagnosticStage::stage_aperture;
  } else if (iree_string_view_equal(value, IREE_SV("submit"))) {
    *out_stage = DiagnosticStage::submit;
  } else if (iree_string_view_equal(value, IREE_SV("trace"))) {
    *out_stage = DiagnosticStage::trace;
  } else {
    return false;
  }
  return true;
}

bool parse_submit_mode(iree_string_view_t value, SubmitMode* out_mode) {
  *out_mode = SubmitMode::direct;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("direct"))) {
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("qhdl"))) {
    *out_mode = SubmitMode::qhdl;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("aperture"))) {
    *out_mode = SubmitMode::aperture;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("pathb"))) {
    *out_mode = SubmitMode::pathb;
    return true;
  }
  return false;
}

iree_status_t require_submit_mode_opt_in(SubmitMode submit_mode) {
  switch (submit_mode) {
    case SubmitMode::direct:
      return iree_ok_status();
    case SubmitMode::aperture:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna_mcdm_submit_mode=aperture is disabled because "
          "the Windows MCDM exec-BO staging contract is not fully mapped yet; "
          "use the standalone MCDM probe tools for controlled experiments");
    case SubmitMode::qhdl:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna_mcdm_submit_mode=qhdl is disabled because the "
          "0x268 qhdl block is an internal XRT object layout, not the public "
          "KMT submit packet; use the standalone MCDM probe tools for "
          "controlled experiments");
    case SubmitMode::pathb:
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unknown amdxdna Windows MCDM submit mode");
}

iree_status_t validate_device_size_fits_u64(iree_device_size_t size) {
  if (IREE_UNLIKELY(size > std::numeric_limits<uint64_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native allocation size is too large");
  }
  return iree_ok_status();
}

iree_status_t parse_power_mode(
    iree_string_view_t power_mode,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_should_set_power_mode = false;
  *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  if (iree_string_view_is_empty(power_mode)) return iree_ok_status();

  *out_should_set_power_mode = true;
  if (iree_string_view_equal(power_mode, IREE_SV("default"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  } else if (iree_string_view_equal(power_mode, IREE_SV("low"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::low;
  } else if (iree_string_view_equal(power_mode, IREE_SV("medium"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::medium;
  } else if (iree_string_view_equal(power_mode, IREE_SV("high"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::high;
  } else if (iree_string_view_equal(power_mode, IREE_SV("turbo"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::turbo;
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_power_mode' expected to be default | low | "
        "medium | high | turbo but got '%.*s'",
        static_cast<int>(power_mode.size), power_mode.data);
  }
  return iree_ok_status();
}

mcdm::BufferKind to_mcdm_buffer_kind(
    iree_hal_amdxdna_native_buffer_type_t type) {
  switch (type) {
    case iree_hal_amdxdna_native_buffer_type_t::host_only:
      return mcdm::BufferKind::host_only;
    case iree_hal_amdxdna_native_buffer_type_t::cacheable:
      return mcdm::BufferKind::cacheable;
  }
  return mcdm::BufferKind::host_only;
}

uint32_t to_ert_opcode(iree_hal_amdxdna_native_command_opcode_t opcode) {
  switch (opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return ERT_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu:
      // XRT's non-ELF DPU/TXN path submits DPU kernels as START_CU packets
      // with type ERT_CU. The NPU operation selector is arg0 in the xclbin XML
      // register map, not the ERT packet opcode.
      return ERT_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf:
      return ERT_START_NPU;
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      return ERT_CMD_CHAIN;
  }
  return ERT_START_CU;
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command);

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command);

uint32_t first_set_bit(uint32_t value) {
  for (uint32_t i = 0; i < 32; ++i) {
    if (value & (uint32_t{1} << i)) return i;
  }
  return 0;
}

size_t align_up_size(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  mcdm::KmtApi api;
  mcdm::Device device;
  DiagnosticStage diagnostic_stop_after = DiagnosticStage::none;
  SubmitMode submit_mode = SubmitMode::direct;
  bool pathb_context_ready = false;
  std::vector<iree_hal_amdxdna_native_buffer_ptr> partial_elf_dummy_buffers;

  iree_hal_amdxdna_native_device_t(iree_allocator_t host_allocator,
                                   DiagnosticStage diagnostic_stop_after,
                                   SubmitMode submit_mode)
      : host_allocator(host_allocator),
        diagnostic_stop_after(diagnostic_stop_after),
        submit_mode(submit_mode) {}
};

struct iree_hal_amdxdna_native_buffer_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Buffer buffer;
  iree_hal_amdxdna_native_buffer_type_t type =
      iree_hal_amdxdna_native_buffer_type_t::host_only;
  bool deferred = false;
  std::vector<uint8_t> deferred_storage;

  iree_hal_amdxdna_native_buffer_t(
      iree_hal_amdxdna_native_device_t* device, mcdm::Buffer buffer)
      : device(device),
        buffer(buffer),
        type(iree_hal_amdxdna_native_buffer_type_t::host_only) {}

  iree_hal_amdxdna_native_buffer_t(
      iree_hal_amdxdna_native_device_t* device,
      iree_hal_amdxdna_native_buffer_type_t type, uint64_t size)
      : device(device),
        type(type),
        deferred(true),
        deferred_storage(static_cast<size_t>(size)) {
    buffer.kind = to_mcdm_buffer_kind(type);
    buffer.size = size;
    buffer.cpu_ptr = deferred_storage.data();
  }

  iree_hal_amdxdna_native_buffer_t(
      iree_hal_amdxdna_native_device_t* device, mcdm::BufferKind kind,
      uint64_t size)
      : device(device),
        type(iree_hal_amdxdna_native_buffer_type_t::cacheable),
        deferred(true),
        deferred_storage(static_cast<size_t>(size)) {
    buffer.kind = kind;
    buffer.size = size;
    buffer.cpu_ptr = deferred_storage.data();
  }
};

struct iree_hal_amdxdna_native_queue_t {
  iree_hal_amdxdna_native_context_t* context = nullptr;
  uint64_t exec_command_count = 0;
};

struct iree_hal_amdxdna_native_context_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Context context;
  mcdm::CommandControlBuffer command_control;
  bool has_command_control = false;
  mcdm::CommandAperture command_aperture;
  bool has_command_aperture = false;
  bool pathb_single_code_staged = false;
  iree_device_size_t pathb_single_code_staged_size = 0;
  std::vector<uint32_t> pathb_single_code_words;
  mcdm::ContextBlobInfo info;
  iree_hal_amdxdna_native_queue_t queue;

  iree_hal_amdxdna_native_context_t(
      iree_hal_amdxdna_native_device_t* device, mcdm::Context context,
      mcdm::CommandControlBuffer command_control, bool has_command_control,
      mcdm::CommandAperture command_aperture, bool has_command_aperture,
      mcdm::ContextBlobInfo info)
      : device(device),
        context(context),
        command_control(command_control),
        has_command_control(has_command_control),
        command_aperture(command_aperture),
        has_command_aperture(has_command_aperture),
        info(std::move(info)) {
    queue.context = this;
  }
};

struct iree_hal_amdxdna_native_command_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  iree_hal_amdxdna_native_command_opcode_t opcode;
  iree_hal_amdxdna_native_buffer_ptr exec_buffer;
  iree_hal_amdxdna_native_buffer_t* control_buffer = nullptr;
  iree_device_size_t control_buffer_size = 0;
  ert_start_kernel_cmd* start_packet = nullptr;
  size_t command_size = 0;
  uint32_t reg_idx = 0;
  uint32_t arg_count = 0;
  bool windows_dpu_regmap_finalized = false;
  bool pathb_code_staged = false;
  iree_device_size_t pathb_code_staged_size = 0;
  uint64_t pathb_chain_descriptor_gpu_va = 0;
  uint32_t pathb_chain_descriptor_bytes = 0;
  uint32_t pathb_chain_first_child_opcode = 0;
  uint64_t pathb_chain_code_used_size = 0;
  uint64_t pathb_chain_code_aperture_offset = 0;
  uint64_t pathb_chain_descriptor_aperture_offset = 0;
  bool pathb_chain_allow_code_dedup = true;
  bool pathb_chain_prepared_valid = false;
  bool pathb_chain_code_dirty = false;
  bool pathb_chain_descriptor_dirty = false;
  bool pathb_chain_bound_residency_checked = false;
  std::vector<size_t> pathb_chain_child_code_offsets;
  std::vector<iree_hal_amdxdna_native_command_t*> chain_children;
  std::vector<BoundBuffer> bound_buffers;

  iree_hal_amdxdna_native_command_t(
      iree_hal_amdxdna_native_device_t* device,
      iree_hal_amdxdna_native_command_opcode_t opcode,
      iree_hal_amdxdna_native_buffer_ptr exec_buffer)
      : device(device),
        opcode(opcode),
        exec_buffer(std::move(exec_buffer)),
        start_packet(reinterpret_cast<ert_start_kernel_cmd*>(
            this->exec_buffer->buffer.cpu_ptr)),
        command_size(static_cast<size_t>(this->exec_buffer->buffer.size)) {}
};

namespace {

bool pathb_stage_code_after_presync(
    iree_hal_amdxdna_native_command_t* command) {
  return command && command->device &&
         command->device->submit_mode == SubmitMode::pathb;
}

bool diagnostic_enabled(iree_hal_amdxdna_native_device_t* device) {
  return device &&
         device->diagnostic_stop_after != DiagnosticStage::none;
}

iree_status_t diagnostic_after(iree_hal_amdxdna_native_device_t* device,
                               DiagnosticStage stage) {
  if (!diagnostic_enabled(device)) return iree_ok_status();
  std::fprintf(stderr, "[amdxdna:mcdm] reached stage: %s\n",
               diagnostic_stage_name(stage));
  std::fflush(stderr);
  if (device->diagnostic_stop_after != stage) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_CANCELLED,
      "amdxdna Windows MCDM diagnostic stop after stage '%s'",
      diagnostic_stage_name(stage));
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command) {
  return command->start_packet;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(command_start_packet(command));
}

bool diagnostic_trace_packets(iree_hal_amdxdna_native_device_t* device) {
  return diagnostic_enabled(device);
}

void trace_command_packet(const char* phase,
                          iree_hal_amdxdna_native_command_t* command) {
  if (!diagnostic_trace_packets(command->device)) return;

  const mcdm::Buffer& exec = command->exec_buffer->buffer;
  const mcdm::BufferKindInfo exec_kind = mcdm::GetBufferKindInfo(exec.kind);
  ert_packet* packet = command_packet(command);
  ert_start_kernel_cmd* start_packet = command_start_packet(command);
  const size_t packet_size = std::min<size_t>(
      get_ert_packet_size_bytes(packet), command->command_size);
  const size_t word_count =
      std::min<size_t>((packet_size + sizeof(uint32_t) - 1) / sizeof(uint32_t),
                       24);

  std::fprintf(stderr,
               "[amdxdna:mcdm] packet %s: exec kind=%s alloc=0x%08x "
               "gpu_va=0x%llx size=%llu packet_bytes=%zu valid=%u\n",
               phase, exec_kind.name, static_cast<unsigned>(exec.allocation),
               static_cast<unsigned long long>(exec.gpu_va),
               static_cast<unsigned long long>(exec.size), packet_size,
               ert_valid_opcode(packet) ? 1u : 0u);
  std::fprintf(stderr,
               "[amdxdna:mcdm] packet %s: header=0x%08x state=%u opcode=%u "
               "type=%u count=%u extra_cu_masks=%u cu_mask=0x%08x "
               "reg_idx=%u arg_count=%u bound_count=%zu\n",
               phase, packet->header, packet->state, packet->opcode,
               packet->type, packet->count, start_packet->extra_cu_masks,
               start_packet->cu_mask, command->reg_idx, command->arg_count,
               command->bound_buffers.size());

  if (packet->opcode == ERT_START_NPU) {
    const ert_npu_data* npu_data = get_ert_npu_data(start_packet);
    if (npu_data) {
      std::fprintf(stderr,
                   "[amdxdna:mcdm] packet %s: npu instruction_va=0x%llx "
                   "instruction_size=%u prop_count=%u\n",
                   phase,
                   static_cast<unsigned long long>(
                       npu_data->instruction_buffer),
                   npu_data->instruction_buffer_size,
                   npu_data->instruction_prop_count);
    }
  } else if (command->opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu) {
    const uint32_t* regmap = get_ert_regmap_begin(start_packet);
    const uint64_t instruction_va =
        static_cast<uint64_t>(regmap[kWindowsDpuInstructionRegWord]) |
        (static_cast<uint64_t>(regmap[kWindowsDpuInstructionRegWord + 1])
         << 32);
    std::fprintf(stderr,
                 "[amdxdna:mcdm] packet %s: windows-dpu "
                 "instruction_va=0x%llx instruction_words=%u\n",
                 phase, static_cast<unsigned long long>(instruction_va),
                 regmap[kWindowsDpuInstructionRegWord + 2]);
  }

  const uint32_t* words = reinterpret_cast<const uint32_t*>(packet);
  std::fprintf(stderr, "[amdxdna:mcdm] packet %s: words", phase);
  for (size_t i = 0; i < word_count; ++i) {
    std::fprintf(stderr, " %08x", words[i]);
  }
  std::fprintf(stderr, "\n");

  for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer) continue;
    const mcdm::Buffer& buffer = bound.buffer->buffer;
    const mcdm::BufferKindInfo kind = mcdm::GetBufferKindInfo(buffer.kind);
    std::fprintf(stderr,
                 "[amdxdna:mcdm] packet %s: bound[%zu] pos=%zu kind=%s "
                 "alloc=0x%08x gpu_va=0x%llx offset=%llu size=%llu "
                 "bo_size=%llu\n",
                 phase, i, bound.position, kind.name,
                 static_cast<unsigned>(buffer.allocation),
                 static_cast<unsigned long long>(buffer.gpu_va),
                 static_cast<unsigned long long>(bound.offset),
                 static_cast<unsigned long long>(bound.size),
                 static_cast<unsigned long long>(buffer.size));
  }
  std::fflush(stderr);
}

iree_status_t build_xgq_start_cuidx_words(
    iree_hal_amdxdna_native_command_t* command,
    std::vector<uint32_t>* out_words) {
  IREE_ASSERT_ARGUMENT(command);
  IREE_ASSERT_ARGUMENT(out_words);
  out_words->clear();

  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::start_cu)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM aperture submit only supports START_CU today");
  }

  ert_start_kernel_cmd* packet = command_start_packet(command);
  const uint32_t mask_words = 1 + packet->extra_cu_masks;
  const uint32_t skipped_control_words = 4;
  if (IREE_UNLIKELY(packet->count <
                    mask_words + skipped_control_words)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM START_CU packet is too small for XGQ "
        "translation");
  }

  const uint32_t payload_words =
      packet->count - mask_words - skipped_control_words;
  const uint32_t payload_bytes = payload_words * sizeof(uint32_t);
  if (IREE_UNLIKELY(payload_bytes > 0x7fff)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM XGQ payload is too large");
  }

  out_words->resize(kXgqHeaderWords + payload_words);
  (*out_words)[0] = (kXgqSqCmdNew << 31) | (payload_bytes << 16) |
                    kXgqCmdOpStartCuIdx;
  const uint32_t cu_idx = first_set_bit(packet->cu_mask);
  (*out_words)[1] = (cu_idx << 16) | (kXgqCuDomainPl << 28);

  const uint32_t* regmap = get_ert_regmap_begin(packet);
  const uint32_t* payload = regmap + skipped_control_words;
  if (payload_words > 0) {
    std::memcpy(out_words->data() + kXgqHeaderWords, payload,
                payload_bytes);
  }
  return iree_ok_status();
}

void trace_xgq_words(iree_hal_amdxdna_native_command_t* command,
                     const std::vector<uint32_t>& words,
                     const mcdm::CommandAperture& aperture) {
  if (!diagnostic_trace_packets(command->device)) return;
  const size_t word_count = std::min<size_t>(words.size(), 24);
  std::fprintf(stderr,
               "[amdxdna:mcdm] aperture xgq: alloc=0x%08x gpu_alloc=0x%08x "
               "gpu_va=0x%llx bytes=%zu aperture_size=%llu\n",
               static_cast<unsigned>(aperture.allocation),
               static_cast<unsigned>(aperture.gpu_allocation),
               static_cast<unsigned long long>(aperture.gpu_va),
               words.size() * sizeof(uint32_t),
               static_cast<unsigned long long>(aperture.allocation_size));
  std::fprintf(stderr, "[amdxdna:mcdm] aperture xgq: words");
  for (size_t i = 0; i < word_count; ++i) {
    std::fprintf(stderr, " %08x", words[i]);
  }
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

iree_status_t stage_xgq_command_aperture(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.cpu_ptr || aperture.allocation_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture is not CPU mapped");
  }

  std::vector<uint32_t> xgq_words;
  IREE_RETURN_IF_ERROR(build_xgq_start_cuidx_words(command, &xgq_words));
  const size_t xgq_bytes = xgq_words.size() * sizeof(uint32_t);
  if (IREE_UNLIKELY(xgq_bytes > aperture.allocation_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM XGQ packet does not fit command aperture");
  }

  std::memset(aperture.cpu_ptr, 0,
              static_cast<size_t>(aperture.allocation_size));
  std::memcpy(aperture.cpu_ptr, xgq_words.data(), xgq_bytes);
  flush_host_writes_to_mcdm();
  trace_xgq_words(command, xgq_words, aperture);
  return iree_ok_status();
}

void set_windows_dpu_instruction_arg(
    iree_hal_amdxdna_native_command_t* command, uint64_t instruction_va,
    uint32_t instruction_words);

void trace_windows_dpu_aperture(iree_hal_amdxdna_native_command_t* command,
                                const mcdm::CommandAperture& aperture,
                                size_t packet_bytes,
                                uint64_t instruction_va,
                                size_t instruction_bytes) {
  if (!diagnostic_trace_packets(command->device)) return;
  const size_t word_count = std::min<size_t>(
      (packet_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t), 24);
  const uint32_t* words =
      reinterpret_cast<const uint32_t*>(aperture.cpu_ptr);
  std::fprintf(stderr,
               "[amdxdna:mcdm] aperture windows-dpu: alloc=0x%08x "
               "gpu_alloc=0x%08x gpu_va=0x%llx gpu_size=0x%llx "
               "cpu_size=0x%llx packet_off=0x%llx packet_bytes=%zu "
               "instruction_va=0x%llx instruction_off=0x%llx "
               "instruction_bytes=%zu\n",
               static_cast<unsigned>(aperture.allocation),
               static_cast<unsigned>(aperture.gpu_allocation),
               static_cast<unsigned long long>(aperture.gpu_va),
               static_cast<unsigned long long>(aperture.gpu_va_size),
               static_cast<unsigned long long>(aperture.cpu_ptr_size),
               static_cast<unsigned long long>(kWindowsDpuAperturePacketOffset),
               packet_bytes,
               static_cast<unsigned long long>(instruction_va),
               static_cast<unsigned long long>(
                   kWindowsDpuInstructionApertureOffset),
               instruction_bytes);
  std::fprintf(stderr, "[amdxdna:mcdm] aperture windows-dpu: words");
  for (size_t i = 0; i < word_count; ++i) {
    std::fprintf(stderr, " %08x", words[i]);
  }
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

iree_status_t stage_windows_dpu_command_aperture(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.cpu_ptr || aperture.cpu_ptr_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture is not CPU mapped");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU submit has no CPU-visible control buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control buffer size is not word aligned");
  }

  // Real dispatch staging: aperture+0 is ZERO (no ERT packet there - RE'd: the
  // firmware reads the transaction binary from the resident code BO at
  // aperture+0x8000), and IREE's control code is copied into that code BO. The
  // prior empty-aperture isolation test could never execute a kernel; the
  // opcode-2 kick only runs whatever real command sits in the aperture window.
  uint8_t* aperture_bytes = reinterpret_cast<uint8_t*>(aperture.cpu_ptr);
  std::memset(aperture_bytes, 0, static_cast<size_t>(aperture.cpu_ptr_size));
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM command aperture has no control-code sub-buffer "
        "(CreateCommandAperture code BO must be enabled)");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM control code exceeds the code sub-buffer");
  }
  std::memcpy(aperture.code_cpu_ptr, command->control_buffer->buffer.cpu_ptr,
              static_cast<size_t>(command->control_buffer_size));
  const uint64_t instruction_va = aperture.code_gpu_va;  // 0x04008000
  flush_host_writes_to_mcdm();
  trace_windows_dpu_aperture(command, aperture, 0, instruction_va,
                             static_cast<size_t>(command->control_buffer_size));
  return iree_ok_status();
}

iree_status_t stage_windows_dpu_code_buffer(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds aperture code BO");
  }
  const size_t control_word_count =
      static_cast<size_t>(command->control_buffer_size / sizeof(uint32_t));
  const auto* control_words = static_cast<const uint32_t*>(
      command->control_buffer->buffer.cpu_ptr);
  auto set_partial_elf_instruction_fields = [&]() -> iree_status_t {
    if (command->opcode !=
        iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf) {
      return iree_ok_status();
    }
    ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
    if (IREE_UNLIKELY(!npu_data)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
    }
    npu_data->instruction_buffer = aperture.code_gpu_va;
    npu_data->instruction_buffer_size =
        zero_instruction_size_enabled()
            ? 0
            : static_cast<uint32_t>(command->control_buffer_size);
    npu_data->instruction_prop_count = 0;
    return iree_ok_status();
  };
  auto context_code_matches = [&]() {
    if (!queue->context->pathb_single_code_staged ||
        queue->context->pathb_single_code_staged_size !=
            command->control_buffer_size ||
        queue->context->pathb_single_code_words.size() !=
            control_word_count) {
      return false;
    }
    return control_word_count == 0 ||
           std::memcmp(queue->context->pathb_single_code_words.data(),
                       control_words,
                       control_word_count * sizeof(uint32_t)) == 0;
  };
  if (context_code_matches()) {
    command->pathb_code_staged = true;
    command->pathb_code_staged_size = command->control_buffer_size;
    IREE_RETURN_IF_ERROR(set_partial_elf_instruction_fields());
    return iree_ok_status();
  }
  std::memset(aperture.code_cpu_ptr, 0, static_cast<size_t>(aperture.code_size));
  std::memcpy(aperture.code_cpu_ptr, command->control_buffer->buffer.cpu_ptr,
              static_cast<size_t>(command->control_buffer_size));
  flush_host_writes_to_mcdm();
  if (xrt_code_stage_readback_enabled()) {
    volatile const uint32_t* code_words =
        reinterpret_cast<volatile const uint32_t*>(aperture.code_cpu_ptr);
    volatile uint32_t checksum = 0;
    const size_t readback_words =
        std::min<size_t>(command->control_buffer_size / sizeof(uint32_t), 64);
    for (size_t i = 0; i < readback_words; ++i) {
      checksum ^= code_words[i];
    }
    if (trace_qhdl_enabled()) {
      std::fprintf(stderr,
                   "[amdxdna:mcdm] pathb code-readback: words=%zu "
                   "checksum=0x%08x\n",
                   readback_words, static_cast<uint32_t>(checksum));
      std::fflush(stderr);
    }
  }
  if (command && command->device &&
      command->device->submit_mode == SubmitMode::pathb) {
    std::string error;
    if (!mcdm::SyncCommandApertureCode(
            command->device->api, command->device->device, aperture,
            kWindowsDpuInstructionApertureOffset,
            static_cast<uint64_t>(command->control_buffer_size), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM aperture code sync failed", error);
    }
  }
  if (command && command->device &&
      command->device->submit_mode == SubmitMode::pathb) {
    std::string error;
    if (!mcdm::RefreshCommandApertureGpuMapping(
            command->device->api, command->device->device, &aperture,
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM aperture code relock failed", error);
    }
  }
  command->pathb_code_staged = true;
  command->pathb_code_staged_size = command->control_buffer_size;
  queue->context->pathb_single_code_staged = true;
  queue->context->pathb_single_code_staged_size = command->control_buffer_size;
  queue->context->pathb_single_code_words.assign(
      control_words, control_words + control_word_count);
  IREE_RETURN_IF_ERROR(set_partial_elf_instruction_fields());
  trace_windows_dpu_aperture(command, aperture, 0, aperture.code_gpu_va,
                             static_cast<size_t>(command->control_buffer_size));
  return iree_ok_status();
}

iree_status_t stage_command_aperture(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (command->opcode == iree_hal_amdxdna_native_command_opcode_t::start_npu) {
    return stage_windows_dpu_command_aperture(queue, command);
  }
  return stage_xgq_command_aperture(queue, command);
}

iree_status_t check_pkt_count_capacity(
    iree_hal_amdxdna_native_command_t* command, uint32_t bytes) {
  if (!command || !command->start_packet) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native command is not initialized");
  }
  uint32_t next_count =
      command->start_packet->count + bytes / sizeof(uint32_t);
  if (command->command_size <
      sizeof(command->start_packet->header) +
          static_cast<size_t>(next_count) * sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna native command packet is full");
  }
  return iree_ok_status();
}

iree_status_t inc_pkt_count(iree_hal_amdxdna_native_command_t* command,
                            uint32_t bytes) {
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, bytes));
  command->start_packet->count += bytes / sizeof(uint32_t);
  return iree_ok_status();
}

void bind_buffer_ref(iree_hal_amdxdna_native_command_t* command,
                     size_t position, iree_hal_amdxdna_native_buffer_t* buffer,
                     iree_device_size_t offset, iree_device_size_t size) {
  if (position == 0 &&
      command->opcode !=
          iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    command->bound_buffers.clear();
  }
  command->bound_buffers.push_back(BoundBuffer{position, buffer, offset, size});
}

bool is_pathb_partial_elf_control_binding(
    iree_hal_amdxdna_native_command_t* command, const BoundBuffer& bound) {
  return command && command->device &&
         command->device->submit_mode == SubmitMode::pathb &&
         command->opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf &&
         bound.position == 0 &&
         bound.buffer == command->control_buffer;
}

void readback_pathb_bound_buffers(iree_hal_amdxdna_native_command_t* command,
                                  const char* phase, bool outputs_only) {
  if (!(outputs_only ? xrt_output_readback_enabled()
                     : xrt_bound_readback_enabled())) {
    return;
  }
  for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer || !bound.buffer->buffer.cpu_ptr) continue;
    // Current Windows DPU regmap ABI is:
    //   opcode, ifm, param, ofm, ...
    // so bound positions 1 and 2 are host-written inputs, while 3 is the first
    // device-written output. Keep the diagnostic from pre-reading output zeros.
    const bool output_like_bound = bound.position >= 3;
    if (outputs_only) {
      if (!output_like_bound) continue;
    } else if (output_like_bound &&
               !xrt_bound_readback_include_outputs_enabled()) {
      continue;
    }
    const uint64_t buffer_size = bound.buffer->buffer.size;
    if (bound.offset >= buffer_size) continue;
    const uint64_t available = buffer_size - bound.offset;
    const uint64_t read_size =
        std::min<uint64_t>(bound.size ? bound.size : available, available);
    const size_t readback_words =
        static_cast<size_t>(read_size / sizeof(uint32_t));
    volatile const uint32_t* words = reinterpret_cast<volatile const uint32_t*>(
        static_cast<const uint8_t*>(bound.buffer->buffer.cpu_ptr) +
        static_cast<size_t>(bound.offset));
    volatile uint32_t checksum = 0;
    size_t nonzero_words = 0;
    for (size_t j = 0; j < readback_words; ++j) {
      const uint32_t word = words[j];
      checksum ^= word;
      if (word) ++nonzero_words;
    }
    if (trace_qhdl_enabled()) {
      std::fprintf(stderr,
                   "[amdxdna:mcdm] pathb %s-readback[%zu]: alloc=0x%08x "
                   "pos=%zu words=%zu nonzero=%zu checksum=0x%08x\n",
                   phase,
                   i, static_cast<unsigned>(bound.buffer->buffer.allocation),
                   bound.position, readback_words, nonzero_words,
                   static_cast<uint32_t>(checksum));
      std::fflush(stderr);
    }
  }
}

bool uses_windows_dpu_regmap(iree_hal_amdxdna_native_command_t* command) {
  return command &&
         command->opcode == iree_hal_amdxdna_native_command_opcode_t::start_npu;
}

bool uses_partial_elf_npu_packet(
    iree_hal_amdxdna_native_command_t* command) {
  return command && command->opcode ==
                        iree_hal_amdxdna_native_command_opcode_t::
                            start_npu_partial_elf;
}

iree_status_t ensure_partial_elf_dummy_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_partial_elf_npu_packet(command) ||
      !partial_elf_dummy_bos_enabled()) {
    return iree_ok_status();
  }
  auto& dummy_buffers = command->device->partial_elf_dummy_buffers;
  while (dummy_buffers.size() < 3) {
    iree_hal_amdxdna_native_buffer_ptr dummy;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_alloc_buffer(
        command->device, /*size=*/4,
        iree_hal_amdxdna_native_buffer_type_t::host_only, &dummy));
    void* ptr = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(dummy.get(), &ptr));
    std::memset(ptr, 0, 4);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        dummy.get(), iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    dummy_buffers.push_back(std::move(dummy));
  }
  return iree_ok_status();
}

iree_status_t maybe_write_partial_elf_bo_table(
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_partial_elf_npu_packet(command) ||
      !partial_elf_bo_table_enabled()) {
    return iree_ok_status();
  }

  // XRT's module-style command BO carries an out-of-packet table of kernel BO
  // GPU VAs after the 32-byte ERT_START_NPU packet. The inline private packet
  // still advertises only the ERT packet bytes, but the miniport can inspect the
  // command BO payload for dependency/binding metadata. Match that table shape:
  // word 11 starts six 64-bit BO VA slots for the DPU memory args.
  constexpr size_t kBoTableWordOffset = 11;
  constexpr size_t kBoTableEntries = 6;
  constexpr size_t kBoTableWords = 2 * kBoTableEntries;
  const size_t table_bytes =
      (kBoTableWordOffset + kBoTableWords) * sizeof(uint32_t);
  if (IREE_UNLIKELY(command->command_size < table_bytes)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM PARTIAL_ELF command BO is too small for the "
        "XRT-style BO table");
  }

  auto* words = reinterpret_cast<uint32_t*>(command->start_packet);
  std::fill(words + kBoTableWordOffset,
            words + kBoTableWordOffset + kBoTableWords, 0);
  IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
  for (const BoundBuffer& bound : command->bound_buffers) {
    if (!bound.buffer || bound.position == 0) continue;
    const size_t table_index = bound.position - 1;
    if (table_index >= kBoTableEntries) continue;
    uint64_t gpu_va =
        iree_hal_amdxdna_native_buffer_device_address(bound.buffer) +
        bound.offset;
    words[kBoTableWordOffset + 2 * table_index] =
        static_cast<uint32_t>(gpu_va);
    words[kBoTableWordOffset + 2 * table_index + 1] =
        static_cast<uint32_t>(gpu_va >> 32);
  }
  const auto& dummy_buffers = command->device->partial_elf_dummy_buffers;
  for (size_t i = 0; i < dummy_buffers.size() &&
                     i < kBoTableEntries - 3;
       ++i) {
    uint64_t gpu_va = iree_hal_amdxdna_native_buffer_device_address(
        dummy_buffers[i].get());
    const size_t table_index = 3 + i;
    words[kBoTableWordOffset + 2 * table_index] =
        static_cast<uint32_t>(gpu_va);
    words[kBoTableWordOffset + 2 * table_index + 1] =
        static_cast<uint32_t>(gpu_va >> 32);
  }
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (command->reg_idx >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (command->reg_idx + 1 >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

void set_windows_dpu_instruction_arg(
    iree_hal_amdxdna_native_command_t* command, uint64_t instruction_va,
    uint32_t instruction_words) {
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[kWindowsDpuInstructionRegWord] = static_cast<uint32_t>(instruction_va);
  regmap[kWindowsDpuInstructionRegWord + 1] =
      static_cast<uint32_t>(instruction_va >> 32);
  regmap[kWindowsDpuInstructionRegWord + 2] = instruction_words;
}

iree_status_t validate_windows_dpu_regmap_inputs(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires a command aperture");
  }
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds aperture code BO");
  }
  // The Windows wrapper DPU ABI carries the TXN selector plus a staged
  // instruction pointer. Execute TXNs add three data buffer VAs (arg_count=4,
  // reg_idx=8 before rewrite). Control-packet reconfiguration TXNs add only the
  // control-packet sequence/MC buffer VA (arg_count=2, reg_idx=4); the common
  // rewrite below maps that single VA into the first data slot and leaves the
  // others zero.
  const bool has_execute_args = command->arg_count >= 4 && command->reg_idx >= 8;
  const bool has_reconfigure_args =
      command->arg_count == 2 && command->reg_idx == 4;
  if (IREE_UNLIKELY(!has_execute_args && !has_reconfigure_args)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU packet is missing selector/data args");
  }
  return iree_ok_status();
}

iree_status_t rewrite_windows_dpu_regmap_to_instruction(
    iree_hal_amdxdna_native_command_t* command, uint64_t instruction_va) {
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  if (IREE_UNLIKELY(command->control_buffer_size / sizeof(uint32_t) >
                    std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM DPU instruction word count is too large");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  const uint64_t selector =
      static_cast<uint64_t>(regmap[0]) | (static_cast<uint64_t>(regmap[1]) << 32);
  const uint64_t ifm_va =
      static_cast<uint64_t>(regmap[2]) | (static_cast<uint64_t>(regmap[3]) << 32);
  const uint64_t param_va =
      static_cast<uint64_t>(regmap[4]) | (static_cast<uint64_t>(regmap[5]) << 32);
  const uint64_t ofm_va =
      static_cast<uint64_t>(regmap[6]) | (static_cast<uint64_t>(regmap[7]) << 32);

  std::memset(regmap, 0, kWindowsDpuRegmapWords * sizeof(uint32_t));
  regmap[0] = static_cast<uint32_t>(selector);
  regmap[1] = static_cast<uint32_t>(selector >> 32);
  const uint32_t instruction_words =
      zero_instruction_size_enabled()
          ? 0
          : static_cast<uint32_t>(command->control_buffer_size /
                                  sizeof(uint32_t));
  set_windows_dpu_instruction_arg(
      command, instruction_va, instruction_words);
  regmap[5] = static_cast<uint32_t>(ifm_va);
  regmap[6] = static_cast<uint32_t>(ifm_va >> 32);
  regmap[7] = static_cast<uint32_t>(param_va);
  regmap[8] = static_cast<uint32_t>(param_va >> 32);
  regmap[9] = static_cast<uint32_t>(ofm_va);
  regmap[10] = static_cast<uint32_t>(ofm_va >> 32);
  command->reg_idx = kWindowsDpuRegmapWords;
  command->windows_dpu_regmap_finalized = true;
  return iree_ok_status();
}

iree_status_t finalize_windows_dpu_regmap(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_windows_dpu_regmap(command)) return iree_ok_status();
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, command));
  mcdm::CommandAperture& aperture = queue->context->command_aperture;

  if (!pathb_stage_code_after_presync(command)) {
    IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
  }

  IREE_RETURN_IF_ERROR(rewrite_windows_dpu_regmap_to_instruction(
      command, aperture.code_gpu_va));
  return iree_ok_status();
}

iree_status_t append_pathb_start_cu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B chain descriptor only supports "
        "START_CU children");
  }

  const uint32_t extra_cu_masks = start->extra_cu_masks;
  const uint32_t packet_words = 1u + start->count;
  const uint32_t copy_start_word = 2u + extra_cu_masks;
  if (IREE_UNLIKELY(packet_words < copy_start_word)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet is too short");
  }

  const uint32_t* words = reinterpret_cast<const uint32_t*>(start);
  uint32_t cu_index = 0;
  bool found_cu = false;
  for (uint32_t mask_index = 0; mask_index <= extra_cu_masks; ++mask_index) {
    const uint32_t mask = words[1 + mask_index];
    if (!mask) continue;
    cu_index = mask_index * 32u + first_set_bit(mask);
    found_cu = true;
    break;
  }
  if (IREE_UNLIKELY(!found_cu)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet has no CU "
        "mask bit set");
  }

  const uint32_t copy_words = packet_words - copy_start_word;
  const size_t copy_bytes = static_cast<size_t>(copy_words) * sizeof(uint32_t);
  const size_t descriptor_bytes =
      kWindowsDpuChainDescriptorHeaderSize + copy_bytes;
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    descriptor_bytes > descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds 0x%zx "
        "bytes",
        descriptor_capacity);
  }

  uint8_t* descriptor = descriptor_base + *descriptor_used;
  std::memset(descriptor, 0, descriptor_bytes);
  uint32_t value = 1;
  std::memcpy(descriptor + 0x00, &value, sizeof(value));
  value = cu_index;
  std::memcpy(descriptor + 0x2c, &value, sizeof(value));
  value = copy_words;
  std::memcpy(descriptor + 0x30, &value, sizeof(value));
  std::memcpy(descriptor + kWindowsDpuChainDescriptorHeaderSize,
              words + copy_start_word, copy_bytes);
  *descriptor_used += descriptor_bytes;
  return iree_ok_status();
}

iree_status_t append_pathb_start_npu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_NPU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor received a "
        "non-START_NPU child");
  }
  ert_npu_data* npu_data = get_ert_npu_data(start);
  if (IREE_UNLIKELY(!npu_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B START_NPU chain child has no NPU data");
  }
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    kWindowsDpuStartNpuChainDescriptorSize >
                        descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor block exceeds "
        "0x%zx bytes",
        descriptor_capacity);
  }

  const uint32_t* args = get_ert_regmap_begin(start);
  const uint32_t* args_end = get_ert_regmap_end(start);
  const ptrdiff_t arg_words = args_end >= args ? args_end - args : 0;
  const uint32_t selector = arg_words > 0 ? args[0] : 0;
  const uint32_t selector_hi = arg_words > 1 ? args[1] : 0;

  uint32_t words[kWindowsDpuStartNpuChainDescriptorSize / sizeof(uint32_t)] =
      {};
  words[0] = 2;
  words[1] = static_cast<uint32_t>(npu_data->instruction_buffer);
  words[2] = static_cast<uint32_t>(npu_data->instruction_buffer >> 32);
  words[7] = npu_data->instruction_buffer_size;
  words[12] = 2;
  words[13] = selector;
  words[14] = selector_hi;

  std::memcpy(descriptor_base + *descriptor_used, words, sizeof(words));
  *descriptor_used += sizeof(words);
  return iree_ok_status();
}

iree_status_t get_pathb_chain_region_sizes(
    iree_hal_amdxdna_native_command_t* chain_command, size_t* out_code_bytes,
    size_t* out_descriptor_bytes) {
  size_t code_offset = 0;
  size_t descriptor_bytes = 0;
  for (iree_hal_amdxdna_native_command_t* child :
       chain_command->chain_children) {
    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    code_offset += static_cast<size_t>(child->control_buffer_size);

    ert_start_kernel_cmd* start = command_start_packet(child);
    if (start->opcode == ERT_START_NPU) {
      descriptor_bytes += kWindowsDpuStartNpuChainDescriptorSize;
      continue;
    }
    if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support START_CU or "
          "START_NPU children");
    }
    const uint32_t packet_words = 1u + start->count;
    const uint32_t copy_start_word = 2u + start->extra_cu_masks;
    if (IREE_UNLIKELY(packet_words < copy_start_word)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child START_CU packet is too "
          "short");
    }
    descriptor_bytes +=
        kWindowsDpuChainDescriptorHeaderSize +
        static_cast<size_t>(packet_words - copy_start_word) * sizeof(uint32_t);
  }
  // XRT's module-runlist aperture layout treats each child instruction stream
  // as occupying a full 0x8000 slot. Reserve through the final slot boundary so
  // batched parent chunks start on the same cadence and opcode-9 marker offsets
  // include the last child slot.
  *out_code_bytes = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
  *out_descriptor_bytes = descriptor_bytes;
  return iree_ok_status();
}

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command,
    bool sync_aperture) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires a command aperture");
  }
  if (IREE_UNLIKELY(chain_command->chain_children.empty())) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM command chain has no child commands");
  }

  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  queue->context->pathb_single_code_staged = false;
  queue->context->pathb_single_code_staged_size = 0;
  queue->context->pathb_single_code_words.clear();
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires an aperture code BO");
  }
  if (IREE_UNLIKELY(!aperture.gpu_cpu_ptr ||
                    aperture.gpu_va_size <=
                        kWindowsDpuChainDescriptorApertureOffset)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires the locked command "
        "aperture GPU view");
  }

  if (!sync_aperture && chain_command->pathb_chain_prepared_valid) {
    if (chain_command->pathb_chain_code_dirty) {
      if (IREE_UNLIKELY(
              chain_command->pathb_chain_child_code_offsets.size() !=
              chain_command->chain_children.size())) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "amdxdna Windows MCDM path-B prepared chain is missing child code "
            "offsets");
      }
      const uint64_t code_base_offset =
          chain_command->pathb_chain_code_aperture_offset;
      if (IREE_UNLIKELY(code_base_offset >= aperture.code_size)) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "amdxdna Windows MCDM prepared path-B chain code base offset %" PRIu64
            " exceeds aperture code BO (%" PRIu64 " bytes)",
            code_base_offset, aperture.code_size);
      }
      uint8_t* code = static_cast<uint8_t*>(aperture.code_cpu_ptr) +
                      static_cast<size_t>(code_base_offset);
      const size_t code_capacity =
          static_cast<size_t>(aperture.code_size - code_base_offset);
      for (size_t child_index = 0;
           child_index < chain_command->chain_children.size();
           ++child_index) {
        iree_hal_amdxdna_native_command_t* child =
            chain_command->chain_children[child_index];
        ert_packet* child_packet = command_packet(child);
        child_packet->state = ERT_CMD_STATE_NEW;
        const size_t code_offset =
            chain_command->pathb_chain_child_code_offsets[child_index];
        const size_t child_code_size =
            static_cast<size_t>(child->control_buffer_size);
        if (IREE_UNLIKELY(code_offset > code_capacity ||
                          child_code_size > code_capacity - code_offset)) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "amdxdna Windows MCDM prepared path-B chain control code "
              "exceeds aperture code BO (%zu-byte child at offset %zu, "
              "capacity %zu)",
              child_code_size, code_offset, code_capacity);
        }
        if (IREE_UNLIKELY(!child->control_buffer ||
                          !child->control_buffer->buffer.cpu_ptr ||
                          child_code_size == 0)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "amdxdna Windows MCDM prepared path-B chain child has no "
              "control-code buffer");
        }
        std::memcpy(code + code_offset, child->control_buffer->buffer.cpu_ptr,
                    child_code_size);
        if (uses_partial_elf_npu_packet(child)) {
          ert_npu_data* npu_data =
              get_ert_npu_data(command_start_packet(child));
          if (IREE_UNLIKELY(!npu_data)) {
            return iree_make_status(
                IREE_STATUS_INTERNAL,
                "amdxdna Windows MCDM PARTIAL_ELF prepared chain child has "
                "no NPU data");
          }
          npu_data->instruction_buffer =
              aperture.code_gpu_va + code_base_offset + code_offset;
          npu_data->instruction_buffer_size =
              zero_instruction_size_enabled()
                  ? 0
                  : static_cast<uint32_t>(child->control_buffer_size);
          npu_data->instruction_prop_count = 0;
          IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(child));
          IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(child));
        }
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            child->exec_buffer.get(),
            iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      }
      chain_command->pathb_chain_descriptor_dirty = false;
      return iree_ok_status();
    }
    for (iree_hal_amdxdna_native_command_t* child :
         chain_command->chain_children) {
      command_packet(child)->state = ERT_CMD_STATE_NEW;
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          child->exec_buffer.get(),
          iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    }
    chain_command->pathb_chain_descriptor_dirty = false;
    return iree_ok_status();
  }

  const uint64_t code_base_offset =
      chain_command->pathb_chain_code_aperture_offset;
  if (IREE_UNLIKELY(code_base_offset >= aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain code base offset %" PRIu64
        " exceeds aperture code BO (%" PRIu64 " bytes)",
        code_base_offset, aperture.code_size);
  }
  uint8_t* code = static_cast<uint8_t*>(aperture.code_cpu_ptr) +
                  static_cast<size_t>(code_base_offset);
  const size_t code_capacity =
      static_cast<size_t>(aperture.code_size - code_base_offset);
  size_t code_offset = 0;
  size_t code_used = 0;
  std::vector<size_t> child_code_offsets;
  std::vector<bool> child_code_needs_copy;
  struct UniqueChainCodeSlot {
    iree_hal_amdxdna_native_command_t* child = nullptr;
    size_t offset = 0;
  };
  std::vector<UniqueChainCodeSlot> unique_code_slots;
  child_code_offsets.reserve(chain_command->chain_children.size());
  child_code_needs_copy.reserve(chain_command->chain_children.size());
  unique_code_slots.reserve(chain_command->chain_children.size());
  const bool allow_code_dedup =
      chain_command->pathb_chain_allow_code_dedup &&
      !pathb_chain_sync9_enabled();
  for (iree_hal_amdxdna_native_command_t* child :
       chain_command->chain_children) {
    const size_t child_code_size =
        static_cast<size_t>(child->control_buffer_size);
    bool found_duplicate_code = false;
    if (allow_code_dedup) {
      for (const UniqueChainCodeSlot& slot : unique_code_slots) {
        iree_hal_amdxdna_native_command_t* previous = slot.child;
        if (!previous ||
            previous->control_buffer_size != child->control_buffer_size) {
          continue;
        }
        if (std::memcmp(previous->control_buffer->buffer.cpu_ptr,
                        child->control_buffer->buffer.cpu_ptr,
                        child_code_size) != 0) {
          continue;
        }
        child_code_offsets.push_back(slot.offset);
        child_code_needs_copy.push_back(false);
        found_duplicate_code = true;
        break;
      }
    }
    if (found_duplicate_code) continue;

    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    if (IREE_UNLIKELY(code_offset > code_capacity ||
                      child_code_size > code_capacity - code_offset)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM path-B chain control code exceeds aperture "
          "code BO (%zu-byte child at offset %zu, capacity %zu)",
          child_code_size, code_offset, code_capacity);
    }
    child_code_offsets.push_back(code_offset);
    child_code_needs_copy.push_back(true);
    unique_code_slots.push_back(UniqueChainCodeSlot{child, code_offset});
    code_used = code_offset + child_code_size;
    code_offset = code_used;
  }
  // Do not clear the full padded code range: START_NPU descriptors carry the
  // exact instruction byte count, and XRT leaves the 0x8000-spaced gaps as
  // allocator slack. Clearing those gaps dominated host-side chain prep.

  const size_t descriptor_offset =
      chain_command->pathb_chain_descriptor_aperture_offset
          ? static_cast<size_t>(
                chain_command->pathb_chain_descriptor_aperture_offset)
          : align_up_size(
                std::max<size_t>(
                    static_cast<size_t>(
                        kWindowsDpuChainDescriptorApertureOffset),
                    static_cast<size_t>(
                        kWindowsDpuInstructionApertureOffset +
                        code_base_offset) +
                        code_used),
                0x1000);
  if (IREE_UNLIKELY(descriptor_offset >= aperture.gpu_va_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds command "
        "aperture after %zu bytes of staged code",
        code_used);
  }
  uint8_t* descriptor_base =
      static_cast<uint8_t*>(aperture.gpu_cpu_ptr) + descriptor_offset;
  const size_t descriptor_capacity =
      static_cast<size_t>(aperture.gpu_va_size - descriptor_offset);
  const size_t descriptor_clear_bytes = std::min<size_t>(
      descriptor_capacity,
      chain_command->chain_children.size() *
          std::max<size_t>(kWindowsDpuStartNpuChainDescriptorSize,
                           kWindowsDpuChainDescriptorHeaderSize +
                               kWindowsDpuRegmapWords * sizeof(uint32_t)));
  std::memset(descriptor_base, 0, descriptor_clear_bytes);

  size_t descriptor_used = 0;
  chain_command->pathb_chain_descriptor_gpu_va = 0;
  chain_command->pathb_chain_descriptor_bytes = 0;
  chain_command->pathb_chain_first_child_opcode =
      command_packet(chain_command->chain_children.front())->opcode;
  for (size_t child_index = 0; child_index < chain_command->chain_children.size();
       ++child_index) {
    iree_hal_amdxdna_native_command_t* child =
        chain_command->chain_children[child_index];
    ert_packet* child_packet = command_packet(child);
    child_packet->state = ERT_CMD_STATE_NEW;
    if (IREE_UNLIKELY(child_packet->opcode !=
                      chain_command->pathb_chain_first_child_opcode)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains do not support mixed child ERT "
          "opcodes yet");
    }
    const bool is_start_cu_child = uses_windows_dpu_regmap(child);
    const bool is_start_npu_child = uses_partial_elf_npu_packet(child);
    if (IREE_UNLIKELY(!is_start_cu_child && !is_start_npu_child)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support DPU commands");
    }
    if (is_start_cu_child && child->windows_dpu_regmap_finalized) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child was already finalized");
    }
    if (IREE_UNLIKELY(!child->control_buffer ||
                      !child->control_buffer->buffer.cpu_ptr ||
                      child->control_buffer_size == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child has no control-code "
          "buffer");
    }
    if (IREE_UNLIKELY(child->control_buffer_size % sizeof(uint32_t) != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child control-code size is not "
          "word aligned");
    }
    code_offset = child_code_offsets[child_index];
    const size_t child_code_size =
        static_cast<size_t>(child->control_buffer_size);
    if (child_code_needs_copy[child_index]) {
      std::memcpy(code + code_offset, child->control_buffer->buffer.cpu_ptr,
                  child_code_size);
    }
    const uint64_t instruction_va =
        aperture.code_gpu_va + code_base_offset + code_offset;
    if (is_start_cu_child) {
      IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, child));
      IREE_RETURN_IF_ERROR(
          rewrite_windows_dpu_regmap_to_instruction(child, instruction_va));
      IREE_RETURN_IF_ERROR(append_pathb_start_cu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    } else {
      ert_npu_data* npu_data = get_ert_npu_data(command_start_packet(child));
      if (IREE_UNLIKELY(!npu_data)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "amdxdna Windows MCDM PARTIAL_ELF chain child has no NPU data");
      }
      npu_data->instruction_buffer = instruction_va;
      npu_data->instruction_buffer_size =
          zero_instruction_size_enabled()
              ? 0
              : static_cast<uint32_t>(child->control_buffer_size);
      npu_data->instruction_prop_count = 0;
      IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(child));
      IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(child));
      IREE_RETURN_IF_ERROR(append_pathb_start_npu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        child->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    trace_command_packet("chain-slot-final", child);
  }

  if (sync_aperture) {
    flush_host_writes_to_mcdm();
    std::string error;
    if (!mcdm::SyncCommandApertureCode(
            chain_command->device->api, chain_command->device->device, aperture,
            kWindowsDpuInstructionApertureOffset + code_base_offset,
            static_cast<uint64_t>(code_used), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B chain aperture code sync failed",
          error);
    }
    if (descriptor_used) {
      if (!mcdm::SyncCommandApertureCode(
              chain_command->device->api, chain_command->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_used), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B chain descriptor sync failed", error);
      }
    }
    if (!mcdm::RefreshCommandApertureGpuMapping(
            chain_command->device->api, chain_command->device->device,
            &aperture, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B chain aperture code relock failed",
          error);
    }
  }
  chain_command->pathb_chain_descriptor_gpu_va =
      aperture.gpu_va + descriptor_offset;
  chain_command->pathb_chain_descriptor_bytes =
      static_cast<uint32_t>(descriptor_used);
  chain_command->pathb_chain_code_used_size =
      allow_code_dedup ? code_used
                       : align_up_size(code_used,
                                       kWindowsDpuChainCodeAlignment);
  chain_command->pathb_chain_child_code_offsets = std::move(child_code_offsets);
  chain_command->pathb_chain_prepared_valid = true;
  chain_command->pathb_chain_code_dirty = true;
  chain_command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command) {
  return prepare_pathb_chain_code(queue, chain_command, true);
}

iree_status_t sync_prepared_pathb_chain_batch(
    iree_hal_amdxdna_native_queue_t* queue, iree_host_size_t command_count,
    size_t code_bytes, size_t descriptor_offset, size_t descriptor_bytes) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  flush_host_writes_to_mcdm();
  std::string error;
  const bool use_sync9 =
      command_count > 1 || pathb_chain_sync9_enabled();
  if (use_sync9) {
    size_t last_sync_offset = 0;
    auto submit_sync9 = [&](size_t end_offset) -> iree_status_t {
      if (!end_offset || end_offset == last_sync_offset) return iree_ok_status();
      if (!mcdm::SubmitPathBApertureSync(
              queue->context->device->api, queue->context->device->device,
              &queue->context->context, aperture,
              static_cast<uint64_t>(end_offset), /*wait_for_cpu=*/false,
              &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch sync9 failed", error);
      }
      last_sync_offset = end_offset;
      return iree_ok_status();
    };
    // XRT's module-runlist path emits opcode-9 markers at instruction-slot
    // boundaries (0x10000, 0x18000, ...), then submits the ERT_CMD_CHAIN
    // parents. It does not emit a marker for the parent descriptor metadata;
    // our descriptor block is still aperture-resident, so keep the conservative
    // invalidate/relock for that region after the instruction markers.
    const size_t code_end = align_up_size(
        static_cast<size_t>(kWindowsDpuInstructionApertureOffset) + code_bytes,
        kWindowsDpuChainCodeAlignment);
    for (size_t sync_offset = kWindowsDpuInstructionApertureOffset +
                              kWindowsDpuChainCodeAlignment;
         sync_offset <= code_end; sync_offset += kWindowsDpuChainCodeAlignment) {
      IREE_RETURN_IF_ERROR(submit_sync9(sync_offset));
    }
    if (descriptor_bytes) {
      if (!mcdm::SyncCommandApertureCode(
              queue->context->device->api, queue->context->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_bytes), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch descriptor sync failed", error);
      }
      if (!mcdm::RefreshCommandApertureGpuMapping(
              queue->context->device->api, queue->context->device->device,
              &aperture, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch aperture relock failed", error);
      }
    }
    return iree_ok_status();
  }
  if (code_bytes) {
    if (!mcdm::SyncCommandApertureCode(
            queue->context->device->api, queue->context->device->device,
            aperture, kWindowsDpuInstructionApertureOffset,
            static_cast<uint64_t>(code_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch aperture code sync failed",
          error);
    }
  }
  if (descriptor_bytes) {
    if (!mcdm::SyncCommandApertureCode(
            queue->context->device->api, queue->context->device->device,
            aperture, static_cast<uint64_t>(descriptor_offset),
            static_cast<uint64_t>(descriptor_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch descriptor sync failed", error);
    }
  }
  if (!mcdm::RefreshCommandApertureGpuMapping(
          queue->context->device->api, queue->context->device->device,
          &aperture, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM path-B batch aperture relock failed", error);
  }
  return iree_ok_status();
}

}  // namespace

void iree_hal_amdxdna_native_buffer_deleter_t::operator()(
    iree_hal_amdxdna_native_buffer_t* buffer) const {
  iree_hal_amdxdna_native_buffer_destroy(buffer);
}

void iree_hal_amdxdna_native_command_deleter_t::operator()(
    iree_hal_amdxdna_native_command_t* command) const {
  iree_hal_amdxdna_native_command_destroy(command);
}

iree_status_t materialize_deferred_buffer(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer || !buffer->deferred) return iree_ok_status();
  if (IREE_UNLIKELY(!buffer->device->pathb_context_ready)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM deferred BO materialized before pathb context "
        "setup");
  }

  mcdm::Buffer real_buffer;
  std::string error;
  if (!mcdm::CreateBuffer(buffer->device->api, buffer->device->device,
                          buffer->buffer.kind, buffer->buffer.size,
                          &real_buffer, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM deferred BO allocation failed", error);
  }
  if (real_buffer.cpu_ptr && !buffer->deferred_storage.empty()) {
    const uint64_t copy_size = std::min<uint64_t>(
        buffer->buffer.size,
        static_cast<uint64_t>(buffer->deferred_storage.size()));
    std::memcpy(real_buffer.cpu_ptr, buffer->deferred_storage.data(),
                static_cast<size_t>(copy_size));
  }
  buffer->buffer = real_buffer;
  buffer->deferred = false;
  buffer->deferred_storage.clear();
  buffer->deferred_storage.shrink_to_fit();
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_options = *options;
  out_device_path_storage->clear();
  if (options->n_core_rows < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_rows' expected a non-negative int32_t but "
        "got %d",
        options->n_core_rows);
  }
  if (options->n_core_cols < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_cols' expected a non-negative int32_t but "
        "got %d",
        options->n_core_cols);
  }
  DiagnosticStage diagnostic_stop_after = DiagnosticStage::none;
  if (!parse_diagnostic_stage(options->mcdm_diagnostic_stop_after,
                              &diagnostic_stop_after)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_mcdm_diagnostic_stop_after' has invalid value "
        "'%.*s'",
        static_cast<int>(options->mcdm_diagnostic_stop_after.size),
        options->mcdm_diagnostic_stop_after.data);
  }
  (void)diagnostic_stop_after;
  SubmitMode submit_mode = SubmitMode::direct;
  if (!parse_submit_mode(options->mcdm_submit_mode, &submit_mode)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_mcdm_submit_mode' has invalid value '%.*s'",
        static_cast<int>(options->mcdm_submit_mode.size),
        options->mcdm_submit_mode.data);
  }
  IREE_RETURN_IF_ERROR(require_submit_mode_opt_in(submit_mode));
  (void)submit_mode;
  return parse_power_mode(options->power_mode, out_power_mode,
                          out_should_set_power_mode);
}

// One-time, per-process XRT firmware warmup. Opening an XRT device performs the
// device/firmware initialization without which D3DKMTSubmitCommandToHwQueue (the
// aperture kick) is rejected with 0xc01e0200 (STATUS_GRAPHICS_ALLOCATION_BUSY).
// Verified with mcdm_replay --xrt-warmup: the identical KMT submit returns
// status=0 only while an XRT device is held open. The handle is intentionally
// leaked so the initialization persists for the process lifetime.
static void EnsureXrtFirmwareWarmup() {
  static bool tried = false;
  if (tried) return;
  tried = true;
  HMODULE xrt = LoadLibraryW(L"xrt_coreutil.dll");
  if (!xrt) {
    xrt = LoadLibraryW(
        L"C:\\Windows\\System32\\DriverStore\\FileRepository\\"
        L"kipudrv.inf_amd64_b3e90d6455884a5f\\xrt_coreutil.dll");
  }
  if (!xrt) {
    std::fprintf(stderr, "[amdxdna:mcdm] xrt warmup: xrt_coreutil.dll not found\n");
    return;
  }
  using XrtDeviceOpenFn = void*(__cdecl*)(unsigned int);
  auto open =
      reinterpret_cast<XrtDeviceOpenFn>(GetProcAddress(xrt, "xrtDeviceOpen"));
  if (open) {
    void* warmup_device = open(0);
    std::fprintf(stderr, "[amdxdna:mcdm] xrt firmware warmup device=%p\n",
                 warmup_device);
    (void)warmup_device;  // held open for the process lifetime
  } else {
    std::fprintf(stderr, "[amdxdna:mcdm] xrt warmup: xrtDeviceOpen missing\n");
  }
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  DiagnosticStage diagnostic_stop_after = DiagnosticStage::none;
  if (!parse_diagnostic_stage(options->mcdm_diagnostic_stop_after,
                              &diagnostic_stop_after)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_mcdm_diagnostic_stop_after' has invalid value "
        "'%.*s'",
        static_cast<int>(options->mcdm_diagnostic_stop_after.size),
        options->mcdm_diagnostic_stop_after.data);
  }
  SubmitMode submit_mode = SubmitMode::direct;
  if (!parse_submit_mode(options->mcdm_submit_mode, &submit_mode)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_mcdm_submit_mode' has invalid value '%.*s'",
        static_cast<int>(options->mcdm_submit_mode.size),
        options->mcdm_submit_mode.data);
  }
  IREE_RETURN_IF_ERROR(require_submit_mode_opt_in(submit_mode));

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  device = new (device) iree_hal_amdxdna_native_device_t(
      host_allocator, diagnostic_stop_after, submit_mode);

  std::string error;
  mcdm::Adapter adapter;
  if (!device->api.Load(&error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM KMT API load failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  iree_status_t status = diagnostic_after(device, DiagnosticStage::load_api);
  if (!iree_status_is_ok(status)) {
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  // NO XRT warmup: the pure-KMT replay works deterministically without any XRT
  // (3/3 status=0 on clean firmware). A held-open XRT device actually CONFLICTS
  // with our own kick by taking the NPU context. Talk to the driver directly.

  if (!mcdm::FindNpuAdapter(device->api, &adapter, &error)) {
    status = status_from_mcdm_error(
        "amdxdna Windows MCDM adapter discovery failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  status = diagnostic_after(device, DiagnosticStage::find_adapter);
  if (!iree_status_is_ok(status)) {
    if (adapter.handle) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = adapter.handle;
      device->api.close_adapter(&close);
    }
    for (D3DKMT_HANDLE handle : adapter.retained_handles) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = handle;
      device->api.close_adapter(&close);
    }
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  if (!mcdm::CreateDevice(device->api, adapter, &device->device, &error)) {
    status = status_from_mcdm_error(
        "amdxdna Windows MCDM device creation failed", error);
    if (adapter.handle) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = adapter.handle;
      device->api.close_adapter(&close);
    }
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  status = diagnostic_after(device, DiagnosticStage::create_device);
  if (!iree_status_is_ok(status)) {
    mcdm::DestroyDevice(device->api, &device->device);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  *out_device = device;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device) return;
  iree_allocator_t host_allocator = device->host_allocator;
  mcdm::DestroyDevice(device->api, &device->device);
  device->~iree_hal_amdxdna_native_device_t();
  iree_allocator_free(host_allocator, device);
}

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_power_mode_t power_mode) {
  (void)device;
  if (power_mode == iree_hal_amdxdna_native_power_mode_t::default_mode) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna Windows MCDM power-mode control is not implemented");
}

bool iree_hal_amdxdna_native_device_supports_partial_elf_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_PARTIAL_ELF) != 0;
}

bool iree_hal_amdxdna_native_device_uses_npu_payload_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0;
}

bool iree_hal_amdxdna_native_device_syncs_bindings_on_submit(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return caps.buffer_sync_model ==
         iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings;
}

iree_hal_amdxdna_native_command_opcode_t
iree_hal_amdxdna_native_device_dispatch_opcode(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return iree_hal_amdxdna_native_command_opcode_t::start_npu;
  }
  return caps.default_dispatch_opcode;
}

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_device_caps_t* out_caps) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_caps);
  iree_hal_amdxdna_native_device_caps_t caps;
  caps.ddi_version = 1;
  caps.max_effective_queues = 1;
  const size_t chain_exec_bo_size =
      device->submit_mode == SubmitMode::pathb
          ? static_cast<size_t>(windows_dpu_pathb_chain_exec_bo_size())
          : static_cast<size_t>(kMaxExecBoSize);
  caps.max_command_chain_slots = chain_slot_capacity(chain_exec_bo_size);
  caps.context_image_models =
      IREE_HAL_AMDXDNA_NATIVE_CONTEXT_IMAGE_MODEL_XCLBIN;
  caps.dispatch_models = IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_CU |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_PARTIAL_ELF |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_COMMAND_CHAIN;
  caps.buffer_sync_model =
      device->submit_mode == SubmitMode::pathb && !skip_pathb_bound_sync_enabled()
          ? iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings
          : iree_hal_amdxdna_native_buffer_sync_model_t::caller_syncs_bindings;
  caps.completion_models =
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_SYNCHRONOUS_WAIT |
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_PROGRESS_FENCE |
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_COMPLETION_SLOT;
  caps.supports_command_chain = true;
  caps.supports_submit_many = true;
  caps.supports_async_submit = false;
  caps.supports_external_buffer_import = false;
  caps.supports_external_buffer_export = false;
  caps.supports_real_multi_queue = false;
  caps.default_dispatch_opcode =
      iree_hal_amdxdna_native_command_opcode_t::start_npu;
  *out_caps = caps;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  out_buffer->reset();
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));

  const bool defer_cacheable_pathb_alloc =
      device->submit_mode == SubmitMode::pathb &&
      type == iree_hal_amdxdna_native_buffer_type_t::cacheable;
  if (device->submit_mode == SubmitMode::pathb &&
      (!device->pathb_context_ready || defer_cacheable_pathb_alloc)) {
    out_buffer->reset(new iree_hal_amdxdna_native_buffer_t(
        device, type, static_cast<uint64_t>(size)));
    return diagnostic_after(device, DiagnosticStage::alloc_buffer);
  }

  mcdm::Buffer buffer;
  std::string error;
  if (!mcdm::CreateBuffer(device->api, device->device, to_mcdm_buffer_kind(type),
                          static_cast<uint64_t>(size), &buffer, &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO allocation failed",
                                  error);
  }
  iree_status_t status =
      diagnostic_after(device, DiagnosticStage::alloc_buffer);
  if (!iree_status_is_ok(status)) {
    mcdm::DestroyBuffer(device->api, device->device, &buffer);
    return status;
  }
  out_buffer->reset(new iree_hal_amdxdna_native_buffer_t(device, buffer));
  (*out_buffer)->type = type;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_context_image_t* image,
    iree_hal_amdxdna_native_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(image);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = nullptr;
  if (IREE_UNLIKELY(image->type !=
                    iree_hal_amdxdna_native_context_image_type_t::xclbin)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires an xclbin context "
        "image; compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }
  iree_const_byte_span_t pdi = image->pdi;
  iree_const_byte_span_t xclbin = image->xclbin;
  if (IREE_UNLIKELY(xclbin.data_length == 0 || !xclbin.data)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires xclbin data; "
        "compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }

  std::vector<uint8_t> private_data;
  mcdm::ContextBlobInfo info;
  std::string error;
  if (!mcdm::BuildContextPrivateDataFromXclbin(
          xclbin.data, xclbin.data_length, GetCurrentProcessId(), &private_data,
          &info, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context blob generation failed", error);
  }
  IREE_RETURN_IF_ERROR(diagnostic_after(device, DiagnosticStage::context_blob));

  mcdm::Context context;
  if (!mcdm::CreateContext(device->api, device->device, private_data, &context,
                           &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context creation failed", error);
  }
  iree_status_t status =
      diagnostic_after(device, DiagnosticStage::create_context);
  if (!iree_status_is_ok(status)) {
    mcdm::DestroyContext(device->api, &context);
    return status;
  }

  mcdm::CommandControlBuffer command_control = {};
  bool has_command_control = false;

  mcdm::CommandAperture command_aperture = {};
  bool has_command_aperture = false;
  if (device->submit_mode == SubmitMode::aperture ||
      device->submit_mode == SubmitMode::qhdl ||
      device->submit_mode == SubmitMode::pathb) {
    if (!mcdm::CreateCommandAperture(device->api, device->device, context,
                                     &command_aperture, &error)) {
      if (has_command_control) {
        mcdm::DestroyCommandControlBuffer(device->api, device->device,
                                          &command_control);
      }
      mcdm::DestroyContext(device->api, &context);
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command aperture creation failed", error);
    }
    has_command_aperture = true;
    if (device->submit_mode == SubmitMode::qhdl) {
      if (command_aperture.cpu_ptr) {
        uint64_t initialized = 1;
        std::memset(command_aperture.cpu_ptr, 0,
                    static_cast<size_t>(command_aperture.allocation_size));
        std::memcpy(command_aperture.cpu_ptr, &initialized,
                    sizeof(initialized));
      }
      command_control.size = command_aperture.allocation_size;
      command_control.allocation = command_aperture.allocation;
      command_control.resource = command_aperture.resource;
      command_control.cpu_ptr = command_aperture.cpu_ptr;
      command_control.next_slot_offset = 8;
      has_command_control = true;
    }
    if (device->submit_mode == SubmitMode::qhdl) {
      if (!mcdm::SubmitAndWaitCommandAperture(
              device->api, device->device, &context, &command_aperture,
              &error)) {
        if (has_command_control) {
          mcdm::DestroyCommandControlBuffer(device->api, device->device,
                                            &command_control);
        }
        mcdm::DestroyCommandAperture(device->api, device->device,
                                     &command_aperture);
        mcdm::DestroyContext(device->api, &context);
        return status_from_mcdm_error(
            "amdxdna Windows MCDM command aperture bootstrap failed", error);
      }
    }

    if (device->submit_mode == SubmitMode::pathb) {
      context.completion_ring.kind = mcdm::BufferKind::cacheable;
      context.completion_ring.size = command_aperture.allocation_size;
      context.completion_ring.allocation = command_aperture.allocation;
      context.completion_ring.resource = command_aperture.resource;
      context.completion_ring.cpu_ptr = command_aperture.cpu_ptr;
      context.completion_ring_ready = true;
      context.completion_ring_offset = 8;
      if (!mcdm::SubmitAndWaitPathBSetup(device->api, device->device, &context,
                                         &command_aperture, pdi.data,
                                         pdi.data_length, &error)) {
        mcdm::DestroyCommandAperture(device->api, device->device,
                                     &command_aperture);
        mcdm::DestroyContext(device->api, &context);
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb setup failed", error);
      }
      device->pathb_context_ready = true;
    }
  }

  *out_context = new iree_hal_amdxdna_native_context_t(
      device, context, command_control, has_command_control, command_aperture,
      has_command_aperture, info);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_t** out_context) {
  iree_hal_amdxdna_native_context_image_t image;
  image.type = iree_hal_amdxdna_native_context_image_type_t::xclbin;
  image.pdi = pdi;
  image.xclbin = xclbin;
  image.kernel_name = kernel_name;
  return iree_hal_amdxdna_native_device_create_context(device, &image,
                                                       out_context);
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return;
  const bool command_control_aliases_aperture =
      context->has_command_control && context->has_command_aperture &&
      context->command_control.allocation ==
          context->command_aperture.allocation &&
      context->command_control.cpu_ptr == context->command_aperture.cpu_ptr;
  if (context->has_command_control && !command_control_aliases_aperture) {
    mcdm::DestroyCommandControlBuffer(context->device->api,
                                      context->device->device,
                                      &context->command_control);
  }
  if (context->has_command_aperture) {
    mcdm::DestroyCommandAperture(context->device->api,
                                 context->device->device,
                                 &context->command_aperture);
  }
  mcdm::DestroyContext(context->device->api, &context->context);
  delete context;
}

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_max_slots);
  iree_hal_amdxdna_native_device_caps_t caps;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_device_query_caps(device, &caps));
  *out_max_slots = caps.max_command_chain_slots;
  return iree_ok_status();
}

size_t iree_hal_amdxdna_native_command_arg_binding_capacity() { return 1024; }

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer) return;
  if (!buffer->deferred) {
    mcdm::DestroyBuffer(buffer->device->api, buffer->device->device,
                        &buffer->buffer);
  }
  delete buffer;
}

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = nullptr;
  if (IREE_UNLIKELY(!buffer || !buffer->buffer.cpu_ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not host-mapped");
  }
  *out_ptr = buffer->buffer.cpu_ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction, iree_device_size_t size,
    iree_device_size_t offset) {
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(offset));
  if (buffer->deferred) {
    return diagnostic_after(buffer->device, DiagnosticStage::sync_buffer);
  }
  if (direction == iree_hal_amdxdna_native_sync_direction_t::host_to_device) {
    flush_host_writes_to_mcdm();
    return diagnostic_after(buffer->device, DiagnosticStage::sync_buffer);
  }
  std::string error;
  if (!mcdm::SyncBuffer(buffer->device->api, buffer->device->device,
                        buffer->buffer, static_cast<uint64_t>(offset),
                        static_cast<uint64_t>(size), &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO sync failed",
                                  error);
  }
  return diagnostic_after(buffer->device, DiagnosticStage::sync_buffer);
}

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction) {
  return iree_hal_amdxdna_native_buffer_sync(
      buffer, direction, iree_hal_amdxdna_native_buffer_size(buffer), 0);
}

iree_status_t iree_hal_amdxdna_native_buffer_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return materialize_deferred_buffer(buffer);
}

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return buffer->buffer.gpu_va;
}

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return static_cast<iree_device_size_t>(buffer->buffer.size);
}

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_cu_index_t* out_cu_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cu_index);

  const std::vector<std::string>& kernel_names = context->info.kernel_names;
  if (kernel_names.empty()) {
    out_cu_index->index = 0;
    return diagnostic_after(context->device, DiagnosticStage::open_cu);
  }

  std::string requested = normalize_cu_name(string_view_to_string(kernel_name));
  for (size_t i = 0; i < kernel_names.size(); ++i) {
    if (requested == kernel_names[i]) {
      out_cu_index->index = static_cast<uint32_t>(i);
      return diagnostic_after(context->device, DiagnosticStage::open_cu);
    }
  }

  if (kernel_names.size() == 1) {
    out_cu_index->index = 0;
    return diagnostic_after(context->device, DiagnosticStage::open_cu);
  }

  std::string available;
  for (size_t i = 0; i < kernel_names.size(); ++i) {
    if (i) available += ", ";
    available += kernel_names[i];
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "amdxdna Windows MCDM context does not contain requested CU '%s'; "
      "available CUs: %s",
      requested.c_str(), available.c_str());
}

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context) {
  return &context->queue;
}

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->exec_command_count;
}

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_ptr* out_command) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command);
  out_command->reset();

  iree_hal_amdxdna_native_buffer_ptr exec_buffer;
  iree_status_t status = iree_ok_status();
  uint64_t exec_buffer_size = kMaxExecBoSize;
  if (device->submit_mode == SubmitMode::pathb &&
      opcode == iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    exec_buffer_size = windows_dpu_pathb_chain_exec_bo_size();
  } else if (device->submit_mode == SubmitMode::pathb &&
             compact_execbuf_enabled()) {
    exec_buffer_size = kWindowsDpuPathBExecBoSize;
  }
  if (device->submit_mode == SubmitMode::pathb) {
    exec_buffer.reset(new iree_hal_amdxdna_native_buffer_t(
        device, mcdm::BufferKind::execbuf, exec_buffer_size));
    status = diagnostic_after(device, DiagnosticStage::create_command);
    if (!iree_status_is_ok(status)) return status;
  } else {
    mcdm::Buffer buffer;
    std::string error;
    if (!mcdm::CreateBuffer(device->api, device->device,
                            mcdm::BufferKind::execbuf, exec_buffer_size, &buffer,
                            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM execbuf allocation failed", error);
    }
    status = diagnostic_after(device, DiagnosticStage::create_command);
    if (!iree_status_is_ok(status)) {
      mcdm::DestroyBuffer(device->api, device->device, &buffer);
      return status;
    }
    exec_buffer.reset(new iree_hal_amdxdna_native_buffer_t(device, buffer));
  }

  auto* command = new iree_hal_amdxdna_native_command_t(
      device, opcode, std::move(exec_buffer));
  std::memset(command->start_packet, 0, command->command_size);
  command->start_packet->state = ERT_CMD_STATE_NEW;
  command->start_packet->opcode = to_ert_opcode(opcode);
  command->start_packet->type = ERT_CU;
  status = inc_pkt_count(command, sizeof(uint32_t));
  if (!iree_status_is_ok(status)) {
    delete command;
    return status;
  }
  if (opcode == iree_hal_amdxdna_native_command_opcode_t::start_npu) {
    // XRT creates the full DPU register-map payload up front:
    // header + CU mask + 15 register words from the xclbin XML metadata.
    command->start_packet->count = 1 + kWindowsDpuRegmapWords;
  } else if (opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf) {
    // Match XRT's module-style ERT_START_NPU packet:
    // header + CU mask + ert_npu_data + selector word + trailing return word.
    command->start_packet->count =
        1 + sizeof(ert_npu_data) / sizeof(uint32_t) + 2;
  }
  out_command->reset(command);
  return iree_ok_status();
}

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  delete command;
}

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_cu_index_t cu_index) {
  command->start_packet->cu_mask = 0x1u << cu_index.index;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  switch (command->opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return iree_ok_status();
    case iree_hal_amdxdna_native_command_opcode_t::start_npu: {
      if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM instruction buffer is empty");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        iree_hal_amdxdna_native_buffer_size(control_buffer))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM instruction byte count exceeds BO size");
      }
      if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM instruction byte count is not word aligned");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        std::numeric_limits<uint32_t>::max())) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM instruction buffer is too large");
      }
      command->control_buffer = control_buffer;
      command->control_buffer_size = control_buffer_size;
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
      return iree_ok_status();
    }
    case iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf: {
      if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM PARTIAL_ELF instruction buffer is empty");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        iree_hal_amdxdna_native_buffer_size(control_buffer))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM PARTIAL_ELF instruction byte count exceeds "
            "BO size");
      }
      if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM PARTIAL_ELF instruction byte count is not "
            "word aligned");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        std::numeric_limits<uint32_t>::max())) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM PARTIAL_ELF instruction buffer is too large");
      }
      command->control_buffer = control_buffer;
      command->control_buffer_size = control_buffer_size;
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
      ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
      if (IREE_UNLIKELY(!npu_data)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
      }
      if (command->device && command->device->submit_mode == SubmitMode::pathb) {
        // Path-B mirrors XRT's module path: instruction bytes are staged into
        // the command aperture code window just before submit, not submitted as
        // a standalone BO. This avoids an otherwise unused KMT allocation and
        // keeps the command packet's instruction VA tied to the aperture.
        npu_data->instruction_buffer = 0;
        npu_data->instruction_buffer_size =
            zero_instruction_size_enabled()
                ? 0
                : static_cast<uint32_t>(control_buffer_size);
        npu_data->instruction_prop_count = 0;
        bind_buffer_ref(command, /*position=*/0, control_buffer, /*offset=*/0,
                        control_buffer_size);
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(materialize_deferred_buffer(control_buffer));
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          control_buffer,
          iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      npu_data->instruction_buffer =
          iree_hal_amdxdna_native_buffer_device_address(control_buffer);
      npu_data->instruction_buffer_size =
          zero_instruction_size_enabled()
              ? 0
              : static_cast<uint32_t>(control_buffer_size);
      npu_data->instruction_prop_count = 0;
      bind_buffer_ref(command, /*position=*/0, control_buffer, /*offset=*/0,
                      control_buffer_size);
      return iree_ok_status();
    }
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported control buffer command opcode");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (uses_windows_dpu_regmap(command)) {
    if (command->arg_count == 0) {
      return write_windows_dpu_regmap_u64(command, value);
    }
    return write_windows_dpu_regmap_u32(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free arg slots");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = value;
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (uses_windows_dpu_regmap(command)) {
    return write_windows_dpu_regmap_u64(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx + 1 >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free u64 arg slot");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = static_cast<uint32_t>(value);
    args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = static_cast<uint32_t>(value);
  args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(command,
                                                                  buffer, 0);
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer offset too large");
  }
  if (uses_partial_elf_npu_packet(command)) {
    bind_buffer_ref(command, command->arg_count, buffer, offset,
                    iree_hal_amdxdna_native_buffer_size(buffer) - offset);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, sizeof(uint64_t)));
  bind_buffer_ref(command, command->arg_count, buffer, offset,
                  iree_hal_amdxdna_native_buffer_size(buffer) - offset);
  return iree_hal_amdxdna_native_command_add_arg_64(
      command, iree_hal_amdxdna_native_buffer_device_address(buffer) + offset);
}

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer) ||
      size > iree_hal_amdxdna_native_buffer_size(buffer) - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer binding range is "
                            "out of bounds");
  }
  bind_buffer_ref(command, position, buffer, offset, size);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  command->bound_buffers.clear();
  if (uses_partial_elf_npu_packet(command) && command->control_buffer) {
    bind_buffer_ref(command, /*position=*/0, command->control_buffer,
                    /*offset=*/0, command->control_buffer_size);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  command->pathb_chain_prepared_valid = false;
  command->pathb_chain_code_dirty = true;
  command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (!command->pathb_chain_prepared_valid) {
    command->pathb_chain_descriptor_dirty = true;
  }
  command->pathb_chain_code_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (IREE_UNLIKELY(command_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command chain is too large");
  }
  const size_t chain_bytes = offsetof(ert_packet, data) +
                             sizeof(ert_cmd_chain_data) +
                             command_count * sizeof(uint64_t);
  if (chain_bytes > command->command_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna cmd-chain: %" PRIhsz
                            " slots exceed exec buffer (%zu > %zu bytes)",
                            command_count, chain_bytes, command->command_size);
  }

  ert_packet* packet = command_packet(command);
  std::memset(packet, 0, command->command_size);
  command->chain_children.clear();
  command->bound_buffers.clear();
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(commands[i]->exec_buffer.get()));
    // Materializing a deferred exec BO replaces the temporary host-storage
    // mapping with the real KMT allocation mapping. Refresh the cached packet
    // pointer before any later descriptor construction reads the child.
    commands[i]->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        commands[i]->exec_buffer->buffer.cpu_ptr);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        commands[i]->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    // XRT's source-level runlist path stores the child run BO's kernel-mode
    // handle here and then calls bind_at() on the parent chain BO.
    chain_data->data[i] = commands[i]->exec_buffer->buffer.allocation;
    command->chain_children.push_back(commands[i]);
    trace_command_packet("chain-slot", commands[i]);
    command->bound_buffers.push_back(BoundBuffer{
        i, commands[i]->exec_buffer.get(), 0,
        iree_hal_amdxdna_native_buffer_size(commands[i]->exec_buffer.get())});
  }
  packet->count =
      (sizeof(ert_cmd_chain_data) + command_count * sizeof(uint64_t)) /
      sizeof(uint32_t);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command);
  ert_packet* packet = command_packet(command);
  packet->state = ERT_CMD_STATE_NEW;
  NativeSubmitTimingScope total_timing(NativeSubmitTimingSection::total);

  {
    NativeSubmitTimingScope timing(
        NativeSubmitTimingSection::finalize_regmap);
    IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));
  }

  if (command->device->submit_mode != SubmitMode::pathb) {
    NativeSubmitTimingScope timing(
        NativeSubmitTimingSection::non_pathb_exec_materialize);
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(command->exec_buffer.get()));
    command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        command->exec_buffer->buffer.cpu_ptr);
    packet = command_packet(command);
    packet->state = ERT_CMD_STATE_NEW;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        command->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  }

  std::string error;
  {
    NativeSubmitTimingScope timing(NativeSubmitTimingSection::bound_residency);
    for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
      const BoundBuffer& bound = command->bound_buffers[i];
      if (!bound.buffer) continue;
      if (is_pathb_partial_elf_control_binding(command, bound)) continue;
      IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
      std::string label = "bound[" + std::to_string(i) + "]";
      if (!mcdm::WaitForBufferResidency(
              command->device->api, command->device->device,
              queue->context->context, bound.buffer->buffer, label.c_str(),
              &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM bound BO residency wait failed", error);
      }
    }
  }

  trace_command_packet("before-submit", command);
  IREE_RETURN_IF_ERROR(
      diagnostic_after(command->device, DiagnosticStage::ready_submit));

  bool packet_state_from_completion_slot = false;
  if (command->device->submit_mode == SubmitMode::aperture) {
    if (!queue->context->has_command_aperture) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM aperture submit requested without command "
          "aperture");
    }
    IREE_RETURN_IF_ERROR(stage_command_aperture(queue, command));
    IREE_RETURN_IF_ERROR(
        diagnostic_after(command->device, DiagnosticStage::stage_aperture));
    if (!mcdm::SubmitAndWaitCommandAperture(
            command->device->api, command->device->device,
            &queue->context->context, &queue->context->command_aperture,
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command aperture submit failed", error);
    }
    if (command->opcode ==
            iree_hal_amdxdna_native_command_opcode_t::start_npu &&
        queue->context->command_aperture.cpu_ptr) {
      uint32_t aperture_header = 0;
      std::memcpy(&aperture_header, queue->context->command_aperture.cpu_ptr,
                  sizeof(aperture_header));
      packet->header = (packet->header & ~uint32_t{0xf}) |
                       (aperture_header & uint32_t{0xf});
      packet_state_from_completion_slot = true;
    }
  } else if (command->device->submit_mode == SubmitMode::qhdl) {
    if (command->opcode ==
        iree_hal_amdxdna_native_command_opcode_t::command_chain) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM qhdl submit does not support command chains "
          "yet; use --amdxdna_cmd_chain=false for the first e2e dispatch");
    }
    if (!queue->context->has_command_control) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM qhdl submit requested without command control "
          "buffer");
    }
    for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
      iree_hal_amdxdna_native_buffer_t* bound = command->bound_buffers[i].buffer;
      if (!bound) continue;
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          bound, iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    }
    const uint32_t command_bytes = (packet->count + 1) * sizeof(uint32_t);
    if (!mcdm::SubmitAndWaitQhdlCommand(
            command->device->api, command->device->device,
            &queue->context->context, &queue->context->command_control,
            command->exec_buffer->buffer, command_bytes, 3,
            static_cast<uint32_t>(command->exec_buffer->buffer.allocation),
            &packet->header, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM qhdl command submit failed", error);
    }
    packet_state_from_completion_slot = true;
  } else if (command->device->submit_mode == SubmitMode::pathb) {
    const bool is_pathb_chain =
        command->opcode ==
        iree_hal_amdxdna_native_command_opcode_t::command_chain;
    const bool is_pathb_partial_elf =
        !is_pathb_chain && uses_partial_elf_npu_packet(command);
    const uint32_t command_bytes = (packet->count + 1) * sizeof(uint32_t);
    const bool skip_bound_sync =
        !is_pathb_chain &&
        (is_pathb_partial_elf || skip_pathb_bound_sync_enabled());
    // The NPU is not cache-coherent: flush every bound buffer (instruction
    // control code + input args) host->device BEFORE the dispatch so the
    // firmware reads real data, not stale device memory. (Output is synced
    // device->host after.)
    if (!skip_bound_sync) {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_pre_bound_sync);
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        iree_hal_amdxdna_native_buffer_t* bound =
            command->bound_buffers[i].buffer;
        if (!bound) continue;
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            bound, iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      }
    }
    readback_pathb_bound_buffers(command, "input", /*outputs_only=*/false);
    if (trace_qhdl_enabled() &&
        command->control_buffer && command->control_buffer->buffer.cpu_ptr) {
      const uint32_t* c =
          static_cast<const uint32_t*>(command->control_buffer->buffer.cpu_ptr);
      uint32_t nz = 0;
      for (size_t i = 0; i < command->control_buffer_size / 4; ++i)
        if (c[i]) ++nz;
      std::fprintf(stderr,
                   "[amdxdna:mcdm] control-code: bytes=%llu nonzero_words=%u "
                   "first=%08x %08x %08x %08x %08x %08x\n",
                   static_cast<unsigned long long>(command->control_buffer_size),
                   nz, c[0], c[1], c[2], c[3], c[4], c[5]);
      std::fflush(stderr);
    }
    if (!queue->context->has_command_aperture) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM pathb submit requested without command "
          "aperture");
    }
    const bool skip_non_chain_presync =
        !is_pathb_chain && is_pathb_partial_elf;
    if (!skip_non_chain_presync) {
      NativeSubmitTimingScope timing(NativeSubmitTimingSection::pathb_pre_sync);
      const bool wait_presync =
          !is_pathb_chain && pathb_stage_code_after_presync(command) && false;
      if (!mcdm::SubmitPathBApertureSync(
              command->device->api, command->device->device,
              &queue->context->context, queue->context->command_aperture,
              /*offset=*/0x10000, wait_presync, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb pre-dispatch sync failed", error);
      }
    }
    {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_stage_code);
      if (is_pathb_chain) {
        IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command));
      } else if (pathb_stage_code_after_presync(command)) {
        IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
      }
    }
    {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_ensure_dummy);
      IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
    }
    {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_exec_materialize);
      IREE_RETURN_IF_ERROR(
          materialize_deferred_buffer(command->exec_buffer.get()));
      command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
          command->exec_buffer->buffer.cpu_ptr);
      packet = command_packet(command);
    }
    {
      NativeSubmitTimingScope timing(NativeSubmitTimingSection::pathb_bo_table);
      IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(command));
    }
    // XRT's module path writes the state-3 command BO through its CPU mapping
    // and submits it directly; there is no per-dispatch D3DKMTInvalidateCache
    // for the command BO in the captured path. For partial-ELF path-B commands
    // mirror that by using a CPU fence instead of the generic host->device
    // sync. Keep the env knob for non-partial probes.
    const bool skip_exec_sync =
        !is_pathb_chain &&
        (is_pathb_partial_elf || skip_pathb_exec_sync_enabled());
    {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_exec_sync);
      if (!skip_exec_sync) {
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            command->exec_buffer.get(),
            iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      } else {
        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
    }
    readback_pathb_bound_buffers(command, "submit", /*outputs_only=*/false);
    trace_command_packet("pathb-before-submit", command);
    if (is_pathb_chain) {
      mcdm::PathBChainSubmitInfo chain_info = {};
      chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
      chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
      chain_info.command_count =
          reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
      chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
      NativeSubmitTimingScope timing(NativeSubmitTimingSection::pathb_submit);
      uint64_t submit_t0 = 0;
      const bool measure_submit = submit_timing_enabled();
      if (measure_submit) submit_t0 = now_ns();
      if (!mcdm::SubmitAndWaitPathBChain(
              command->device->api, command->device->device,
              &queue->context->context, command->exec_buffer->buffer, packet,
              command_bytes, chain_info, &packet->header, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb chain submit failed", error);
      }
      if (measure_submit) record_submit_timing(now_ns() - submit_t0);
    } else {
      NativeSubmitTimingScope timing(NativeSubmitTimingSection::pathb_submit);
      uint64_t submit_t0 = 0;
      const bool measure_submit = submit_timing_enabled();
      if (measure_submit) submit_t0 = now_ns();
      if (!mcdm::SubmitAndWaitPathB(
              command->device->api, command->device->device,
              &queue->context->context, command->exec_buffer->buffer, packet,
              command_bytes, /*command_state=*/3, &packet->header, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb command submit failed", error);
      }
      if (measure_submit) record_submit_timing(now_ns() - submit_t0);
    }
    packet_state_from_completion_slot = true;
    const bool skip_non_chain_postsync =
        !is_pathb_chain && is_pathb_partial_elf;
    if (!skip_non_chain_postsync) {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_post_sync);
      if (!mcdm::SubmitPathBApertureSync(
              command->device->api, command->device->device,
              &queue->context->context, queue->context->command_aperture,
              /*offset=*/0x8000, /*wait_for_cpu=*/true, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb post-dispatch sync failed", error);
      }
    }
    // The NPU is not cache-coherent: invalidate every bound buffer (incl. the
    // output) device->host so the host reads the firmware's results, not stale
    // cache. This was missing and could masquerade as "no execution".
    if (!skip_bound_sync) {
      NativeSubmitTimingScope timing(
          NativeSubmitTimingSection::pathb_post_bound_sync);
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        iree_hal_amdxdna_native_buffer_t* bound =
            command->bound_buffers[i].buffer;
        if (!bound) continue;
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            bound, iree_hal_amdxdna_native_sync_direction_t::device_to_host));
      }
    }
    readback_pathb_bound_buffers(command, "output", /*outputs_only=*/true);
  } else {
    if (!mcdm::SubmitAndWaitBuffer(command->device->api,
                                   command->device->device,
                                   &queue->context->context,
                                   command->exec_buffer->buffer, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command submit failed", error);
    }
  }
  queue->exec_command_count++;
  IREE_RETURN_IF_ERROR(diagnostic_after(command->device,
                                        DiagnosticStage::submit));

  if (!packet_state_from_completion_slot) {
    NativeSubmitTimingScope timing(NativeSubmitTimingSection::final_exec_sync);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        command->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::device_to_host));
  }
  trace_command_packet("after-readback", command);

  if (packet->state == ERT_CMD_STATE_COMPLETED) return iree_ok_status();

  if (command->opcode ==
      iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s did not complete: ert state %u (error_index %u, "
        "submit_index %u)",
        static_cast<int>(label.size), label.data, packet->state,
        chain_data->error_index, chain_data->submit_index);
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL, "amdxdna %.*s did not complete: ert state %u",
      static_cast<int>(label.size), label.data, packet->state);
}

iree_status_t iree_hal_amdxdna_native_queue_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(commands);
  if (command_count == 0) return iree_ok_status();

  iree_hal_amdxdna_native_device_t* device = queue->context->device;
  if (device->submit_mode != SubmitMode::pathb) {
    for (iree_host_size_t i = 0; i < command_count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdxdna_native_queue_submit_and_wait(queue, commands[i],
                                                        label));
    }
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (commands[i]->opcode !=
        iree_hal_amdxdna_native_command_opcode_t::command_chain) {
      for (iree_host_size_t j = 0; j < command_count; ++j) {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdxdna_native_queue_submit_and_wait(queue, commands[j],
                                                          label));
      }
      return iree_ok_status();
    }
  }

  if (!queue->context->has_command_aperture) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM pathb batch submit requested without command "
        "aperture");
  }

  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  std::vector<size_t> code_sizes(command_count);
  std::vector<size_t> descriptor_sizes(command_count);
  size_t code_cursor = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    // Deduplicating identical child instruction streams within one parent chain
    // is valid and matches the driver's per-descriptor byte-count model. Keep
    // multi-parent batches conservative: submitting multiple parents before a
    // wait failed when they shared compacted aperture slots, so give each child
    // its own XRT-style 0x8000 slot until that firmware ordering rule is mapped.
    commands[i]->pathb_chain_allow_code_dedup = command_count == 1;
    IREE_RETURN_IF_ERROR(get_pathb_chain_region_sizes(
        commands[i], &code_sizes[i], &descriptor_sizes[i]));
    const size_t code_base = align_up_size(code_cursor, 0x1000);
    commands[i]->pathb_chain_code_aperture_offset = code_base;
    code_cursor = code_base + align_up_size(code_sizes[i], 0x1000);
  }
  if (IREE_UNLIKELY(code_cursor > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B batch chain code uses %zu bytes, "
        "exceeding aperture code BO (%" PRIu64 " bytes)",
        code_cursor, aperture.code_size);
  }

  size_t descriptor_cursor = align_up_size(
      std::max<size_t>(
          static_cast<size_t>(kWindowsDpuChainDescriptorApertureOffset),
          static_cast<size_t>(kWindowsDpuInstructionApertureOffset) +
              code_cursor),
      0x1000);
  const size_t descriptor_batch_offset = descriptor_cursor;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    commands[i]->pathb_chain_descriptor_aperture_offset = descriptor_cursor;
    descriptor_cursor += align_up_size(descriptor_sizes[i], 0x1000);
  }
  if (IREE_UNLIKELY(descriptor_cursor > aperture.gpu_va_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B batch chain descriptors use aperture "
        "through byte %zu, exceeding command aperture (%" PRIu64 " bytes)",
        descriptor_cursor, aperture.gpu_va_size);
  }

  std::vector<mcdm::PathBPendingSubmit> pending(command_count);
  std::string error;
  const bool batch_phase_timing = false;
  const uint64_t batch_total_t0 = batch_phase_timing ? now_ns() : 0;
  uint64_t batch_prepare_ns = 0;
  uint64_t batch_sync_ns = 0;
  uint64_t batch_submit_ns = 0;
  uint64_t batch_wait_ns = 0;
  uint64_t batch_post_ns = 0;
  size_t batch_code_sync_bytes = 0;
  uint64_t submit_t0 = 0;
  const bool measure_submit = submit_timing_enabled();
  if (measure_submit) submit_t0 = now_ns();
  uint64_t phase_t0 = batch_phase_timing ? now_ns() : 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    packet->state = ERT_CMD_STATE_NEW;

    IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));

    if (!command->pathb_chain_bound_residency_checked) {
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        const BoundBuffer& bound = command->bound_buffers[i];
        if (!bound.buffer) continue;
        IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
        std::string residency_label = "batch-bound[" + std::to_string(i) + "]";
        if (!mcdm::WaitForBufferResidency(
                command->device->api, command->device->device,
                queue->context->context, bound.buffer->buffer,
                residency_label.c_str(), &error)) {
          return status_from_mcdm_error(
              "amdxdna Windows MCDM bound BO residency wait failed", error);
        }
      }
      command->pathb_chain_bound_residency_checked = true;
    }

    IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command, false));
    if (!skip_clean_chain_sync_enabled() ||
        command->pathb_chain_code_dirty ||
        command->pathb_chain_descriptor_dirty) {
      batch_code_sync_bytes = std::max<size_t>(
          batch_code_sync_bytes,
          static_cast<size_t>(command->pathb_chain_code_aperture_offset +
                              command->pathb_chain_code_used_size));
    }
    IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(command->exec_buffer.get()));
    command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        command->exec_buffer->buffer.cpu_ptr);
    packet = command_packet(command);

    for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
      iree_hal_amdxdna_native_buffer_t* bound =
          command->bound_buffers[i].buffer;
      if (!bound) continue;
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          bound, iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        command->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  }
  if (batch_phase_timing) batch_prepare_ns = now_ns() - phase_t0;

  size_t dirty_descriptor_begin = std::numeric_limits<size_t>::max();
  size_t dirty_descriptor_end = 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    if (!command->pathb_chain_descriptor_dirty) continue;
    const size_t begin =
        static_cast<size_t>(command->pathb_chain_descriptor_aperture_offset);
    const size_t end =
        begin + static_cast<size_t>(command->pathb_chain_descriptor_bytes);
    dirty_descriptor_begin = std::min(dirty_descriptor_begin, begin);
    dirty_descriptor_end = std::max(dirty_descriptor_end, end);
  }
  const size_t descriptor_sync_offset =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? descriptor_batch_offset
          : dirty_descriptor_begin;
  const size_t descriptor_sync_bytes =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? 0
          : dirty_descriptor_end - dirty_descriptor_begin;

  phase_t0 = batch_phase_timing ? now_ns() : 0;
  IREE_RETURN_IF_ERROR(sync_prepared_pathb_chain_batch(
      queue, command_count, batch_code_sync_bytes, descriptor_sync_offset,
      descriptor_sync_bytes));
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    commands[command_index]->pathb_chain_code_dirty = false;
    commands[command_index]->pathb_chain_descriptor_dirty = false;
  }
  if (batch_phase_timing) batch_sync_ns = now_ns() - phase_t0;

  phase_t0 = batch_phase_timing ? now_ns() : 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    mcdm::PathBChainSubmitInfo chain_info = {};
    chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
    chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
    chain_info.command_count =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
    chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
    if (!mcdm::SubmitPathBChain(command->device->api, command->device->device,
                                &queue->context->context,
                                command->exec_buffer->buffer, packet,
                                (packet->count + 1) * sizeof(uint32_t),
                                chain_info, &packet->header,
                                &pending[command_index], &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain batch submit failed", error);
    }
  }
  if (batch_phase_timing) batch_submit_ns = now_ns() - phase_t0;

  phase_t0 = batch_phase_timing ? now_ns() : 0;
  if (!mcdm::WaitForPathBSubmits(device->api, device->device,
                                 &queue->context->context, pending.data(),
                                 pending.size(), &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb chain batch wait failed", error);
  }
  if (batch_phase_timing) batch_wait_ns = now_ns() - phase_t0;
  if (measure_submit) record_submit_timing(now_ns() - submit_t0);

  phase_t0 = batch_phase_timing ? now_ns() : 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    // The direct command-buffer chain flush invalidates the exact I/O binding
    // ranges once after the whole group completes. Avoid invalidating every
    // bound BO for every native parent chunk here; that duplicates work and was
    // the dominant batched-chain host overhead.
    queue->exec_command_count++;
    trace_command_packet("batch-after-readback", command);
    if (packet->state == ERT_CMD_STATE_COMPLETED) continue;
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s batch command %" PRIhsz
        " did not complete: ert state %u (error_index %u, submit_index %u)",
        static_cast<int>(label.size), label.data, command_index,
        packet->state, chain_data->error_index, chain_data->submit_index);
  }
  if (batch_phase_timing) {
    batch_post_ns = now_ns() - phase_t0;
    const uint64_t batch_total_ns = now_ns() - batch_total_t0;
    std::fprintf(stderr,
                 "[amdxdna:mcdm-batch-phase] chunks=%" PRIhsz
                 " prepare_us=%.3f aperture_sync_us=%.3f submit_us=%.3f "
                 "wait_us=%.3f post_sync_us=%.3f total_us=%.3f\n",
                 command_count, static_cast<double>(batch_prepare_ns) / 1000.0,
                 static_cast<double>(batch_sync_ns) / 1000.0,
                 static_cast<double>(batch_submit_ns) / 1000.0,
                 static_cast<double>(batch_wait_ns) / 1000.0,
                 static_cast<double>(batch_post_ns) / 1000.0,
                 static_cast<double>(batch_total_ns) / 1000.0);
    std::fflush(stderr);
  }
  return iree_ok_status();
}
