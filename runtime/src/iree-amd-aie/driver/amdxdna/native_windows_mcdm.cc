// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/native.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr char kAllowUnsafeApertureSubmitEnv[] =
    "IREE_AMDXDNA_MCDM_ALLOW_UNSAFE_APERTURE_SUBMIT";
constexpr char kAllowUnsafeQhdlSubmitEnv[] =
    "IREE_AMDXDNA_MCDM_ALLOW_UNSAFE_QHDL_SUBMIT";
constexpr uint32_t kXgqCmdOpStartCuIdx = 0x100;
constexpr uint32_t kXgqSqCmdNew = 1;
constexpr uint32_t kXgqCuDomainPl = 0;
constexpr size_t kXgqHeaderWords = 2;

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

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value) return false;
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0;
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
  return false;
}

iree_status_t require_submit_mode_opt_in(SubmitMode submit_mode) {
  switch (submit_mode) {
    case SubmitMode::direct:
      return iree_ok_status();
    case SubmitMode::aperture:
      if (env_flag_enabled(kAllowUnsafeApertureSubmitEnv)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna_mcdm_submit_mode=aperture is disabled by default because "
          "the Windows MCDM exec-BO staging contract is not fully mapped yet; "
          "set %s=1 to opt in to this unsafe probe mode",
          kAllowUnsafeApertureSubmitEnv);
    case SubmitMode::qhdl:
      if (env_flag_enabled(kAllowUnsafeQhdlSubmitEnv)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna_mcdm_submit_mode=qhdl is disabled by default because the "
          "0x268 qhdl block is an internal XRT object layout, not the public "
          "KMT submit packet; set %s=1 only for controlled probe runs",
          kAllowUnsafeQhdlSubmitEnv);
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

}  // namespace

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  mcdm::KmtApi api;
  mcdm::Device device;
  DiagnosticStage diagnostic_stop_after = DiagnosticStage::none;
  SubmitMode submit_mode = SubmitMode::direct;

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

  iree_hal_amdxdna_native_buffer_t(
      iree_hal_amdxdna_native_device_t* device, mcdm::Buffer buffer)
      : device(device), buffer(buffer) {}
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
  ert_start_kernel_cmd* start_packet = nullptr;
  size_t command_size = 0;
  uint32_t reg_idx = 0;
  uint32_t arg_count = 0;
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
                       16);

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
  std::atomic_thread_fence(std::memory_order_seq_cst);
  FlushProcessWriteBuffers();
  trace_xgq_words(command, xgq_words, aperture);
  return iree_ok_status();
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
  if (position == 0) command->bound_buffers.clear();
  command->bound_buffers.push_back(BoundBuffer{position, buffer, offset, size});
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
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  if (!mcdm::CreateDevice(device->api, adapter.handle, &device->device,
                          &error)) {
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

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  out_buffer->reset();
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));

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
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_t** out_context) {
  (void)pdi;
  (void)kernel_name;
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = nullptr;
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
      device->submit_mode == SubmitMode::qhdl) {
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
      if (!mcdm::SubmitAndWaitCommandAperture(
              device->api, device->device, &context, command_aperture,
              &error)) {
        mcdm::DestroyCommandAperture(device->api, device->device,
                                     &command_aperture);
        mcdm::DestroyContext(device->api, &context);
        return status_from_mcdm_error(
            "amdxdna Windows MCDM command aperture bootstrap failed", error);
      }
    }
  }

  *out_context = new iree_hal_amdxdna_native_context_t(
      device, context, command_control, has_command_control, command_aperture,
      has_command_aperture, info);
  return iree_ok_status();
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
    mcdm::DestroyCommandAperture(context->device->api, context->device->device,
                                 &context->command_aperture);
  }
  mcdm::DestroyContext(context->device->api, &context->context);
  delete context;
}

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  IREE_ASSERT_ARGUMENT(out_max_slots);
  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      device, iree_hal_amdxdna_native_command_opcode_t::command_chain,
      &command));
  const size_t capacity =
      static_cast<size_t>(iree_hal_amdxdna_native_buffer_size(
          command->exec_buffer.get()));
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  *out_max_slots =
      capacity > header
          ? static_cast<uint32_t>((capacity - header) / sizeof(uint64_t))
          : 1;
  return iree_ok_status();
}

size_t iree_hal_amdxdna_native_command_arg_binding_capacity() { return 1024; }

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer) return;
  mcdm::DestroyBuffer(buffer->device->api, buffer->device->device,
                      &buffer->buffer);
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
  (void)direction;
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(offset));
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
  (void)context;
  (void)kernel_name;
  IREE_ASSERT_ARGUMENT(out_cu_index);
  // Single-PDI prototype contexts expose one DPU/CU. Multi-PDI xclbins will
  // need metadata-driven CU lookup.
  out_cu_index->index = 0;
  return diagnostic_after(context->device, DiagnosticStage::open_cu);
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
  mcdm::Buffer buffer;
  std::string error;
  if (!mcdm::CreateBuffer(device->api, device->device,
                          mcdm::BufferKind::execbuf, kMaxExecBoSize, &buffer,
                          &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM execbuf allocation failed", error);
  }
  iree_status_t status =
      diagnostic_after(device, DiagnosticStage::create_command);
  if (!iree_status_is_ok(status)) {
    mcdm::DestroyBuffer(device->api, device->device, &buffer);
    return status;
  }
  exec_buffer.reset(new iree_hal_amdxdna_native_buffer_t(device, buffer));

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
    iree_hal_amdxdna_native_buffer_t* control_buffer) {
  switch (command->opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return iree_ok_status();
    case iree_hal_amdxdna_native_command_opcode_t::start_npu: {
      iree_device_size_t instruction_buffer_size =
          iree_hal_amdxdna_native_buffer_size(control_buffer);
      if (IREE_UNLIKELY(instruction_buffer_size >
                        std::numeric_limits<uint32_t>::max())) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM instruction buffer is too large");
      }
      IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(ert_npu_data)));
      ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
      npu_data->instruction_buffer =
          iree_hal_amdxdna_native_buffer_device_address(control_buffer);
      npu_data->instruction_buffer_size =
          static_cast<uint32_t>(instruction_buffer_size);
      npu_data->instruction_prop_count = 0;
      bind_buffer_ref(command, command->arg_count, control_buffer, 0,
                      instruction_buffer_size);
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
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
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
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer offset too large");
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
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer) ||
      size > iree_hal_amdxdna_native_buffer_size(buffer) - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer binding range is "
                            "out of bounds");
  }
  bind_buffer_ref(command, position, buffer, offset, size);
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
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        commands[i]->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    chain_data->data[i] = commands[i]->exec_buffer->buffer.allocation;
    trace_command_packet("chain-slot", commands[i]);
    command->bound_buffers.push_back(BoundBuffer{
        i, commands[i]->exec_buffer.get(), 0,
        iree_hal_amdxdna_native_buffer_size(commands[i]->exec_buffer.get())});
    for (const BoundBuffer& bound : commands[i]->bound_buffers) {
      command->bound_buffers.push_back(bound);
    }
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

  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
      command->exec_buffer.get(),
      iree_hal_amdxdna_native_sync_direction_t::host_to_device));

  std::string error;
  for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer) continue;
    std::string label = "bound[" + std::to_string(i) + "]";
    if (!mcdm::WaitForBufferResidency(
            command->device->api, command->device->device,
            queue->context->context, bound.buffer->buffer, label.c_str(),
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM bound BO residency wait failed", error);
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
    IREE_RETURN_IF_ERROR(stage_xgq_command_aperture(queue, command));
    IREE_RETURN_IF_ERROR(
        diagnostic_after(command->device, DiagnosticStage::stage_aperture));
    if (!mcdm::SubmitAndWaitCommandAperture(
            command->device->api, command->device->device,
            &queue->context->context, queue->context->command_aperture,
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM command aperture submit failed", error);
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
