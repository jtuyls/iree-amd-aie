// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/native.h"

#include <algorithm>
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

struct BoundBuffer {
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  iree_device_size_t offset = 0;
  iree_device_size_t size = 0;
};

std::string string_view_to_string(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_status_t status_from_mcdm_error(const char* label,
                                     const std::string& error) {
  return iree_make_status(IREE_STATUS_INTERNAL, "%s: %s", label,
                          error.c_str());
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

}  // namespace

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  mcdm::KmtApi api;
  mcdm::Device device;

  explicit iree_hal_amdxdna_native_device_t(iree_allocator_t host_allocator)
      : host_allocator(host_allocator) {}
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
  mcdm::ContextBlobInfo info;
  iree_hal_amdxdna_native_queue_t queue;

  iree_hal_amdxdna_native_context_t(
      iree_hal_amdxdna_native_device_t* device, mcdm::Context context,
      mcdm::ContextBlobInfo info)
      : device(device), context(context), info(std::move(info)) {
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

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command) {
  return command->start_packet;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(command_start_packet(command));
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
  command->bound_buffers.push_back(BoundBuffer{buffer, offset, size});
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
  return parse_power_mode(options->power_mode, out_power_mode,
                          out_should_set_power_mode);
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  (void)options;
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  device = new (device) iree_hal_amdxdna_native_device_t(host_allocator);

  std::string error;
  mcdm::Adapter adapter;
  if (!device->api.Load(&error) ||
      !mcdm::FindNpuAdapter(device->api, &adapter, &error) ||
      !mcdm::CreateDevice(device->api, adapter.handle, &device->device,
                          &error)) {
    iree_status_t status =
        status_from_mcdm_error("amdxdna Windows MCDM device creation failed",
                               error);
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

  mcdm::Context context;
  if (!mcdm::CreateContext(device->api, device->device, private_data, &context,
                           &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context creation failed", error);
  }

  *out_context = new iree_hal_amdxdna_native_context_t(device, context, info);
  return iree_ok_status();
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return;
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
  return iree_ok_status();
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
  return iree_ok_status();
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
  exec_buffer.reset(new iree_hal_amdxdna_native_buffer_t(device, buffer));

  auto* command = new iree_hal_amdxdna_native_command_t(
      device, opcode, std::move(exec_buffer));
  std::memset(command->start_packet, 0, command->command_size);
  command->start_packet->state = ERT_CMD_STATE_NEW;
  command->start_packet->opcode = to_ert_opcode(opcode);
  command->start_packet->type = ERT_CU;
  iree_status_t status = inc_pkt_count(command, sizeof(uint32_t));
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
    chain_data->data[i] = commands[i]->exec_buffer->buffer.allocation;
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
  if (!mcdm::SubmitAndWaitBuffer(command->device->api, command->device->device,
                                 &queue->context->context,
                                 command->exec_buffer->buffer, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM command submit failed", error);
  }
  queue->exec_command_count++;

  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
      command->exec_buffer.get(),
      iree_hal_amdxdna_native_sync_direction_t::device_to_host));

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
