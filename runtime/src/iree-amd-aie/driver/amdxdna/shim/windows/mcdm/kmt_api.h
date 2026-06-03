// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winternl.h>

#include <d3dkmthk.h>

#include <cstdint>
#include <string>
#include <vector>

namespace iree::hal::amdxdna::mcdm {

enum class BufferKind {
  host_only,
  cacheable,
  execbuf,
};

struct BufferKindInfo {
  const char* name;
  uint32_t private_type;
  uint32_t xcl_flags;
};

BufferKindInfo GetBufferKindInfo(BufferKind kind);

struct KmtApi {
  PFND3DKMT_ENUMADAPTERS3 enum_adapters3 = nullptr;
  PFND3DKMT_QUERYADAPTERINFO query_adapter_info = nullptr;
  PFND3DKMT_CLOSEADAPTER close_adapter = nullptr;
  PFND3DKMT_CREATEDEVICE create_device = nullptr;
  PFND3DKMT_DESTROYDEVICE destroy_device = nullptr;
  PFND3DKMT_CREATEPAGINGQUEUE create_paging_queue = nullptr;
  PFND3DKMT_DESTROYPAGINGQUEUE destroy_paging_queue = nullptr;
  PFND3DKMT_CREATEALLOCATION2 create_allocation2 = nullptr;
  PFND3DKMT_DESTROYALLOCATION2 destroy_allocation2 = nullptr;
  PFND3DKMT_MAPGPUVIRTUALADDRESS map_gpu_virtual_address = nullptr;
  PFND3DKMT_FREEGPUVIRTUALADDRESS free_gpu_virtual_address = nullptr;
  PFND3DKMT_MAKERESIDENT make_resident = nullptr;
  PFND3DKMT_LOCK2 lock2 = nullptr;
  PFND3DKMT_UNLOCK2 unlock2 = nullptr;
  PFND3DKMT_INVALIDATECACHE invalidate_cache = nullptr;
  PFND3DKMT_CREATECONTEXTVIRTUAL create_context_virtual = nullptr;
  PFND3DKMT_DESTROYCONTEXT destroy_context = nullptr;
  PFND3DKMT_CREATEHWQUEUE create_hw_queue = nullptr;
  PFND3DKMT_DESTROYHWQUEUE destroy_hw_queue = nullptr;
  PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait_from_gpu = nullptr;
  PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait_from_cpu = nullptr;
  PFND3DKMT_SUBMITCOMMANDTOHWQUEUE submit_command_to_hw_queue = nullptr;

  bool Load(std::string* out_error);
};

struct Adapter {
  D3DKMT_HANDLE handle = 0;
  std::wstring description;
};

struct Device {
  D3DKMT_HANDLE adapter = 0;
  D3DKMT_HANDLE device = 0;
  D3DKMT_HANDLE paging_queue = 0;
  D3DKMT_HANDLE paging_sync_object = 0;
  void* paging_fence_cpu = nullptr;
};

struct Buffer {
  BufferKind kind = BufferKind::host_only;
  uint64_t size = 0;
  D3DKMT_HANDLE allocation = 0;
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  void* cpu_ptr = nullptr;
  UINT64 paging_fence_value = 0;
};

struct Context {
  D3DKMT_HANDLE context = 0;
  D3DKMT_HANDLE hw_queue = 0;
  D3DKMT_HANDLE progress_fence = 0;
  void* progress_fence_cpu = nullptr;
  D3DGPU_VIRTUAL_ADDRESS progress_fence_gpu = 0;
  uint64_t next_fence_id = 1;
};

struct CommandAperture {
  uint64_t allocation_size = 0;
  uint64_t gpu_va_size = 0;
  D3DKMT_HANDLE allocation = 0;
  D3DKMT_HANDLE resource = 0;
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  void* cpu_ptr = nullptr;
};

bool FindNpuAdapter(const KmtApi& api, Adapter* out_adapter,
                    std::string* out_error);

bool CreateDevice(const KmtApi& api, D3DKMT_HANDLE adapter, Device* out_device,
                  std::string* out_error);

void DestroyDevice(const KmtApi& api, Device* device);

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, std::string* out_error);

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, std::string* out_error);

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer);

bool CreateContext(const KmtApi& api, const Device& device,
                   const std::vector<uint8_t>& private_data,
                   Context* out_context, std::string* out_error);

void DestroyContext(const KmtApi& api, Context* context);

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture,
                           std::string* out_error);

bool SubmitAndWaitCommandAperture(const KmtApi& api, const Device& device,
                                  Context* context,
                                  const CommandAperture& aperture,
                                  std::string* out_error);

bool SubmitAndWaitBuffer(const KmtApi& api, const Device& device,
                         Context* context, const Buffer& buffer,
                         std::string* out_error);

void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture);

std::string NtStatusToString(NTSTATUS status);

}  // namespace iree::hal::amdxdna::mcdm

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_
