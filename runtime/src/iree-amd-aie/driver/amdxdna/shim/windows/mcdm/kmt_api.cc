// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103);
constexpr uint64_t kPageSize = 4096;
constexpr uint64_t kCommandApertureBase = 0x04000000;
constexpr uint64_t kCommandApertureAllocationSize = 0x1000;
constexpr uint64_t kCommandApertureGpuVaSize = 0x04000000;
constexpr uint32_t kCommandAperturePrivateType = 0x332b;
constexpr uint32_t kSubmitPrivateQwords = 13;  // 104 bytes on current driver.

template <typename Fn>
Fn ResolveKmtProc(const char* name) {
  HMODULE modules[] = {
      LoadLibraryW(L"win32u.dll"),
      LoadLibraryW(L"gdi32.dll"),
      LoadLibraryW(L"dxcore.dll"),
  };
  for (HMODULE module : modules) {
    if (!module) continue;
    FARPROC proc = GetProcAddress(module, name);
    if (proc) return reinterpret_cast<Fn>(proc);
  }
  return nullptr;
}

bool CheckStatus(const char* call_name, NTSTATUS status,
                 std::string* out_error) {
  if (status == 0) return true;
  if (out_error) {
    std::ostringstream os;
    os << call_name << " failed with " << NtStatusToString(status);
    *out_error = os.str();
  }
  return false;
}

bool CheckStatusOrPending(const char* call_name, NTSTATUS status,
                          std::string* out_error) {
  if (status == 0 || status == kStatusPending) return true;
  return CheckStatus(call_name, status, out_error);
}

uint32_t Flags32(const D3DKMT_CREATEALLOCATIONFLAGS& flags) {
  uint32_t value = 0;
  static_assert(sizeof(value) <= sizeof(flags), "flag storage mismatch");
  std::memcpy(&value, &flags, sizeof(value));
  return value;
}

struct AllocPrivate {
  uint64_t reserved0 = 0;
  uint64_t requested_size = 0;
  uint64_t aligned_size = 0;
  uint32_t reserved1 = 0;
  uint32_t private_type = 0;
  uint32_t policy = 2;
  uint32_t reserved2 = 0;
  uint32_t xcl_flags = 0;
  uint32_t reserved3 = 0;
  uint64_t reserved4 = 0;
};

static_assert(sizeof(AllocPrivate) == 56,
              "Windows MCDM BO private packet must remain 56 bytes");

uint64_t AlignUpToPage(uint64_t value) {
  return (value + 4095u) & ~uint64_t{4095u};
}

}  // namespace

BufferKindInfo GetBufferKindInfo(BufferKind kind) {
  switch (kind) {
    case BufferKind::host_only:
      return {"host_only", 0x3329, 0x20000000};
    case BufferKind::cacheable:
      return {"cacheable", 0x3323, 0x01000000};
    case BufferKind::execbuf:
      return {"execbuf", 0x3328, 0x80000000};
  }
  return {"host_only", 0x3329, 0x20000000};
}

bool KmtApi::Load(std::string* out_error) {
  enum_adapters3 = ResolveKmtProc<PFND3DKMT_ENUMADAPTERS3>(
      "D3DKMTEnumAdapters3");
  query_adapter_info = ResolveKmtProc<PFND3DKMT_QUERYADAPTERINFO>(
      "D3DKMTQueryAdapterInfo");
  close_adapter =
      ResolveKmtProc<PFND3DKMT_CLOSEADAPTER>("D3DKMTCloseAdapter");
  create_device =
      ResolveKmtProc<PFND3DKMT_CREATEDEVICE>("D3DKMTCreateDevice");
  destroy_device =
      ResolveKmtProc<PFND3DKMT_DESTROYDEVICE>("D3DKMTDestroyDevice");
  create_paging_queue = ResolveKmtProc<PFND3DKMT_CREATEPAGINGQUEUE>(
      "D3DKMTCreatePagingQueue");
  destroy_paging_queue = ResolveKmtProc<PFND3DKMT_DESTROYPAGINGQUEUE>(
      "D3DKMTDestroyPagingQueue");
  create_allocation2 = ResolveKmtProc<PFND3DKMT_CREATEALLOCATION2>(
      "D3DKMTCreateAllocation2");
  destroy_allocation2 = ResolveKmtProc<PFND3DKMT_DESTROYALLOCATION2>(
      "D3DKMTDestroyAllocation2");
  map_gpu_virtual_address = ResolveKmtProc<PFND3DKMT_MAPGPUVIRTUALADDRESS>(
      "D3DKMTMapGpuVirtualAddress");
  free_gpu_virtual_address = ResolveKmtProc<PFND3DKMT_FREEGPUVIRTUALADDRESS>(
      "D3DKMTFreeGpuVirtualAddress");
  make_resident =
      ResolveKmtProc<PFND3DKMT_MAKERESIDENT>("D3DKMTMakeResident");
  lock2 = ResolveKmtProc<PFND3DKMT_LOCK2>("D3DKMTLock2");
  unlock2 = ResolveKmtProc<PFND3DKMT_UNLOCK2>("D3DKMTUnlock2");
  invalidate_cache = ResolveKmtProc<PFND3DKMT_INVALIDATECACHE>(
      "D3DKMTInvalidateCache");
  create_context_virtual = ResolveKmtProc<PFND3DKMT_CREATECONTEXTVIRTUAL>(
      "D3DKMTCreateContextVirtual");
  destroy_context =
      ResolveKmtProc<PFND3DKMT_DESTROYCONTEXT>("D3DKMTDestroyContext");
  create_hw_queue =
      ResolveKmtProc<PFND3DKMT_CREATEHWQUEUE>("D3DKMTCreateHwQueue");
  destroy_hw_queue =
      ResolveKmtProc<PFND3DKMT_DESTROYHWQUEUE>("D3DKMTDestroyHwQueue");
  wait_from_gpu =
      ResolveKmtProc<PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU>(
          "D3DKMTWaitForSynchronizationObjectFromGpu");
  wait_from_cpu =
      ResolveKmtProc<PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU>(
          "D3DKMTWaitForSynchronizationObjectFromCpu");
  submit_command_to_hw_queue =
      ResolveKmtProc<PFND3DKMT_SUBMITCOMMANDTOHWQUEUE>(
          "D3DKMTSubmitCommandToHwQueue");

  if (enum_adapters3 && query_adapter_info && close_adapter && create_device &&
      destroy_device && create_paging_queue && destroy_paging_queue &&
      create_allocation2 && destroy_allocation2 && map_gpu_virtual_address &&
      free_gpu_virtual_address && make_resident && lock2 && unlock2 &&
      invalidate_cache && create_context_virtual && destroy_context &&
      create_hw_queue && destroy_hw_queue && wait_from_gpu && wait_from_cpu &&
      submit_command_to_hw_queue) {
    return true;
  }

  if (out_error) {
    *out_error = "failed to resolve one or more required KMT entry points";
  }
  return false;
}

bool FindNpuAdapter(const KmtApi& api, Adapter* out_adapter,
                    std::string* out_error) {
  D3DKMT_ENUMADAPTERS3 enum_args = {};
  enum_args.Filter.IncludeComputeOnly = 1;
  NTSTATUS status = api.enum_adapters3(&enum_args);
  if (!CheckStatus("D3DKMTEnumAdapters3(count)", status, out_error)) {
    return false;
  }
  if (enum_args.NumAdapters == 0) {
    if (out_error) *out_error = "D3DKMTEnumAdapters3 returned no adapters";
    return false;
  }

  std::vector<D3DKMT_ADAPTERINFO> adapters(enum_args.NumAdapters);
  enum_args.pAdapters = adapters.data();
  status = api.enum_adapters3(&enum_args);
  if (!CheckStatus("D3DKMTEnumAdapters3(list)", status, out_error)) {
    return false;
  }
  adapters.resize(enum_args.NumAdapters);

  Adapter exact;
  Adapter fallback;
  Adapter loose;
  for (const D3DKMT_ADAPTERINFO& adapter : adapters) {
    D3DKMT_DRIVER_DESCRIPTION description = {};
    D3DKMT_QUERYADAPTERINFO query = {};
    query.hAdapter = adapter.hAdapter;
    query.Type = KMTQAITYPE_DRIVER_DESCRIPTION;
    query.pPrivateDriverData = &description;
    query.PrivateDriverDataSize = sizeof(description);
    status = api.query_adapter_info(&query);
    if (status != 0) continue;

    std::wstring text = description.DriverDescription;
    if (text == L"AMD XDNA(TM) NPU") {
      exact = {adapter.hAdapter, text};
      break;
    }
    if (!fallback.handle && text == L"NPU Compute Accelerator Device") {
      fallback = {adapter.hAdapter, text};
    }
    if (!loose.handle && text.find(L"NPU") != std::wstring::npos) {
      loose = {adapter.hAdapter, text};
    }
  }

  Adapter selected =
      exact.handle ? exact : (fallback.handle ? fallback : loose);
  if (!selected.handle) {
    for (const D3DKMT_ADAPTERINFO& adapter : adapters) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = adapter.hAdapter;
      api.close_adapter(&close);
    }
    if (out_error) *out_error = "no NPU adapter was found";
    return false;
  }

  for (const D3DKMT_ADAPTERINFO& adapter : adapters) {
    if (adapter.hAdapter == selected.handle) continue;
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    api.close_adapter(&close);
  }

  *out_adapter = selected;
  return true;
}

bool CreateDevice(const KmtApi& api, D3DKMT_HANDLE adapter, Device* out_device,
                  std::string* out_error) {
  Device device = {};
  device.adapter = adapter;

  D3DKMT_CREATEDEVICE create_device = {};
  create_device.hAdapter = adapter;
  NTSTATUS status = api.create_device(&create_device);
  if (!CheckStatus("D3DKMTCreateDevice", status, out_error)) return false;
  device.device = create_device.hDevice;

  D3DKMT_CREATEPAGINGQUEUE paging = {};
  paging.hDevice = device.device;
  paging.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;
  paging.PhysicalAdapterIndex = 0;
  status = api.create_paging_queue(&paging);
  if (!CheckStatus("D3DKMTCreatePagingQueue", status, out_error)) {
    D3DKMT_DESTROYDEVICE destroy_device = {};
    destroy_device.hDevice = device.device;
    api.destroy_device(&destroy_device);
    return false;
  }

  device.paging_queue = paging.hPagingQueue;
  device.paging_sync_object = paging.hSyncObject;
  device.paging_fence_cpu = paging.FenceValueCPUVirtualAddress;
  *out_device = device;
  return true;
}

void DestroyDevice(const KmtApi& api, Device* device) {
  if (!device) return;
  if (device->paging_queue) {
    D3DDDI_DESTROYPAGINGQUEUE destroy_paging = {};
    destroy_paging.hPagingQueue = device->paging_queue;
    api.destroy_paging_queue(&destroy_paging);
    device->paging_queue = 0;
  }
  if (device->device) {
    D3DKMT_DESTROYDEVICE destroy_device = {};
    destroy_device.hDevice = device->device;
    api.destroy_device(&destroy_device);
    device->device = 0;
  }
  if (device->adapter) {
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = device->adapter;
    api.close_adapter(&close);
    device->adapter = 0;
  }
}

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, std::string* out_error) {
  BufferKindInfo kind_info = GetBufferKindInfo(kind);
  uint64_t aligned_size = AlignUpToPage(std::max<uint64_t>(size, 1));
  uint64_t size_pages = aligned_size / 4096;

  AllocPrivate alloc_private = {};
  alloc_private.requested_size = aligned_size;
  alloc_private.aligned_size = aligned_size;
  alloc_private.private_type = kind_info.private_type;
  alloc_private.xcl_flags = kind_info.xcl_flags;

  D3DDDI_ALLOCATIONINFO2 alloc_info = {};
  alloc_info.pPrivateDriverData = &alloc_private;
  alloc_info.PrivateDriverDataSize = sizeof(alloc_private);

  D3DKMT_CREATEALLOCATION create = {};
  create.hDevice = device.device;
  create.NumAllocations = 1;
  create.pAllocationInfo2 = &alloc_info;

  NTSTATUS status = api.create_allocation2(&create);
  if (!CheckStatus("D3DKMTCreateAllocation2", status, out_error)) {
    return false;
  }

  Buffer buffer = {};
  buffer.kind = kind;
  buffer.size = aligned_size;
  buffer.allocation = alloc_info.hAllocation;

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = buffer.allocation;
  map.SizeInPages = size_pages;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.gpu_va = map.VirtualAddress;

  D3DKMT_HANDLE resident_allocs[1] = {buffer.allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  resident.Flags.CantTrimFurther = 1;
  resident.Flags.MustSucceed = 1;
  status = api.make_resident(&resident);
  if (!CheckStatusOrPending("D3DKMTMakeResident", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer.allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.cpu_ptr = lock.pData;

  // Exercise a minimal write to prove the mapping is CPU-accessible. The probe
  // syncs the range before teardown.
  if (buffer.cpu_ptr && buffer.size >= sizeof(uint32_t)) {
    uint32_t pattern = 0xa11e0000u | kind_info.private_type;
    std::memcpy(buffer.cpu_ptr, &pattern, sizeof(pattern));
  }

  *out_buffer = buffer;
  return true;
}

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, std::string* out_error) {
  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = buffer.allocation;
  invalidate.Offset = offset;
  invalidate.Length = length;
  NTSTATUS status = api.invalidate_cache(&invalidate);
  return CheckStatus("D3DKMTInvalidateCache", status, out_error);
}

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer) {
  if (!buffer || !buffer->allocation) return;
  if (buffer->cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = buffer->allocation;
    api.unlock2(&unlock);
    buffer->cpu_ptr = nullptr;
  }
  if (buffer->gpu_va) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = buffer->gpu_va;
    free_va.Size = buffer->size;
    api.free_gpu_virtual_address(&free_va);
    buffer->gpu_va = 0;
  }
  D3DKMT_HANDLE allocs[1] = {buffer->allocation};
  D3DKMT_DESTROYALLOCATION2 destroy = {};
  destroy.hDevice = device.device;
  destroy.AllocationCount = 1;
  destroy.phAllocationList = allocs;
  destroy.Flags.AssumeNotInUse = 1;
  api.destroy_allocation2(&destroy);
  buffer->allocation = 0;
  buffer->size = 0;
}

bool CreateContext(const KmtApi& api, const Device& device,
                   const std::vector<uint8_t>& private_data,
                   Context* out_context, std::string* out_error) {
  if (!out_context) {
    if (out_error) *out_error = "CreateContext called with null output";
    return false;
  }

  Context context = {};
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {};
  create_context.hDevice = device.device;
  create_context.NodeOrdinal = 0;
  create_context.EngineAffinity = 1;
  create_context.Flags.HwQueueSupported = 1;
  create_context.pPrivateDriverData =
      const_cast<uint8_t*>(private_data.data());
  create_context.PrivateDriverDataSize =
      static_cast<UINT>(private_data.size());
  create_context.ClientHint = D3DKMT_CLIENTHINT_VITIS;
  NTSTATUS status = api.create_context_virtual(&create_context);
  if (!CheckStatus("D3DKMTCreateContextVirtual", status, out_error)) {
    return false;
  }
  context.context = create_context.hContext;

  D3DKMT_CREATEHWQUEUE create_queue = {};
  create_queue.hHwContext = context.context;
  status = api.create_hw_queue(&create_queue);
  if (!CheckStatus("D3DKMTCreateHwQueue", status, out_error)) {
    DestroyContext(api, &context);
    return false;
  }
  context.hw_queue = create_queue.hHwQueue;
  context.progress_fence = create_queue.hHwQueueProgressFence;
  context.progress_fence_cpu =
      create_queue.HwQueueProgressFenceCPUVirtualAddress;
  context.progress_fence_gpu =
      create_queue.HwQueueProgressFenceGPUVirtualAddress;

  *out_context = context;
  return true;
}

void DestroyContext(const KmtApi& api, Context* context) {
  if (!context) return;
  if (context->hw_queue) {
    D3DKMT_DESTROYHWQUEUE destroy_queue = {};
    destroy_queue.hHwQueue = context->hw_queue;
    api.destroy_hw_queue(&destroy_queue);
    context->hw_queue = 0;
  }
  if (context->context) {
    D3DKMT_DESTROYCONTEXT destroy_context = {};
    destroy_context.hContext = context->context;
    api.destroy_context(&destroy_context);
    context->context = 0;
  }
  context->progress_fence = 0;
  context->progress_fence_cpu = nullptr;
  context->progress_fence_gpu = 0;
  context->next_fence_id = 1;
}

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture,
                           std::string* out_error) {
  if (!out_aperture) {
    if (out_error) *out_error = "CreateCommandAperture called with null output";
    return false;
  }

  AllocPrivate alloc_private = {};
  alloc_private.requested_size = kCommandApertureAllocationSize;
  alloc_private.aligned_size = kCommandApertureAllocationSize;
  alloc_private.private_type = kCommandAperturePrivateType;
  alloc_private.policy = 0;

  D3DDDI_ALLOCATIONINFO2 alloc_info = {};
  alloc_info.pPrivateDriverData = &alloc_private;
  alloc_info.PrivateDriverDataSize = sizeof(alloc_private);

  D3DKMT_CREATEALLOCATION create = {};
  create.hDevice = device.device;
  create.Flags.CreateResource = 1;
  create.Flags.CreateShared = 1;
  create.NumAllocations = 1;
  create.pAllocationInfo2 = &alloc_info;

  NTSTATUS status = api.create_allocation2(&create);
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture)", status,
                   out_error)) {
    return false;
  }

  CommandAperture aperture = {};
  aperture.allocation_size = kCommandApertureAllocationSize;
  aperture.gpu_va_size = kCommandApertureGpuVaSize;
  aperture.allocation = alloc_info.hAllocation;
  aperture.resource = create.hResource;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = aperture.allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2(command aperture)", status, out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.cpu_ptr = lock.pData;
  if (aperture.cpu_ptr) {
    std::memset(aperture.cpu_ptr, 0,
                static_cast<size_t>(aperture.allocation_size));
  }

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture.allocation;
  map.BaseAddress = kCommandApertureBase;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress(command aperture)",
                            status, out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.gpu_va = map.VirtualAddress;

  D3DKMT_HANDLE resident_allocs[1] = {aperture.allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  resident.Flags.CantTrimFurther = 1;
  resident.Flags.MustSucceed = 1;
  status = api.make_resident(&resident);
  if (!CheckStatusOrPending("D3DKMTMakeResident(command aperture)", status,
                            out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
  UINT64 wait_values[1] = {resident.PagingFenceValue};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
  wait.hContext = context.context;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.MonitoredFenceValueArray = wait_values;
  status = api.wait_from_gpu(&wait);
  if (!CheckStatus("D3DKMTWaitForSynchronizationObjectFromGpu(command aperture)",
                   status, out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  *out_aperture = aperture;
  return true;
}

bool SubmitAndWaitCommandAperture(const KmtApi& api, const Device& device,
                                  Context* context,
                                  const CommandAperture& aperture,
                                  std::string* out_error) {
  if (!context || !context->hw_queue) {
    if (out_error) *out_error = "SubmitAndWait called without an HW queue";
    return false;
  }

  uint64_t submit_private[kSubmitPrivateQwords] = {};
  submit_private[0] = 2;
  submit_private[1] = aperture.allocation;
  submit_private[2] = aperture.gpu_va;

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = aperture.gpu_va;
  submit.CommandLength = static_cast<UINT>(aperture.gpu_va_size);
  submit.PrivateDriverDataSize = sizeof(submit_private);
  submit.pPrivateDriverData = submit_private;
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue", status, out_error)) {
    return false;
  }

  HANDLE wait_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  D3DKMT_HANDLE wait_objects[1] = {context->progress_fence};
  UINT64 wait_values[1] = {fence_id + 1};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
  wait.hDevice = device.device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.FenceValueArray = wait_values;
  wait.hAsyncEvent = wait_event;
  status = api.wait_from_cpu(&wait);
  if (status == kStatusPending && wait_event) {
    DWORD wait_result = WaitForSingleObject(wait_event, 5000);
    if (wait_result != WAIT_OBJECT_0) {
      status = static_cast<NTSTATUS>(WAIT_TIMEOUT);
    } else {
      status = 0;
    }
  }
  if (wait_event) CloseHandle(wait_event);
  return CheckStatus("D3DKMTWaitForSynchronizationObjectFromCpu", status,
                     out_error);
}

void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture) {
  if (!aperture || (!aperture->allocation && !aperture->resource)) return;
  if (aperture->gpu_va) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = aperture->gpu_va;
    free_va.Size = aperture->gpu_va_size;
    api.free_gpu_virtual_address(&free_va);
    aperture->gpu_va = 0;
  }
  if (aperture->cpu_ptr && aperture->allocation) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = aperture->allocation;
    api.unlock2(&unlock);
    aperture->cpu_ptr = nullptr;
  }

  D3DKMT_DESTROYALLOCATION2 destroy = {};
  destroy.hDevice = device.device;
  destroy.Flags.AssumeNotInUse = 1;
  if (aperture->resource) {
    destroy.hResource = aperture->resource;
  } else if (aperture->allocation) {
    D3DKMT_HANDLE allocs[1] = {aperture->allocation};
    destroy.AllocationCount = 1;
    destroy.phAllocationList = allocs;
  }
  api.destroy_allocation2(&destroy);
  *aperture = {};
}

std::string NtStatusToString(NTSTATUS status) {
  std::ostringstream os;
  os << "0x" << std::hex << static_cast<uint32_t>(status);
  if (status == kStatusPending) {
    os << " (STATUS_PENDING)";
  }
  return os.str();
}

}  // namespace iree::hal::amdxdna::mcdm
