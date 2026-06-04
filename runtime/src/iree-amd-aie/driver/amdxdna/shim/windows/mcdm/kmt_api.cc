// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103);
constexpr uint64_t kPageSize = 4096;
constexpr uint64_t kCommandApertureAllocationSize = 0x1000;
constexpr D3DGPU_VIRTUAL_ADDRESS kCommandApertureGpuVaBase = 0x04000000;
constexpr uint64_t kCommandApertureGpuVaSize = 0x04000000;
constexpr uint32_t kCommandAperturePrivateType = 0x332b;
constexpr uint64_t kCommandControlBufferSize = 0x1000;
constexpr uint32_t kQhdlSubmitPrivateSize = 0x268;
constexpr uint32_t kQhdlSubmitPacketOffset = 0x68;
constexpr uint32_t kQhdlCompletionSlotSize = 8;
constexpr uint32_t kSubmitPrivateQwords = 13;  // 104 bytes on current driver.
constexpr char kApertureLockHandleDeltaEnv[] =
    "IREE_AMDXDNA_MCDM_APERTURE_LOCK_HANDLE_DELTA";
constexpr char kApertureGpuHandleDeltaEnv[] =
    "IREE_AMDXDNA_MCDM_APERTURE_GPU_HANDLE_DELTA";
constexpr char kTraceQhdlEnv[] = "IREE_AMDXDNA_MCDM_TRACE_QHDL";

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

bool TraceQhdlEnabled() {
  const char* value = std::getenv(kTraceQhdlEnv);
  return value && value[0] && value[0] != '0';
}

void WriteU32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU64(std::vector<uint8_t>* data, size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
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

uint32_t ReadHandleDeltaEnv(const char* name, uint32_t default_value) {
  const char* text = std::getenv(name);
  if (!text || !*text) return default_value;
  char* end = nullptr;
  unsigned long value = std::strtoul(text, &end, 0);
  if (!end || *end != '\0') return default_value;
  return static_cast<uint32_t>(value);
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
  buffer.paging_fence_value = resident.PagingFenceValue;

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

bool WaitForBufferResidency(const KmtApi& api, const Device& device,
                            const Context& context, const Buffer& buffer,
                            const char* label, std::string* out_error) {
  if (buffer.paging_fence_value == 0) return true;

  D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
  UINT64 wait_values[1] = {buffer.paging_fence_value};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
  wait.hContext = context.context;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.MonitoredFenceValueArray = wait_values;
  NTSTATUS status = api.wait_from_gpu(&wait);
  std::string call_name = "D3DKMTWaitForSynchronizationObjectFromGpu";
  if (label && label[0]) {
    call_name += "(";
    call_name += label;
    call_name += ")";
  }
  return CheckStatus(call_name.c_str(), status, out_error);
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

  AllocPrivate command_private = {};
  command_private.requested_size = kCommandApertureAllocationSize;
  command_private.aligned_size = kCommandApertureAllocationSize;
  command_private.private_type = kCommandAperturePrivateType;
  command_private.policy = 0;

  D3DDDI_ALLOCATIONINFO2 command_info = {};
  command_info.pPrivateDriverData = &command_private;
  command_info.PrivateDriverDataSize = sizeof(command_private);

  D3DKMT_CREATEALLOCATION create_command = {};
  create_command.hDevice = device.device;
  create_command.Flags.CreateResource = 1;
  create_command.Flags.CreateShared = 1;
  create_command.NumAllocations = 1;
  create_command.pAllocationInfo2 = &command_info;

  NTSTATUS status = api.create_allocation2(&create_command);
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture)", status,
                   out_error)) {
    return false;
  }

  CommandAperture aperture = {};
  aperture.allocation_size = kCommandApertureAllocationSize;
  aperture.gpu_va_size = kCommandApertureGpuVaSize;
  aperture.allocation = command_info.hAllocation;
  aperture.resource = create_command.hResource;

  const uint32_t lock_handle_delta =
      ReadHandleDeltaEnv(kApertureLockHandleDeltaEnv, 0);
  const uint32_t gpu_handle_delta =
      ReadHandleDeltaEnv(kApertureGpuHandleDeltaEnv, 0x40);
  const D3DKMT_HANDLE lock_allocation =
      aperture.allocation + lock_handle_delta;
  aperture.cleanup_allocation = lock_allocation;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = lock_allocation;
  status = api.lock2(&lock);
  if (status != 0) {
    if (out_error) {
      std::ostringstream os;
      os << "D3DKMTLock2(command aperture) failed with "
         << NtStatusToString(status) << " allocation=0x" << std::hex
         << aperture.allocation << " lock_allocation=0x" << lock_allocation
         << " resource=0x" << aperture.resource;
      *out_error = os.str();
    }
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.cpu_ptr = lock.pData;
  if (aperture.cpu_ptr) {
    std::memset(aperture.cpu_ptr, 0,
                static_cast<size_t>(aperture.allocation_size));
  }

  aperture.gpu_allocation = aperture.allocation + gpu_handle_delta;

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture.gpu_allocation;
  map.BaseAddress = kCommandApertureGpuVaBase;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (status != 0 && status != kStatusPending) {
    if (out_error) {
      std::ostringstream os;
      os << "D3DKMTMapGpuVirtualAddress(command aperture) failed with "
         << NtStatusToString(status) << " allocation=0x" << std::hex
         << aperture.allocation << " gpu_allocation=0x"
         << aperture.gpu_allocation << " lock_allocation=0x"
         << lock_allocation << " resource=0x" << aperture.resource
         << " base=0x" << kCommandApertureGpuVaBase << " pages=0x"
         << map.SizeInPages << " lock_delta=0x" << lock_handle_delta
         << " gpu_delta=0x" << gpu_handle_delta;
      *out_error = os.str();
    }
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.gpu_va = map.VirtualAddress;
  if (aperture.gpu_va != kCommandApertureGpuVaBase) {
    if (out_error) {
      std::ostringstream os;
      os << "D3DKMTMapGpuVirtualAddress(command aperture) returned VA 0x"
         << std::hex << aperture.gpu_va << ", expected 0x"
         << kCommandApertureGpuVaBase;
      *out_error = os.str();
    }
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  D3DKMT_HANDLE resident_allocs[1] = {aperture.gpu_allocation};
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

bool CreateCommandControlBuffer(const KmtApi& api, const Device& device,
                                CommandControlBuffer* out_buffer,
                                std::string* out_error) {
  if (!out_buffer) {
    if (out_error) {
      *out_error = "CreateCommandControlBuffer called with null output";
    }
    return false;
  }

  AllocPrivate command_private = {};
  command_private.requested_size = kCommandControlBufferSize;
  command_private.aligned_size = kCommandControlBufferSize;
  command_private.private_type = kCommandAperturePrivateType;
  command_private.policy = 0;

  D3DDDI_ALLOCATIONINFO2 command_info = {};
  command_info.pPrivateDriverData = &command_private;
  command_info.PrivateDriverDataSize = sizeof(command_private);

  D3DKMT_CREATEALLOCATION create_command = {};
  create_command.hDevice = device.device;
  create_command.Flags.CreateResource = 1;
  create_command.Flags.CreateShared = 1;
  create_command.NumAllocations = 1;
  create_command.pAllocationInfo2 = &command_info;

  NTSTATUS status = api.create_allocation2(&create_command);
  if (!CheckStatus("D3DKMTCreateAllocation2(command control)", status,
                   out_error)) {
    return false;
  }

  CommandControlBuffer buffer = {};
  buffer.size = kCommandControlBufferSize;
  buffer.allocation = command_info.hAllocation;
  buffer.resource = create_command.hResource;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer.allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2(command control)", status, out_error)) {
    DestroyCommandControlBuffer(api, device, &buffer);
    return false;
  }
  buffer.cpu_ptr = lock.pData;
  buffer.next_slot_offset = kQhdlCompletionSlotSize;
  if (buffer.cpu_ptr) {
    std::memset(buffer.cpu_ptr, 0, static_cast<size_t>(buffer.size));
    uint64_t initialized = 1;
    std::memcpy(buffer.cpu_ptr, &initialized, sizeof(initialized));
  }

  *out_buffer = buffer;
  return true;
}

void DestroyCommandControlBuffer(const KmtApi& api, const Device& device,
                                 CommandControlBuffer* buffer) {
  if (!buffer || !buffer->allocation) return;
  if (buffer->cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = buffer->allocation;
    api.unlock2(&unlock);
    buffer->cpu_ptr = nullptr;
  }
  if (buffer->resource) {
    D3DKMT_DESTROYALLOCATION2 destroy = {};
    destroy.hDevice = device.device;
    destroy.hResource = buffer->resource;
    destroy.Flags.AssumeNotInUse = 1;
    api.destroy_allocation2(&destroy);
  } else {
    D3DKMT_HANDLE allocs[1] = {buffer->allocation};
    D3DKMT_DESTROYALLOCATION2 destroy = {};
    destroy.hDevice = device.device;
    destroy.AllocationCount = 1;
    destroy.phAllocationList = allocs;
    destroy.Flags.AssumeNotInUse = 1;
    api.destroy_allocation2(&destroy);
  }
  buffer->allocation = 0;
  buffer->resource = 0;
  buffer->size = 0;
  buffer->next_slot_offset = 0;
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
  submit_private[1] = aperture.gpu_allocation;
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

  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = aperture.gpu_allocation;
  invalidate.Offset = 0;
  invalidate.Length = aperture.gpu_va_size;
  status = api.invalidate_cache(&invalidate);
  // XRT invalidates the GPU-side sibling handle after submit. The KMT-only
  // fallback maps the returned allocation handle directly, where this cache
  // operation may be rejected even though submit/wait succeeds.
  (void)status;

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

bool SubmitAndWaitQhdlCommand(const KmtApi& api, const Device& device,
                              Context* context,
                              CommandControlBuffer* control,
                              const Buffer& command_buffer,
                              uint32_t command_bytes, uint32_t command_state,
                              uint32_t command_allocation_tag,
                              uint32_t* packet_header,
                              std::string* out_error) {
  if (!context || !context->hw_queue) {
    if (out_error) {
      *out_error = "SubmitAndWaitQhdlCommand called without an HW queue";
    }
    return false;
  }
  if (!control || !control->allocation || !control->cpu_ptr ||
      control->size < kQhdlCompletionSlotSize * 2) {
    if (out_error) {
      *out_error = "SubmitAndWaitQhdlCommand called without a command control buffer";
    }
    return false;
  }
  if (!command_buffer.allocation || !command_buffer.gpu_va ||
      command_buffer.size == 0) {
    if (out_error) {
      *out_error = "SubmitAndWaitQhdlCommand called with an invalid command BO";
    }
    return false;
  }
  if (!packet_header || command_bytes == 0 ||
      command_bytes > kQhdlSubmitPrivateSize - kQhdlSubmitPacketOffset) {
    if (out_error) {
      std::ostringstream os;
      os << "SubmitAndWaitQhdlCommand called with invalid packet bytes "
         << command_bytes;
      *out_error = os.str();
    }
    return false;
  }

  if (!WaitForBufferResidency(api, device, *context, command_buffer,
                              "qhdl-command", out_error)) {
    return false;
  }

  uint32_t completion_offset = control->next_slot_offset;
  if (completion_offset + kQhdlCompletionSlotSize > control->size) {
    completion_offset = 0;
  }
  control->next_slot_offset =
      completion_offset + kQhdlCompletionSlotSize >= control->size
          ? 0
          : completion_offset + kQhdlCompletionSlotSize;

  uint8_t* completion_slot =
      static_cast<uint8_t*>(control->cpu_ptr) + completion_offset;
  std::memset(completion_slot, 0, kQhdlCompletionSlotSize);

  std::vector<uint8_t> private_data(kQhdlSubmitPrivateSize, 0);
  // XRT's qhdl submit block is intentionally sparse for the normal ERT path:
  // +0x00 = command state/type, +0x08 = command allocation, +0x28..0x38 =
  // the completion slot in the armed command aperture, +0x68 = packet bytes.
  WriteU32(&private_data, 0x00, command_state);
  WriteU64(&private_data, 0x08, command_allocation_tag
                                ? command_allocation_tag
                                : command_buffer.allocation);
  WriteU64(&private_data, 0x28, control->allocation);
  WriteU32(&private_data, 0x30, completion_offset);
  WriteU32(&private_data, 0x34, kQhdlCompletionSlotSize);
  WriteU64(&private_data, 0x38,
           reinterpret_cast<uint64_t>(completion_slot));
  std::memcpy(private_data.data() + kQhdlSubmitPacketOffset, packet_header,
              command_bytes);

  std::atomic_thread_fence(std::memory_order_seq_cst);
  FlushProcessWriteBuffers();

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = command_buffer.gpu_va;
  submit.CommandLength =
      static_cast<UINT>(command_buffer.size + kQhdlSubmitPacketOffset);
  submit.PrivateDriverDataSize = static_cast<UINT>(private_data.size());
  submit.pPrivateDriverData = private_data.data();
  if (TraceQhdlEnabled()) {
    std::fprintf(stderr,
                 "[amdxdna:mcdm] qhdl submit: hwq=0x%08x fence=%llu "
                 "cmd_alloc=0x%08x cmd_va=0x%llx cmd_len=%u "
                 "slot_alloc=0x%08x slot_off=0x%x slot_ptr=0x%llx "
                 "private_size=%u header=0x%08x\n",
                 static_cast<unsigned>(context->hw_queue),
                 static_cast<unsigned long long>(fence_id),
                 static_cast<unsigned>(command_buffer.allocation),
                 static_cast<unsigned long long>(command_buffer.gpu_va),
                 static_cast<unsigned>(submit.CommandLength),
                 static_cast<unsigned>(control->allocation),
                 static_cast<unsigned>(completion_offset),
                 static_cast<unsigned long long>(
                     reinterpret_cast<uintptr_t>(completion_slot)),
                 static_cast<unsigned>(submit.PrivateDriverDataSize),
                 packet_header ? *packet_header : 0);
    std::fflush(stderr);
  }
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(qhdl)", status, out_error)) {
    return false;
  }

  uint32_t completion_header = 0;
  const uint64_t deadline = GetTickCount64() + 5000;
  do {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::memcpy(&completion_header, completion_slot,
                sizeof(completion_header));
    if ((completion_header & 0xFu) >= 4) break;
    Sleep(1);
  } while (GetTickCount64() < deadline);
  if (packet_header) {
    uint32_t state_delta = (*packet_header ^ completion_header) & 0xFu;
    *packet_header ^= state_delta;
  }
  if ((completion_header & 0xFu) < 4) {
    if (out_error) {
      std::ostringstream os;
      os << "qhdl completion slot did not complete: header=0x" << std::hex
         << completion_header << " offset=0x" << completion_offset;
      *out_error = os.str();
    }
    return false;
  }
  return true;
}

bool SubmitAndWaitBuffer(const KmtApi& api, const Device& device,
                         Context* context, const Buffer& buffer,
                         std::string* out_error) {
  if (!context || !context->hw_queue) {
    if (out_error) *out_error = "SubmitAndWaitBuffer called without an HW queue";
    return false;
  }
  if (!buffer.allocation || !buffer.gpu_va || buffer.size == 0) {
    if (out_error) *out_error = "SubmitAndWaitBuffer called with invalid buffer";
    return false;
  }

  if (!WaitForBufferResidency(api, device, *context, buffer, "buffer",
                              out_error)) {
    return false;
  }

  uint64_t submit_private[kSubmitPrivateQwords] = {};
  submit_private[0] = 2;
  submit_private[1] = buffer.allocation;
  submit_private[2] = buffer.gpu_va;

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = buffer.gpu_va;
  submit.CommandLength = static_cast<UINT>(buffer.size);
  submit.PrivateDriverDataSize = sizeof(submit_private);
  submit.pPrivateDriverData = submit_private;
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(buffer)", status,
                   out_error)) {
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
  return CheckStatus("D3DKMTWaitForSynchronizationObjectFromCpu(buffer)",
                     status, out_error);
}

void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture) {
  if (!aperture ||
      (!aperture->allocation && !aperture->resource &&
       !aperture->gpu_allocation && !aperture->gpu_resource &&
       !aperture->cleanup_allocation)) {
    return;
  }
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
    unlock.hAllocation = aperture->cleanup_allocation
                             ? aperture->cleanup_allocation
                             : aperture->allocation;
    api.unlock2(&unlock);
    aperture->cpu_ptr = nullptr;
  }

  bool owns_separate_gpu_allocation = aperture->gpu_resource != 0;
  if (owns_separate_gpu_allocation) {
    D3DKMT_DESTROYALLOCATION2 destroy_gpu = {};
    destroy_gpu.hDevice = device.device;
    destroy_gpu.Flags.AssumeNotInUse = 1;
    if (aperture->gpu_resource) {
      destroy_gpu.hResource = aperture->gpu_resource;
    } else if (aperture->gpu_allocation) {
      D3DKMT_HANDLE allocs[1] = {aperture->gpu_allocation};
      destroy_gpu.AllocationCount = 1;
      destroy_gpu.phAllocationList = allocs;
    }
    api.destroy_allocation2(&destroy_gpu);
    aperture->gpu_allocation = 0;
    aperture->gpu_resource = 0;
  }

  if (aperture->resource || aperture->cleanup_allocation ||
      aperture->allocation) {
    D3DKMT_DESTROYALLOCATION2 destroy_command = {};
    destroy_command.hDevice = device.device;
    destroy_command.Flags.AssumeNotInUse = 1;
    if (aperture->resource) {
      destroy_command.hResource = aperture->resource;
    } else if (aperture->cleanup_allocation) {
      D3DKMT_HANDLE allocs[1] = {aperture->cleanup_allocation};
      destroy_command.AllocationCount = 1;
      destroy_command.phAllocationList = allocs;
    } else if (aperture->allocation) {
      D3DKMT_HANDLE allocs[1] = {aperture->allocation};
      destroy_command.AllocationCount = 1;
      destroy_command.phAllocationList = allocs;
    }
    api.destroy_allocation2(&destroy_command);
  }
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
