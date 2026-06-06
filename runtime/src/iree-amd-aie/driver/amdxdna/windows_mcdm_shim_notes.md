# Windows MCDM shim notes

This note records the current investigation for implementing an XRT-free Windows
transport for the `amdxdna` HAL backend. The goal is to keep XRT as a reference
only, while talking directly to the AMD MCDM kernel driver through the public KMT
ABI and the smallest set of AMD-private payloads needed for Ryzen AI execution.

## Local machine facts

The tested machine exposes the NPU as:

* PnP device: `NPU Compute Accelerator Device`
* PCI ID: `VEN_1022&DEV_17F0`, BDF reported by XRT/KMT as `00c5:00:01.1`
* Driver service: `IpuMcdmDriver`
* Driver binary: `ipustack.sys`
* Driver version: `32.0.203.280`
* Firmware version reported by XRT: `1.0.18.3`
* XRT package version in the AMD driver package: `2.19.0`

The installed XRT binaries are useful for reference, but should not be linked
from IREE.

## Public KMT surface verified

A no-XRT PowerShell/C# probe directly called KMT entry points from `gdi32.dll`.
The public KMT path can reach the NPU without XRT:

1. `D3DKMTEnumAdapters3` with `IncludeComputeOnly = 1`.
2. `D3DKMTQueryAdapterInfo(KMTQAITYPE_DRIVER_DESCRIPTION)`.
3. Select `NPU Compute Accelerator Device`.
4. `D3DKMTCreateDevice`.
5. `D3DKMTCreateContextVirtual` with `D3DDDI_CREATECONTEXTFLAGS.HwQueueSupported`.
6. `D3DKMTCreateHwQueue`.
7. Destroy queue, context, device, adapter.

Observed results:

```text
D3DKMTEnumAdapters3 IncludeComputeOnly status=0x00000000 count=3
adapter[1] bdf=00c5:00:1 driverDescription=NPU Compute Accelerator Device
D3DKMTCreateDevice status=0x00000000
D3DKMTCreateContextVirtual flags=0x10 status=0x00000000
D3DKMTCreateHwQueue flags=0x0 status=0x00000000
```

Context creation without `HwQueueSupported` failed:

```text
D3DKMTCreateContextVirtual flags=0x0 status=0xc0000001
```

A 4 KB public standard allocation probe failed:

```text
D3DKMTCreateAllocation2 standard status=0xc000000d
```

This confirms that buffer creation needs AMD-private allocation data, which
matches the installed XRT strings for `arg_bo`, `cmd_bo`, `instr_bo`,
`status_bo`, and `carveout_bo`.

## Installed XRT-MCDM evidence

The installed `xrt_core.dll` imports or dynamically resolves the expected KMT
entry points:

* `D3DKMTEnumAdapters3`
* `D3DKMTQueryAdapterInfo`
* `D3DKMTCreateDevice`
* `D3DKMTCreateAllocation2`
* `D3DKMTLock2`
* `D3DKMTMapGpuVirtualAddress`
* `D3DKMTMakeResident`
* `D3DKMTCreateContextVirtual`
* `D3DKMTCreateHwQueue`
* `D3DKMTSubmitCommandToHwQueue`
* `D3DKMTSubmitWaitForSyncObjectsToHwQueue`
* `D3DKMTSubmitSignalSyncObjectsToHwQueue`
* `D3DKMTEscape`

It also contains MCDM-specific class names:

* `xrt_core::umd::mcdm::device`
* `xrt_core::umd::mcdm::system`
* `xrt_core::umd::shim::bo`
* `xrt_core::umd::shim::arg_bo`
* `xrt_core::umd::shim::cmd_bo`
* `xrt_core::umd::shim::status_bo`
* `xrt_core::umd::shim::instr_bo`
* `xrt_core::umd::shim::hwcontext_aie4`
* `xrt_core::umd::shim::hwqueue_aie4`

Useful error strings include:

* `CMD_CONFIG_CU command failed!`
* `update_qos command failed!`
* `XRT_CMD_CREATE_AIE4_CTX command failed!`
* `internal error: bad ert_packet size`
* `internal error: unsupported opcode`
* `aie4 unsupported ERT opcode`

This points to a standard KMT shell with AMD-private packets for AIE-specific
BO creation, context configuration, and command submission.

Disassembly of the installed `xrt_core.dll` confirms the per-allocation private
type values that were inferred from captures:

| XRT BO flag/use | Private type | Notes |
| --- | ---: | --- |
| cacheable | `0x3323` | Maps successfully with the returned allocation handle. |
| execbuf | `0x3328` | Linux `AMDXDNA_BO_CMD` equivalent; still a host/share BO internally. |
| host-only | `0x3329` | Linux `AMDXDNA_BO_SHARE` equivalent. |
| kern/dev heap-ish | `0x332c` | Present in the disassembly, not yet needed for the minimal path. |
| command aperture | `0x332b` | Special 64 MiB command aperture used by context validation submit. |

The `0x332b` constructor in XRT creates a 56-byte private allocation packet,
calls the driver's create-allocation wrapper, then locks the visible returned
allocation. The map/resident handle used later is not visible in the public
`D3DKMTCreateAllocation2` return fields.

## Captured XRT validation path

A local debugger harness launched `xrt-smi validate --run latency` and trapped
KMT entry/return calls in the XRT process. This provided private packet bytes
that ETW did not expose. The best current full capture is:

```text
C:\Users\jornt\workspace\iree-ai\kmt-capture-xrt-validate-latency-openresource.jsonl
```

`xrt-smi examine` only exercised adapter/query/device/escape paths. The
allocation and submit path appears in the validation workload.

Observed call sequence:

1. `D3DKMTCreateDevice`.
2. `D3DKMTCreatePagingQueue`.
3. `D3DKMTEscape` with a 112-byte private packet.
4. `D3DKMTCreateContextVirtual`.
5. `D3DKMTCreateHwQueue`.
6. `D3DKMTCreateAllocation2`.
7. `D3DKMTLock2`.
8. `D3DKMTMapGpuVirtualAddress`.
9. `D3DKMTMakeResident`.
10. `D3DKMTWaitForSynchronizationObjectFromGpu`.
11. `D3DKMTSubmitCommandToHwQueue`.
12. `D3DKMTWaitForSynchronizationObjectFromCpu`.
13. `D3DKMTUnlock2`.
14. `D3DKMTDestroyAllocation2`.
15. `D3DKMTDestroyHwQueue`.
16. `D3DKMTDestroyContext`.

Important captured details:

* `D3DKMTCreatePagingQueue` returns a paging queue, sync object, and CPU fence
  address. `D3DKMTMapGpuVirtualAddress` and `D3DKMTMakeResident` both use this
  paging queue.
* The 112-byte `D3DKMTEscape` packet changes one qword on return: offset 80
  changes from `ffffffffffffffff` to `0100000000000000`.
* `D3DKMTCreateContextVirtual` uses `HwQueueSupported`, `EngineAffinity = 1`,
  and `D3DKMT_CLIENTHINT_VITIS = 25`. Its private blob is 49,896 bytes for the
  latency workload and contains `xclbin2`, `dummy_bitstream`, AIE/IP/connectivity
  sections, and xclbin JSON. This is not a tiny open-context packet; XRT passes
  xclbin/PDI-style metadata to the driver.
* The first 16 bytes of the context blob are the embedded AXLF UUID. Offset
  `0x58` is the creator process id and is the only byte range that changed
  across the two full-context captures.
* `D3DKMTCreateHwQueue` did not use private driver data in this capture. It
  returns the hardware queue handle plus progress fence CPU/GPU addresses.
* `D3DKMTCreateAllocation2` used resource flags `0x3` (`CreateResource` and
  `CreateShared`) with one allocation and a 56-byte per-allocation private
  packet:

```text
000000000000000000100000000000000010000000000000000000002b330000000000000000000000000000000000000000000000000000000
```

* `D3DKMTMapGpuVirtualAddress` mapped a 64 MiB command buffer to GPU VA
  `0x04000000` and returned `STATUS_PENDING`, followed by `D3DKMTMakeResident`.
* `D3DKMTSubmitCommandToHwQueue` submitted fence id `1`, command buffer GPU VA
  `0x04000000`, command length `67108864`, and a 96-byte private packet:

```text
020000000000000040040040000000000000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

Interpreting this packet as little-endian qwords gives:

```text
qword[0] = 0x0000000000000002
qword[1] = 0x0000000040000440
qword[2] = 0x0000000004000000
qword[3..] = 0
```

`qword[1]` matches the allocation handle used by map/resident, and `qword[2]`
matches the command buffer GPU VA.

Follow-up captures with resource/share/sync hooks found no `D3DKMTShareObjects`,
`D3DKMTQueryResourceInfo*`, `D3DKMTOpenResource*`, or sync-open calls in the
latency path. They did reveal the missing waits:

* After `D3DKMTMakeResident`, XRT calls
  `D3DKMTWaitForSynchronizationObjectFromGpu` on the paging queue sync object,
  waiting for the paging fence returned by `MakeResident`.
* After `D3DKMTSubmitCommandToHwQueue`, XRT calls
  `D3DKMTWaitForSynchronizationObjectFromCpu` on the HW queue progress fence,
  waiting for fence value `2`.

The command BO contents sampled at submit time were zero-filled for the latency
test. The submit behavior is therefore driven by `D3DKMTSubmitCommandToHwQueue`
and its 96-byte private packet, not by a visible ERT packet at the start of the
locked command BO.

One important gap remains: allocation handles differ across create, map/resident,
and teardown even though no open/share KMT calls occur in this process. XRT uses
the created allocation handle for `Lock2`, the next adjacent handle for
map/resident/submit, and a later adjacent handle for unlock/destroy. Treat these
as opaque until the allocation private schema explains them.

The current offline decoder is:

```text
C:\Users\jornt\workspace\iree-ai\scripts\mcdm_decode.py
```

Comparing the latency captures with this decoder showed:

* `D3DKMTEscape`, `D3DKMTCreateAllocation2`, and
  `D3DKMTSubmitCommandToHwQueue` private packets are byte-identical across the
  captures.
* The full `D3DKMTCreateContextVirtual` blobs are byte-identical except for the
  process id at offset `0x58`.
* The allocation handle flow in the richest capture is:

```text
CreateAllocation2 return hAllocation = 0x40000400
Lock2                         uses = 0x40000400
MapGpuVirtualAddress          uses = 0x40000440
MakeResident                  uses = 0x40000440
Submit private qword[1]       uses = 0x40000440
Unlock2                       uses = 0x400004c0
DestroyAllocation2            uses = 0x400004c0
```

These adjacent handles should still be treated as driver-created opaque aliases,
not as arithmetic to bake into the shim, until a create-allocation return path
or private packet field explains them.

### Direct no-XRT replay status

The in-tree probe can already talk to the Windows MCDM driver without XRT for the
non-submit surface:

* Adapter discovery works through `D3DKMTEnumAdapters3` and
  `KMTQAITYPE_DRIVER_DESCRIPTION`.
* `D3DKMTCreateDevice` plus `D3DKMTCreatePagingQueue` works.
* Regular BO creation works for `0x3329`, `0x3323`, and `0x3328` using the
  returned allocation handle for map/resident/lock/sync/destroy.
* Captured `D3DKMTCreateContextVirtual` and `D3DKMTCreateHwQueue` private data
  can be replayed far enough to create and destroy a context/queue.

Historical note: this section is superseded by the 2026-06-03 full-access
update below. At this point in the investigation, the replay was not
submit-ready because the special `0x332b` command aperture could be created and
locked with the visible returned allocation handle, while attempting to map the
`created + 0x40` handle observed in XRT returned `STATUS_INVALID_PARAMETER`.
The visible return fields were checked and were not the source of that XRT
handle:

* `D3DDDI_ALLOCATIONINFO2::GpuVirtualAddress` is zero.
* WDDM 2.2 `Unused` and `Reserved[]` are zero.
* `D3DKMT_CREATEALLOCATION::hPrivateRuntimeResourceHandle`, when supplied, is
  returned as null.

The later fixed-base probe showed that the visible allocation handle is usable
for map/resident/submit when `D3DDDI_MAPGPUVIRTUALADDRESS::BaseAddress` is set
to `0x04000000`, and that the resource can be destroyed through
`D3DKMT_DESTROYALLOCATION2::hResource`.

## Linux `amd/xdna-driver` mapping

The Linux UAPI in `include/uapi/drm/amdxdna_accel.h` provides a clean semantic
model for the Windows shim:

| Linux ioctl | Windows KMT/MCDM equivalent |
| --- | --- |
| `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` | `D3DKMTCreateContextVirtual`, then `D3DKMTCreateHwQueue` for KMQ mode. |
| `DRM_IOCTL_AMDXDNA_DESTROY_HWCTX` | `D3DKMTDestroyHwQueue`, then `D3DKMTDestroyContext`. |
| `DRM_IOCTL_AMDXDNA_CREATE_BO` | `D3DKMTCreateAllocation2` with 56-byte AMD private allocation data. |
| `DRM_IOCTL_AMDXDNA_GET_BO_INFO` | Returned KMT allocation handle plus `D3DKMTMapGpuVirtualAddress` GPU VA. |
| `DRM_IOCTL_AMDXDNA_SYNC_BO` | `D3DKMTInvalidateCache` or CPU cache flush, depending on coherency and direction. |
| `DRM_IOCTL_AMDXDNA_EXEC_CMD` | `D3DKMTSubmitCommandToHwQueue` with private submit packet. |
| `DRM_IOCTL_AMDXDNA_WAIT_CMD` | `D3DKMTWaitForSynchronizationObjectFromCpu` on the HW queue progress fence. |

Linux `AMDXDNA_BO_CMD` is created as a SHARE BO with an internal bit, which
matches Windows `0x3328` behaving as a normal host-mappable allocation. The
special Windows `0x332b` command aperture is not the ordinary Linux command BO;
it is the KMT HW-queue command-buffer address range used by the Windows MCDM
submission path.

For IREE, the right implementation strategy remains a lightweight Windows KMT
shim rather than a custom kernel driver. The shim should expose Linux-like
operations (`open`, `create_hwctx`, `create_bo`, `sync_bo`, `submit`, `wait`,
`destroy`) and hide the KMT mechanics internally. A custom kernel driver would
need to reproduce firmware ownership, context scheduling, power management,
PASID/IOMMU, queueing, and signing/installation, while the shipped MCDM driver
already owns all of that.

## Implementation readiness

Ready to implement:

* Adapter/device/paging queue lifetime.
* Regular host/cacheable/execbuf BO creation, map, residency, CPU mapping,
  cache invalidation, and teardown.
* Static BO private type table and 56-byte private allocation packet.
* Context/queue lifetime scaffolding, provided the context private blob can be
  constructed from an xclbin rather than replayed.
* Direct no-op/bootstrap submit through the Windows command aperture, using the
  visible allocation handle, fixed `0x04000000` command VA, and resource
  teardown.

Not ready to implement as production behavior:

* Arbitrary IREE workload submit until ERT packet construction, argument BO
  residency/binding, and AXLF packaging have been validated with a tiny real
  dispatch.
* Arithmetic use of adjacent allocation handles (`created + 0x40`,
  `created + 0xc0`). Those are observed facts from XRT, not an API contract.
* Replaying a captured context blob as a real context creation strategy. The
  blob embeds an AXLF/xclbin and some process-local fields; the shim must build
  this from the actual executable/metadata handed to IREE.

Next low-risk mapping work:

1. Decode the `D3DKMTCreateContextVirtual` private blob schema enough to build
   it from an AXLF/xclbin and current process fields.
2. Wire the proven KMT calls into the Windows `amdxdna` native boundary without
   adding an XRT runtime dependency.
3. Validate a tiny real IREE dispatch, then command-chain dispatch, using the
   same ERT command semantics as the Linux XDNA path.

### Embedded AXLF in the context blob

The context private blob embeds a normal AXLF/xclbin container at offset `0xe0`.
For the latency capture:

```text
AXLF base             : 0xe0
AXLF length           : 0xbcd8
platform_vbnv         : xilinx_v1_ipu_0_0
version               : 2.11.598
section count         : 11
```

The decoded section table is:

```text
[00] BITSTREAM              dummy_bitstream  offset=0x380  size=0x0
[01] MEM_TOPOLOGY           mem_topology      offset=0x380  size=0x58
[02] BUILD_METADATA         vadd.link_build  offset=0x3d8  size=0x8e2
[03] SYSTEM_METADATA        packagedSystemD  offset=0xcc0  size=0x15ec
[04] ASK_GROUP_TOPOLOGY                       offset=0x22b0 size=0x58
[05] AIE_METADATA           aie_control_con  offset=0x2308 size=0x32bc
[06] AIE_PARTITION          aie              offset=0x55c8 size=0x358
[07] IP_LAYOUT              ip               offset=0x5920 size=0xa8
[08] ASK_GROUP_CONNECTIVITY conn             offset=0x59c8 size=0xdc
[09] CONNECTIVITY           conn_only        offset=0x5aa8 size=0x70
[10] EMBEDDED_METADATA      emb              offset=0x5b18 size=0x1612
```

Important decoded binary sections:

```text
AIE_PARTITION:
  column_width = 4
  start_columns = [0]
  pdi[0].uuid = 00000000000000000000000000001111
  pdi[0].image_size = 432
  pdi[0].cdo[0].name = DPU_PDI_0
  pdi[0].cdo[0].type = 3
  pdi[0].cdo[0].pdi_id = 0xf0
  pdi[0].cdo[0].dpu_kernel_ids = [0x100]
  pdi[0].cdo[0].pre_cdo_groups = [0xc0]

IP_LAYOUT:
  ip[0] base = 0x0008000000000101
  ip[1] base = 0xffffffff01000001, name = DPU_PDI_0:IPUV1CNN

CONNECTIVITY:
  9 entries binding args 0,1,2 to ip[0]/HOST and args 1,2,3,4,5,7
  to ip[1], with arg 5 mapped to SRAM.
```

The `MEM_TOPOLOGY` section matches the local compiler's `XCLBinGen.cpp`
template:

```text
mem[0] type=2 used=1 size=0x10000 base=0x04000000 tag=HOST
mem[1] type=2 used=1 size=0x0c000 base=0x04000000 tag=SRAM
```

This is important for IREE: `XCLBinGen.cpp` already knows how to create the
sections Windows MCDM appears to consume (`MEM_TOPOLOGY`, `AIE_PARTITION`,
kernel JSON, and PDI attachment). The Windows shim should not invent a new
metadata format; it should either receive an AXLF from the executable path or
construct the same AXLF wrapper from the existing `amdaie-pdi-fb` contents.

The same decoder also parses shipped AMD overlay xclbins, for example:

```text
C:\Windows\System32\AMD\AMD_AIE2P_4x4_Overlay_3.5.0.0-2354_ipu_2.xclbin
```

That file has the same AXLF header style, the same `MEM_TOPOLOGY` values, and
additional real sections such as:

```text
[07] PDI              full_pm14       offset=0x8aa0  size=0x1eac0
[10] AIE_METADATA     aie_control_con offset=0x27e90 size=0x32f9
[11] AIE_PARTITION    aie_partition   offset=0x2b190 size=0x367408
```

Decoded overlay details:

```text
IP_LAYOUT count = 17
  DPU_PDI_0..DPU_PDI_14:IPUV1CNN plus XDP_KERNEL:IPUV1CNN

AIE_PARTITION:
  start_columns = [0, 4]
  pdi_count = 15
  pdi ids = 0xf0..0xfe
  dpu_kernel_ids = 0x100..0x10e
```

A static survey of all installed `C:\Windows\System32\AMD\*.xclbin` files shows
the context builder cannot assume the one-PDI latency shape:

```text
xclbin count surveyed      : 32
AXLF section count range   : 11..12
IP_LAYOUT count range      : 2..19
CONNECTIVITY count range   : 8..110
AIE_PARTITION pdi_count    : 1..17
observed start_columns     : [0], [1], [0,4], [1,2,3,4],
                             [0,1,2,3], [0,1,2,3,4],
                             [0,1,2,3,4,5,6,7]
```

This strengthens the implementation assumption: Windows MCDM context creation
is fed by ordinary AXLF/xclbin metadata wrapped in an AMD-private
`D3DKMTCreateContextVirtual` prefix.

### Working private packet schemas

These are not ABI headers yet; they are the current structural model from
captured bytes.

```c
// D3DKMTEscape private packet, 112 bytes.
// Stable across latency captures. qword[11] is an in/out field:
// entry: 0xffffffffffffffff, return: 1.
struct mcdm_escape_112 {
  uint64_t q[14];
};
```

```c
// D3DKMTCreateContextVirtual private blob prefix before the embedded AXLF.
// The embedded AXLF starts at byte 0xe0.
struct mcdm_context_prefix_observed {
  uint8_t axlf_uuid[16];        // Matches AXLF header UUID.
  uint8_t reserved0[0x30];
  uint64_t aperture_base;       // Observed 0x04000000.
  uint64_t unknown_48;          // Observed 0x48.
  uint64_t payload_size_like;   // Observed private_size - 0x78.
  uint64_t process_id;          // Must match current process id.
  uint8_t reserved1[0x60];
  uint64_t command_bo_size;     // Observed 0x1000 in latency path.
  uint64_t axlf_length;         // Observed 0xbcd8.
  uint64_t tail_offset0;        // Observed 0xc1b8.
  uint64_t tail_offset1;        // Observed 0xc208.
  uint8_t axlf_data[];          // Starts with "xclbin2\0".
};
```

```c
// D3DKMTCreateAllocation2 per-allocation private packet, 56 bytes.
// Captured for the latency command allocation only.
struct mcdm_allocation_private_56_observed {
  uint64_t reserved0;           // 0.
  uint64_t size;                // 0x1000.
  uint64_t alignment_or_size;   // 0x1000.
  uint32_t reserved1;           // 0.
  uint32_t type_or_flags;       // 0x332b, stable in latency captures.
  uint64_t reserved2[3];        // 0.
};
```

```c
// D3DKMTSubmitCommandToHwQueue private packet, 96 bytes.
// Captured for the latency/nop submission only.
struct mcdm_submit_private_96_observed {
  uint64_t opcode_or_count;     // 2.
  uint64_t allocation_alias;    // Handle used by MapGpuVirtualAddress.
  uint64_t command_gpu_va;      // 0x04000000 in latency path.
  uint64_t reserved[9];         // 0.
};
```

The allocation and submit schemas are only proven for the latency path. Real
IREE dispatch will likely need additional allocation types and submit fields for
ERT command BOs and argument BO handles.

## Replay probe status and safety

A local no-XRT replay probe was added outside the repository at:

```text
C:\Users\jornt\workspace\iree-ai\scripts\mcdm_replay.cpp
```

It reads captured private blobs from JSONL and attempts to replay the KMT path.
The first live replay reached `D3DKMTCreateContextVirtual` and failed with:

```text
STATUS_GRAPHICS_DRIVER_MISMATCH (0xc01e0009)
```

One process-specific field was identified in the context private blob: an early
qword contains the creating process id. In the captured `xrt-smi` run this
field matched the `process_started` PID exactly. The replay source now patches
that early PID field and defaults to dry-run mode unless an explicit
`--execute-risky` flag is passed. The staged probe results below show that this
PID patch is necessary but not sufficient for context replay.

The replay harness was then tightened into an explicit staged probe:

```text
mcdm_replay.exe <capture.jsonl> --execute-risky --stage=<name>

stages:
  discover, device, paging, escape, context, queue, allocation, lock,
  resident, submit
```

The default dry-run refuses live KMT calls. A live run also refuses to execute
without an explicit `--stage`, and `--stage=submit` requires the extra
`--allow-submit` flag. The harness unwinds in reverse order and calls
`D3DKMTFreeGpuVirtualAddress`, `Unlock2`, `DestroyAllocation2`,
`DestroyHwQueue`, `DestroyContext`, `DestroyPagingQueue`, `DestroyDevice`, and
`CloseAdapter` when those resources were created.

Live staged results so far:

```text
--stage=discover  status: passed, selected NPU Compute Accelerator Device
--stage=device    status: passed, CreateDevice/DestroyDevice
--stage=paging    status: passed, CreatePagingQueue/DestroyPagingQueue
--stage=escape    status: passed, captured 112-byte escape packet returned 0
--stage=context   status: failed at CreateContextVirtual with 0xc01e0009
```

The context failure happens even after patching the captured PID. The public
`D3DKMT_CREATECONTEXTVIRTUAL` fields match the capture:

```text
NodeOrdinal = 0
EngineAffinity = 1
Flags.HwQueueSupported = 1
ClientHint = D3DKMT_CLIENTHINT_VITIS (25)
PrivateDriverDataSize = 49896
```

Static checks found no embedded KMT adapter/device handles or obvious user-mode
pointers in the context blob. `CreateContextVirtual` does mutate its private
blob on successful XRT return:

```text
offset 0x003c: 0x00 -> 0x07
offset 0xc2e0: 0x04 -> 0x08
```

So the context packet is an in/out AMD-private structure, not just passive AXLF
metadata. The remaining mismatch likely comes from a hidden process/session
handshake, a context-private field not yet identified, or an XRT-specific
registration path before context creation.

Live replay on this laptop should be tiered. Incorrect private packets can
plausibly crash the MCDM kernel driver or NPU firmware and cause a system
reset/reboot. Do not jump straight to `D3DKMTSubmitCommandToHwQueue` replay.
The safer order is:

1. Static capture/binary decode only.
2. Discovery-only KMT (`EnumAdapters3`, `QueryAdapterInfo`) with no contexts.
3. Create/destroy-only (`CreateDevice`, `CreatePagingQueue`,
   `CreateContextVirtual`, `CreateHwQueue`, then teardown) using known-good
   private blobs patched only for current PID.
4. Allocation-only (`CreateAllocation2`, `Lock2`, `Unlock2`,
   `DestroyAllocation2`) without GPU VA mapping or residency.
5. Map/resident/wait on the paging queue, then immediate teardown.
6. Submit only after the previous tiers are stable across multiple runs.

## Relation to Linux XDNA

The Linux `amdxdna` DRM UAPI remains the best semantic reference. The Windows
shim should mirror the meaning of these operations, not the Linux ioctl numbers:

* `CREATE_BO`
* `GET_BO_INFO`
* `SYNC_BO`
* `CREATE_HWCTX`
* `CONFIG_HWCTX`
* `EXEC_CMD`
* `WAIT_CMD`
* `GET_INFO`
* `SET_STATE`

The local IREE `amdxdna` backend already has the right native boundary in
`native.h`, so the HAL layer should not need a rewrite.

## Windows API surface map

The XRT-free shim should wrap public KMT calls and own all AMD-private packet
construction. Current confidence by call family:

| Family | Windows KMT calls | Linux XDNA semantic | Status |
| --- | --- | --- | --- |
| Adapter discovery | `D3DKMTEnumAdapters3`, `D3DKMTQueryAdapterInfo`, `D3DKMTCloseAdapter` | Device open / `GET_INFO` preflight | Verified by no-XRT probe |
| Device lifetime | `D3DKMTCreateDevice`, `D3DKMTDestroyDevice` | DRM fd lifetime | Verified by no-XRT probe and XRT capture |
| Paging queue | `D3DKMTCreatePagingQueue`, `D3DKMTDestroyPagingQueue` | Residency/mapping synchronization | Captured in latency path |
| Driver queries | `D3DKMTEscape` | `GET_INFO` / `SET_STATE`-like private commands | One 112-byte query decoded; more query opcodes unknown |
| Context | `D3DKMTCreateContextVirtual`, `D3DKMTDestroyContext`, possibly `D3DKMTSetContextSchedulingPriority` | `CREATE_HWCTX` plus `CONFIG_HWCTX` | Captured; private context blob embeds AXLF |
| Hardware queue | `D3DKMTCreateHwQueue`, `D3DKMTDestroyHwQueue` | MCDM HW queue/context scheduling for Strix KMQ | Captured; no private data in latency path |
| BO create/map | `D3DKMTCreateAllocation2`, `D3DKMTDestroyAllocation2`, `D3DKMTLock2`, `D3DKMTUnlock2`, `D3DKMTMapGpuVirtualAddress`, `D3DKMTFreeGpuVirtualAddress` | `CREATE_BO`, `GET_BO_INFO`, mmap, free | Captured for one command BO shape |
| Residency | `D3DKMTMakeResident`, `D3DKMTEvict` | BO residency before submit | `MakeResident` captured; `Evict` not needed yet |
| Cache sync | `D3DKMTInvalidateCache` | `SYNC_BO` | Surfaced by XRT; not captured in latency path |
| Submit | `D3DKMTSubmitCommandToHwQueue` | `EXEC_CMD` | Captured for latency/nop private packet |
| Queue waits/signals | `D3DKMTSubmitWaitForSyncObjectsToHwQueue`, `D3DKMTSubmitSignalSyncObjectsToHwQueue` | Linux fence dependency/signal submission | Surfaced by XRT; not captured in latency path |
| Completion wait | `D3DKMTWaitForSynchronizationObjectFromCpu`, `D3DKMTWaitForSynchronizationObjectFromGpu` | `WAIT_CMD` / syncobj timeline wait | Captured for residency and command completion |
| Sync object lifetime | `D3DKMTCreateSynchronizationObject2`, `D3DKMTDestroySynchronizationObject`, `D3DKMTOpenSyncObjectFromNtHandle2` | Explicit fence objects / imports | Surfaced by XRT; not captured in latency path |
| Sharing/import | `D3DKMTShareObjects`, `D3DKMTQueryResourceInfo*`, `D3DKMTOpenResource*` | PRIME/export/import equivalents | Hooked but not seen in latency path |

Mapping to `native.h`:

* `native_device_create`: adapter discovery, `CreateDevice`, `CreatePagingQueue`.
* `native_device_alloc_buffer`: `CreateAllocation2`, `Lock2`,
  `MapGpuVirtualAddress`, `MakeResident`, then wait on the paging queue sync
  object when residency is pending.
* `native_device_create_context`: build MCDM context private blob, call
  `CreateContextVirtual`, then `CreateHwQueue`.
* `native_context_open_cu`: resolve kernel name/CU index from AXLF/IP-layout
  metadata. Windows may not need a separate KMT call after context creation.
* `native_command_create`: create a command allocation and prepare the ERT or
  command-chain BO contents.
* `native_command_bind_buffer`: retain the KMT allocation handles needed by the
  submit-private packet and by driver residency tracking.
* `native_queue_submit_and_wait`: `SubmitCommandToHwQueue`, then CPU wait on
  the HW queue progress fence returned by `CreateHwQueue`.

The runtime must not depend on XRT. If Windows MCDM requires AXLF instead of
raw PDI, the clean options are:

1. Extend the compiler/runtime executable payload to carry an AXLF alongside
   the existing `amdaie-pdi-fb` data.
2. Add a small in-tree AXLF builder for the subset of sections Windows MCDM
   needs, using the same section content generated by `XCLBinGen.cpp`.
3. Keep using `iree-aie-xclbinutil` only at compile time, not in the runtime
   shim.

## Proposed implementation shape

Add a Windows native implementation next to the existing Linux native layer:

```text
runtime/src/iree-amd-aie/driver/amdxdna/
  native_windows_mcdm.cc
  shim/windows/mcdm/
    kmt_api.h
    kmt_api.cc
    adapter.h
    adapter.cc
    allocation.h
    allocation.cc
    hw_context.h
    hw_context.cc
    hw_queue.h
    hw_queue.cc
    private_packets.h
```

Keep the IREE-facing API in `native.h` unchanged:

* `iree_hal_amdxdna_native_device_create`
* `iree_hal_amdxdna_native_device_alloc_buffer`
* `iree_hal_amdxdna_native_device_create_context`
* `iree_hal_amdxdna_native_context_open_cu`
* `iree_hal_amdxdna_native_queue_submit_and_wait`

The Windows implementation should initially provide:

1. Adapter discovery using `D3DKMTEnumAdapters3` and optionally DXCore.
2. KMT device/context/hwqueue lifetime management.
3. A BO path modeled on `D3DKMTCreateAllocation2`, `D3DKMTLock2`,
   `D3DKMTMapGpuVirtualAddress`, and `D3DKMTMakeResident`.
4. A context path that can construct the captured xclbin/PDI-style private blob.
5. A submit path modeled on `D3DKMTSubmitCommandToHwQueue` and hardware queue
   progress fence polling.

## Remaining packets to recover

The remaining reverse-engineering target is the schema of the AMD-private data
passed through:

* `D3DKMTCreateAllocation2` for every BO type, not just the latency command BO.
* `D3DKMTCreateContextVirtual` xclbin/PDI metadata construction for real IREE
  artifacts.
* `D3DKMTSubmitCommandToHwQueue` for non-trivial ERT/AIE command streams.
* `D3DKMTEscape` queries for firmware/AIE metadata, partition information, and
  power mode.
* Handle sharing/opening calls that explain allocation handle aliases.

Likely semantic fields, based on Linux XDNA and XRT strings:

* BO type: arg, cmd, instr, status, carveout, userptr/imported
* BO size and alignment
* returned GPU/device virtual address
* AIE partition and column mask
* QoS/priority
* xclbin UUID/PDI metadata
* ERT opcode and payload size
* argument BO handle list
* progress fence value

## Practical next steps

1. Land a minimal Windows KMT wrapper and adapter probe in-tree.
2. Add CMake gating so `amdxdna` can build on Windows with the MCDM shim.
3. Implement discovery, create device, create hwqueue, and clean teardown.
4. Extend the local call-capture harness to hook share/open-resource and sync
   object APIs.
5. Map the captured private packets back to Linux XDNA semantics and implement
   BO/context/submit one operation at a time.

Avoid implementing a replacement kernel driver. The existing AMD `ipustack.sys`
already owns firmware boot, scheduling, power, memory isolation, and Windows
device integration. The right target is an XRT-free user-mode KMT/MCDM shim.

## 2026-06-03 full-access update

This section supersedes the older uncertainty above where it conflicts with the
newer captures. The investigation now has enough confidence for an in-tree
Windows MCDM shim prototype for discovery, BO allocation, mapping, residency,
locking, and cache sync. The remaining high-risk gap is real workload submit
with argument binding, not basic driver access.

Current driver state after reboot/update:

```text
OS release             : 26200
XRT version/hash       : 2.19.0 / 77c7088d804602a53c3eb489b9cb37b709bcd751
XRT build date         : 2025-10-10 17:23:25
NPU driver version     : 32.0.203.314
NPU firmware version   : 1.0.21.43
Driver package         : C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_b3e90d6455884a5f
Validation xclbin      : validate_17f0_00.xclbin
```

The previous `32.0.203.280` / `1.0.18.3` notes remain useful for comparison,
but packet offsets are driver-version-sensitive. For example, the old context
blob had its AXLF at `0xe0` and process id at `0x58`; the current driver uses
AXLF offset `0xe8` and process id offset `0x60`.

### Tooling updates

The local capture harness now hooks `D3DKMTInvalidateCache` and records:

```text
hDevice
hAllocation
Offset
Length
return NTSTATUS
```

It also resolves generic hooks for the surfaced but not-yet-observed
`D3DKMTGetDeviceState`, `D3DKMTEvict`, and
`D3DKMTSetContextSchedulingPriority` calls so future traces will not miss them.

Updated tooling and evidence:

```text
scripts\kmt_capture.cpp
scripts\kmt_capture.exe
scripts\mcdm_alloc_probe.cpp
scripts\mcdm_alloc_probe.exe
scripts\mcdm_replay.cpp
scripts\mcdm_replay.exe
kmt-capture-mcdm-alloc-host-invalidate.jsonl
kmt-capture-mcdm-alloc-cacheable-invalidate.jsonl
kmt-capture-mcdm-alloc-execbuf-invalidate.jsonl
```

The direct allocation probe uses no XRT APIs. It loads KMT entry points from
`win32u.dll`/`gdi32.dll`/`dxcore.dll`, selects the NPU adapter with
`D3DKMTEnumAdapters3` + `KMTQAITYPE_DRIVER_DESCRIPTION`, creates a device and
paging queue, creates one AMD-private allocation, maps it to a GPU VA, makes it
resident, locks it to CPU memory, invalidates the cache range, and tears down.

The replay harness was also updated so dry-run parsing works for both the old
latency capture and the current C++ context capture:

```text
current context capture : escape=0   context=49904 allocation=56 submit=104
older latency capture   : escape=112 context=49896 allocation=56 submit=96
```

Live replay remains gated behind `--execute-risky` and explicit stages. The
current update was parsing-only; no live context replay or command submit was
run as part of this pass.

### Buffer allocation packet

The AMD-private `D3DKMTCreateAllocation2` packet for regular device/user BOs is
now mapped. It is 56 bytes:

```c
struct mcdm_alloc_private_56 {
  uint64_t reserved0;       // 0
  uint64_t requested_size;  // page-aligned BO size
  uint64_t aligned_size;    // page-aligned BO size
  uint32_t reserved1;       // 0
  uint32_t private_type;    // 0x3323/0x3328/0x3329/0x332c
  uint32_t policy;          // 2 for regular BOs
  uint32_t reserved2;       // 0
  uint32_t xcl_flags;       // XCL_BO_FLAGS_*
  uint32_t reserved3;       // 0
  uint64_t reserved4;       // 0
};
```

Confirmed private type mapping:

| IREE/Linux meaning | XRT flag | Windows private type | Direct KMT status |
| --- | --- | --- | --- |
| host/shared BO | `XCL_BO_FLAGS_HOST_ONLY` (`0x20000000`) | `0x3329` | create/map/resident/lock/invalidate OK |
| cacheable/device BO | `XCL_BO_FLAGS_CACHEABLE` (`0x01000000`) | `0x3323` | create/map/resident/lock/invalidate OK |
| command/exec BO | `XCL_BO_FLAGS_EXECBUF` (`0x80000000`) | `0x3328` | create/map/resident/lock/invalidate OK |
| carveout/kernel buffer | `XCL_BO_FLAGS_KERNBUF` (`0x02000000`) | `0x332c` | type recovered statically, not yet needed |
| AIE4 context command BO | context-internal | `0x332b` | captured in context path |

The mapping is independently supported by static disassembly of installed
`xrt_core.dll`: the high XRT flag byte branches to constants `0x3323`, `0x3328`,
`0x3329`, and `0x332c`; a separate AIE4 context path hardcodes `0x332b`.

Observed regular BO flow:

```text
D3DKMTCreateAllocation2(flags=0, one allocation, private_size=56)
D3DKMTMapGpuVirtualAddress -> STATUS_PENDING, paging fence
D3DKMTMakeResident         -> STATUS_PENDING, paging fence
D3DKMTLock2                -> CPU pointer
D3DKMTInvalidateCache      -> STATUS_SUCCESS
D3DKMTUnlock2
D3DKMTDestroyAllocation2
```

`D3DKMTInvalidateCache` succeeds on host, cacheable, and execbuf BOs. This is
the Windows analog of Linux `DRM_IOCTL_AMDXDNA_SYNC_BO`; the Linux driver
explicitly documents that the NPU is not cache coherent. The KMT call does not
carry a direction field, so `host_to_device` and `device_to_host` can initially
map to the same invalidation call, with direction preserved only at the IREE API
boundary in case a later driver-private path is found.

Observed GPU VA placement from fresh direct KMT probes:

```text
host_only  : gpu_va=0x00010000
execbuf    : gpu_va=0x00010000
cacheable  : gpu_va=0x04000000
```

The cacheable address matches the AXLF memory topology base used by the shipped
validation xclbin (`0x04000000`). Treat returned GPU VAs as driver-assigned
values, not constants.

The XRT C++ `xrt::bo(device, ..., XCL_BO_FLAGS_CACHEABLE)` path crashed its
child process before `CreateAllocation2`, both before and after xclbin
registration. Direct KMT cacheable allocation succeeds, so this is evidence
against a driver limitation and in favor of a bug/assumption in that XRT C++
probe path.

### Current context packet

Current-driver `D3DKMTCreateContextVirtual` capture:

```text
hDevice                 : XRT-created KMT device
NodeOrdinal             : 0
EngineAffinity          : 1
ClientHint              : D3DKMT_CLIENTHINT_VITIS (25)
Flags                   : HwQueueSupported (0x10)
PrivateDriverDataSize   : 49904
Embedded AXLF offset    : 0xe8
Creator pid offset      : 0x60
```

Only two bytes changed on successful return:

```text
offset 0x0040 : 00 -> 09
offset 0xc2e8 : 04 -> 08
```

Current prefix qwords:

```text
q[0]  @0x0000 = 0x3db88bbc4782e00b
q[1]  @0x0008 = 0x7fd70760cdd80999
q[8]  @0x0040 = 0
q[9]  @0x0048 = 0x04000000
q[10] @0x0050 = 0x48
q[11] @0x0058 = 0xc270
q[12] @0x0060 = current pid
q[16] @0x0080 = 1
q[25] @0x00c8 = 0x1000
q[26] @0x00d0 = 0xbcd8
q[27] @0x00d8 = 0xc1b8
q[28] @0x00e0 = 0xc208
```

The embedded AXLF is an ordinary `xclbin2` container. For the shipped validation
xclbin:

```text
platform_vbnv = xilinx_v1_ipu_0_0
uuid          = 0be08247bc8bb83d9909d8cd6007d77f
version       = 2.11.598
mode          = 4
action_mask   = 0x1
sections      = 11
```

Critical sections:

```text
MEM_TOPOLOGY:
  HOST size=0x10000 base=0x04000000
  SRAM size=0x0c000 base=0x04000000

AIE_PARTITION:
  column_width=4
  start_columns=[0]
  pdi_count=1
  pdi[0].image_size=432
  cdo[0].name=DPU_PDI_0
  cdo[0].type=3
  cdo[0].pdi_id=0xf0
  cdo[0].dpu_kernel_ids=[0x100]

IP_LAYOUT:
  two IP_MB entries; second is DPU_PDI_0:IPUV1CNN

CONNECTIVITY:
  9 entries; arg 5 maps to SRAM, the rest observed args map to HOST
```

Implementation implication: Windows context creation wants AXLF/xclbin metadata,
not just raw PDI bytes. The least fragile runtime design is to make the
executable payload carry an AXLF or enough section data to build one in-tree.
Using `iree-aie-xclbinutil` at compile time is fine; linking XRT at runtime is
not needed.

### Context-internal command allocation and submit

The current context setup path creates one special allocation:

```text
D3DKMTCreateAllocation2:
  flags              = CreateResource | CreateShared (0x3)
  private type       = 0x332b
  size               = 0x1000
  returned resource  = nonzero
  returned share     = nonzero
```

The context submit uses `D3DKMTSubmitCommandToHwQueue` with a 104-byte private
packet on the current driver:

```text
q[0]  = 2
q[1]  = allocation handle used by MapGpuVirtualAddress/MakeResident
q[2]  = command GPU VA (0x04000000 in the validation capture)
q[3..12] = 0
```

This differs from the older 96-byte submit private packet and is another reason
to keep packet builders version-aware and assert sizes from captures/probes.

Completion flow:

```text
MapGpuVirtualAddress -> pending paging fence
MakeResident         -> pending paging fence
WaitFromGpu          -> wait on paging queue sync object/fence
SubmitCommandToHwQueue(fence id 1)
WaitFromCpu          -> wait on HW queue progress fence value 2
```

The locked command BO was zero-filled in the validation context setup capture,
so this path is best treated as driver/context bootstrap, not proof of an IREE
ERT workload submit. For IREE, the runtime should create an execbuf BO, write
the same ERT packets used by Linux (`ERT_START_CU`, `ERT_START_NPU`,
`ERT_CMD_CHAIN`), make the command and argument BOs resident, submit command VA
and allocation handle through the HW queue packet, and wait on the progress
fence. The no-op/bootstrap submit is now directly replayed without XRT; a tiny
real IREE dispatch is still the next workload gate.

### Linux XDNA semantic crosswalk

The local `xdna-driver` clone at `b3a2865` confirms the semantic API surface:

```text
CREATE_HWCTX  -> Windows CreateContextVirtual + CreateHwQueue
DESTROY_HWCTX -> DestroyHwQueue + DestroyContext
CONFIG_HWCTX  -> likely encoded into AXLF/context creation on Windows
CREATE_BO     -> CreateAllocation2
GET_BO_INFO   -> returned KMT allocation handle + returned GPU VA + CPU lock ptr
SYNC_BO       -> D3DKMTInvalidateCache
EXEC_CMD      -> SubmitCommandToHwQueue
WAIT_CMD      -> WaitForSynchronizationObjectFromCpu on progress fence
GET_INFO      -> D3DKMTEscape / QueryAdapterInfo diagnostics
GET_ARRAY     -> D3DKMTEscape diagnostics for contexts, BO usage, logs, coredumps
SET_STATE     -> D3DKMTEscape diagnostics/admin state
```

Linux `EXEC_CMD` currently supports one command BO and an optional array of arg
BO handles. That arg list is used by the kernel for object lookup/residency and
job tracking. Windows has no captured separate arg-list KMT field yet. The
working hypothesis is that explicit KMT residency plus device addresses inside
the ERT packet are sufficient, while the submit private packet identifies the
command allocation and command VA. This is the main runtime validation gap.

Linux BO details also explain the cache discipline:

```text
AMDXDNA_BO_SHARE / SHMEM -> host-visible shared memory
AMDXDNA_BO_DEV           -> device BO backed by heap ranges
AMDXDNA_BO_CMD           -> user + driver accessible command BO
SYNC_BO                  -> CPU cache flush/invalidate because NPU is not coherent
```

### Windows API surface status

For an IREE runtime shim, the mapped surface is:

| Area | Calls | Status for shim |
| --- | --- | --- |
| Adapter/device | `EnumAdapters3`, `QueryAdapterInfo`, `CloseAdapter`, `CreateDevice`, `DestroyDevice` | Implement now |
| Paging queue | `CreatePagingQueue`, `DestroyPagingQueue` | Implement now |
| BO lifecycle | `CreateAllocation2`, `DestroyAllocation2`, `Lock2`, `Unlock2`, `MapGpuVirtualAddress`, `FreeGpuVirtualAddress`, `MakeResident` | Implement now |
| Cache sync | `InvalidateCache` | Implement now |
| Context | `CreateContextVirtual`, `DestroyContext` | Implement once AXLF packaging is wired |
| HW queue | `CreateHwQueue`, `DestroyHwQueue` | Implement with context |
| Submit/wait | `SubmitCommandToHwQueue`, `WaitForSynchronizationObjectFromCpu`, `WaitForSynchronizationObjectFromGpu` | Implement initial synchronous path; real IREE dispatch still needs a workload gate |
| Queue dependency waits/signals | `SubmitWaitForSyncObjectsToHwQueue`, `SubmitSignalSyncObjectsToHwQueue` | Optional; not needed for synchronous `submit_and_wait` |
| Explicit sync-object import/export | `CreateSynchronizationObject2`, `DestroySynchronizationObject`, `OpenSyncObjectFromNtHandle2` | Optional unless HAL async interop needs it |
| Resource sharing/import | `ShareObjects`, `QueryResourceInfo*`, `OpenResource*` | Optional; not observed in local execution path |
| Diagnostics/admin | `Escape`, admin ETW, power/query state | Optional; useful for device info and debugging |

### Implementation plan

Use the existing native boundary in `native.h`; do not add XRT to the runtime
dependency graph.

Suggested file layout:

```text
runtime/src/iree-amd-aie/driver/amdxdna/
  native_windows_mcdm.cc
  shim/windows/mcdm/
    kmt_api.h
    kmt_api.cc
    adapter.h
    adapter.cc
    allocation.h
    allocation.cc
    context.h
    context.cc
    queue.h
    queue.cc
    private_packets.h
```

Implementation stages:

1. Add `kmt_api` dynamic loading for the public KMT functions from
   `win32u.dll`/`gdi32.dll`/`dxcore.dll`; no XRT includes or libs.
2. Implement adapter/device/paging queue creation and teardown.
3. Implement `native_device_alloc_buffer`, `native_buffer_map`,
   `native_buffer_sync`, `native_buffer_device_address`, and buffer teardown
   using the proven 56-byte allocation packet.
4. Add unit/probe tests that allocate host/cacheable/execbuf BOs, map, sync,
   and destroy without creating a context.
5. Change executable packaging so Windows receives AXLF/xclbin metadata instead
   of only raw PDI. Prefer compile-time AXLF generation over runtime XRT use.
6. Implement `native_device_create_context` by building the current
   `CreateContextVirtual` private prefix around the AXLF and creating a HW
   queue.
7. Implement command exec BO creation and ERT packet filling by reusing the
   Linux native command logic as closely as possible.
8. Implement synchronous `native_queue_submit_and_wait` with explicit
   make-resident/wait, `SubmitCommandToHwQueue`, and CPU wait on the HW queue
   progress fence.
9. Validate with the direct no-op/bootstrap submit first, then a tiny IREE
   dispatch, then command-chain dispatch.

Do not implement a custom Windows kernel driver for this. `ipustack.sys` already
owns firmware boot, scheduling, power management, memory isolation, and Windows
MCDM integration. Replacing it would require driver signing and firmware
protocol ownership, and would add far more risk than an XRT-free KMT user-mode
shim.

## 2026-06-03 in-tree MCDM probe

Added a first Windows-only in-tree shim/probe:

```text
runtime/src/iree-amd-aie/driver/amdxdna/shim/windows/mcdm/
  CMakeLists.txt
  kmt_api.h
  kmt_api.cc
  mcdm_probe_main.cc
```

The top-level `amdxdna/CMakeLists.txt` now allows this Windows probe subdir to
build while still returning before registering the Linux-only HAL backend on
Windows.

Manual MSVC build command used:

```text
cl /std:c++17 /EHsc /W4 /O2 kmt_api.cc mcdm_probe_main.cc /link /out:scripts\mcdm_probe.exe
```

Stage results:

```text
scripts\mcdm_probe.exe --stage=discover
  kmt.load=ok
  adapter.handle=1073741888 desc="NPU Compute Accelerator Device"

scripts\mcdm_probe.exe --stage=device
  kmt.load=ok
  adapter.handle=1073741888 desc="NPU Compute Accelerator Device"
  device.handle=1073742016
  paging_queue=1073742144
  paging_sync=1073742080

scripts\mcdm_probe.exe --stage=all-bos --size=4096
  host_only  private_type=0x3329 gpu_va=0x00010000 sync=ok destroy=ok
  cacheable  private_type=0x3323 gpu_va=0x04000000 sync=ok destroy=ok
  execbuf    private_type=0x3328 gpu_va=0x00010000 sync=ok destroy=ok
```

This confirms the repo-local Windows MCDM layer can directly talk to the driver
without XRT for:

```text
EnumAdapters3
QueryAdapterInfo
CreateDevice / DestroyDevice
CreatePagingQueue / DestroyPagingQueue
CreateAllocation2 / DestroyAllocation2
MapGpuVirtualAddress / FreeGpuVirtualAddress
MakeResident
Lock2 / Unlock2
InvalidateCache
```

No context creation, hardware queue creation, or submit was run in this in-tree
BO probe. The staged replay below exercises those gates.

### Context, HW queue, and submit replay gate

After the in-tree BO probe passed, the current-driver context capture was replayed
through the staged `scripts\mcdm_replay.exe` harness. This uses the captured
`CreateContextVirtual` private blob and patches the captured PID to the current
process PID.

Successful context-only gate:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=context
  loaded escape=0 context=49904 allocation=56 submit=104
  context_pid_patches=1
  D3DKMTCreateDevice status=0x0
  D3DKMTCreatePagingQueue status=0x0
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTDestroyContext status=0x0
  teardown status=0x0
```

Successful HW queue gate:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=queue
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTCreateHwQueue status=0x0
  hwqueue=1073742336 progress_fence=1073742272 fence_gpu=0x10040
  teardown status=0x0
```

The first alias replay confirmed that adjacent XRT handles must not be treated
as arithmetic API handles:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=resident plus40 plus40
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTCreateHwQueue status=0x0
  D3DKMTCreateAllocation2 status=0x0
  D3DKMTLock2 status=0x0
  D3DKMTMapGpuVirtualAddress status=0xc000000d
```

The working path is simpler and does not need the XRT adjacent handles:

* Use the visible `D3DDDI_ALLOCATIONINFO2::hAllocation` returned by
  `D3DKMTCreateAllocation2` for `Lock2`, `MapGpuVirtualAddress`,
  `MakeResident`, and submit private `q[1]`.
* Set `D3DDDI_MAPGPUVIRTUALADDRESS::BaseAddress = 0x04000000` for the
  context-internal `0x332b` command aperture. With base zero, the driver maps
  the same allocation at a different VA, which does not match the captured
  submit packet/context metadata.
* Tear down the special resource through
  `D3DKMT_DESTROYALLOCATION2::hResource` with zero allocation count. Destroying
  it by the visible allocation-list handle returns `STATUS_INVALID_PARAMETER`.

Successful fixed-base map gate:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=map --map-base=0x4000000 created resource
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTCreateHwQueue status=0x0
  D3DKMTCreateAllocation2 status=0x0
  D3DKMTLock2 status=0x0
  D3DKMTMapGpuVirtualAddress status=0x103
  gpu_va=0x4000000 paging_fence=0x1b59
  D3DKMTDestroyAllocation2(resource) status=0x0
  teardown status=0x0
```

Successful fixed-base resident gate:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=resident --map-base=0x4000000 created resource
  D3DKMTMapGpuVirtualAddress status=0x103
  gpu_va=0x4000000 paging_fence=0x1b59
  D3DKMTMakeResident status=0x103
  resident_fence=0x1b5a
  D3DKMTWaitForSynchronizationObjectFromGpu status=0x0
  D3DKMTDestroyAllocation2(resource) status=0x0
  teardown status=0x0
```

Successful direct no-op/bootstrap submit gate:

```text
scripts\mcdm_replay.exe kmt-capture-xrt-cpp-context-validate17f0.jsonl --execute-risky --stage=submit --allow-submit --map-base=0x4000000 created resource
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTCreateHwQueue status=0x0
  D3DKMTCreateAllocation2 status=0x0
  D3DKMTLock2 status=0x0
  D3DKMTMapGpuVirtualAddress status=0x103
  gpu_va=0x4000000 paging_fence=0x1b59
  D3DKMTMakeResident status=0x103
  D3DKMTWaitForSynchronizationObjectFromGpu status=0x0
  D3DKMTSubmitCommandToHwQueue status=0x0
  D3DKMTWaitForSynchronizationObjectFromCpu status=0x0
  D3DKMTFreeGpuVirtualAddress status=0x0
  D3DKMTUnlock2 status=0x0
  D3DKMTDestroyAllocation2(resource) status=0x0
  teardown status=0x0
```

This means the driver-facing Windows API surface for context bootstrap is now
directly exercised without XRT. The remaining runtime work is not discovering
more KMT calls; it is constructing the `CreateContextVirtual` AXLF wrapper and
real ERT command payloads from IREE artifacts.

A post-submit health check with `scripts\mcdm_probe.exe --stage=all-bos
--size=4096` still passed for host-only, cacheable, and execbuf BOs, including
sync and destroy, so the direct no-op/bootstrap submit did not leave the local
driver path in an obviously bad state.

### In-tree xclbin context builder and submit gate

The replay-only context gate is now superseded by in-tree shim code that builds
the current-driver `D3DKMTCreateContextVirtual` private blob directly from an
AXLF/xclbin.

Added:

```text
runtime/src/iree-amd-aie/driver/amdxdna/shim/windows/mcdm/context_blob.h
runtime/src/iree-amd-aie/driver/amdxdna/shim/windows/mcdm/context_blob.cc
```

The builder currently supports the validated one-PDI xclbin shape:

* `xclbin2` AXLF section table parsing.
* `BUILD_METADATA` first user-region kernel name extraction.
* `AIE_PARTITION` column width, start column, CDO/PDI name, and DPU kernel id
  extraction.
* Current-driver context prefix with AXLF at offset `0xe8` and process id at
  offset `0x60`.
* The observed `0x530` tail records for kernel and PDI/DPU metadata.

The Python reference builder:

```text
scripts\mcdm_context_builder.py
```

now byte-matches the captured XRT context blob for:

```text
C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_b3e90d6455884a5f\validate_17f0_00.xclbin
```

Result:

```text
context_size=49904 xclbin_size=48344 pid=28640
compare_equal=True
```

The synthetic replay capture generated from that builder also submitted cleanly
without XRT:

```text
scripts\mcdm_replay.exe kmt-capture-synth-context-validate17f0.jsonl --execute-risky --stage=submit --allow-submit --map-base=0x4000000 created resource
  D3DKMTCreateContextVirtual status=0x0
  D3DKMTCreateHwQueue status=0x0
  D3DKMTCreateAllocation2 status=0x0
  D3DKMTMapGpuVirtualAddress status=0x103
  D3DKMTMakeResident status=0x103
  D3DKMTSubmitCommandToHwQueue status=0x0
  D3DKMTWaitForSynchronizationObjectFromCpu status=0x0
  teardown status=0x0
```

The in-tree `mcdm_probe` now resolves and wraps:

```text
D3DKMTCreateContextVirtual / D3DKMTDestroyContext
D3DKMTCreateHwQueue / D3DKMTDestroyHwQueue
D3DKMTWaitForSynchronizationObjectFromGpu
D3DKMTSubmitCommandToHwQueue
D3DKMTWaitForSynchronizationObjectFromCpu
```

It also implements the special current-driver command aperture path:

* `D3DKMTCreateAllocation2` with private type `0x332b`,
  `CreateResource | CreateShared`, and a `0x1000` allocation packet.
* CPU lock of the visible returned allocation handle.
* Fixed GPU VA mapping at `0x04000000` with `SizeInPages = 0x4000`.
* `D3DKMTMakeResident`, followed by a GPU wait on the paging fence.
* `D3DKMTSubmitCommandToHwQueue` with a 104-byte private packet:
  `q[0]=2`, `q[1]=allocation`, `q[2]=gpu_va`.
* CPU wait on the HW queue progress fence value `submit_fence_id + 1`.
* Resource teardown through `D3DKMT_DESTROYALLOCATION2::hResource`.

Successful in-tree context creation from the shipped validation xclbin:

```text
scripts\mcdm_probe.exe --stage=context --xclbin=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_b3e90d6455884a5f\validate_17f0_00.xclbin
  context.private_size=49904 xclbin_size=48344 kernel="vadd" pdi="DPU_PDI_0"
  context.handle=1073742208 hwqueue=1073742336 progress_fence=1073742272
  context.destroy=ok
```

Successful in-tree bootstrap submit from the same xclbin:

```text
scripts\mcdm_probe.exe --stage=submit --xclbin=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_b3e90d6455884a5f\validate_17f0_00.xclbin
  context.private_size=49904 xclbin_size=48344 kernel="vadd" pdi="DPU_PDI_0"
  command_aperture.allocation=1073742464 resource=1073742400 gpu_va=0x4000000
  submit.wait=ok
```

A post-submit health check still passed:

```text
scripts\mcdm_probe.exe --stage=all-bos --size=4096
  host_only sync=ok destroy=ok
  cacheable sync=ok destroy=ok
  execbuf sync=ok destroy=ok
```

This moves the implementation from "replay XRT private bytes" to "derive the
driver private context bytes from an xclbin in IREE-owned code". The remaining
work for a real IREE dispatch is now:

1. Make the Windows `amdxdna` executable payload carry AXLF/xclbin bytes, or
   enough metadata and PDI data to build the same AXLF wrapper at runtime.
2. Add multi-PDI/tile-partition support to the context builder before targeting
   overlay xclbins or wider partitions.
3. Port the Linux KMQ ERT packet construction into the Windows native path:
   `ERT_START_CU`, `ERT_START_NPU`, and `ERT_CMD_CHAIN`.
4. Make command BOs and argument BOs resident before submit, then validate
   whether explicit Windows residency is sufficient in place of Linux
   `EXEC_CMD`'s arg BO handle array.
5. Validate a tiny real IREE dispatch, then the command-chain path.

### Payload packaging conclusion

The local public XRT checkout under `C:\Users\jornt\workspace\iree-ai\XRT`
does not contain the shipped Windows MCDM UMD implementation that appears in the
installed `xrt_core.dll` (`hwcontext_aie4`, `hwqueue_aie4`, MCDM BO classes,
and the KMT wrappers are not present in that source tree). It is still useful
for common ERT packet layout and high-level XRT kernel behavior, but the
Windows-specific KMT packet surface remains based on local captures plus direct
probes.

The compiler/runtime payload situation is now clear:

* `--iree-amdaie-device-hal=xrt` already emits `amdaie-xclbin-fb`, containing
  the AXLF/xclbin bytes needed by Windows MCDM context creation.
* `--iree-amdaie-device-hal=amdxdna` emits `amdaie-pdi-fb`, containing raw PDI
  bytes plus the amdxdna-only host patch table used by the cmd-chain path.
* `aie2xclbin` currently branches: the AMDXDNA path copies `design.pdi` to the
  artifact path and returns; the XRT path continues into `generateXCLBin`.
  It does not emit both artifacts in one pass today.

The best packaging change for Windows MCDM is therefore not to make the
amdxdna runtime consume XRT's flatbuffer wholesale. Instead, extend the
existing `amdaie-pdi-fb` schema with an optional AXLF/xclbin payload, and add a
compiler option or new Windows-MCDM HAL mode that asks `aie2xclbin` to keep
building the PDI/runlists/patch tables but also continue through
`generateXCLBin` and store the resulting xclbin bytes. That preserves:

* Linux amdxdna behavior and tests: existing PDI path remains valid.
* Windows context creation: the shim receives real AXLF bytes for
  `BuildContextPrivateDataFromXclbin`.
* Windows command-chain support: the amdxdna-specific patch table remains
  available, which the XRT flatbuffer format does not carry.

An interim bring-up path can compile with the XRT HAL mode only to get
`amdaie-xclbin-fb` and then manually feed the xclbin to `mcdm_probe`, but a
production Windows MCDM HAL should keep the amdxdna executable format and add
the optional AXLF field.

### Unified amdxdna path for one-PDI and command-chain workloads

Linux command chaining already works through one amdxdna HAL path: executable
side-channel metadata carries PDI/runlists/patch tables, the command buffer
accumulates dispatches into groups keyed by native context/queue, and the
native backend submits Linux-equivalent ERT packets.

Windows should keep that shape. The driver-specific difference belongs below
the native boundary:

```text
IREE amdxdna executable
  PDI bytes
  optional AXLF/xclbin context wrapper
  runlists
  patch tables
        |
        v
IREE amdxdna command buffer
  same ERT_START_CU / ERT_START_NPU / ERT_CMD_CHAIN construction
        |
        v
native backend
  Linux  : create context from raw PDI
  Windows: create context from AXLF/xclbin wrapper
```

This avoids two HAL paths. The command-buffer logic should not care whether the
context was created from a PDI or an AXLF wrapper.

For one-PDI-per-dispatch compilation, each PDI can be wrapped in a generated
single-PDI xclbin and attached to the same `amdaie-pdi-fb` entry. The runtime
can cache contexts by `(PDI bytes, optional xclbin bytes, kernel name)` so the
xclbin detail is hidden from users and repeated dispatches do not need to
recreate contexts.

For command chaining across multiple PDI-bearing dispatches, the same rule as
Linux applies: a single `ERT_CMD_CHAIN` is submitted to one native
context/queue. Therefore:

* If all chained dispatches resolve to the same context, one chain is valid.
  On Windows that may require those dispatches to point at the same multi-PDI
  xclbin wrapper.
* If dispatches resolve to different contexts, the existing grouping behavior
  should split them into multiple chain submissions in recorded order.
* A "multi-xclbin chain" is not the right primitive. Multiple xclbins imply
  multiple contexts. Use one multi-PDI xclbin for one shared context, or split
  the chain by context.

Implementation plumbing started:

```text
runtime/src/iree-amd-aie/schemas/pdi_executable_def.fbs
  appended optional xclbin_indices/xclbins fields

runtime/src/iree-amd-aie/driver/amdxdna/executable_internal.h
  kernel params now carry optional xclbin bytes

runtime/src/iree-amd-aie/driver/amdxdna/native.h
  native create_context now receives PDI + optional xclbin

runtime/src/iree-amd-aie/driver/amdxdna/device.cc
  context cache key includes PDI, optional xclbin, and kernel name

runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc
  dispatch passes both spans through one unified context path
```

The compiler still needs to fill the optional xclbin fields. The likely
compiler change is to extend the AMDXDNA packaging mode so it keeps generating
the raw PDI/runlists/patch tables and also continues through `generateXCLBin`
to produce an AXLF wrapper. For multi-PDI command-chain workloads, that wrapper
should be shared by the entry points that must chain in one Windows context.
