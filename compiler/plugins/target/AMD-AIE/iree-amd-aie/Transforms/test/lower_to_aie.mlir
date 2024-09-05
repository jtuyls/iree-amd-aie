// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-lower-to-aie)" --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
module {
}

// -----

// CHECK: module
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: aiex.runtime_sequence @empty_func
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @empty_func() {
    return
  }
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: aiex.runtime_sequence @workgroup
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @workgroup() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aiex.runtime_sequence @hal_bindings
// CHECK-SAME:  %{{.+}}: memref<32x1024xi32>
// CHECK-SAME:  %{{.+}}: memref<1024x64xi32>
// CHECK-SAME:  %{{.+}}: memref<32x64xi32>
// CHECK-NOT:   memref.assume_alignment
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hal_bindings() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether aie.objectfifo is linked correctly,
// this test checks two `amdaie.connection` operations, so they can be linked
// correctly.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       aiex.runtime_sequence @connections_and_link
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @connections_and_link() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%obj0, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.logicalobjectfifo.link[%conn0] -> [%conn1] ()
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether aie.objectfifo is linked correctly,
// this test checks two `amdaie.connection` operations, so they can be linked
// correctly.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]] toStream [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       aiex.runtime_sequence @circular_dma_cpy_sizes_and_strides
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_dma_cpy_sizes_and_strides() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%obj0, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.logicalobjectfifo.link[%conn0] -> [%conn1] ()
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [%c0, %c0, %c0] [%c32, %c4, %c8] [%c8, %c256, %c1])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Produce
// CHECK:         %[[ACCESS:.+]] = aie.objectfifo.subview.access %[[ACQUIRE]]
// CHECK:         %[[REINTERPRET:.+]] = memref.reinterpret_cast %[[ACCESS]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[REINTERPRET]] : memref<32x32xi32, 1>)
// CHECK:       aiex.runtime_sequence @tile_and_core_and_acquire
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tile_and_core_and_acquire() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%obj0, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%conn0]) {
        %0 = amdaie.logicalobjectfifo.acquire(%conn0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
        %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 1>> -> memref<1024xi32, 1>
        %2 = memref.reinterpret_cast %1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 1> to memref<32x32xi32, 1>
        linalg.fill ins(%c0_i32 : i32) outs(%2 : memref<32x32xi32, 1>)
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device
// CHECK-DAG:   func.func private @ukernel_A(memref<i32, 2>, index) attributes {llvm.bareptr = true}
// CHECK-DAG:   func.func private @ukernel_B(memref<i32, 2>, index, memref<f32, 2>, index) attributes {llvm.bareptr = true}
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Produce
// CHECK:         %[[ACCESS:.+]] = aie.objectfifo.subview.access %[[ACQUIRE]]
// CHECK:         %[[REINTERPRET:.+]] = memref.reinterpret_cast %[[ACCESS]]
// CHECK:         %[[ACQUIRE0:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Produce
// CHECK:         %[[ACCESS0:.+]] = aie.objectfifo.subview.access %[[ACQUIRE0]]
// CHECK:         %[[REINTERPRET0:.+]] = memref.reinterpret_cast %[[ACCESS0]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[REINTERPRET]] : memref<32x32xi32, 2>)
// CHECK:         %[[BASE_BUFFER:.*]], %{{.+}}, %{{.+}}:2, %{{.+}}:2 = memref.extract_strided_metadata %[[REINTERPRET]] :
// CHECK:         %[[BASE_BUFFER0:.*]], %{{.+}}, %{{.+}}:2, %{{.+}}:2 = memref.extract_strided_metadata %[[REINTERPRET0]] :
// CHECK:         func.call @ukernel_A(%[[BASE_BUFFER]], %[[C0]]) : (memref<i32, 2>, index) -> ()
// CHECK:         func.call @ukernel_B(%[[BASE_BUFFER]], %[[C0]], %[[BASE_BUFFER0]], %[[C0]]) : (memref<i32, 2>, index, memref<f32, 2>, index) -> ()
// CHECK:         aie.end
// CHECK:       } {link_with = "/path/to/ukernel.o"}
// CHECK:       aiex.runtime_sequence @lower_to_aie_ukernel
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func private @ukernel_A(memref<i32, 2>, index) attributes {link_with = "/path/to/ukernel.o", llvm.bareptr = true}
  func.func private @ukernel_B(memref<i32, 2>, index, memref<f32, 2>, index) attributes {link_with = "/path/to/ukernel.o", llvm.bareptr = true}
  func.func @lower_to_aie_ukernel() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 1>
      %alloc_3 = memref.alloc() : memref<32x32xf32, 2>
      %alloc_4 = memref.alloc() : memref<4x8x4x8xf32, 1>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_0_1} : memref<32x32xf32, 2> -> !amdaie.logicalobjectfifo<memref<1024xf32, 2>>
      %obj4 = amdaie.logicalobjectfifo.from_memref %alloc_4, {%tile_0_2} : memref<4x8x4x8xf32, 1> -> !amdaie.logicalobjectfifo<memref<1024xf32, 1>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %conn1 = amdaie.connection(%obj3, %obj4) : (!amdaie.logicalobjectfifo<memref<1024xf32, 2>>, !amdaie.logicalobjectfifo<memref<1024xf32, 1>>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%conn0, %conn1]) {
        %0 = amdaie.logicalobjectfifo.acquire(%conn0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
        %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        %reinterpret_0 = memref.reinterpret_cast %1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2> to memref<32x32xi32, 2>
        %2 = amdaie.logicalobjectfifo.acquire(%conn1, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xf32, 2>>
        %3 = amdaie.logicalobjectfifo.access(%2, Write) : !amdaie.logicalobjectfifo<memref<1024xf32, 2>> -> memref<1024xf32, 2>
        %reinterpret_1 = memref.reinterpret_cast %3 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xf32, 2> to memref<32x32xf32, 2>
        linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_0 : memref<32x32xi32, 2>)
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_0 : memref<32x32xi32, 2> -> memref<i32, 2>, index, index, index, index, index
        %base_buffer0, %offset0, %sizes0:2, %strides0:2 = memref.extract_strided_metadata %reinterpret_1 : memref<32x32xf32, 2> -> memref<f32, 2>, index, index, index, index, index
        func.call @ukernel_A(%base_buffer, %c0) : (memref<i32, 2>, index) -> ()
        func.call @ukernel_B(%base_buffer, %c0, %base_buffer0, %c0) : (memref<i32, 2>, index, memref<f32, 2>, index) -> ()
        amdaie.end
      } {link_with = "/path/to/ukernel.o"}
      memref.dealloc %alloc_4 : memref<4x8x4x8xf32, 1>
      memref.dealloc %alloc_3 : memref<32x32xf32, 2>
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 1>
      memref.dealloc %alloc_1 : memref<32x32xi32, 2>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----


// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE_0:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Consume
// CHECK:         aie.objectfifo.subview.access
// CHECK-SAME:    %[[ACQUIRE_0]]
// CHECK:       aie.core(%[[TILE_1_2]])
// CHECK:         %[[ACQUIRE_1:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Consume
// CHECK:         aie.objectfifo.subview.access
// CHECK-SAME:    %[[ACQUIRE_1]]
// CHECK:       aiex.runtime_sequence @tile_and_core_and_acquire_broadcast
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tile_and_core_and_acquire_broadcast() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2, %tile_1_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj0) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %conn1 = amdaie.connection(%obj2, %obj1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %core_0_2 = amdaie.core(%tile_0_2, in : [%conn1], out : []) {
        %0 = amdaie.logicalobjectfifo.acquire(%conn1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
        amdaie.end
      }
      %core_1_2 = amdaie.core(%tile_1_2, in : [%conn1], out : []) {
        %0 = amdaie.logicalobjectfifo.acquire(%conn1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         aie.objectfifo.release
// CHECK:       aiex.runtime_sequence @tile_and_core_and_release
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tile_and_core_and_release() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%obj0, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%conn0]) {
        amdaie.logicalobjectfifo.release(%conn0, Produce) {size = 1 : i32}
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @invalid_npu_dma_cpy_nd() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %2, 64 : memref<32x64xi32>
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%placeholder, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.logicalobjectfifo.link[%conn0] -> [%conn1] ()
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      // expected-error @+1 {{could not convert to AIEDialect ops}}
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        %obj0 = amdaie.logicalobjectfifo.from_memref %2, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        // expected-error @+1 {{'amdaie.npu.dma_cpy_nd' op must have a target BD ID op to lower to the AIE dialect}}
        %npu_dma_2 = amdaie.npu.dma_cpy_nd %conn1(%obj0[%c0, %c32] [%c32, %c32] [%c64, %c1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%npu_dma_2, S2MM)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device
// CHECK:       aiex.runtime_sequence @npu_dma_cpy_nd_with_repeat_already_on_outer_dim(%[[ARG0:.+]]: memref<32x64xi32>
// CHECK:       aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 32][2, 1, 2, 32][2, 0, 16, 1])
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_with_repeat_already_on_outer_dim() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %2, 64 : memref<32x64xi32>
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %bd_id_0 = amdaie.bd_id(%tile_0_0, 0)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<2048xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %conn0 = amdaie.connection(%obj1, %obj2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %conn1 = amdaie.connection(%placeholder, %obj1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.logicalobjectfifo.link[%conn0] -> [%conn1] ()
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      amdaie.controlcode {
        %npu_dma_0 = amdaie.npu.circular_dma_cpy_nd %conn0([] [] [], [] [] [])
        %npu_dma_1 = amdaie.npu.circular_dma_cpy_nd %conn1([] [] [], [] [] [])
        %obj0 = amdaie.logicalobjectfifo.from_memref %2, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        %npu_dma_2 = amdaie.npu.dma_cpy_nd %conn1(%obj0[0, 0, 0, 32] [2, 1, 2, 32] [2, 0, 16, 1] bd_id = %bd_id_0, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.end
      }
    }
    return
  }
}

// -----

// Test to show mix of implicit/explicit source/target addressing in amdaie.npu.dma_cpy_nd.

// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ2:.+]](%[[TILE_0_0]], {%[[TILE_0_1]]}
// CHECK:       aie.objectfifo.link [@[[OBJ0]]] -> [@[[OBJ1]]]
// CHECK:       aiex.runtime_sequence @controlcode(%[[ARG0:.+]]: memref<32x64xi32>
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:            %[[ARG0]][0, 0, 0, 32][1, 1, 32, 32][0, 0, 64, 1]
// CHECK-SAME:            issue_token = true
// CHECK-SAME:            metadata = @[[OBJ1]]
// CHECK-NEXT:    aiex.npu.dma_wait {symbol = @[[OBJ1]]}
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:            %[[ARG0]][0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:            issue_token = true
// CHECK-SAME:            metadata = @[[OBJ1]]
// CHECK-NEXT:    aiex.npu.dma_wait {symbol = @[[OBJ1]]}
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:            %[[ARG0]][0, 0, 0, 32][1, 1, 32, 32][0, 0, 64, 1]
// CHECK-SAME:            issue_token = true
// CHECK-SAME:            metadata = @[[OBJ2]]
// CHECK-NEXT:    aiex.npu.dma_wait {symbol = @[[OBJ2]]}
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:            %[[ARG0]][0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]
// CHECK-SAME:            issue_token = true
// CHECK-SAME:            metadata = @[[OBJ2]]
// CHECK-NEXT:    aiex.npu.dma_wait {symbol = @[[OBJ2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<2, storage_buffer>]>]>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @controlcode() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %0, 64 : memref<32x64xi32>
      %tile = amdaie.tile(%c0, %c0)
      %tile_0 = amdaie.tile(%c0, %c1)
      %tile_1 = amdaie.tile(%c0, %c2)
      %bd_id = amdaie.bd_id(%tile, 0)
      %alloc = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %1 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<2048xi32>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_1} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %4 = amdaie.connection(%2, %3) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %5 = amdaie.connection(%1, %2) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %6 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      amdaie.logicalobjectfifo.link[%4] -> [%5] ()
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc : memref<32x32xi32, 1>
      amdaie.controlcode {
        %7 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [] [] [])
        %8 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
        %9 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [] [] [])
        %10 = amdaie.logicalobjectfifo.from_memref %0, {%tile} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        %11 = amdaie.npu.dma_cpy_nd %5(%10[0, 0, 0, 32] [1, 1, 32, 32] [0, 0, 64, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%11, S2MM)
        %12 = amdaie.npu.dma_cpy_nd %5(%10[0, 0, 0, 0] [1, 1, 1, 1024] [0, 0, 0, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%12, S2MM)
        %13 = amdaie.npu.dma_cpy_nd %6([] [] [], %10[0, 0, 0, 32] [1, 1, 32, 32] [0, 0, 64, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%13, MM2S)
        %14 = amdaie.npu.dma_cpy_nd %6([] [] [], %10[0, 0, 0, 0] [1, 1, 1, 2048] [0, 0, 0, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%14, MM2S)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device(npu1_4col) {
// CHECK-DAG:     %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-DAG:     %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:     %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:         aie.objectfifo @[[OBJ0:.*]](%[[TILE_0_0]], {%[[TILE_0_1]]}, 2 : i32) : !aie.objectfifo<memref<1024xbf16, 1 : i32>>
// CHECK:         aie.objectfifo @[[OBJ1:.*]](%[[TILE_1_0]], {%[[TILE_0_1]]}, 2 : i32) : !aie.objectfifo<memref<1024xbf16, 1 : i32>>
// CHECK:         aie.objectfifo @[[OBJ2:.*]](%[[TILE_0_1]]
// CHECK-SAME:    {%[[TILE_1_0]]}, 2 : i32) : !aie.objectfifo<memref<1024xf32>>
// CHECK:         aiex.runtime_sequence @bf16_f32_lit_test
// CHECK-SAME:         (%[[LHS:.*]]: memref<32x32xbf16>, %[[RHS:.*]]: memref<32x32xbf16>, %[[OUT:.*]]: memref<32x32xf32>) {
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:          %[[OUT]][0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:          issue_token = true
// CHECK-SAME:          metadata = @[[OBJ2]]
// CHECK-SAME:          memref<32x32xf32>
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:          %[[RHS]][0, 0, 1, 2][1, 2, 32, 16][0, 16, 32, 1]
// CHECK-SAME:          metadata = @[[OBJ1]]
// CHECK-SAME:          memref<32x32xbf16>
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:          %[[LHS]][0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:          metadata = @[[OBJ0]]
// CHECK-SAME:          memref<32x32xbf16>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @bf16_f32_lit_test() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %alloc = memref.alloc() : memref<2x2x16x16xf32, 1 : i32>
      %alloc_0 = memref.alloc() : memref<1x2x32x16xbf16, 1 : i32>
      %tile = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x2x16x16xf32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x16x16xf32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x16xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x16xbf16, 1 : i32>, 2>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x16xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x16x32xbf16, 1 : i32>, 2>
      %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x32xbf16>
      %tile_1 = amdaie.tile(%c0, %c0)
      %tile_2 = amdaie.tile(%c1, %c0)
      %bd_id = amdaie.bd_id(%tile_1, 2)
      %bd_id_3 = amdaie.bd_id(%tile_1, 1)
      %bd_id_4 = amdaie.bd_id(%tile_1, 0)
      %4 = amdaie.logicalobjectfifo.placeholder{%tile_1} : !amdaie.logicalobjectfifo<memref<32x32xbf16>>
      memref.assume_alignment %3, 64 : memref<32x32xbf16>
      %5 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x32xbf16>
      %6 = amdaie.logicalobjectfifo.placeholder{%tile_2} : !amdaie.logicalobjectfifo<memref<32x32xbf16>>
      memref.assume_alignment %5, 64 : memref<32x32xbf16>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x32xf32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_2} : !amdaie.logicalobjectfifo<memref<1024xf32>>
      %9 = amdaie.connection(%2, %4) : (!amdaie.logicalobjectfifo<memref<2x1x16x32xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<32x32xbf16>>)
      %10 = amdaie.connection(%1, %6) : (!amdaie.logicalobjectfifo<memref<1x2x32x16xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<32x32xbf16>>)
      %11 = amdaie.connection(%8, %0) : (!amdaie.logicalobjectfifo<memref<1024xf32>>, !amdaie.logicalobjectfifo<memref<2x2x16x16xf32, 1 : i32>, 2>)
      amdaie.controlcode {
        %12 = amdaie.npu.circular_dma_cpy_nd %9([] [] [], [] [] [])
        %13 = amdaie.npu.circular_dma_cpy_nd %10([] [] [], [] [] [])
        %14 = amdaie.npu.circular_dma_cpy_nd %11([] [] [], [0, 0, 0, 0] [2, 16, 2, 16] [512, 16, 256, 1])
        %15 = amdaie.logicalobjectfifo.from_memref %3, {%tile_1} : memref<32x32xbf16> -> !amdaie.logicalobjectfifo<memref<32x32xbf16>>
        %16 = amdaie.logicalobjectfifo.from_memref %5, {%tile_1} : memref<32x32xbf16> -> !amdaie.logicalobjectfifo<memref<32x32xbf16>>
        %17 = amdaie.logicalobjectfifo.from_memref %7, {%tile_1} : memref<32x32xf32> -> !amdaie.logicalobjectfifo<memref<1024xf32>>
        %18 = amdaie.npu.dma_cpy_nd %11(%17[0, 0, 0, 0] [1, 1, 1, 1024] [0, 0, 0, 1] bd_id = %bd_id_4, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<1024xf32>>
        %19 = amdaie.npu.dma_cpy_nd %10([] [] [], %16[0, 0, 1, 2] [1, 2, 32, 16] [0, 16, 32, 1] bd_id = %bd_id_3) : source_type = !amdaie.logicalobjectfifo<memref<32x32xbf16>>
        %20 = amdaie.npu.dma_cpy_nd %9([] [] [], %15[0, 0, 0, 0] [1, 1, 1, 1024] [0, 0, 0, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<32x32xbf16>>
        amdaie.npu.dma_wait(%18, S2MM)
        amdaie.npu.dma_wait(%19, MM2S)
        amdaie.npu.dma_wait(%20, MM2S)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_0]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_2]], %[[TILE_1_2]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE_0:.+]] = aie.objectfifo.acquire @[[OBJ1]](Consume, 1)
// CHECK:         %[[ACCESS_0:.+]] = aie.objectfifo.subview.access %[[ACQUIRE_0]]
// CHECK:         %[[REINTERPRET_0:.+]] = memref.reinterpret_cast %[[ACCESS_0]]
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           linalg.fill
// CHECK-SAME:      %[[REINTERPRET_0]]
// CHECK:         }
// CHECK:         aie.objectfifo.release
// CHECK-SAME:    @[[OBJ1]]
// CHECK:       aie.core(%[[TILE_1_2]])
// CHECK:         %[[ACQUIRE_1:.+]] = aie.objectfifo.acquire @[[OBJ1]](Consume, 1)
// CHECK:         %[[ACCESS_1:.+]] = aie.objectfifo.subview.access %[[ACQUIRE_1]]
// CHECK:         %[[REINTERPRET_1:.+]] = memref.reinterpret_cast %[[ACCESS_1]]
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           linalg.fill
// CHECK-SAME:      %[[REINTERPRET_1]]
// CHECK:         }
// CHECK:         aie.objectfifo.release
// CHECK-SAME:    @[[OBJ1]]
// CHECK:       aiex.runtime_sequence @large_example
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:    %[[ARG0]]
// CHECK-SAME:    [0, 0, 0, 32]
// CHECK-SAME:    [1, 1, 32, 32]
// CHECK-SAME:    [0, 0, 64, 1]
// CHECK-SAME:    issue_token = true
// CHECK-SAME:    @[[OBJ0]]
// CHECK-NEXT:    aiex.npu.dma_wait
// CHECK-SAME:    @[[OBJ0]]
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<2, storage_buffer>]>]>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @large_example() {
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c0)
      %tile_0 = amdaie.tile(%c0, %c1)
      %tile_1 = amdaie.tile(%c0, %c2)
      %tile_2 = amdaie.tile(%c1, %c2)
      %bd_id = amdaie.bd_id(%tile, 0)
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %0, 64 : memref<32x64xi32>
      %alloc = memref.alloc() : memref<32x32xi32, 1>
      %alloc_3 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %1 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<2048xi32>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_1, %tile_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %4 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %5 = amdaie.connection(%3, %2) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.logicalobjectfifo.link[%4] -> [%5] ()
      %6 = amdaie.core(%tile_1, in : [%5], out : []) {
        %8 = amdaie.logicalobjectfifo.acquire(%5, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
        %9 = amdaie.logicalobjectfifo.access(%8, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        %reinterpret_cast = memref.reinterpret_cast %9 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<1024xi32, 2> to memref<4x8x4x8xi32, 2>
        scf.for %arg0 = %c0 to %c8 step %c1 {
          linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<4x8x4x8xi32, 2>)
        }
        amdaie.logicalobjectfifo.release(%5, Consume) {size = 1 : i32}
        amdaie.end
      }
      %7 = amdaie.core(%tile_2, in : [%5], out : []) {
        %8 = amdaie.logicalobjectfifo.acquire(%5, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
        %9 = amdaie.logicalobjectfifo.access(%8, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        %reinterpret_cast = memref.reinterpret_cast %9 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<1024xi32, 2> to memref<4x8x4x8xi32, 2>
        scf.for %arg0 = %c0 to %c8 step %c1 {
          linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<4x8x4x8xi32, 2>)
        }
        amdaie.logicalobjectfifo.release(%5, Consume) {size = 1 : i32}
        amdaie.end
      }
      memref.dealloc %alloc_3 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc : memref<32x32xi32, 1>
      amdaie.controlcode {
        %8 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [] [] [])
        %9 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
        %10 = amdaie.logicalobjectfifo.from_memref %0, {%tile} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        %11 = amdaie.npu.dma_cpy_nd %4([] [] [], %10[0, 0, 0, 32] [1, 1, 32, 32] [0, 0, 64, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%11, MM2S)
        amdaie.end
      }
    }
    return
  }
}

