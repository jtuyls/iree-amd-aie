// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-loop-subsumption,canonicalize))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Sanity check: ensure no modification in case of no loop depedency
// CHECK-LABEL: @npu_dma_cpy_nd_without_loop_dependency
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%{{.+}}, %{{.+}}) in (2, 2)
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:             %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_without_loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.forall (%arg2, %arg3) in (2, 2) {
        scf.for %arg4 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg4)
          %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 0] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2, S2MM)
        }
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of an affine expression with a constant.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK:       #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 16 + 3)>
// CHECK:       #[[$MAP2:.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL: @invalid_affine_expr
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C1]] to %[[C6]] step %[[C2]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[APPLY2:.+]] = affine.apply #[[$MAP1]](%[[ARG2]])
// CHECK:           %[[NPU_DMA1:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[APPLY2]]] [16] [1], [%[[APPLY1]]] [16] [1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA1]], S2MM)
// CHECK:           %[[APPLY3:.+]] = affine.apply #[[$MAP2]](%[[ARG2]])
// CHECK:           %[[NPU_DMA2:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[APPLY3]]] [16] [1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA2]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 16 + 3)>
#map2 = affine_map<(d0) -> (d0 + 3)>
func.func @invalid_affine_expr(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c1 to %c6 step %c2 {
        %1 = affine.apply #map(%arg2)
        %2 = affine.apply #map1(%arg2)
        %3 = amdaie.npu.dma_cpy_nd %0([%2] [16] [1], [%1] [16] [1])
        amdaie.npu.dma_wait(%3, S2MM)
        %4 = affine.apply #map2(%arg2)
        %5 = amdaie.npu.dma_cpy_nd %0([%4] [16] [1], [] [] [])
        amdaie.npu.dma_wait(%5, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L3.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, %[[APPLY]]] [1, 1, 8, 16] [128, 128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, %1] [1, 1, 8, 16] [128, 128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L3.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, %[[APPLY]]] [1, 1, 8, 16] [128, 128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, %1] [1, 1, 8, 16] [128, 128, 16, 1])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L2.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target_on_l2
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, %[[APPLY]]] [1, 1, 8, 16] [128, 128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_target_on_l2(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, %1] [1, 1, 8, 16] [128, 128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L2.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source_on_l2
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, %[[APPLY]]] [1, 1, 8, 16] [128, 128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_source_on_l2(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, %1] [1, 1, 8, 16] [128, 128, 16, 1])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 3 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L1.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target_on_l1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_target_on_l1(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 3 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L1.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source_on_l1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_too_many_dims_source_on_l1(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 2>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 2>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1] [1, 8, 16] [128, 16, 1])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_with_for_dependency_on_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C6]], %[[C1]], %[[C8]], %[[C16]]] [%[[C16]], %[[C128]], %[[C16]], %[[C1]]], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_with_for_dependency_on_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_with_forall_dependency_on_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C6]], %[[C1]], %[[C8]], %[[C16]]] [%[[C16]], %[[C128]], %[[C16]], %[[C1]]], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (16 * d0)>
func.func @npu_dma_cpy_nd_with_forall_dependency_on_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.forall (%arg2, %arg3) in (2, 6) {
        %1 = affine.apply #map(%arg3)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_with_for_dependency_on_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C6]], %[[C1]], %[[C8]], %[[C16]]] [%[[C16]], %[[C128]], %[[C16]], %[[C1]]])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_with_for_dependency_on_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1] [1, 8, 16] [128, 16, 1])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_with_forall_dependency_on_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C6]], %[[C1]], %[[C8]], %[[C16]]] [%[[C16]], %[[C128]], %[[C16]], %[[C1]]])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_with_forall_dependency_on_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.forall (%arg2, %arg3) in (2, 6) {
        %1 = affine.apply #map(%arg3)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1] [1, 8, 16] [128, 16, 1])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Check with multiple `affine.apply` usages in a `amdaie.npu.dma_cpy_nd` operation.
// CHECK-LABEL: @npu_dma_cpy_nd_with_multiple_for_dependencies
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C6]], %[[C6]], %[[C8]], %[[C16]]] [%[[C256]], %[[C16]], %[[C16]], %[[C1]]], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @npu_dma_cpy_nd_with_multiple_for_dependencies(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([%1, %1] [8, 16] [16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_with_multiple_forall_dependencies
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C2]], %[[C6]], %[[C8]], %[[C16]]] [%[[C16]], %[[C512]], %[[C16]], %[[C1]]], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 32)>
func.func @npu_dma_cpy_nd_with_multiple_forall_dependencies(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.forall (%arg2, %arg3) in (2, 6) {
        %1 = affine.apply #map(%arg2)
        %2 = affine.apply #map1(%arg3)
        %3 = amdaie.npu.dma_cpy_nd %0([%2, %1] [8, 16] [16, 1], [] [] [])
        amdaie.npu.dma_wait(%3, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @non_normalized_loop
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([%[[C0]], %[[C16]]] [%[[C3]], %[[C16]]] [%[[C16]], %[[C1]]], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @non_normalized_loop(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c1 to %c6 step %c2 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([%1] [16] [1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of multiple npu.dma_cpy_nd users with the same source in the same scope.
// CHECK-LABEL: @for_with_multiple_npu_dma_cpy_nd_same_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]], S2MM)
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @for_with_multiple_npu_dma_cpy_nd_same_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.for %arg2 = %c0 to %c6 step %c1 {
        %1 = affine.apply #map(%arg2)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
        %3 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%3, S2MM)
      }
      amdaie.end
    }
  }
  return
}

// -----

// Ensure no modification in case of multiple npu.dma_cpy_nd users with the same source in the same scope.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @forall_with_multiple_npu_dma_cpy_nd_same_source
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 6)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]], S2MM)
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]], S2MM)
#map = affine_map<(d0) -> (d0 * 16)>
func.func @forall_with_multiple_npu_dma_cpy_nd_same_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  amdaie.workgroup {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    amdaie.controlcode {
      scf.forall (%arg2, %arg3) in (2, 6) {
        %1 = affine.apply #map(%arg3)
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%2, S2MM)
        %3 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
        amdaie.npu.dma_wait(%3, S2MM)
      }
      amdaie.end
    }
  }
  return
}
