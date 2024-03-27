export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build4}
export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${WORK}/versal/iree-amd-aie} 

export PASSES="fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-to-objectfifo"
export PASSES="${PASSES},iree-amdaie-dma-to-control-code"
export PASSES="${PASSES},iree-amdaie-dma-to-objectfifo,iree-amdaie-logical-objectfifo-from-memref-cleanup"
export PASSES="${PASSES},iree-amdaie-fuse-dma-copy-into-aie-region,iree-amdaie-fuse-from-memref-into-aie-region"
export PASSES="${PASSES},iree-amdaie-fuse-scf-for-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-core"
export PASSES="${PASSES},iree-amdaie-fuse-aie-regions,iree-amdaie-fuse-from-memref-into-aie-region"
export PASSES="${PASSES},iree-amdaie-logical-objectfifo-from-memref-cleanup,iree-amdaie-simplify-aie-regions"
export PASSES="${PASSES},iree-amdaie-logical-objectfifo-from-memref-cleanup"
export PASSES="${PASSES},iree-amdaie-to-aie"
${IREE_BUILD_DIR}/tools/iree-opt \
    --pass-pipeline="builtin.module(${PASSES})" \
    --mlir-print-ir-after-all matmul_7.mlir
    # --pass-pipeline="builtin.module(${passes})" \
    #--pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-to-objectfifo,iree-amdaie-dma-to-objectfifo,iree-amdaie-logical-objectfifo-from-memref-cleanup,iree-amdaie-fuse-dma-copy-into-aie-region,iree-amdaie-fuse-from-memref-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-core,iree-amdaie-fuse-aie-regions,iree-amdaie-fuse-from-memref-into-aie-region,iree-amdaie-logical-objectfifo-from-memref-cleanup)" \
    

    
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     matmul_1.mlir \
#     --pass-pipeline="builtin.module(iree-amdaie-aie-lowering-pipeline)" \
#     --mlir-print-ir-after-all \
#     > module_dump.mlir 2>&1