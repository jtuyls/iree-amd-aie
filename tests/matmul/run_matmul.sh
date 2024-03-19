export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build4}
export IREE_AMD_AIE_DIR=${IREE_AMD_AIE_DIR:-${WORK}/versal/iree-amd-aie} 

${IREE_BUILD_DIR}/tools/iree-opt \
    --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-to-objectfifo,iree-amdaie-dma-to-objectfifo,iree-amdaie-logical-objectfifo-from-memref-cleanup,iree-amdaie-fuse-dma-copy-into-aie-region,iree-amdaie-fuse-from-memref-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-region,iree-amdaie-fuse-scf-for-into-aie-core)" \
    --mlir-print-ir-after-all matmul_1.mlir

    
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     matmul_1.mlir \
#     --pass-pipeline="builtin.module(iree-amdaie-aie-lowering-pipeline)" \
#     --mlir-print-ir-after-all \
#     > module_dump.mlir 2>&1