export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build5}

# ${IREE_BUILD_DIR}/tools/iree-opt --iree-transform-dialect-interpreter matmul_fill_spec_pad_pack_peel.mlir

# ${IREE_BUILD_DIR}/tools/iree-compile \
#     test.mlir \
#     --iree-hal-target-backends=amd-aie \
#     --compile-to=executable-sources |
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-amdaie-lower-executable-target, fold-memref-alias-ops)))' \
#     --iree-codegen-transform-dialect-library=matmul_fill_spec_pad_pack_peel.mlir
#     # --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" \
#     # --iree-codegen-transform-dialect-library=matmul_fill_spec_pad_pack_peel.mlir \

# pack_peel_pipeline_matmul.mlir \
# -debug-only=iree-amdaie-unroll-and-distribute-workgroup \
# ${IREE_BUILD_DIR}/tools/iree-compile \
#     pack_peel_pipeline_matmul.mlir  \
#     --iree-hal-target-backends=amd-aie \
#     --compile-to=executable-sources |
# ${IREE_BUILD_DIR}/tools/iree-opt \
#     --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" \
#     --iree-amdaie-use-pipeline=pack-peel \
#     --mlir-print-ir-before-all \
#     --iree-amdaie-enable-vectorization-passes=false \
#     -debug-only=iree-amdaie-unroll-and-distribute-workgroup \
#     > Log.cc

# matmul_peeled_objectfifo.mlir
# matmul_peeled_2x2.mlir
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2_small.mlir --mlir-print-ir-before-all --debug-only=iree-amdaie-create-aie-workgroup --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-aie-workgroup,iree-amdaie-fuse-logicalobjectfifo-into-workgroup,cse,iree-amdaie-unroll-and-distribute-workgroup,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-consume-produce-to-acquire-release,cse,canonicalize,iree-amdaie-controlcode-loop-unroll,cse,canonicalize,iree-amdaie-lower-to-aie,canonicalize,convert-linalg-to-loops)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2.mlir --mlir-print-ir-before-all --debug-only=iree-amdaie-unroll-and-distribute-workgroup --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-aie-workgroup,iree-amdaie-fuse-logicalobjectfifo-into-workgroup,cse,iree-amdaie-unroll-and-distribute-workgroup,cse,iree-amdaie-dma-to-circular-dma,func.func(iree-amdaie-create-aie-workgroup),canonicalize,cse,iree-amdaie-canonicalize-doubly-strided-op,iree-amdaie-consume-produce-to-acquire-release,cse,canonicalize,iree-amdaie-dma-loop-subsumption)"
# ${IREE_BUILD_DIR}/tools/iree-opt matmul_peeled_2x2.mlir --mlir-print-ir-before-all --debug-only=iree-amdaie-unroll-and-distribute-workgroup --pass-pipeline="builtin.module(fold-memref-alias-ops,iree-amdaie-pack-to-dma,air-copy-to-dma,iree-amdaie-air-dma-to-amdaie-dma,iree-amdaie-insert-aie-workgroup,iree-amdaie-fuse-logicalobjectfifo-into-workgroup,cse,iree-amdaie-unroll-and-distribute-workgroup)"


${IREE_BUILD_DIR}/tools/iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources matmul_sample.mlir | 
${IREE_BUILD_DIR}/tools/iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-enable-vectorization-passes=false --iree-amdaie-use-pipeline=pack-peel --mlir-print-ir-before-all &> debug_matmul_new.mlir