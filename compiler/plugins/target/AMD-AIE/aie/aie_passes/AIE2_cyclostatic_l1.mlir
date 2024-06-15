
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @fifo : memref<i32>
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[FIFO_BUFF_0:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_1:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_2:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK:           %[[FIFO_BUFF_3:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK:           %[[FIFO_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:           %[[FIFO_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:           %[[BUF23:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "buf23"} : memref<4xi32>
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C55_I32:.*]] = arith.constant 55 : i32
// CHECK:             %[[C66_I32:.*]] = arith.constant 66 : i32
// CHECK:             %[[C77_I32:.*]] = arith.constant 77 : i32
// CHECK:             %[[C88_I32:.*]] = arith.constant 88 : i32
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C55_I32]], %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C66_I32]], %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C77_I32]], %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[C88_I32]], %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_3:.*]] = aie.core(%[[TILE_2_3]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_0:.*]] = memref.load %[[FIFO_BUFF_0]][] : memref<i32>
// CHECK:             memref.store %[[VAL_0]], %[[BUF23]]{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             %[[VAL_1:.*]] = memref.load %[[FIFO_BUFF_1]][] : memref<i32>
// CHECK:             %[[VAL_2:.*]] = memref.load %[[FIFO_BUFF_2]][] : memref<i32>
// CHECK:             memref.store %[[VAL_1]], %[[BUF23]]{{\[}}%[[C1]]] : memref<4xi32>
// CHECK:             memref.store %[[VAL_2]], %[[BUF23]]{{\[}}%[[C2]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 2)
// CHECK:             aie.use_lock(%[[FIFO_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_3:.*]] = memref.load %[[FIFO_BUFF_3]][] : memref<i32>
// CHECK:             memref.store %[[VAL_3]], %[[BUF23]]{{\[}}%[[C3]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[FIFO_PROD_LOCK]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_l1 {
    aie.device(xcve2302) {
        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile
        %buf23  = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>
        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>
        // Producer core
        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            // Push 55
            %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 66
            %subview1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 77
            %subview2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            // Push 88
            %subview3 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)
            aie.end
        }
        // Consumer core
        %core23 = aie.core(%tile23) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index
            // Pop 1 object off queue
            %subview0 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf23[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)
            // Pop 2 objects off queue
            %subview1 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf23[%i1] : memref<4xi32>
            memref.store %v77, %buf23[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 2)
            // Pop 1 object off queue
            %subview2 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf23[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)
            aie.end
        }
    }
}
