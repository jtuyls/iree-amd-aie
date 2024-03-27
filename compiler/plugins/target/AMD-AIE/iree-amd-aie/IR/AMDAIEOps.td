// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
#define IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "iree-amd-aie/IR/AMDAIEAttrs.td"
include "iree-amd-aie/IR/AMDAIEDialect.td"
include "iree-amd-aie/IR/AMDAIETypes.td"


//===----------------------------------------------------------------------===//
// IREE Codegen dialect
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"


class AMDAIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AMDAIE_Dialect, mnemonic, traits>;

def AIERegionOp : AMDAIE_Op<"aie.region", [SymbolTable]> {
  let regions = (region AnyRegion);
  let assemblyFormat = [{ regions attr-dict }];
}

def ControlCodeRegionOp : AMDAIE_Op<"controlcode", [HasParent<"AIERegionOp">]> {
  let regions = (region AnyRegion);
  let assemblyFormat = [{ regions attr-dict }];
}

// Experimental to debug an issue with xilinx::AIE::CoreOp
def TileOp: AMDAIE_Op<"tile", []>, Results<(outs Index)> {
  let summary = "AIE tile";
  let arguments = (
    ins ConfinedAttr<I32Attr, [IntMinValue<0>]>:$col,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$row
  );
  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];
  let description = [{
    AIE tile
  }];
  let extraClassDeclaration = [{
    int colIndex() { return getCol(); }
    int rowIndex() { return getRow(); }
  }];
  let builders = [
    OpBuilder<(ins "long":$col, "long":$row), [{
      build($_builder, $_state, $_builder.getIndexType(), col, row);
    }]>
  ];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}
// Experimental to debug an issue with xilinx::AIE::CoreOp
def CoreOp: AMDAIE_Op<"core", []>, Results<(outs Index)> {
  let summary = "AIE core";
  let arguments = (
    ins Index:$tile
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let description = [{
    AIE core
  }];
  let extraClassDeclaration = [{
    TileOp getTileOp();
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
  ];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

// TODO(jornt): Use AIE dialect end op instead?
def EndOp: AMDAIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let description = [{
    A generic terminator operation for AIE ops' regions.
  }];
  let assemblyFormat = [{ attr-dict }];
}

def LogicalObjectFifoFromMemref : AMDAIE_Op<"logicalobjectfifo.from_memref", []> {
  let summary = "ObjectFifoSubview type accessor method";
  let description = [{
    Access the Nth element of a value of `ObjectFifoSubview` type.

    Example:
    ```
      %subview = aie.objectfifo.acquire @of1 (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
    ```
    In this example, `%elem` is the first object of the subview. Note that this may not correspond to the first element of
    the `objectFifo` if other acquire operations took place beforehand.

  }];

  let arguments = (
    ins AnyMemRef:$memref,
        // ConfinedAttr<I32Attr, []>:$id,
        Variadic<Index>:$tiles
  ); 

  let hasVerifier = 0;

  let results = (outs AMDAIE_ObjectFifoType:$output);

  let assemblyFormat = [{
    $memref  `,``{` $tiles `}` attr-dict `:` type($memref) `->` type($output)
  }];

  // Allow building an AIE_ObjectFifoSubviewAccessOp with just a subview value and an index.
  let builders = [
    OpBuilder<(ins "mlir::Value":$memref)>
    // OpBuilder<(ins "Type":$result_type, "mlir::Value":$memref)>
    // OpBuilder<(ins "Type":$result_type, "mlir::Value":$memref), [{
    //   build($_builder, $_state, result_type, memref, $_builder.getI32IntegerAttr(-1), ValueRange{});
    // }]>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemrefType() { return getOutput().getType().cast<AMDAIEObjectFifoType>().getElementType(); }
    // void setTile(Value tile) { (*this)->setOperand(1, tile); };
  }];

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}


def DmaCpyNdOp: AMDAIE_Op<"dma_cpy_nd", [AttrSizedOperandSegments]>, Results<(outs Index)> {
  let summary = "dma operator";
  let arguments = (
    ins AnyAMDAIEObjectFifoType:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyAMDAIEObjectFifoType:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );
  // let results = (outs Optional<air_AsyncToken>:$async_token);
  // let results = (outs Index:$output);
  let assemblyFormat = [{
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,`
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `,` type($src) `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
    Type getSrcType() { return getSrc().getType(); }
    Type getDstType() { return getDst().getType(); }
    LogicalObjectFifoFromMemref getSrcObjectFifo();
    LogicalObjectFifoFromMemref getDstObjectFifo();
    // Type getSrcMemrefType() { return getSrcType().cast<AMDAIEObjectFifoType>().getElementType(); }
    // bool containsValueInOperands(Value value);
    // int32_t getId() {
    //   if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
    //     return id_attr.getInt();
    //   }
    //   return -1;
    // }
  }];

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoLink: AMDAIE_Op<"logicalobjectfifo.link", [AttrSizedOperandSegments]> {
  let summary = "Explicitly link dma ops";
  let arguments = (
    ins Variadic<Index>:$ins,
        Variadic<Index>:$outs
  );
  let assemblyFormat = [{
    `[` ($ins^)? `]` `->` `[` ($outs^)? `]` `(` `)` attr-dict
  }];
  // let extraClassDeclaration = [{
  // }];
  let description = [{
    Explicitly link dma ops
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoConsume: AMDAIE_Op<"logicalobjectfifo.consume", []> {
  let summary = "Consume dma result";
  let arguments = (
    // ins AnyAMDAIEObjectFifoType:$objectfifo
    ins Index:$dma
  );
  let assemblyFormat = [{
    `(` $dma `)`  attr-dict
  }];
  let extraClassDeclaration = [{
    Value getObjectfifo() { return dyn_cast<DmaCpyNdOp>(getDma().getDefiningOp()).getDst(); }
    ObjectFifoPort getPort() { return ObjectFifoPort::Consume; }
  }];
  let description = [{
    acquire logical objectfifo
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoProduce: AMDAIE_Op<"logicalobjectfifo.produce", []> {
  let summary = "release logical objectfifo";
  let arguments = (
    // ins AnyAMDAIEObjectFifoType:$objectfifo
    ins Index:$dma
  );
  let assemblyFormat = [{
    `(` $dma `)`  attr-dict
  }];
  let extraClassDeclaration = [{
    Value getObjectfifo() { return dyn_cast<DmaCpyNdOp>(getDma().getDefiningOp()).getSrc(); }
    ObjectFifoPort getPort() { return ObjectFifoPort::Produce; }
  }];
  let description = [{
    release logical objectfifo
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoAcquire: AMDAIE_Op<"logicalobjectfifo.acquire", []> {
  let summary = "Acquire dma result";
  let arguments = (
    ins Index:$dma,
        ObjectFifoPort:$port
  );
  let assemblyFormat = [{
    `(` $dma `,` $port `)`  attr-dict
  }];
  let extraClassDeclaration = [{
    Value getObjectfifo() { return dyn_cast<DmaCpyNdOp>(getDma().getDefiningOp()).getDst(); }
  }];
  let description = [{
    acquire dma result
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoRelease: AMDAIE_Op<"logicalobjectfifo.release", []> {
  let summary = "Release dma result";
  let arguments = (
    ins Index:$dma,
        ObjectFifoPort:$port
  );
  let assemblyFormat = [{
    `(` $dma `,` $port `)`  attr-dict
  }];
  let extraClassDeclaration = [{
    Value getObjectfifo() { return dyn_cast<DmaCpyNdOp>(getDma().getDefiningOp()).getDst(); }
  }];
  let description = [{
    release dma result
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoWait: AMDAIE_Op<"logicalobjectfifo.wait", []> {
  let summary = "logicalobjectfifo synchronization point";
  let arguments = (
    ins AnyAMDAIEObjectFifoType:$objectfifo
  );
  let assemblyFormat = [{
    `(` $objectfifo `)`  attr-dict `:` `(` type($objectfifo) `)`
  }];
  let description = [{
    Wait until all queued operations on this objectfifo are done.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LoadCoreOp: AMDAIE_Op<"ipu.load", []> {
  let summary = "load an AIE core";
  let arguments = (
    ins Index:$core
  );
  let assemblyFormat = [{ `(` $core `)` regions attr-dict }];
  let description = [{
    load an AIE core
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def IpuDmaCpyNdOp: AMDAIE_Op<"ipu.dma_cpy_nd", [AttrSizedOperandSegments]> {
  let summary = "ipu dma operator";
  let arguments = (
    ins AnyAMDAIEObjectFifoType:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyAMDAIEObjectFifoType:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );
  // let results = (outs Optional<air_AsyncToken>:$async_token);
  let assemblyFormat = [{
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,`
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `,` type($src) `)`
  }];
  let description = [{
    ipu dma operator
  }];
  let extraClassDeclaration = [{
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
    Type getSrcType() { return getSrc().getType(); }
    Type getDstType() { return getDst().getType(); }
    // Type getSrcMemrefType() { return getSrcType().cast<AMDAIEObjectFifoType>().getElementType(); }
  }];

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}


#endif // IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
