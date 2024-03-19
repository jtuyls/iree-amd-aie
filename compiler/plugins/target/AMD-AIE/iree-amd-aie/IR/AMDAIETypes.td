// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_TYPES
#define AMDAIE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "iree-amd-aie/IR/AMDAIEDialect.td"



class AMDAIEDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMDAIE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


// def AMDAIE_ObjectFifoType : AMDAIEDialect_Type<"ObjectFifo", "objectfifotype"> {
//   let summary = "AMDAIE objectFifo type";
// }

def AMDAIE_ObjectFifoType :
    DialectType<AMDAIE_Dialect, CPred<"$_self.isa<AMDAIEObjectFifoType>()">,
                "AMDAIE objectFifo type">;

def AnyAMDAIEObjectFifoType : AnyTypeOf<[AMDAIE_ObjectFifoType]>;
    

#endif // AMDAIE_TYPES
