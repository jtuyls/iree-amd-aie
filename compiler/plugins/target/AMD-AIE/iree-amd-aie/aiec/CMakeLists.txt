# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# In-tree aiec frontend: lexer/parser/AST (dependency-free) + an
# OpBuilder-based CodeGen that constructs AMDAIE IR directly (no textual
# string assembly). The IREE HAL/Stream wrapper around the kernel is still
# emitted textually (it is IREE runtime plumbing, not AIE codegen).

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::aiec")
iree_add_all_subdirs()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

iree_cc_binary(
  NAME
    aiec-compile
  SRCS
    "Basic/Diagnostic.cpp"
    "Basic/DiagnosticIDs.cpp"
    "Basic/SourceLocation.cpp"
    "Basic/TokenKinds.cpp"
    "Lex/Lexer.cpp"
    "Parse/Parser.cpp"
    "CodeGen/ConstExpr.cpp"
    "CodeGen/Emitter.cpp"
    "Frontend/CompilerInvocation.cpp"
    "Frontend/CompilerInstance.cpp"
    "tools/aiec-compile.cpp"
  DEPS
    iree::target::amd-aie::IR::AMDAIEDialect
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRVectorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRParser
)
