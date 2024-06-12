//===- LLVMLink.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This is basically copy-pasta from llvm/tools/llvm-link/llvm-link.cpp but
// rewritten to take strings as input instead of files. Comments are preserved
// in case you need to go back to the source and compare/contrast.

#include "AIEChessHack.h"

#include <memory>
#include <utility>

#define mlir mlir15
#define llvm llvm15

#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace llvm;
using namespace mlir;

static std::unique_ptr<Module> loadFile(std::unique_ptr<MemoryBuffer> Buffer,
                                        LLVMContext &Context, bool Verbose,
                                        bool MaterializeMetadata = true) {
  SMDiagnostic Err;
  if (Verbose) errs() << "Loading '" << Buffer->getBufferIdentifier() << "'\n";
  std::unique_ptr<Module> Result;
  Result = parseIR(*Buffer, Err, Context);
  if (!Result) {
    Err.print("aie-llvm-Link", errs());
    return nullptr;
  }

  if (MaterializeMetadata) {
    // this call return an error that convert true if there was in fact an
    // error...
    if (Result->materializeMetadata()) {
      Err.print("aie-llvm-link", errs());
      return nullptr;
    }
    UpgradeDebugInfo(*Result);
  }

  return Result;
}

mlir::LogicalResult linkFiles(std::vector<std::string> Files,
                              LLVMContext &Context, Linker &L, unsigned Flags,
                              bool DisableDITypeMap, bool NoVerify,
                              bool Internalize, bool Verbose) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;
  // Similar to some flags, internalization doesn't apply to the first file.
  bool InternalizeLinkedSymbols = false;
  for (const auto &File : Files) {
    std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBufferCopy(File);

    std::unique_ptr<Module> maybeModule =
        loadFile(std::move(Buffer), Context, Verbose);
    if (!maybeModule) {
      WithColor::error() << " loading file '" << File << "'\n";
      return mlir::failure();
    }

    // Note that when ODR merging types cannot verify input files in here When
    // doing that debug metadata in the src module might already be pointing to
    // the destination.
    if (DisableDITypeMap && !NoVerify && verifyModule(*maybeModule, &errs())) {
      WithColor::error() << "input module is broken!\n";
      return mlir::failure();
    }

    if (Verbose) errs() << "Linking in '" << File << "'\n";

    bool Err;
    if (InternalizeLinkedSymbols)
      Err = L.linkInModule(
          std::move(maybeModule), ApplicableFlags,
          [](Module &M, const llvm::StringSet<> &GVS) {
            internalizeModule(M, [&GVS](const GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    else
      Err = L.linkInModule(std::move(maybeModule), ApplicableFlags);

    if (Err) {
      errs() << "couldn't link.\n";
      return mlir::failure();
    }

    // Internalization applies to linking of subsequent files.
    InternalizeLinkedSymbols = Internalize;

    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }

  return mlir::success();
}

std::optional<std::string> xilinx::AIE::AIELLVMLink(
    std::vector<std::string> Files, bool DisableDITypeMap, bool NoVerify,
    bool Internalize, bool OnlyNeeded, bool PreserveAssemblyUseListOrder,
    bool Verbose) {
  std::string output;
  raw_string_ostream llvmirStream(output);
  LLVMContext Context;

  if (!DisableDITypeMap) Context.enableDebugTypeODRUniquing();

  auto Composite = std::make_unique<Module>("aie-llvm-link", Context);
  Linker L(*Composite);

  unsigned Flags = Linker::Flags::None;
  if (OnlyNeeded) Flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (failed(linkFiles(Files, Context, L, Flags, DisableDITypeMap, NoVerify,
                       Internalize, Verbose)))
    return std::nullopt;

  Composite->print(llvmirStream, nullptr, PreserveAssemblyUseListOrder);
  return {output};
}

std::optional<std::string> xilinx::AIE::AIETranslateModuleToLLVMIR(
    const std::string &moduleOp) {
  MLIRContext context;
  DialectRegistry registry;
  registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  llvm::StringRef module(moduleOp);
  OwningOpRef<ModuleOp> owning = parseSourceString<ModuleOp>(module, &context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*owning, llvmContext);
  if (!llvmModule) return std::nullopt;
  std::string llvmir;
  llvm::raw_string_ostream os(llvmir);
  llvmModule->print(os, nullptr);
  //  char *cStr = static_cast<char *>(malloc(llvmir.size()));
  //  llvmir.copy(cStr, llvmir.size());
  return {llvmir};
}