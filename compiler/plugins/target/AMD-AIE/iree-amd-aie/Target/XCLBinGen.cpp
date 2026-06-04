// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "XCLBinGen.h"

#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <system_error>
#include <vector>

#include "AMDAIETargets.h"
#include "aie/Passes.h"
#include "air/Conversion/AIRToAIEPass.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "ukernels.h"

#define DEBUG_TYPE "amdaie-xclbingen"

#ifndef IREE_AMD_AIE_BOOTGEN_EXECUTABLE
extern int iree_aie_bootgen_main(int argc, const char *argv[]);
#endif  // IREE_AMD_AIE_BOOTGEN_EXECUTABLE

// https://stackoverflow.com/a/60198074
using namespace std::placeholders;
using namespace llvm;
using namespace mlir;
using namespace xilinx;
using Path = std::filesystem::path;

namespace mlir::iree_compiler::AMDAIE {
namespace detail {

FailureOr<std::vector<std::string>> flagStringToVector(
    const std::string &flags) {
  if (flags.empty()) return std::vector<std::string>{};
  // Check that flags string is of the form "-flag1 -flag2".
  // i.e. that it starts and ends with ".
  if (flags.size() < 2 || flags.front() != '"' || flags.back() != '"') {
    llvm::errs()
        << "additional peano opt flags must be of the form "
           "\"-flag1 -flag2 ...\". Specifically it must start and end with \".";
    return failure();
  }
  // Split the additional flags on whitespace, and then add to the default args.
  std::istringstream iss(flags.substr(1, flags.size() - 2));
  return std::vector<std::string>{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
}

// Extract an integer from a string, if possible.
std::optional<int> safeStoi(std::string_view intString) {
  size_t start = intString.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) return std::nullopt;
  int value = 0;
  const char *d0 = intString.data() + start;
  const char *d1 = intString.data() + intString.size();
  auto [ptr, ec] = std::from_chars(d0, d1, value);
  if (ec == std::errc()) return value;
  return std::nullopt;
}

static bool isBlank(const std::string &line) {
  return std::all_of(line.begin(), line.end(), [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
}

/// Returns the maximum stack size by parsing the output string of 'llvm-readelf
/// --stack-sizes'.
///
/// We assume that stack sizes output is of the form:
///
/// ```
/// Stack Sizes:
///      Size     Functions
///        64     core_0_5, main
///        32     some_func
///        64     some_other_func
/// ```
///
/// In terms of how we estimate stack sizes, we assume that all functions
/// could be called in nested fashion and the maximum stack size is the
/// sum of all functions' stack sizes.
///
/// TODO(jornt): For the exact stack size of the program we need the function
/// call graph as well as all functions' stack sizes. Once the retrieval of this
/// is supported in peano, we can use the exact stack size of the program.
FailureOr<uint32_t> getMaxStackSize(const std::string &stackSizesOutput) {
  std::istringstream stackSizesStream(stackSizesOutput);
  std::string line;

  // Skip 'Stack Sizes:', possibly preceded by blank lines
  while (std::getline(stackSizesStream, line)) {
    if (!isBlank(line)) {
      break;
    }
  }

  // Skip the header lines:
  //      Size     Functions
  std::getline(stackSizesStream, line);

  uint32_t maxStackSize = 0;
  while (std::getline(stackSizesStream, line)) {
    std::istringstream linestream(line);
    uint32_t size;
    if (linestream >> size) {
      maxStackSize += size;
    } else {
      return failure();
    }
  }
  return maxStackSize;
}

// Peano's `opt` program optimizes llvm-ir (.ll files). We run it with a system
// call. This functions constructs the flags to pass to `opt`. There are some
// default flags, most of which are copied from llvm-aie. See
//
// clang-format off
// https://github.com/nod-ai/iree-amd-aie/pull/622
// https://github.com/Xilinx/llvm-aie/blob/0be095354faa49985cd031661853f6d9b9b787f2/clang/lib/Driver/ToolChains/AIE.cpp#L97-L121
// clang-format on
//
// There are also additional flags which have been passed down from the user,
// `additionalFlags`. This function appends these user specific flags,
// and checks that they are valid. If they are not, it returns failure.
FailureOr<std::vector<std::string>> makePeanoOptArgs(
    const std::vector<std::string> &additionalFlags) {
  std::vector<std::string> args{
      // peano has no proper vectorization cost model for AIE
      "-vectorize-loops=false",
      //
      "-vectorize-slp=false",
      // An if-then-else cascade requires at least 5 delay slots for
      // evaluating the condition and 5 delay slots for one of the
      // branches, thus speculating 10 instructions should be fine
      "--two-entry-phi-node-folding-threshold=10",
      // Make sure to perform most optimizations before mandatory
      // inlinings, otherwise noalias attributes can get lost and
      // hurt AA results.
      "-mandatory-inlining-before-opt=false",
      // complete AA analysis on phi nodes.
      "-basic-aa-full-phi-analysis=true",
      // Extend the max limit of the search depth in BasicAA
      "-basic-aa-max-lookup-search-depth=10",
      //
      "-O3",
      //
      "--inline-threshold=10",
      // missing from libc
      "--disable-builtin=memset",
  };

  if (additionalFlags.empty()) return args;

  // Return true if `flag` is an optimization level flag, like -O2.
  auto isOptLevelFlag = [](const std::string &flag) {
    bool isOptFlag = flag.size() == 3 && flag[0] == '-' && flag[1] == 'O';
    return isOptFlag;
  };

  // Return true if flags `a` and `b` cannot coexist when passed to `opt`.
  auto isContention = [&](const std::string &a, const std::string &b) {
    // If both flags are optimization level flags, they cannot coexist, because
    // llvm-opt will fail to run if it sees two different optimization levels.
    if (isOptLevelFlag(a) && isOptLevelFlag(b)) return true;
    return false;
  };

  // Append the additional flags, unless they conflict with an existing flag,
  // in which case replace the existing flag.
  args.reserve(args.size() + additionalFlags.size());
  for (const auto &flag : additionalFlags) {
    auto iter = std::find_if(args.begin(), args.end(),
                             std::bind(isContention, _1, flag));
    if (iter == args.end()) {
      args.push_back(flag);
    } else {
      *iter = flag;
    }
  }

  // Adding cse after the default O2 pipeline eliminates repeated
  // ```
  // %49 = trunc i64 %38 to i20
  // ```
  // for certain matmuls (outlining, phoenix), and results in dramatic
  // improvements in performance.
  for (std::string &flag : args) {
    if (isOptLevelFlag(flag)) {
      auto optLevel = flag.substr(1);
      auto passes = "default<" + optLevel + ">,early-cse,dce";
      flag = "-passes=" + passes;
    }
  }
  return args;
}
}  // namespace detail
}  // namespace mlir::iree_compiler::AMDAIE

namespace {
namespace uuid {
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> dis(0, 15);
static std::uniform_int_distribution<> dis2(8, 11);

std::string getUUIDString() {
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  };
  return ss.str();
}
}  // namespace uuid

FailureOr<std::string> getTargetDir(const std::string &npuVersion) {
  if (npuVersion == "npu1") return std::string{"target_aie_ml"};
  if (npuVersion == "npu4") return std::string{"target_aie2p"};
  llvm::errs() << "unsupported NPUVersion: " << npuVersion;
  return failure();
}

// Apply the pass manager specific options of the XCLBinGenConfig to the pass
// manager. These control when (if ever) and what IR gets printed between
// passes, and whether the pass manager uses multi-theading.
void applyConfigToPassManager(PassManager &pm, bool printIRBeforeAll,
                              bool printIRAfterAll, bool printIRModuleScope,
                              bool timing) {
  auto shouldPrintBeforePass = [printIRBeforeAll](Pass *, Operation *) {
    return printIRBeforeAll;
  };

  auto shouldPrintAfterPass = [printIRAfterAll](Pass *, Operation *) {
    return printIRAfterAll;
  };

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printIRModuleScope);

  if (timing) pm.enableTiming();
}

FailureOr<Path> findVitis(std::optional<Path> &vitisDir,
                          const std::string &npuVersion) {
  if (!vitisDir) {
    const char *envVitis = ::getenv("VITIS");
    if (!envVitis) {
      if (auto vpp = sys::findProgramByName("v++")) {
        SmallString<64> realVpp;
        std::error_code err = sys::fs::real_path(vpp.get(), realVpp);
        if (!err) {
          sys::path::remove_filename(realVpp);
          sys::path::remove_filename(realVpp);
          vitisDir = realVpp.str().str();
          LLVM_DEBUG(dbgs() << "Found Vitis at " << realVpp.c_str() << "\n");
        }
      }
    }
  }
  if (!vitisDir) {
    llvm::errs() << "ERROR: couldn't find vitis directory\n";
    return failure();
  }

  const char *licenseFile = ::getenv("XILINXD_LICENSE_FILE");
  if (!licenseFile) {
    licenseFile = ::getenv("LM_LICENSE_FILE");
    if (!licenseFile) {
      llvm::errs() << "ERROR: either XILINXD_LICENSE_FILE or LM_LICENSE_FILE "
                      "must be set\n";
      return failure();
    }
    if (!std::filesystem::exists(licenseFile)) {
      llvm::errs() << "ERROR: license file" << licenseFile << " does not exist";
      return failure();
    }
  }

  Path aieToolsPath = *vitisDir / "aietools";
  if (!std::filesystem::exists(aieToolsPath)) {
    llvm::errs() << "ERROR: couldn't find aietools directory\n";
    return failure();
  }

  Path chessccPath = aieToolsPath / "tps" / "lnx64" /
                     *getTargetDir(npuVersion) / "bin" / "LNa64bin";

  std::string chessccPathStr = chessccPath.string();

  if (!std::filesystem::exists(chessccPath / "chess-clang")) {
    llvm::errs() << "Tried to find chess-clang at:\n  " << chessccPathStr
                 << "\nERROR: couldn't find chess-clang\n";
    return failure();
  }
  if (!std::filesystem::exists(chessccPath / "chess-llvm-link")) {
    llvm::errs() << "Tried to find chess-llvm-link at:\n  " << chessccPathStr
                 << "\nERROR: couldn't find chess-llvm-link\n";
    return failure();
  }

  return *vitisDir;
}

FailureOr<Path> findAMDAIETool(std::string toolName,
                               const Path &amdAIEInstallDir) {
#if defined(_WIN32)
  toolName += ".exe";
#endif  // _WIN32
  Path toolBinExe;
  if (!amdAIEInstallDir.empty()) {
    toolBinExe = amdAIEInstallDir / toolName;
    if (std::filesystem::exists(toolBinExe)) return toolBinExe;

    toolBinExe = amdAIEInstallDir / "bin" / toolName;
    if (std::filesystem::exists(toolBinExe)) return toolBinExe;

    toolBinExe = amdAIEInstallDir / "tools" / toolName;
    if (std::filesystem::exists(toolBinExe)) return toolBinExe;
  }

  toolBinExe = mlir::iree_compiler::findTool(toolName);
  if (std::filesystem::exists(toolBinExe)) return toolBinExe;

  llvm::errs() << "Could not find " << toolName
               << ". Check your --iree-amd-aie-install-dir flag\n";
  return failure();
}

std::pair<std::string, std::vector<std::string>> makeChessArgs(
    Path &vitisDir, Path &tempDir, const std::string &npuVersion,
    bool verbose) {
  std::string archVersion;
  std::string modelDir;
  if (npuVersion == "npu1") {
    archVersion = "20";
    modelDir = "aie_ml";
  } else if (npuVersion == "npu4") {
    archVersion = "21";
    modelDir = "aie2p";
  } else {
    llvm::errs() << "unsupported NPU version: " << npuVersion;
    llvm::report_fatal_error("unsupported NPU version");
  }

  Path aieToolsDir = vitisDir / "aietools";
  std::vector<std::string> flags{
      // -j <threads> : parallel compilation (function + file level)
      "-j1",
      // -p <name> : processor
      "-pme",
      // -P <dir> : processor model directory
      "-P" + (aieToolsDir / "data" / modelDir / "lib").string(),
      // -f : use LLVM frontend (chess-clang)
      "-f",
      // -C <cfg> : configuration (for chess-clang)
      "-CRelease_LLVM",
      // +w <dir> : work directory
      "+w" + tempDir.string(),
      // for adf headers
      "-D__AIENGINE__",
      // for aie_api headers
      "-D__AIE_ARCH__=" + archVersion, "-D__AIEARCH__=" + archVersion,
      // for aie_api headers
      "-I" + (aieToolsDir / "include").string()};
  // disassemble output
  if (verbose) flags.emplace_back("-d");
  return {(aieToolsDir / "bin" / "unwrapped" / "lnx64.o" / "xchesscc").string(),
          flags};
}

std::vector<std::string> makeChessEnv(Path &vitisDir,
                                      const std::string &npuVersion) {
  Path aieToolsPath = vitisDir / "aietools";
  Path chessccPath = aieToolsPath / "tps" / "lnx64" /
                     *getTargetDir(npuVersion) / "bin" / "LNa64bin";
  Path path(::getenv("PATH"));
  Path lnx64o = aieToolsPath / "lib" / "lnx64.o";
  Path dotLib = aieToolsPath / "lnx64" / "tools" / "dot" / "lib";
  Path ldLibraryPath;
  if (char *ldLibraryPath_ = ::getenv("LD_LIBRARY_PATH")) {
    ldLibraryPath = ldLibraryPath_;
  }
  std::string pathEnv = "PATH=" + chessccPath.string() +
                        std::string{sys::EnvPathSeparator} + path.string();
  std::string ldLibEnv = "LD_LIBRARY_PATH=" + lnx64o.string() +
                         std::string{sys::EnvPathSeparator} + dotLib.string() +
                         std::string{sys::EnvPathSeparator} +
                         ldLibraryPath.string();
  std::string rdiDataEnv = "RDI_DATADIR=" + (aieToolsPath / "data").string();
  const char *licenseFile = ::getenv("XILINXD_LICENSE_FILE");
  if (!licenseFile) licenseFile = ::getenv("LM_LICENSE_FILE");
  std::string licenseFileEnv =
      "XILINXD_LICENSE_FILE=" + std::string(licenseFile);
  return {pathEnv, ldLibEnv, rdiDataEnv, licenseFileEnv};
}

std::optional<std::string> dumpStrToDisk(const std::string &payload,
                                         const std::string &outputPath) {
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(outputPath, &errorMessage);
  if (!outputFile) return errorMessage;
  outputFile->os() << payload;
  outputFile->keep();
  return {};
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return fullString.compare(fullString.length() - ending.length(),
                              ending.length(), ending) == 0;
  }
  return false;
}

LogicalResult runTool(
    std::string program, ArrayRef<std::string> args, bool verbose,
    std::optional<std::vector<std::string>> env = std::nullopt,
    std::optional<std::string> userProvidedLogFilename = std::nullopt) {
#if defined(_WIN32)
  if (!hasEnding(program, ".exe")) program = program + ".exe";
#endif  // _WIN32
  if (verbose) {
    llvm::outs() << '\n';
    if (env) {
      llvm::outs() << "Environment variables:";
      for (auto &s : *env) llvm::outs() << " " << s;
      llvm::outs() << "\n";
    }
    llvm::outs() << "Running: \n" << program;
    for (auto &s : args) llvm::outs() << " " << s;
    llvm::outs() << "\n";
  }

  // Check that 'program' is a valid path, if not, fail immediately.
  if (!std::filesystem::exists(program)) {
    llvm::errs() << "Program " << program << " does not exist\n";
    return failure();
  }

  // Run the program, piping any output to a file.
  SmallVector<StringRef, 8> pArgs = {program};
  pArgs.append(args.begin(), args.end());
  SmallVector<char> logPath;
  if (userProvidedLogFilename.has_value()) {
    std::string lfn = userProvidedLogFilename.value();
    logPath.append(lfn.begin(), lfn.end());
    if (!std::filesystem::exists(lfn)) {
      std::ofstream ofs(lfn);
      ofs.close();
    }
  } else {
    std::string prefix{"tmpRunTool"};
    std::string suffix{"Logging"};
    auto errorCode =
        llvm::sys::fs::createTemporaryFile(prefix, suffix, logPath);
    if (errorCode) {
      llvm::errs() << "Failed to create temporary file: " << errorCode.message()
                   << "\n";
      return failure();
    }
  }

  SmallVector<std::optional<StringRef>> redirects;
#ifdef _WIN32
  redirects = {{}, {}, {}};
  // Explicit type but this never actually constructs an ArrayRef
  std::optional<ArrayRef<StringRef>> envSmallVec = std::nullopt;
#else
  std::string logPathStr = std::string(logPath.begin(), logPath.size());
  StringRef logPathRef(logPathStr);
  llvm::SmallVector<llvm::StringRef> envSmallVec;
  if (env) envSmallVec.append(env->begin(), env->end());
  auto tp = std::optional<StringRef>(logPathRef);
  redirects = {tp, tp, tp};
#endif

  bool executionFailed;
  std::string errMsg;

  sys::ProcessStatistics stats_;
  std::optional<sys::ProcessStatistics> optStats = std::move(stats_);

  int exitCode = sys::ExecuteAndWait(program, pArgs, envSmallVec,
                                     /* redirects */ redirects,
                                     /*SecondsToWait*/ 0, /*MemoryLimit*/ 0,
                                     &errMsg, &executionFailed, &optStats);

#ifndef _WIN32
  auto maybeOutputFromFile = [&]() -> std::optional<std::string> {
    std::ifstream t(logPathRef.str());
    std::stringstream buffer;
    if (t.is_open() && t.good()) {
      buffer << t.rdbuf();
      return buffer.str();
    }
    return nullptr;
  }();

  if (!maybeOutputFromFile) {
    llvm::errs() << "Failed to open temporary file " << logPathRef.str()
                 << "\n";
  }
  const std::string &outputFromFile = maybeOutputFromFile.value();
#endif

  if (verbose) {
    std::chrono::microseconds microSecondsTotal = optStats->TotalTime;
    std::chrono::microseconds microSecondsUser = optStats->UserTime;
    std::string exitStatusStr = exitCode == 0 ? "Succeeded" : "Failed";
    llvm::outs() << exitStatusStr
                 << ". Total time = " << microSecondsTotal.count() / 1e6
                 << " [s] and user time = " << microSecondsUser.count() / 1e6
                 << " [s].\n";
    if (exitCode != 0) llvm::outs() << "Exit code : " << exitCode << "\n";
#ifndef _WIN32
    if (!outputFromFile.empty()) {
      llvm::outs() << "The logging in file " << logPathRef.str() << " is:\n";
      llvm::outs() << outputFromFile << "\n";
    }
#endif
  }

  if (exitCode) {
    llvm::errs() << "Failed to run tool: " << program << ". Error: '" << errMsg
                 << "'\n";
#ifndef _WIN32
    llvm::errs() << outputFromFile;
#endif
    return failure();
  }
  return success();
}

static LogicalResult assembleFileUsingPeano(
    const std::string &inputFile, const std::string &outputFile,
    const std::vector<std::string> &extraArgs, Path &_tempDir, Path &peanoDir,
    const std::string &npuVersion, bool verbose) {
  std::vector<std::string> args;
  args.reserve(args.size() + std::distance(extraArgs.begin(), extraArgs.end()));
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());
  // Use O2 by default as this is recommended by peano folks.
  args.emplace_back("-O2");
  // The following flag is needed to prevent peano from inlining memset, which
  // results in slow scalar code for the vectorized zeroization ukernel.
  args.emplace_back("-fno-builtin-memset");
  // The `-ffunction-sections` and `-fdata-sections` flags are needed to put
  // each function and data item into their own section so any unused sections
  // can be discarded later during linking with `-Wl,--gc-sections`.
  args.emplace_back("-ffunction-sections");
  args.emplace_back("-fdata-sections");
  // The `-fstack-size-section` adds stack size metadata to a special section of
  // the object file, so we can retrieve it later for stack size checking
  // purposes.
  args.emplace_back("-fstack-size-section");
  args.emplace_back("-c");
  args.emplace_back(inputFile);
  args.emplace_back("-o");
  args.emplace_back(outputFile);
  if (verbose) args.emplace_back("-v");
  if (failed(runTool((peanoDir / "bin" / "clang").string(), args, verbose))) {
    llvm::errs() << "Failed to assemble " << outputFile << ".o with peano";
    return failure();
  }
  return success();
}

LogicalResult assembleFileUsingChess(const std::string &inputFile,
                                     const std::string &outputFile,
                                     const std::vector<std::string> &extraArgs,
                                     Path &tempDir, Path &vitisDir,
                                     const std::string &npuVersion,
                                     bool verbose) {
  auto [xChessCCExe, args] =
      makeChessArgs(vitisDir, tempDir, npuVersion, verbose);
  args.reserve(args.size() + extraArgs.size());
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());
  args.emplace_back("-c");
  args.emplace_back(inputFile);
  args.emplace_back("-o");
  args.emplace_back(outputFile);
  std::vector<std::string> env = makeChessEnv(vitisDir, npuVersion);
  return runTool(xChessCCExe, args, verbose, env);
}

using FileAssemblerT = std::function<decltype(assembleFileUsingChess)>;

FailureOr<Path> assembleStringUsing(
    const FileAssemblerT &assembler, const std::string &inputFileStr,
    const std::string &inputFileName, const std::string &outputFileName,
    Path &outputDir, const std::vector<std::string> &extraArgs, Path &workDir,
    Path &toolDir, const std::string &npuVersion, bool verbose = false) {
  Path inputFile = workDir / inputFileName;
  if (auto maybeErr = dumpStrToDisk(inputFileStr, inputFile.string());
      maybeErr.has_value()) {
    llvm::errs() << "Failed to dump to disk " << inputFile.string()
                 << " because: " << maybeErr;
    return failure();
  }

  Path outputFile;
  if (!sys::path::is_absolute(outputFileName)) {
    outputFile = Path(outputDir) / outputFileName;
  } else {
    outputFile = outputFileName;
  }
  if (failed(assembler(inputFile.string(), outputFile.string(), extraArgs,
                       workDir, toolDir, npuVersion, verbose))) {
    llvm::errs() << "Failed to assemble " << outputFileName << ".o";
    return failure();
  }
  return outputFile;
}

static auto assembleStringUsingChess =
    std::bind(assembleStringUsing, assembleFileUsingChess, _1, _2, _3, _4, _5,
              _6, _7, _8, _9);

static auto assembleStringUsingPeano =
    std::bind(assembleStringUsing, assembleFileUsingPeano, _1, _2, _3, _4, _5,
              _6, _7, _8, _9);

/// Gets the contents of a uKernel file that has been embedded using
/// `iree_c_embed_data`. The location of the uKernel files are specified in the
/// local CMakeLists.txt and the contents are embedded as const uint8_t arrays
/// during the build process.
FailureOr<std::string> getUkernelFileContent(StringRef fileName) {
  const iree_file_toc_t *file_start = iree_amd_aie_ukernels_create();
  const iree_file_toc_t *file_end = file_start + iree_amd_aie_ukernels_size();
  for (const iree_file_toc_t *file = file_start; file < file_end; ++file) {
    if (fileName == file->name) return StringRef(file->data, file->size).str();
  }
  return failure();
}

/// Retrieves the maximum stack size for the provided program (ELF file) using
/// 'llvm-readelf'.
FailureOr<uint32_t> getMaxStackSizeFromExecutable(const std::string &outputFile,
                                                  bool verbose, Path peanoDir) {
  std::string stackSizesFile = outputFile + ".stacksizes";
  std::vector<std::string> args{outputFile, "--stack-sizes"};
  if (failed(runTool((peanoDir / "bin" / "llvm-readelf").string(), args,
                     verbose, std::nullopt, stackSizesFile))) {
    llvm::errs() << "Failed to get stack sizes with peano\n";
    return failure();
  }
  // Read the contents of the file stackSizesFile.
  std::ifstream stackSizesFileStream(stackSizesFile);
  std::stringstream stackSizesBuffer;
  stackSizesBuffer << stackSizesFileStream.rdbuf();
  std::string stackSizes = stackSizesBuffer.str();
  return mlir::iree_compiler::AMDAIE::detail::getMaxStackSize(stackSizes);
}

// Generate the elf files for the core
LogicalResult generateCoreElfFiles(AIE::DeviceOp deviceOp,
                                   const std::string &objFile, Path &tempDir,
                                   bool useChess, bool useChessForUKernel,
                                   std::optional<Path> vitisDir,
                                   const std::string &targetArch, bool verbose,
                                   Path peanoDir,
                                   const std::string &npuVersion) {
  auto tileOps = deviceOp.getOps<AIE::TileOp>();
  std::string errorMessage;

  // Get all the core ops.
  SmallVector<AIE::CoreOp> coreOps;
  for (AIE::TileOp tileOp : tileOps) {
    AIE::CoreOp coreOp = AIE::getCoreOp(tileOp);
    if (coreOp) coreOps.push_back(coreOp);
  }
  uint32_t nCoreOps = coreOps.size();

  // Keep track of the ukernel object file that has been generated, so that we
  // don't need to regenerate it for every core.
  llvm::DenseMap<StringRef, Path> ukernelObjectNameToPath;
  for (auto iter : llvm::enumerate(coreOps)) {
    // Control logging verbosity: lower verbosing for all but the first core.
    bool verboseForThisIteration = verbose && (iter.index() == 0);
    AIE::CoreOp coreOp = iter.value();
    int col = coreOp.getTileOp().getCol();
    int row = coreOp.getTileOp().getRow();

    if (verbose) {
      llvm::outs() << "Generating elf for core " << 1 + iter.index() << " / "
                   << nCoreOps;
      std::string tail =
          verboseForThisIteration ? "" : ", won't print full log";
      llvm::outs() << tail << ".\n";
    }

    std::string elfFileName;
    if (auto fileAttr = coreOp.getElfFileAttr()) {
      elfFileName = std::string(fileAttr.getValue());
    } else {
      elfFileName = std::string("core_") + std::to_string(col) + "_" +
                    std::to_string(row) + ".elf";
      coreOp.setElfFile(elfFileName);
    }

    Path elfFile = tempDir / elfFileName;
    // For each core, its linkWith attribute is a comma separated list of
    // ukernel object files. Example: linkWith = "matmul.o,zero_fill.o".
    std::optional<StringRef> linkWithStr = coreOp.getLinkWith();
    SmallVector<StringRef> ukernelObjectNames;
    if (linkWithStr.has_value())
      llvm::SplitString(linkWithStr.value(), ukernelObjectNames, ",");
    // Generate all the ukernel object files for this core.
    SmallVector<Path> ukernelObjectFilePaths;
    for (StringRef ukernelObjectName : ukernelObjectNames) {
      // If already exists, skip.
      if (ukernelObjectNameToPath.contains(ukernelObjectName)) continue;
      // Get the ukernel source file name by substituting the '.o' with '.cc'.
      llvm::Regex re("\\.o$");
      std::string ukernelFileName = re.sub(".cc", ukernelObjectName);
      Path ukernelFilePath = Path(npuVersion) /
                             (useChessForUKernel ? "chess" : "peano") /
                             ukernelFileName;
      // Get the ukernel file content as a string.
      FailureOr<std::string> ukernelFileContent =
          getUkernelFileContent(ukernelFilePath.string());
      if (failed(ukernelFileContent)) {
        llvm::errs() << "Failed to get ukernel file content for "
                     << ukernelFilePath.string() << '\n';
        return failure();
      }
      // Generate the ukernel object file using either chess or peano.
      FailureOr<Path> ukernelObjectFilePath;
      if (useChessForUKernel) {
        FailureOr<Path> maybeVitisDir = findVitis(vitisDir, npuVersion);
        if (failed(maybeVitisDir)) {
          llvm::errs() << "compiling ukernels with chess requires Vitis to "
                          "be found"
                       << '\n';
          return failure();
        }
        ukernelObjectFilePath = assembleStringUsingChess(
            /*inputFileStr=*/*ukernelFileContent,
            /*inputFileName=*/ukernelFileName,
            /*outputFileName=*/ukernelObjectName.str(),
            /*outputDir=*/tempDir,
            /*extraArgs=*/std::vector<std::string>{},
            /*workDir=*/tempDir,
            /*vitisDir=*/*maybeVitisDir,
            /*npuVersion*/ npuVersion, verboseForThisIteration);
      } else {
        std::string targetLower = StringRef(targetArch).lower();
        std::vector<std::string> extraArgs{"--target=" + targetLower +
                                           "-none-unknown-elf"};
        ukernelObjectFilePath = assembleStringUsingPeano(
            /*inputFileStr=*/*ukernelFileContent,
            /*inputFileName=*/ukernelFileName,
            /*outputFileName=*/ukernelObjectName.str(),
            /*outputDir=*/tempDir,
            /*extraArgs=*/extraArgs,
            /*workDir=*/tempDir,
            /*vitisDir=*/peanoDir,
            /*npuVersion*/ npuVersion, verboseForThisIteration);
      }
      if (failed(ukernelObjectFilePath)) return failure();
      ukernelObjectNameToPath[ukernelObjectName] = *ukernelObjectFilePath;
    }

    if (useChess) {
      FailureOr<Path> maybeVitisDir = findVitis(vitisDir, npuVersion);
      if (failed(maybeVitisDir)) return failure();
      static constexpr llvm::StringLiteral chessIntrinsicWrapperFileName =
          "chess_intrinsic_wrapper.cpp";
      static constexpr llvm::StringLiteral chessIntrinsicWrapperObjectName =
          "chess_intrinsic_wrapper.o";
      if (!ukernelObjectNameToPath.contains(chessIntrinsicWrapperObjectName)) {
        // Get the chess intrinsic wrapper file content as a string.
        Path chessIntrinsicsFilePath =
            Path(npuVersion) / "chess" / chessIntrinsicWrapperFileName.str();
        FailureOr<std::string> chessIntrinsicWrapperFileContent =
            getUkernelFileContent(chessIntrinsicsFilePath.string());
        if (failed(chessIntrinsicWrapperFileContent)) {
          llvm::errs() << "Failed to get chess intrinsic wrapper file content "
                       << "for " << chessIntrinsicsFilePath.string() << '\n';
          return failure();
        }
        // Generate the chess intrinsic wrapper object file.
        FailureOr<Path> chessIntrinsicsObjFile = assembleStringUsingChess(
            /*inputFileStr=*/*chessIntrinsicWrapperFileContent,
            /*inputFileName=*/chessIntrinsicWrapperFileName.str(),
            /*outputFileName=*/chessIntrinsicWrapperObjectName.str(),
            /*outputDir=*/tempDir,
            /*extraArgs*/ std::vector<std::string>{},
            /*workDir=*/tempDir,
            /*vitisDir=*/*maybeVitisDir,
            /*npuVersion*/ npuVersion, verboseForThisIteration);
        if (failed(chessIntrinsicsObjFile)) return failure();
        ukernelObjectNameToPath[chessIntrinsicWrapperObjectName] =
            *chessIntrinsicsObjFile;
      }

      // Use xbridge (to remove any peano dependency with use-chess option)
      Path bcfPath = tempDir / (elfFileName + ".bcf");

      {
        auto bcfOutput = openOutputFile(bcfPath.string(), &errorMessage);
        if (!bcfOutput) {
          llvm::errs() << "failed to open bcf file because: " << errorMessage;
          return failure();
        }

        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToBCF(
                deviceOp, bcfOutput->os(), col, row))) {
          llvm::errs() << "Failed to generate BCF";
          return failure();
        }
        bcfOutput->keep();
      }

      auto [xChessCCExe, chessArgs] = makeChessArgs(
          *vitisDir, tempDir, npuVersion, verboseForThisIteration);
      chessArgs.emplace_back(objFile);
      chessArgs.emplace_back(
          ukernelObjectNameToPath[chessIntrinsicWrapperObjectName].string());
      for (StringRef ukernelObjectName : ukernelObjectNames) {
        chessArgs.emplace_back(
            ukernelObjectNameToPath[ukernelObjectName].string());
      }
      chessArgs.emplace_back("+l");
      chessArgs.emplace_back(bcfPath.string());
      chessArgs.emplace_back("-o");
      chessArgs.emplace_back(elfFile.string());
      std::vector<std::string> env = makeChessEnv(*vitisDir, npuVersion);
      if (failed(
              runTool(xChessCCExe, chessArgs, verboseForThisIteration, env))) {
        return deviceOp.emitOpError() << "failed to generate elf for core: ("
                                      << col << ", " << row << ")";
      }
    } else {
      Path ldscriptPath = tempDir / (elfFileName + ".ld");
      {
        auto ldscriptOutput =
            openOutputFile(ldscriptPath.string(), &errorMessage);
        if (!ldscriptOutput) {
          llvm::errs() << "Failed to open ldscript file because: "
                       << errorMessage;
          return failure();
        }
        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToLdScript(
                deviceOp, ldscriptOutput->os(), col, row))) {
          return failure();
        }
        ldscriptOutput->keep();
      }

      std::string targetLower = StringRef(targetArch).lower();
      std::string targetTriple = targetLower + "-none-unknown-elf";
      std::vector<std::string> flags;
      flags.emplace_back(objFile);
      for (StringRef ukernelObjectName : ukernelObjectNames)
        flags.emplace_back(ukernelObjectNameToPath[ukernelObjectName].string());
      flags.emplace_back("--target=" + targetTriple);
      flags.emplace_back("-Wl,--gc-sections");

      // Decision to use 'warn' for orphan sections: currently if the preceding
      // call to llc has the flag --stack-size-section, an orphan section
      // is created containing the stack sizes. The linker needs to know how to
      // handle this: options are 'place' or 'warn' or 'error'. 'place' would
      // result in larger binaries. The flag '--exclude-secion' should work
      // but doesn't appear to supported with peano.
      flags.emplace_back("-Wl,--orphan-handling=warn");
#ifdef _WIN32
      // The Windows Peano package ships AIE2P libc/libm as c.lib/m.lib while
      // the clang driver injects GNU-style -lc/-lm. Make per-compilation
      // aliases in the temporary directory instead of mutating the toolchain
      // install. Also ignore host-default .deplibs entries such as libcmt that
      // can be embedded in the packaged crt objects.
      Path peanoTargetLibDir = peanoDir / "lib" / targetTriple;
      for (auto [srcName, aliasName] :
           {std::pair{"c.lib", "libc.a"}, std::pair{"m.lib", "libm.a"}}) {
        Path installedAlias = peanoTargetLibDir / aliasName;
        Path srcPath = peanoTargetLibDir / srcName;
        if (std::filesystem::exists(installedAlias) ||
            !std::filesystem::exists(srcPath)) {
          continue;
        }
        Path aliasPath = tempDir / aliasName;
        std::error_code ec;
        std::filesystem::copy_file(
            srcPath, aliasPath, std::filesystem::copy_options::overwrite_existing,
            ec);
        if (ec) {
          llvm::errs() << "Failed to create Peano library alias "
                       << aliasPath.string() << " from " << srcPath.string()
                       << " because: " << ec.message() << "\n";
          return failure();
        }
      }
      flags.emplace_back("-Wl,-L," + tempDir.string());
      flags.emplace_back("-Wl,--no-dependent-libraries");
#endif
      flags.emplace_back("-Wl,-T," + ldscriptPath.string());
      flags.emplace_back("-o");
      flags.emplace_back(elfFile.string());
      if (verbose) flags.emplace_back("-v");
      // we run clang (ie cc) so that libc, libm, crt0/1 paths are injected
      // automatically into the ld.lld invocation
      if (failed(runTool((peanoDir / "bin" / "clang").string(), flags,
                         verboseForThisIteration))) {
        return failure();
      }

      // If this is not windows, we can do this check. On windows checkTool
      // doesn't pipe logging in the way thay's needed for this to work.
#ifndef _WIN32
      FailureOr<uint32_t> maybeMaxStackSize =
          getMaxStackSizeFromExecutable(elfFile.string(), verbose, peanoDir);
      if (failed(maybeMaxStackSize)) {
        return failure();
      }
      uint32_t stackSize = maybeMaxStackSize.value();
      if (stackSize > coreOp.getStackSize()) {
        llvm::errs() << "An upper bound for the stack size of the core (col="
                     << col << ", row=" << row
                     << "), inferred from the object file, is " << stackSize
                     << " bytes. The assigned memory for the stack is "
                     << coreOp.getStackSize()
                     << " bytes, which is insufficient (" << stackSize << " > "
                     << coreOp.getStackSize() << ").\n";
        return failure();
      }
#endif
    }
  }
  return success();
}

LogicalResult generateCDO(MLIRContext *context, AIE::DeviceOp deviceOp,
                          const Path &tempDir, bool enableCtrlPkt) {
  auto copy = cast<ModuleOp>(deviceOp.getParentOp()->clone());
  deviceOp = *copy.getOps<AIE::DeviceOp>().begin();
  if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToCDODirect(
          deviceOp, tempDir.string(), enableCtrlPkt))) {
    llvm::errs() << "failed to emit CDO";
    return failure();
  }
  copy->erase();
  return success();
}

json::Object makeKernelJSON(const std::string &name, const std::string &id,
                            const std::string &instance) {
  return json::Object{
      {"name", name},
      {"type", "dpu"},
      {"extended-data",
       json::Object{
           {"subtype", "DPU"}, {"functional", "0"}, {"dpu_kernel_id", id}}},
      {"arguments", json::Array{json::Object{{"name", "opcode"},
                                             {"address-qualifier", "SCALAR"},
                                             {"type", "uint64_t"},
                                             {"offset", "0x00"}},
                                json::Object{{"name", "instr"},
                                             {"memory-connection", "SRAM"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x08"}},
                                json::Object{{"name", "ninstr"},
                                             {"address-qualifier", "SCALAR"},
                                             {"type", "uint32_t"},
                                             {"offset", "0x10"}},
                                json::Object{{"name", "bo0"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x14"}},
                                json::Object{{"name", "bo1"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x1c"}},
                                json::Object{{"name", "bo2"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x24"}},
                                json::Object{{"name", "bo3"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x2c"}},
                                json::Object{{"name", "bo4"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x34"}},
                                json::Object{{"name", "bo5"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x3c"}}}},
      {"instances", json::Array{json::Object{{"name", instance}}}}};
}

namespace {

constexpr uint32_t kAxlfSectionBitstream = 0;
constexpr uint32_t kAxlfSectionEmbeddedMetadata = 2;
constexpr uint32_t kAxlfSectionMemTopology = 6;
constexpr uint32_t kAxlfSectionConnectivity = 7;
constexpr uint32_t kAxlfSectionIpLayout = 8;
constexpr uint32_t kAxlfSectionBuildMetadata = 14;
constexpr uint32_t kAxlfSectionAskGroupTopology = 26;
constexpr uint32_t kAxlfSectionAskGroupConnectivity = 27;
constexpr uint32_t kAxlfSectionAiePartition = 32;

constexpr uint8_t kMemDram = 2;
constexpr uint32_t kIpKernel = 1;
constexpr uint32_t kIpPsKernel = 7;
constexpr uint8_t kCdoPrimary = 1;
constexpr uint8_t kCdoPrePost = 3;
constexpr uint16_t kPsSubtypeDpu = 1;
constexpr uint16_t kPsFunctionalDpu = 0;

struct AxlfSectionHeader {
  uint32_t kind = 0;
  char name[16] = {};
  uint64_t offset = 0;
  uint64_t size = 0;
};
static_assert(sizeof(AxlfSectionHeader) == 40);

struct AxlfHeader {
  uint64_t length = 0;
  uint64_t timeStamp = 0;
  uint64_t featureRomTimeStamp = 0;
  uint16_t versionPatch = 0;
  uint8_t versionMajor = 2;
  uint8_t versionMinor = 19;
  uint16_t mode = 0;
  uint16_t actionMask = 0;
  uint8_t interfaceUuid[16] = {};
  char platformVbnv[64] = {};
  uint8_t uuid[16] = {};
  char debugBin[16] = {};
  uint32_t numSections = 0;
};
static_assert(sizeof(AxlfHeader) == 152);

struct AxlfFileHeader {
  char magic[8] = {};
  int32_t signatureLength = -1;
  uint8_t reserved[28] = {};
  uint8_t keyBlock[256] = {};
  uint64_t uniqueId = 0;
  AxlfHeader header;
};
static_assert(sizeof(AxlfFileHeader) == 456);

struct ArrayOffset {
  uint32_t size = 0;
  uint32_t offset = 0;
};
static_assert(sizeof(ArrayOffset) == 8);

struct MemData {
  uint8_t type = 0;
  uint8_t used = 0;
  uint8_t padding[6] = {};
  uint64_t size = 0;
  uint64_t baseAddress = 0;
  uint8_t tag[16] = {};
};
static_assert(sizeof(MemData) == 40);

struct Connection {
  int32_t argIndex = 0;
  int32_t ipLayoutIndex = 0;
  int32_t memDataIndex = 0;
};
static_assert(sizeof(Connection) == 12);

struct IpData {
  uint32_t type = 0;
  uint32_t properties = 0;
  uint64_t baseAddress = 0;
  uint8_t name[64] = {};
};
static_assert(sizeof(IpData) == 80);

struct CdoGroup {
  uint32_t nameOffset = 0;
  uint8_t cdoType = 0;
  uint8_t padding[3] = {};
  uint64_t pdiId = 0;
  ArrayOffset dpuKernelIds;
  ArrayOffset preCdoGroups;
  uint8_t reserved[64] = {};
};
static_assert(sizeof(CdoGroup) == 96);

struct AiePdi {
  uint8_t uuid[16] = {};
  ArrayOffset pdiImage;
  ArrayOffset cdoGroups;
  uint8_t reserved[64] = {};
};
static_assert(sizeof(AiePdi) == 96);

struct AiePartitionInfo {
  uint16_t columnWidth = 0;
  uint8_t padding[6] = {};
  ArrayOffset startColumns;
  uint8_t reserved[72] = {};
};
static_assert(sizeof(AiePartitionInfo) == 88);

struct AiePartition {
  uint8_t schemaVersion = 0;
  uint8_t padding0[3] = {};
  uint32_t nameOffset = 0;
  uint32_t operationsPerCycle = 0;
  uint8_t padding[4] = {};
  uint64_t inferenceFingerprint = 0;
  uint64_t prePostFingerprint = 0;
  AiePartitionInfo info;
  ArrayOffset aiePdi;
  uint32_t kernelCommitId = 0;
  uint8_t reserved[52] = {};
};
static_assert(sizeof(AiePartition) == 184);

struct XclbinSectionPayload {
  uint32_t kind = 0;
  StringRef name;
  std::vector<uint8_t> payload;
};

void appendAlignment(std::vector<uint8_t> &data, size_t alignment = 8) {
  size_t padding = (alignment - data.size() % alignment) % alignment;
  data.insert(data.end(), padding, uint8_t{0});
}

template <typename T>
void appendPod(std::vector<uint8_t> &data, const T &value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  data.insert(data.end(), bytes, bytes + sizeof(T));
}

void appendBytes(std::vector<uint8_t> &data, ArrayRef<uint8_t> bytes) {
  data.insert(data.end(), bytes.begin(), bytes.end());
}

LogicalResult copyCString(MutableArrayRef<uint8_t> dest, StringRef value,
                          const char *fieldName) {
  if (value.size() >= dest.size()) {
    llvm::errs() << fieldName << " is too long for xclbin field: " << value
                 << "\n";
    return failure();
  }
  std::fill(dest.begin(), dest.end(), uint8_t{0});
  std::memcpy(dest.data(), value.data(), value.size());
  return success();
}

LogicalResult copyCString(char *dest, size_t destSize, StringRef value,
                          const char *fieldName) {
  return copyCString(MutableArrayRef<uint8_t>(
                         reinterpret_cast<uint8_t *>(dest), destSize),
                     value, fieldName);
}

FailureOr<uint32_t> appendHeapData(std::vector<uint8_t> &section,
                                   ArrayRef<uint8_t> bytes,
                                   bool align = true) {
  if (section.size() > std::numeric_limits<uint32_t>::max() ||
      bytes.size() >
          std::numeric_limits<uint32_t>::max() - section.size() - 7) {
    llvm::errs() << "xclbin section exceeds 32-bit AIE_PARTITION offsets\n";
    return failure();
  }
  uint32_t offset = static_cast<uint32_t>(section.size());
  appendBytes(section, bytes);
  if (align) appendAlignment(section);
  return offset;
}

FailureOr<uint32_t> appendHeapCString(std::vector<uint8_t> &section,
                                      StringRef value) {
  std::vector<uint8_t> bytes(value.begin(), value.end());
  bytes.push_back(0);
  return appendHeapData(section, bytes);
}

template <typename T>
FailureOr<uint32_t> appendHeapPod(std::vector<uint8_t> &section,
                                  const T &value, bool align = true) {
  ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(&value),
                          sizeof(T));
  return appendHeapData(section, bytes, align);
}

std::array<uint8_t, 16> makeUuidBytes() {
  std::array<uint8_t, 16> uuid = {};
  std::random_device rd;
  std::mt19937_64 rng(rd());
  for (size_t i = 0; i < uuid.size(); i += sizeof(uint64_t)) {
    uint64_t word = rng();
    std::memcpy(uuid.data() + i, &word, sizeof(word));
  }
  uuid[6] = (uuid[6] & 0x0f) | 0x40;
  uuid[8] = (uuid[8] & 0x3f) | 0x80;
  return uuid;
}

FailureOr<uint64_t> parseUInt64(StringRef value, const char *fieldName) {
  uint64_t result = 0;
  if (value.getAsInteger(/*Radix=*/0, result)) {
    llvm::errs() << "failed to parse " << fieldName << " as integer: "
                 << value << "\n";
    return failure();
  }
  return result;
}

std::vector<uint8_t> buildMemTopologySection() {
  std::vector<uint8_t> section;
  int32_t count = 2;
  uint32_t reserved = 0;
  appendPod(section, count);
  appendPod(section, reserved);

  MemData host = {};
  host.type = kMemDram;
  host.used = 1;
  host.size = 0x10000;
  host.baseAddress = 0x4000000;
  (void)copyCString(MutableArrayRef<uint8_t>(host.tag, sizeof(host.tag)),
                    "HOST", "mem tag");
  appendPod(section, host);

  MemData sram = {};
  sram.type = kMemDram;
  sram.used = 1;
  sram.size = 0xc000;
  sram.baseAddress = 0x4000000;
  (void)copyCString(MutableArrayRef<uint8_t>(sram.tag, sizeof(sram.tag)),
                    "SRAM", "mem tag");
  appendPod(section, sram);
  return section;
}

FailureOr<std::vector<uint8_t>> buildIpLayoutSection(
    StringRef kernelName, StringRef dpuName, StringRef instanceName,
    uint64_t kernelId) {
  IpData userIp = {};
  userIp.type = kIpKernel;
  userIp.properties = 0x101;
  userIp.baseAddress = 0x80000;
  std::string userIpName = (kernelName + ":" + instanceName).str();
  if (failed(copyCString(
          MutableArrayRef<uint8_t>(userIp.name, sizeof(userIp.name)),
          userIpName, "IP_LAYOUT user kernel name"))) {
    return failure();
  }

  IpData dpuIp = {};
  dpuIp.type = kIpPsKernel;
  dpuIp.properties = kPsSubtypeDpu | (kPsFunctionalDpu << 4) |
                     ((kernelId & 0xfff) << 16);
  dpuIp.baseAddress = std::numeric_limits<uint64_t>::max();
  std::string dpuIpName = (dpuName + ":" + instanceName).str();
  if (failed(copyCString(
          MutableArrayRef<uint8_t>(dpuIp.name, sizeof(dpuIp.name)), dpuIpName,
          "IP_LAYOUT DPU name"))) {
    return failure();
  }

  std::vector<uint8_t> section;
  int32_t count = 2;
  uint32_t reserved = 0;
  appendPod(section, count);
  appendPod(section, reserved);
  appendPod(section, userIp);
  appendPod(section, dpuIp);
  return section;
}

std::vector<uint8_t> buildConnectivitySection(bool groupConnectivity) {
  std::vector<Connection> connections = {
      {/*argIndex=*/0, /*ipLayoutIndex=*/0, /*memDataIndex=*/0},
      {/*argIndex=*/1, /*ipLayoutIndex=*/0, /*memDataIndex=*/0},
      {/*argIndex=*/2, /*ipLayoutIndex=*/0, /*memDataIndex=*/0},
      {/*argIndex=*/1, /*ipLayoutIndex=*/1, /*memDataIndex=*/0},
      {/*argIndex=*/2, /*ipLayoutIndex=*/1, /*memDataIndex=*/0},
      {/*argIndex=*/3, /*ipLayoutIndex=*/1, /*memDataIndex=*/0},
      {/*argIndex=*/4, /*ipLayoutIndex=*/1, /*memDataIndex=*/0},
      {/*argIndex=*/5, /*ipLayoutIndex=*/1, /*memDataIndex=*/1},
      {/*argIndex=*/7, /*ipLayoutIndex=*/1, /*memDataIndex=*/0},
  };
  if (groupConnectivity) {
    const size_t connectionCount = connections.size();
    connections.reserve(connectionCount * 2);
    for (size_t i = 0; i < connectionCount; ++i) {
      connections.push_back(connections[i]);
    }
  }

  std::vector<uint8_t> section;
  int32_t count = static_cast<int32_t>(connections.size());
  appendPod(section, count);
  for (const Connection &connection : connections) {
    appendPod(section, connection);
  }
  return section;
}

FailureOr<std::vector<uint8_t>> buildAiePartitionSection(ArrayRef<uint8_t> pdi,
                                                         uint64_t kernelId) {
  AiePartition partition = {};
  partition.operationsPerCycle = 2048;
  partition.inferenceFingerprint = 23423;
  partition.prePostFingerprint = 12345;
  partition.info.columnWidth = 4;
  partition.info.startColumns.size = 1;
  partition.aiePdi.size = 1;

  std::vector<uint8_t> section(sizeof(AiePartition), uint8_t{0});

  FailureOr<uint32_t> partitionName = appendHeapCString(section, "QoS");
  if (failed(partitionName)) return failure();
  partition.nameOffset = *partitionName;

  FailureOr<uint32_t> kernelCommitId = appendHeapCString(section, "");
  if (failed(kernelCommitId)) return failure();
  partition.kernelCommitId = *kernelCommitId;

  uint16_t startColumn = 0;
  FailureOr<uint32_t> startColumnsOffset =
      appendHeapPod(section, startColumn, /*align=*/true);
  if (failed(startColumnsOffset)) return failure();
  partition.info.startColumns.offset = *startColumnsOffset;

  AiePdi aiePdi = {};
  std::array<uint8_t, 16> pdiUuid = makeUuidBytes();
  std::memcpy(aiePdi.uuid, pdiUuid.data(), pdiUuid.size());
  aiePdi.pdiImage.size = static_cast<uint32_t>(pdi.size());
  FailureOr<uint32_t> pdiImageOffset =
      appendHeapData(section, pdi, /*align=*/true);
  if (failed(pdiImageOffset)) return failure();
  aiePdi.pdiImage.offset = *pdiImageOffset;

  CdoGroup cdo = {};
  FailureOr<uint32_t> cdoName = appendHeapCString(section, "DPU_PDI_0");
  if (failed(cdoName)) return failure();
  cdo.nameOffset = *cdoName;
  cdo.cdoType = kCdoPrePost;
  cdo.pdiId = 0xf0;
  cdo.dpuKernelIds.size = 1;
  FailureOr<uint32_t> kernelIdsOffset =
      appendHeapPod(section, kernelId, /*align=*/true);
  if (failed(kernelIdsOffset)) return failure();
  cdo.dpuKernelIds.offset = *kernelIdsOffset;
  cdo.preCdoGroups.size = 1;
  uint64_t preCdoGroup = 0xc0;
  FailureOr<uint32_t> preCdoGroupsOffset =
      appendHeapPod(section, preCdoGroup, /*align=*/true);
  if (failed(preCdoGroupsOffset)) return failure();
  cdo.preCdoGroups.offset = *preCdoGroupsOffset;

  aiePdi.cdoGroups.size = 1;
  FailureOr<uint32_t> cdoOffset = appendHeapPod(section, cdo, /*align=*/true);
  if (failed(cdoOffset)) return failure();
  aiePdi.cdoGroups.offset = *cdoOffset;

  FailureOr<uint32_t> pdiOffset =
      appendHeapPod(section, aiePdi, /*align=*/true);
  if (failed(pdiOffset)) return failure();
  partition.aiePdi.offset = *pdiOffset;

  std::memcpy(section.data(), &partition, sizeof(partition));
  return section;
}

std::vector<uint8_t> buildBuildMetadataSection(StringRef kernelName) {
  json::Object xclbin{
      {"xclbin_name", (kernelName + ".link").str()},
      {"user_regions",
       json::Array{json::Object{{"kernels",
                                  json::Array{json::Object{
                                      {"name", kernelName}}}}}}}};
  json::Object metadata{
      {"schema_version", json::Object{{"major", 1}, {"minor", 0}}},
      {"build_metadata", json::Object{{"xclbin", std::move(xclbin)}}}};
  std::string metadataStr =
      llvm::formatv("{0:2}", json::Value(std::move(metadata))).str();
  return std::vector<uint8_t>(metadataStr.begin(), metadataStr.end());
}

std::string xmlEscape(StringRef value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped.push_back(c);
        break;
    }
  }
  return escaped;
}

std::vector<uint8_t> buildEmbeddedMetadataSection(StringRef kernelName,
                                                  StringRef dpuName,
                                                  StringRef instanceName,
                                                  uint64_t kernelId) {
  std::string kernel = xmlEscape(kernelName);
  std::string dpu = xmlEscape(dpuName);
  std::string instance = xmlEscape(instanceName);
  std::string xml;
  llvm::raw_string_ostream os(xml);
  os << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
  os << "<project name=\"" << kernel << ".link\">\n";
  os << "  <platform vendor=\"xilinx\" boardid=\"v1\" name=\"ipu\" "
        "featureRomTime=\"0\">\n";
  os << "    <version major=\"0\" minor=\"0\"/>\n";
  os << "    <device name=\"fpga0\" "
        "fpgaDevice=\"virtex7:xc7vx485t:ffg1157:-1\" addrWidth=\"0\">\n";
  os << "      <core name=\"OCL_REGION_0\" target=\"hw_em\" "
        "type=\"clc_region\" clockFreq=\"0MHz\" numComputeUnits=\"1\">\n";
  os << "        <kernelClocks>\n";
  os << "          <clock port=\"DATA_CLK\" frequency=\"500.000000MHz\"/>\n";
  os << "        </kernelClocks>\n";
  os << "        <kernel name=\"" << kernel
     << "\" language=\"c\" hwControlProtocol=\"ap_ctrl_chain\">\n";
  os << "          <port name=\"S_AXI_CONTROL\" mode=\"slave\" "
        "range=\"0x1000\" dataWidth=\"32\" portType=\"addressable\" "
        "base=\"0x0\"/>\n";
  os << "          <instance name=\"" << instance << "\">\n";
  os << "            <addrRemap base=\"0x00080000\" range=\"0x10000\" "
        "port=\"S_AXI_CONTROL\"/>\n";
  os << "          </instance>\n";
  os << "        </kernel>\n";
  os << "        <kernel name=\"" << dpu
     << "\" language=\"c\" type=\"dpu\">\n";
  os << "          <extended-data subtype=\"1\" functional=\"0\" "
        "dpu_kernel_id=\""
     << llvm::formatv("{0:x}", kernelId) << "\"/>\n";
  os << "          <arg name=\"opcode\" addressQualifier=\"0\" id=\"0\" "
        "size=\"0x8\" offset=\"0x00\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"uint64_t\"/>\n";
  os << "          <arg name=\"ifm\" addressQualifier=\"1\" id=\"1\" "
        "size=\"0x8\" offset=\"0x08\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <arg name=\"param\" addressQualifier=\"1\" id=\"2\" "
        "size=\"0x8\" offset=\"0x10\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <arg name=\"ofm\" addressQualifier=\"1\" id=\"3\" "
        "size=\"0x8\" offset=\"0x18\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <arg name=\"inter\" addressQualifier=\"1\" id=\"4\" "
        "size=\"0x8\" offset=\"0x20\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <arg name=\"instruct\" addressQualifier=\"1\" id=\"5\" "
        "size=\"0x8\" offset=\"0x28\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <arg name=\"nistruct\" addressQualifier=\"0\" id=\"6\" "
        "size=\"0x4\" offset=\"0x30\" hostOffset=\"0x0\" hostSize=\"0x4\" "
        "type=\"uint32_t\"/>\n";
  os << "          <arg name=\"mc\" addressQualifier=\"1\" id=\"7\" "
        "size=\"0x8\" offset=\"0x34\" hostOffset=\"0x0\" hostSize=\"0x8\" "
        "type=\"char *\"/>\n";
  os << "          <instance name=\"" << instance << "\"/>\n";
  os << "        </kernel>\n";
  os << "      </core>\n";
  os << "    </device>\n";
  os << "  </platform>\n";
  os << "</project>\n";
  os.flush();
  return std::vector<uint8_t>(xml.begin(), xml.end());
}

LogicalResult writeAxlf(StringRef outputPath,
                        ArrayRef<XclbinSectionPayload> sections) {
  if (sections.empty() || sections.size() > std::numeric_limits<uint32_t>::max()) {
    llvm::errs() << "invalid xclbin section count\n";
    return failure();
  }

  AxlfFileHeader fileHeader = {};
  std::memcpy(fileHeader.magic, "xclbin2", 8);
  fileHeader.signatureLength = -1;
  std::memset(fileHeader.reserved, 0xff, sizeof(fileHeader.reserved));
  std::memset(fileHeader.keyBlock, 0xff, sizeof(fileHeader.keyBlock));
  uint64_t now = static_cast<uint64_t>(
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  fileHeader.uniqueId = now;
  fileHeader.header.timeStamp = now;
  fileHeader.header.mode = 4;
  fileHeader.header.actionMask = 1;
  if (failed(copyCString(fileHeader.header.platformVbnv,
                         sizeof(fileHeader.header.platformVbnv),
                         "xilinx_v1_ipu_0_0", "AXLF platform VBNV"))) {
    return failure();
  }
  fileHeader.header.numSections = static_cast<uint32_t>(sections.size());
  std::array<uint8_t, 16> axlfUuid = makeUuidBytes();
  std::memcpy(fileHeader.header.uuid, axlfUuid.data(), axlfUuid.size());

  std::vector<AxlfSectionHeader> sectionHeaders(sections.size());
  uint64_t currentOffset =
      sizeof(AxlfFileHeader) + sizeof(AxlfSectionHeader) * sections.size();
  for (size_t i = 0; i < sections.size(); ++i) {
    currentOffset += (8 - currentOffset % 8) % 8;
    sectionHeaders[i].kind = sections[i].kind;
    if (failed(copyCString(sectionHeaders[i].name,
                           sizeof(sectionHeaders[i].name), sections[i].name,
                           "AXLF section name"))) {
      return failure();
    }
    sectionHeaders[i].offset = currentOffset;
    sectionHeaders[i].size = sections[i].payload.size();
    currentOffset += sections[i].payload.size();
  }
  fileHeader.header.length = currentOffset;

  std::vector<uint8_t> xclbin;
  xclbin.reserve(static_cast<size_t>(currentOffset));
  appendPod(xclbin, fileHeader);
  for (const AxlfSectionHeader &sectionHeader : sectionHeaders) {
    appendPod(xclbin, sectionHeader);
  }
  for (size_t i = 0; i < sections.size(); ++i) {
    appendAlignment(xclbin);
    if (xclbin.size() != sectionHeaders[i].offset) {
      llvm::errs() << "internal xclbin offset mismatch\n";
      return failure();
    }
    appendBytes(xclbin, sections[i].payload);
  }

  std::ofstream output(outputPath.str(), std::ios::binary);
  if (!output.is_open()) {
    llvm::errs() << "failed to open context xclbin for writing: "
                 << outputPath << "\n";
    return failure();
  }
  output.write(reinterpret_cast<const char *>(xclbin.data()), xclbin.size());
  if (!output.good()) {
    llvm::errs() << "failed to write context xclbin: " << outputPath << "\n";
    return failure();
  }
  return success();
}

LogicalResult generateAMDXDNAContextXCLBin(
    const std::string &output, const Path &pdiPath,
    const std::string &xclBinKernelID, const std::string &xclBinKernelName,
    const std::string &xclBinInstanceName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> pdiBuffer =
      MemoryBuffer::getFile(pdiPath.string(), /*IsText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!pdiBuffer) {
    llvm::errs() << "failed to open PDI for context xclbin: "
                 << pdiBuffer.getError().message() << "\n";
    return failure();
  }
  if ((*pdiBuffer)->getBufferSize() > std::numeric_limits<uint32_t>::max()) {
    llvm::errs() << "PDI is too large for AIE_PARTITION metadata\n";
    return failure();
  }

  FailureOr<uint64_t> kernelId =
      parseUInt64(xclBinKernelID, "xclbin kernel id");
  if (failed(kernelId)) return failure();

  ArrayRef<uint8_t> pdiBytes(
      reinterpret_cast<const uint8_t *>((*pdiBuffer)->getBufferStart()),
      (*pdiBuffer)->getBufferSize());
  constexpr StringLiteral kDpuName = "DPU_PDI_0";
  FailureOr<std::vector<uint8_t>> ipLayout =
      buildIpLayoutSection(xclBinKernelName, kDpuName, xclBinInstanceName,
                           *kernelId);
  if (failed(ipLayout)) return failure();
  FailureOr<std::vector<uint8_t>> aiePartition =
      buildAiePartitionSection(pdiBytes, *kernelId);
  if (failed(aiePartition)) return failure();

  std::vector<XclbinSectionPayload> sections;
  sections.push_back({kAxlfSectionBitstream, "dummy_bitstream", {}});
  sections.push_back({kAxlfSectionBuildMetadata, "BUILD_METADATA",
                      buildBuildMetadataSection(xclBinKernelName)});
  sections.push_back(
      {kAxlfSectionMemTopology, "MEM_TOPOLOGY", buildMemTopologySection()});
  sections.push_back({kAxlfSectionAskGroupTopology, "",
                      buildMemTopologySection()});
  sections.push_back(
      {kAxlfSectionIpLayout, "IP_LAYOUT", std::move(*ipLayout)});
  sections.push_back({kAxlfSectionConnectivity, "CONNECTIVITY",
                      buildConnectivitySection(/*groupConnectivity=*/false)});
  sections.push_back(
      {kAxlfSectionAskGroupConnectivity, "conn",
       buildConnectivitySection(/*groupConnectivity=*/true)});
  sections.push_back({kAxlfSectionAiePartition, "AIE_PARTITION",
                      std::move(*aiePartition)});
  // AXLF stores section names in a 16-byte C string, so the canonical
  // EMBEDDED_METADATA spelling does not fit. Consumers key on the numeric kind.
  sections.push_back({kAxlfSectionEmbeddedMetadata, "EMBEDDED_META",
                      buildEmbeddedMetadataSection(
                          xclBinKernelName, kDpuName, xclBinInstanceName,
                          *kernelId)});
  return writeAxlf(output, sections);
}

}  // namespace

LogicalResult generatePDI(const std::string &Output, const Path &tempDir,
                          bool enableCtrlPkt) {
  std::string errorMessage;
  // Create design.bif.
  Path designBifFile = tempDir / "design.bif";
  {
    auto designBifOut = openOutputFile(designBifFile.string(), &errorMessage);
    if (!designBifOut) {
      llvm::errs() << "failed to open design.bif because: " << errorMessage;
      return failure();
    }

    designBifOut->os() << "all:\n"
                       << "{\n"
                       << "  id_code = 0x14ca8093\n"
                       << "  extended_id_code = 0x01\n"
                       << "  image\n"
                       << "  {\n"
                       << "    name=aie_image, id=0x1c000000\n"
                       << "    { type=cdo\n";

    if (enableCtrlPkt) {
      // When control packets are enabled, only the switch configuration
      // binary is needed and all other binaries are skipped
      designBifOut->os() << "      file=" << tempDir.string()
                         << "/aie_cdo_switches.bin\n";
    } else {
      // The order of binary is critical for correct behavior, so modify with
      // care!
      designBifOut->os() << "      file=" << tempDir.string()
                         << "/aie_cdo_elfs.bin\n"
                         << "      file=" << tempDir.string()
                         << "/aie_cdo_init.bin\n"
                         << "      file=" << tempDir.string()
                         << "/aie_cdo_switches.bin\n"
                         << "      file=" << tempDir.string()
                         << "/aie_cdo_enable.bin\n";
    }

    designBifOut->os() << "    }\n"
                       << "  }\n"
                       << "}";
    designBifOut->keep();
  }

  // Execute the bootgen command.
  {
#ifdef IREE_AMD_AIE_BOOTGEN_EXECUTABLE
    std::vector<std::string> flags = {"-arch", "versal", "-image",
                                      designBifFile.string(), "-o", Output,
                                      "-w"};
    if (failed(runTool(IREE_AMD_AIE_BOOTGEN_EXECUTABLE, flags,
                       /*verbose=*/false))) {
      llvm::errs() << "failed to execute bootgen";
      return failure();
    }
#else
    // first element is empty string because iree_aie_bootgen_main
    // is the main of bootgen.exe (and argv[0] is typically the name of the exe)
    std::vector<std::string> flags = {
        "",   "-arch", "versal", "-image", designBifFile.string(),
        "-o", Output,  "-w"};
    std::vector<char *> cstrings;
    cstrings.reserve(flags.size());
    for (const auto &inputFlag : flags) {
      cstrings.push_back(const_cast<char *>(inputFlag.c_str()));
    }
    if (iree_aie_bootgen_main(cstrings.size(),
                              const_cast<const char **>(&cstrings[0]))) {
      llvm::errs() << "failed to execute bootgen";
      return failure();
    }
#endif  // IREE_AMD_AIE_BOOTGEN_EXECUTABLE
  }

  return success();
}

LogicalResult generateXCLBin(const std::string &Output, const Path &tempDir,
                             const std::string &xclBinKernelID,
                             const std::string &xclBinKernelName,
                             const std::string &xclBinInstanceName,
                             const Path &amdAIEInstallDir, bool verbose,
                             const std::optional<std::string> &inputXclbin,
                             bool enableCtrlPkt) {
  std::string errorMessage;
  // Create mem_topology.json.
  Path memTopologyJsonFile = tempDir / "mem_topology.json";
  {
    std::string memTopologyData = R"({
      "mem_topology": {
          "m_count": "2",
          "m_mem_data": [
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0x10000",
                  "m_tag": "HOST",
                  "m_base_address": "0x4000000"
              },
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0xc000",
                  "m_tag": "SRAM",
                  "m_base_address": "0x4000000"
              }
          ]
      }
    })";
    if (auto maybeErr =
            dumpStrToDisk(memTopologyData, memTopologyJsonFile.string());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk mem_topology.json because: "
                   << *maybeErr;
      return failure();
    }
  }

  // Create aie_partition.json.
  Path aiePartitionJsonFile = tempDir / "aie_partition.json";
  {
    std::string uuidStr = uuid::getUUIDString();
    std::string aiePartitionJsonData = R"(
      {
        "aie_partition": {
          "name": "QoS",
          "operations_per_cycle": "2048",
          "inference_fingerprint": "23423",
          "pre_post_fingerprint": "12345",
          "partition": {
            "column_width": 4,
            "start_columns": [1]
          },
          "PDIs": [
            {
              "uuid": ")" + uuidStr + R"(",
              "file_name": "./design.pdi",
              "cdo_groups": [
                {
                  "name": "DPU",
                  "type": "PRIMARY",
                  "pdi_id": "0x01",
                  "dpu_kernel_ids": [
                    ")" + xclBinKernelID +
                                       R"("
                  ],
                  "pre_cdo_groups": [
                    "0xC1"
                  ]
                }
              ]
            }
          ]
        }
      }
    )";
    if (auto maybeErr =
            dumpStrToDisk(aiePartitionJsonData, aiePartitionJsonFile.string());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk aie_partition.json because: "
                   << *maybeErr;
      return failure();
    }
  }

  Path kernelsJsonFile = tempDir / "kernels.json";
  {
    // TODO: Support for multiple kernels
    json::Object kernelsData{
        {"ps-kernels",
         json::Object{{"kernels", json::Array{makeKernelJSON(
                                      xclBinKernelName, xclBinKernelID,
                                      xclBinInstanceName)}}}}};

    auto kernelStr =
        llvm::formatv("{0:2}", json::Value(std::move(kernelsData)));
    if (auto maybeErr = dumpStrToDisk(kernelStr, kernelsJsonFile.string());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk kernels.json because: "
                   << *maybeErr;
      return failure();
    }
  }

  if (failed(generatePDI((tempDir / "design.pdi").string(), tempDir,
                         enableCtrlPkt))) {
    return failure();
  }

  std::vector<std::string> flags;
  // Execute the xclbinutil command.
  std::string memArg = "MEM_TOPOLOGY:JSON:" + memTopologyJsonFile.string();
  std::string partArg = "AIE_PARTITION:JSON:" + aiePartitionJsonFile.string();
  FailureOr<Path> xclbinutilBin =
      findAMDAIETool("iree-aie-xclbinutil", amdAIEInstallDir);

  if (failed(xclbinutilBin)) return failure();

  if (!inputXclbin) {
    flags.insert(flags.end(), {"--add-replace-section", memArg});
  } else {
    // Create aie_partition.json.
    Path aieInputPartitionJsonFile = tempDir / "aie_input_partition.json";
    std::string inputPartArg =
        "AIE_PARTITION:JSON:" + aieInputPartitionJsonFile.string();
    std::vector<std::string> inputFlags{"--dump-section", inputPartArg,
                                        "--force", "--input", *inputXclbin};

    if (failed(runTool(xclbinutilBin.value().string(), inputFlags, verbose))) {
      llvm::errs() << "failed to execute xclbinutil";
      return failure();
    }
    auto aieInputPartitionOut =
        openInputFile(aieInputPartitionJsonFile.string(), &errorMessage);
    if (!aieInputPartitionOut) {
      llvm::errs() << "failed to open aie_input_partition.json because: "
                   << errorMessage;
      return failure();
    }
    Expected<json::Value> aieInputPartitionOutValue =
        llvm::json::parse(aieInputPartitionOut->getBuffer());
    json::Array *aieInputPartionPDIs;
    aieInputPartionPDIs = aieInputPartitionOutValue->getAsObject()
                              ->getObject("aie_partition")
                              ->getArray("PDIs");
    auto aiePartitionOut =
        openInputFile(aiePartitionJsonFile.string(), &errorMessage);
    if (!aiePartitionOut) {
      llvm::errs() << "failed to open aie aie_input_partition.json for "
                      "output because: "
                   << errorMessage;
      return failure();
    }
    llvm::Expected<llvm::json::Value> aiePartitionOutValue =
        llvm::json::parse(aiePartitionOut->getBuffer());
    json::Array *aiePartionPDIs;
    aiePartionPDIs = aiePartitionOutValue->getAsObject()
                         ->getObject("aie_partition")
                         ->getArray("PDIs");
    aieInputPartionPDIs->insert(aieInputPartionPDIs->end(),
                                aiePartionPDIs->begin(), aiePartionPDIs->end());
    // rewrite aie partion json file
    if (auto maybeErr =
            dumpStrToDisk(formatv("{0:2}", *aieInputPartitionOutValue),
                          aiePartitionJsonFile.string());
        maybeErr.has_value()) {
      llvm::errs()
          << "failed to dump to disk aie_input_partition.json because: "
          << errorMessage;
      return failure();
    }
    flags.insert(flags.end(), {"--input", *inputXclbin});
  }
  flags.insert(flags.end(), {"--add-kernel", kernelsJsonFile.string(),
                             "--add-replace-section", partArg, "--force",
                             "--output", std::string(Output)});

  return runTool(xclbinutilBin.value().string(), flags, verbose);
}

void addLowerToLLVMPasses(OpPassManager &pm) {
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(opts));
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(
      mlir::iree_compiler::AMDAIE::createAMDAIERemoveWrapFlagFromGepPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

LogicalResult generateUnifiedObject(
    MLIRContext *context, AIE::DeviceOp deviceOp, const std::string &outputFile,
    bool printIRBeforeAll, bool printIRAfterAll, bool printIRModuleScope,
    bool timing, bool useChess, bool verbose, Path &tempDir,
    std::optional<Path> vitisDir, const std::string &targetArch, Path &peanoDir,
    const std::string &npuVersion, const std::string &additionalPeanoOptFlags) {
  assert(deviceOp->getParentOp() && isa<ModuleOp>(deviceOp->getParentOp()) &&
         "DeviceOp must be in a module parent");

  PassManager pm(context, ModuleOp::getOperationName());
  applyConfigToPassManager(pm, printIRBeforeAll, printIRAfterAll,
                           printIRModuleScope, timing);

  mlir::iree_compiler::AMDAIE::AMDAIECoreToStandardOptions options;
  options.lowerToChess = useChess;
  pm.addPass(
      mlir::iree_compiler::AMDAIE::createAMDAIECoreToStandardPass(options));
  addLowerToLLVMPasses(pm);

  if (verbose) {
    llvm::outs() << "\nRunning: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  ModuleOp moduleOpCopy = cast<ModuleOp>(deviceOp->getParentOp()).clone();
  if (failed(pm.run(moduleOpCopy))) {
    llvm::errs() << "Failed to lower to LLVM";
    return failure();
  }

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(moduleOpCopy, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVMIR";
    return failure();
  }

  std::string inputLLStr;
  {
    llvm::raw_string_ostream rso(inputLLStr);
    llvmModule->print(rso, nullptr);
  }

  if (useChess) {
    FailureOr<Path> maybeVitisDir = findVitis(vitisDir, npuVersion);
    if (failed(maybeVitisDir)) return failure();
    FailureOr<Path> objFilePath = assembleStringUsingChess(
        /*inputFileStr=*/inputLLStr,
        /*inputFileName=*/"input.ll",
        /*outputFileName=*/outputFile,
        /*outputDir=*/tempDir,
        /*extraArgs*/ std::vector<std::string>{},
        /*workDir=*/tempDir,
        /*vitisDir=*/*maybeVitisDir,
        /*npuVersion*/ npuVersion,
        /*verbose=*/verbose);
    if (failed(objFilePath)) {
      return failure();
    }
  } else {
    std::string LLVMIRFile = (tempDir / "input.ll").string();
    if (auto maybeErr = dumpStrToDisk(inputLLStr, LLVMIRFile);
        maybeErr.has_value()) {
      llvm::errs() << "Failed to dump to disk input.ll"
                   << " because: " << maybeErr;
      return failure();
    }
    Path peanoOptBin = peanoDir / "bin" / "opt";
    Path peanoLLCBin = peanoDir / "bin" / "llc";
    Path peanoReadElfBin = peanoDir / "bin" / "llvm-readelf";

    std::string OptLLVMIRFile = (tempDir / "input.opt.ll").string();

    FailureOr<std::vector<std::string>> maybeAdditionalPeanoArgs =
        mlir::iree_compiler::AMDAIE::detail::flagStringToVector(
            additionalPeanoOptFlags);
    if (failed(maybeAdditionalPeanoArgs)) {
      llvm::errs() << "Failed to parse additional peano args\n";
      return failure();
    }

    FailureOr<std::vector<std::string>> maybePeanoArgs =
        mlir::iree_compiler::AMDAIE::detail::makePeanoOptArgs(
            maybeAdditionalPeanoArgs.value());
    if (failed(maybePeanoArgs)) {
      llvm::errs() << "Failed to make peano opt args\n";
      return failure();
    }
    std::vector<std::string> peanoArgs = maybePeanoArgs.value();
    // Source file, IR to optimize
    peanoArgs.emplace_back("-S");
    peanoArgs.emplace_back(LLVMIRFile);
    // Output file, optimized IR
    peanoArgs.emplace_back("-o");
    peanoArgs.emplace_back(OptLLVMIRFile);

    if (failed(runTool(peanoOptBin.string(), peanoArgs, verbose))) {
      llvm::errs() << "Failed to optimize ll with peano\n";
      llvm::errs() << "Using peano at provided path: '" << peanoDir.string()
                   << "'\n";
      return failure();
    }

    std::vector<std::string> llcArgs{OptLLVMIRFile,
                                     "-O2",
                                     "--march=" + StringRef(targetArch).lower(),
                                     "--function-sections",
                                     "--filetype=obj",
                                     "-o",
                                     outputFile,
                                     "--stack-size-section"};

    if (failed(runTool(peanoLLCBin.string(), llcArgs, verbose))) {
      llvm::errs() << "Failed to assemble ll with peano\n";
      return failure();
    }
  }

  moduleOpCopy->erase();
  return success();
}

}  // namespace

namespace mlir::iree_compiler::AMDAIE {

/// Pipeline to generate control packets from `xilinx::aie::device`, and dump
/// them into files.
LogicalResult generateControlPackets(
    MLIRContext *context, AIE::DeviceOp deviceOp, const Path &tempDirPath,
    StringRef ctrlpktInstPath, StringRef ctrlpktSeqPath, bool printIRBeforeAll,
    bool printIRAfterAll, bool printIRModuleScope, bool timing) {
  assert(deviceOp->getParentOp() && isa<ModuleOp>(deviceOp->getParentOp()) &&
         "DeviceOp must be in a module parent");
  PassManager pm(context, ModuleOp::getOperationName());
  applyConfigToPassManager(pm, printIRBeforeAll, printIRAfterAll,
                           printIRModuleScope, timing);
  // Assuming the ELF files have already been generated and are stored in
  // `tempDirPath`, use aie-rt to generate control packets.
  {
    AMDAIEConvertDeviceToControlPacketsOptions options;
    options.pathToElfs = tempDirPath.string();
    pm.addPass(createAMDAIEConvertDeviceToControlPacketsPass(options));
  }
  // TODO (zhewen): avoid regeneration?
  // Regenerate the overlay for sending control packets.
  {
    AMDAIEGenerateControlOverlayOptions options;
    options.routeShimToTileCtrl = true;
    pm.addPass(createAMDAIEGenerateControlOverlayPass(options));
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
  }
  // TODO (zhewen): avoid regeneration?
  // Regenerate the flows and packet ids.
  pm.addPass(createAMDAIEConnectionToFlowPass());
  pm.addPass(createAMDAIEAssignPacketIdsPass());
  // Extract the DMA instructions and the DMA data from the control packets.
  pm.addPass(createAMDAIESplitControlPacketDataPass());
  pm.addPass(createAMDAIEControlPacketToNpuDmaPass());
  pm.addPass(createAMDAIEAssignNpuDmaBdIdsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  // Optimize the controlcode size.
  pm.addPass(createAMDAIENpuDmaToHalfDmaCpyNdPass());
  pm.addPass(createAMDAIEInsertDmaBdChainPass());
  pm.addPass(createAMDAIEFoldDmaWaitsPass());
  // Lower the DMA instructions for sending control packets.
  pm.addPass(createAMDAIEControlCodeLoweringPass());
  pm.addPass(createAMDAIEControlCodeToTransactionPass());

  // Run the pipeline.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(deviceOp);
  ModuleOp moduleOpCopy = cast<ModuleOp>(deviceOp->getParentOp()).clone();
  moduleOpCopy->setAttr("hal.executable.target", targetAttr);
  if (failed(pm.run(moduleOpCopy))) {
    llvm::errs() << "Failed to lower to control packets \n";
    return failure();
  }

  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  moduleOpCopy.walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  if (workgroupOps.size() != 1) {
    llvm::errs() << "Expected exactly one workgroup op, found "
                 << workgroupOps.size() << "\n";
    return failure();
  }
  // Dump the control packets sequence (i.e., the data inside the control
  // packets) to a file.
  if (failed(emitDenseArrayAttrToFile(workgroupOps[0], "ctrlpkt_sequence",
                                      ctrlpktSeqPath))) {
    llvm::errs() << "Failed to emit control packets sequence \n";
    return failure();
  }
  // Dump the control packets DMA instructions to a file.
  if (failed(emitDenseArrayAttrToFile(workgroupOps[0], "npu_instructions",
                                      ctrlpktInstPath))) {
    llvm::errs() << "Failed to emit control packets instructions \n";
    return failure();
  }
  return success();
}

LogicalResult emitDenseArrayAttrToFile(Operation *op, StringRef attrName,
                                       StringRef fileName) {
  // Get the attribute from the operation.
  auto maybeAttr = op->getAttrOfType<DenseUI32ResourceElementsAttr>(attrName);
  if (!maybeAttr)
    return op->emitError() << "Failed to get attribute " << attrName << "\n";
  // Get the array ref from the attribute.
  std::optional<ArrayRef<uint32_t>> maybeArrayRef =
      maybeAttr.tryGetAsArrayRef();
  if (!maybeArrayRef) {
    return op->emitError() << "Failed to get values for " << attrName
                           << " in tryGetAsArrayRef \n";
  }
  // Open the output file.
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(fileName, &errorMessage);
  if (!output) {
    llvm::errs() << "Failed to open " << fileName
                 << " for writing because: " << errorMessage << "\n";
    return failure();
  }
  output->keep();
  // Write the values to the output file.
  for (int i = 0; i < maybeArrayRef->size() - 1; ++i) {
    output->os() << llvm::format("%08X\n", maybeArrayRef->operator[](i));
  }
  // Don't emit empty line at the end.
  output->os() << llvm::format("%08X", maybeArrayRef->back());

  return success();
}

LogicalResult aie2xclbin(
    MLIRContext *ctx, AIE::DeviceOp deviceOp,
    const std::optional<std::string> &outputNpuInstPath,
    const std::optional<std::string> &outputCtrlPktInstPath,
    const std::optional<std::string> &outputCtrlPktSeqPath,
    const std::string &artifactPath,
    const std::optional<std::string> &contextXclbinPath,
    bool printIRBeforeAll, bool printIRAfterAll, bool printIRModuleScope,
    bool timing, const std::string &tempDir, bool useChess,
    bool useChessForUKernel, bool verbose,
    const std::optional<std::string> &vitisDir, const std::string &targetArch,
    const std::string &npuVersion, const std::string &peanoDir,
    const mlir::iree_compiler::AMDAIE::AMDAIEOptions::DeviceHAL deviceHal,
    const std::string &xclBinKernelID, const std::string &xclBinKernelName,
    const std::string &xclBinInstanceName, const std::string &amdAIEInstallDir,
    const std::optional<std::string> &InputXCLBin,
    const std::string &additionalPeanoOptFlags, bool enableCtrlPkt) {
  if (outputNpuInstPath.has_value() &&
      failed(emitDenseArrayAttrToFile(deviceOp, "npu_instructions",
                                      outputNpuInstPath.value()))) {
    return failure();
  }

  Path tempDirPath{tempDir};
  tempDirPath.make_preferred();
  Path peanoDirPath{peanoDir};
  peanoDirPath.make_preferred();
  std::optional<Path> vitisDirPath{vitisDir};
  if (vitisDirPath) vitisDirPath->make_preferred();

  Path unifiedObj = tempDirPath / "input.o";
  if (failed(generateUnifiedObject(
          ctx, deviceOp, unifiedObj.string(), printIRBeforeAll, printIRAfterAll,
          printIRModuleScope, timing, useChess, verbose, tempDirPath,
          vitisDirPath, targetArch, peanoDirPath, npuVersion,
          additionalPeanoOptFlags))) {
    llvm::errs() << "Failed to generate unified object\n";
    return failure();
  }

  if (failed(generateCoreElfFiles(deviceOp, unifiedObj.string(), tempDirPath,
                                  useChess, useChessForUKernel, vitisDirPath,
                                  targetArch, verbose, peanoDir, npuVersion))) {
    llvm::errs() << "Failed to generate core ELF file(s)\n";
    return failure();
  }

  if (enableCtrlPkt &&
      failed(generateControlPackets(
          ctx, deviceOp, tempDirPath, outputCtrlPktInstPath.value(),
          outputCtrlPktSeqPath.value(), printIRBeforeAll, printIRAfterAll,
          printIRModuleScope, timing))) {
    llvm::errs() << "Failed to generate control packets MLIR file\n";
    return failure();
  }

  if (failed(generateCDO(ctx, deviceOp, tempDirPath, enableCtrlPkt))) {
    llvm::errs() << "Failed to generate CDO\n";
    return failure();
  }

  Path pdiPath = tempDirPath / "design.pdi";
  if (failed(generatePDI(pdiPath.string(), tempDirPath, enableCtrlPkt))) {
    llvm::errs() << "Failed to generate PDI\n";
    return failure();
  }

  if (deviceHal == AMDAIEOptions::DeviceHAL::AMDXDNA) {
    std::error_code ec;
    if (!std::filesystem::copy_file(
            pdiPath, artifactPath,
            std::filesystem::copy_options::overwrite_existing, ec)) {
      llvm::errs() << "Failed to copy file because: " << ec.message() << "\n";
      return failure();
    }
    if (contextXclbinPath &&
        failed(generateAMDXDNAContextXCLBin(
            *contextXclbinPath, pdiPath, xclBinKernelID, xclBinKernelName,
            xclBinInstanceName))) {
      llvm::errs() << "Failed to generate AMDXDNA context XCLBin\n";
      return failure();
    }
    return success();
  }

  assert(deviceHal == AMDAIEOptions::DeviceHAL::XRT &&
         "generating XCLBin for non-XRT HAL");
  if (failed(generateXCLBin(artifactPath, tempDirPath, xclBinKernelID,
                            xclBinKernelName, xclBinInstanceName,
                            amdAIEInstallDir, verbose, InputXCLBin,
                            enableCtrlPkt))) {
    llvm::errs() << "Failed to generate XCLBin\n";
    return failure();
  }

  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
