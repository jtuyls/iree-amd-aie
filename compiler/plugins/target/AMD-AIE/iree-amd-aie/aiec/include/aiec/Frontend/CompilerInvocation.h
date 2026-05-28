// CompilerInvocation.h - one compilation's options, as a data structure.
//
// Mirrors clang::CompilerInvocation. The principle, copied from clang:
//
//   "argv parsing produces a CompilerInvocation; the compiler doesn't
//    read argv. Different sources (cmdline, in-process API, a future
//    driver) all build a CompilerInvocation and hand it to a
//    CompilerInstance."
//
// This is the seam that lets us add a multi-process driver later
// without rewriting the compiler.

#ifndef AIEC_FRONTEND_COMPILER_INVOCATION_H
#define AIEC_FRONTEND_COMPILER_INVOCATION_H

#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace aiec {

enum class AIEDevice {
  // Add more as needed.
  npu4,
};

enum class FrontendActionKind {
  ParseOnly,  // Lex + Parse + Sema, no codegen.
  EmitMLIR,   // Lex + Parse + Sema + Instantiate + Emit LOF MLIR.
};

// Holds all options for one compilation.
//
// Default-constructed = invalid; populate via createFromArgs() (the
// argv path) or by directly setting fields (in-process API).
struct CompilerInvocation {
  // Inputs
  std::string inputPath;

  // Outputs
  std::string outputPath;

  // Action selection
  FrontendActionKind action = FrontendActionKind::EmitMLIR;

  // Target device
  AIEDevice target = AIEDevice::npu4;

  // Template parameters for the kernel-to-instantiate.
  // Key/value strings; the frontend parses these into typed values.
  std::map<std::string, std::string> templateParams;

  // Name of the kernel within the module to instantiate.
  // Empty means "the only kernel" or "the kernel matching the file's
  // basename" (resolved by Frontend later).
  std::string kernelName;

  // Diagnostic options (placeholder for now).
  bool warningsAsErrors = false;

  // Emit a self-contained hal.executable-sources module (device global +
  // executable + export + host stub) instead of just the placed kernel.
  // Lets `iree-compile --compile-from=executable-sources` run with no --ref.
  bool emitWrapper = false;

  // Parse argv into an Invocation. Returns true on success; writes
  // errors to `err`.
  static bool createFromArgs(int argc, char **argv, CompilerInvocation &out,
                             std::ostream &err);

  // Pretty-print for diagnostics.
  void dump(std::ostream &os) const;
};

} // namespace aiec

#endif // AIEC_FRONTEND_COMPILER_INVOCATION_H
