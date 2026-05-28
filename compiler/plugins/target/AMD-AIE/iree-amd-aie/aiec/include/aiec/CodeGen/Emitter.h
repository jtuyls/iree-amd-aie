// Emitter.h - generic AST → LOF MLIR walker.
//
// Takes a parsed KModuleDecl + template parameter values, walks the AST,
// and emits LOF MLIR. Driven entirely by the AST structure — no kernel
// pattern is hardcoded. The MLIR op patterns for each AIE/LOF construct
// (buffer alloc, route connection, on core block, controlcode) are
// emission rules, not kernel knowledge.

#ifndef AIEC_CODEGEN_EMITTER_H
#define AIEC_CODEGEN_EMITTER_H

#include "aiec/AST/KDecl.h"
#include "aiec/Basic/Diagnostic.h"

#include <map>
#include <string>

namespace aiec {

// Emit LOF MLIR for the given parsed module and parameter values.
// Returns the MLIR text. Reports errors via `diag` (and returns empty
// string if there were any). `sourceDir` is the directory of the .aiec
// source, used to resolve relative `extern fn ... impl "path"` templates.
// Emits the placed kernel module (for the splice-based flow: wrap via
// aie_compile.py --ref). `wrap` (self-contained hal.executable-sources
// wrapper minting) is NOT implemented here yet — passing wrap=true reports
// an error rather than silently emitting an unwrapped module.
std::string emitModule(const KModuleDecl *mod,
                       const std::map<std::string, int64_t> &params,
                       DiagnosticEngine &diag,
                       const std::string &sourceDir = {}, bool wrap = false);

} // namespace aiec

#endif // AIEC_CODEGEN_EMITTER_H
