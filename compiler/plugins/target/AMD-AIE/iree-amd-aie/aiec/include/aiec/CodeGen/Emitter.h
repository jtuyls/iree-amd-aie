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
// When `wrap` is true, the output is a self-contained hal.executable-sources
// module (device global + executable + export + host stub) that can be fed
// straight to `iree-compile --compile-from=executable-sources` with no
// external --ref wrapper. When false, only the placed kernel module is
// emitted (for the splice-based flow).
std::string emitModule(const KModuleDecl *mod,
                       const std::map<std::string, int64_t> &params,
                       DiagnosticEngine &diag,
                       const std::string &sourceDir = {}, bool wrap = false);

} // namespace aiec

#endif // AIEC_CODEGEN_EMITTER_H
