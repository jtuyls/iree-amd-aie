// CompilerInstance.h - one running compilation.
//
// Mirrors clang::CompilerInstance. Owns: SourceManager, DiagnosticEngine,
// and (eventually) ASTContext, MLIRContext. Takes a CompilerInvocation
// and runs the requested action.
//
// CompilerInstance does NOT read argv. argv parsing lives in argv parser
// helpers that produce a CompilerInvocation.

#ifndef AIEC_FRONTEND_COMPILER_INSTANCE_H
#define AIEC_FRONTEND_COMPILER_INSTANCE_H

#include "aiec/Basic/Diagnostic.h"
#include "aiec/Basic/SourceLocation.h"
#include "aiec/Frontend/CompilerInvocation.h"

#include <memory>
#include <ostream>

namespace aiec {

class CompilerInstance {
public:
  // diagOut: where diagnostics go (typically std::cerr).
  CompilerInstance(CompilerInvocation invocation, std::ostream &diagOut);

  // Run the requested action. Returns 0 on success, non-zero on error.
  int run();

  const CompilerInvocation &getInvocation() const { return invocation_; }
  SourceManager &getSourceManager() { return sm_; }
  DiagnosticEngine &getDiagnostics() { return diag_; }

private:
  int runEmitMLIR();
  int runParseOnly();

  CompilerInvocation invocation_;
  std::ostream &diagOut_;
  SourceManager sm_;
  DiagnosticEngine diag_;
};

} // namespace aiec

#endif // AIEC_FRONTEND_COMPILER_INSTANCE_H
