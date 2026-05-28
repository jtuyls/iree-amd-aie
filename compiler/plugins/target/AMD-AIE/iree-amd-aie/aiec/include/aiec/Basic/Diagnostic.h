// Diagnostic.h - emit diagnostics with source location and arguments.
//
// Mirrors clang::DiagnosticBuilder ("streaming" args into a builder) but
// without the layers of clang's text-format machinery.
//
// Usage:
//   diag.report(loc, diag::err_slice_size_mismatch) << sliceSize << bufSize;

#ifndef AIEC_BASIC_DIAGNOSTIC_H
#define AIEC_BASIC_DIAGNOSTIC_H

#include "aiec/Basic/DiagnosticIDs.h"
#include "aiec/Basic/SourceLocation.h"

#include <ostream>
#include <string>
#include <vector>

namespace aiec {

class DiagnosticEngine;

// RAII builder: caller streams in arguments, destructor emits the
// formatted diagnostic to the engine.
class DiagnosticBuilder {
public:
  DiagnosticBuilder(DiagnosticEngine &eng, SourceLocation loc, diag::DiagID id);
  ~DiagnosticBuilder();

  // Non-copyable, movable (so we can return one from report()).
  DiagnosticBuilder(const DiagnosticBuilder &) = delete;
  DiagnosticBuilder &operator=(const DiagnosticBuilder &) = delete;
  DiagnosticBuilder(DiagnosticBuilder &&) noexcept;

  DiagnosticBuilder &operator<<(std::string_view s);
  DiagnosticBuilder &operator<<(int64_t v);
  DiagnosticBuilder &operator<<(int v);
  DiagnosticBuilder &operator<<(unsigned v);
  DiagnosticBuilder &operator<<(const std::string &s);
  DiagnosticBuilder &operator<<(const char *s);

private:
  DiagnosticEngine *eng_;
  SourceLocation loc_;
  diag::DiagID id_;
  std::vector<std::string> args_;
  bool emitted_ = false;
};

class DiagnosticEngine {
public:
  DiagnosticEngine(const SourceManager &sm, std::ostream &out);

  DiagnosticBuilder report(SourceLocation loc, diag::DiagID id) {
    return DiagnosticBuilder(*this, loc, id);
  }

  // Emit a formatted diagnostic. Called by DiagnosticBuilder's destructor.
  void emit(SourceLocation loc, diag::DiagID id,
            const std::vector<std::string> &args);

  uint32_t getNumErrors() const { return numErrors_; }
  uint32_t getNumWarnings() const { return numWarnings_; }
  bool hasErrors() const { return numErrors_ > 0; }

private:
  const SourceManager &sm_;
  std::ostream &out_;
  uint32_t numErrors_ = 0;
  uint32_t numWarnings_ = 0;
};

} // namespace aiec

#endif // AIEC_BASIC_DIAGNOSTIC_H
