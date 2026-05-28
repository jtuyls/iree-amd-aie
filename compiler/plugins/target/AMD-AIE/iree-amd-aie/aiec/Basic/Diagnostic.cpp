#include "aiec/Basic/Diagnostic.h"

#include <cstdio>
#include <iostream>
#include <string>

namespace aiec {

DiagnosticBuilder::DiagnosticBuilder(DiagnosticEngine &eng, SourceLocation loc,
                                     diag::DiagID id)
    : eng_(&eng), loc_(loc), id_(id) {}

DiagnosticBuilder::DiagnosticBuilder(DiagnosticBuilder &&o) noexcept
    : eng_(o.eng_), loc_(o.loc_), id_(o.id_), args_(std::move(o.args_)),
      emitted_(o.emitted_) {
  o.emitted_ = true; // moved-from must not emit
}

DiagnosticBuilder::~DiagnosticBuilder() {
  if (!emitted_ && eng_)
    eng_->emit(loc_, id_, args_);
}

DiagnosticBuilder &DiagnosticBuilder::operator<<(std::string_view s) {
  args_.emplace_back(s);
  return *this;
}
DiagnosticBuilder &DiagnosticBuilder::operator<<(int64_t v) {
  args_.emplace_back(std::to_string(v));
  return *this;
}
DiagnosticBuilder &DiagnosticBuilder::operator<<(int v) {
  args_.emplace_back(std::to_string(v));
  return *this;
}
DiagnosticBuilder &DiagnosticBuilder::operator<<(unsigned v) {
  args_.emplace_back(std::to_string(v));
  return *this;
}
DiagnosticBuilder &DiagnosticBuilder::operator<<(const std::string &s) {
  args_.emplace_back(s);
  return *this;
}
DiagnosticBuilder &DiagnosticBuilder::operator<<(const char *s) {
  args_.emplace_back(s);
  return *this;
}

DiagnosticEngine::DiagnosticEngine(const SourceManager &sm, std::ostream &out)
    : sm_(sm), out_(out) {}

namespace {

std::string formatMessage(std::string_view fmt,
                          const std::vector<std::string> &args) {
  std::string out;
  out.reserve(fmt.size() + 32);
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] == '%' && i + 1 < fmt.size() && fmt[i + 1] >= '0' &&
        fmt[i + 1] <= '9') {
      unsigned idx = fmt[i + 1] - '0';
      if (idx < args.size())
        out += args[idx];
      else
        out += "<?>";
      ++i;
    } else {
      out += fmt[i];
    }
  }
  return out;
}

const char *severityName(DiagSeverity s) {
  switch (s) {
  case DiagSeverity::Note:    return "note";
  case DiagSeverity::Warning: return "warning";
  case DiagSeverity::Error:   return "error";
  }
  return "?";
}

} // namespace

void DiagnosticEngine::emit(SourceLocation loc, diag::DiagID id,
                            const std::vector<std::string> &args) {
  auto severity = getDiagSeverity(id);
  if (severity == DiagSeverity::Error)
    ++numErrors_;
  else if (severity == DiagSeverity::Warning)
    ++numWarnings_;

  std::string msg = formatMessage(getDiagFormat(id), args);

  // Format: path:line:col: severity: msg [-Wdiag-name]
  if (loc.isValid()) {
    auto lc = sm_.getLineCol(loc);
    out_ << sm_.getBufferPath(loc.getFileID()) << ":" << lc.line << ":"
         << lc.column << ": ";
  }
  out_ << severityName(severity) << ": " << msg
       << " [-W" << getDiagName(id) << "]\n";

  // Show the source line + caret for non-note diagnostics.
  if (loc.isValid() && severity != DiagSeverity::Note) {
    auto line = sm_.getLineText(loc);
    auto lc = sm_.getLineCol(loc);
    out_ << "  " << line << "\n";
    out_ << "  ";
    for (uint32_t i = 1; i < lc.column; ++i)
      out_ << ' ';
    out_ << "^\n";
  }
}

} // namespace aiec
