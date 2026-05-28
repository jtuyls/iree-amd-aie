#include "aiec/Basic/DiagnosticIDs.h"

namespace aiec {

namespace {

struct DiagInfo {
  std::string_view name;
  std::string_view format;
  DiagSeverity severity;
};

const DiagInfo &getDiagInfo(diag::DiagID id) {
  static const DiagInfo table[] = {
#define DIAG(ID, SEVERITY, FORMAT) {#ID, FORMAT, DiagSeverity::SEVERITY},
#include "aiec/Basic/Diagnostics.def"
  };
  return table[id];
}

} // namespace

std::string_view getDiagFormat(diag::DiagID id) {
  return getDiagInfo(id).format;
}

DiagSeverity getDiagSeverity(diag::DiagID id) {
  return getDiagInfo(id).severity;
}

std::string_view getDiagName(diag::DiagID id) { return getDiagInfo(id).name; }

} // namespace aiec
