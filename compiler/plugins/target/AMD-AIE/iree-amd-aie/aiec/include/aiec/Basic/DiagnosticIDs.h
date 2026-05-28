// DiagnosticIDs.h - enum of every diagnostic, generated from Diagnostics.def.
//
// Mirrors clang/include/clang/Basic/DiagnosticIDs.h.

#ifndef AIEC_BASIC_DIAGNOSTIC_IDS_H
#define AIEC_BASIC_DIAGNOSTIC_IDS_H

#include <string_view>

namespace aiec {

enum class DiagSeverity { Note, Warning, Error };

namespace diag {

enum DiagID {
#define DIAG(ID, SEVERITY, FORMAT) ID,
#include "aiec/Basic/Diagnostics.def"
  NUM_DIAGS
};

} // namespace diag

// Properties of a diagnostic ID.
std::string_view getDiagFormat(diag::DiagID id);
DiagSeverity getDiagSeverity(diag::DiagID id);
std::string_view getDiagName(diag::DiagID id);

} // namespace aiec

#endif // AIEC_BASIC_DIAGNOSTIC_IDS_H
