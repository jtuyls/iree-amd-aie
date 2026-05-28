#include "aiec/Basic/TokenKinds.h"

#include <string_view>
#include <unordered_map>

namespace aiec {
namespace tok {

std::string_view getTokenName(TokenKind k) {
  switch (k) {
#define TOK(X)                                                                 \
  case X:                                                                      \
    return #X;
#include "aiec/Basic/TokenKinds.def"
  default:
    return "???";
  }
}

std::string_view getPunctuatorSpelling(TokenKind k) {
  switch (k) {
#define TOK(X)
#define PUNCT(NAME, SPELLING)                                                  \
  case NAME:                                                                   \
    return SPELLING;
#include "aiec/Basic/TokenKinds.def"
  default:
    return {};
  }
}

TokenKind keywordKind(std::string_view text) {
  // Build a static keyword map once.
  static const std::unordered_map<std::string_view, TokenKind> kw = []() {
    std::unordered_map<std::string_view, TokenKind> m;
#define TOK(X)
#define KEYWORD(X) m[#X] = kw_##X;
#include "aiec/Basic/TokenKinds.def"
    // Custom mapping for the 'on' dim-label which we name kw_on_label to
    // avoid clashing with the placement keyword 'on'. The 'on' source
    // spelling always lexes to kw_on (placement); the parser disambiguates.
    m.erase("on_label");
    return m;
  }();

  auto it = kw.find(text);
  return it == kw.end() ? identifier : it->second;
}

} // namespace tok
} // namespace aiec
