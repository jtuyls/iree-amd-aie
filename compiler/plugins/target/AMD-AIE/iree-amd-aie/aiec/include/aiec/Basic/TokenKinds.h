// TokenKinds.h - generated from TokenKinds.def via X-macros.
//
// Mirrors clang/include/clang/Basic/TokenKinds.h.

#ifndef AIEC_BASIC_TOKEN_KINDS_H
#define AIEC_BASIC_TOKEN_KINDS_H

#include <string_view>

namespace aiec {
namespace tok {

enum TokenKind {
#define TOK(X) X,
#include "aiec/Basic/TokenKinds.def"
  NUM_TOKENS
};

// Human-readable name (e.g. "kw_kernel", "arrow"). Useful in diagnostics.
std::string_view getTokenName(TokenKind k);

// Spelling for punctuation; "" for non-punctuation.
std::string_view getPunctuatorSpelling(TokenKind k);

// If the identifier text matches a keyword, returns that keyword's
// TokenKind. Otherwise returns tok::identifier.
TokenKind keywordKind(std::string_view text);

} // namespace tok
} // namespace aiec

#endif // AIEC_BASIC_TOKEN_KINDS_H
