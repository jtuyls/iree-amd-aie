// Token.h - lexer token.
//
// Mirrors clang::Token: small POD with kind, location, length. The
// actual text is reconstructable from (loc, length) + SourceManager.

#ifndef AIEC_LEX_TOKEN_H
#define AIEC_LEX_TOKEN_H

#include "aiec/Basic/SourceLocation.h"
#include "aiec/Basic/TokenKinds.h"

#include <cstdint>
#include <string_view>

namespace aiec {

class Token {
public:
  Token() = default;
  Token(tok::TokenKind kind, SourceLocation loc, uint32_t length)
      : kind_(kind), loc_(loc), length_(length) {}

  tok::TokenKind kind() const { return kind_; }
  SourceLocation getLocation() const { return loc_; }
  uint32_t getLength() const { return length_; }
  SourceLocation getEndLocation() const {
    return SourceLocation::make(loc_.getFileID(), loc_.getOffset() + length_);
  }

  bool is(tok::TokenKind k) const { return kind_ == k; }
  bool isNot(tok::TokenKind k) const { return kind_ != k; }
  bool isOneOf(tok::TokenKind k1, tok::TokenKind k2) const {
    return kind_ == k1 || kind_ == k2;
  }
  template <typename... Args>
  bool isOneOf(tok::TokenKind k1, Args... rest) const {
    return kind_ == k1 || isOneOf(rest...);
  }

  // For numeric literals: parsed integer value (set by lexer).
  int64_t getIntValue() const { return int_value_; }
  void setIntValue(int64_t v) { int_value_ = v; }

  // For identifiers / strings: the raw spelling. Set by lexer via raw_text_.
  std::string_view rawText() const { return raw_text_; }
  void setRawText(std::string_view t) { raw_text_ = t; }

private:
  tok::TokenKind kind_ = tok::unknown;
  SourceLocation loc_;
  uint32_t length_ = 0;
  int64_t int_value_ = 0;
  std::string_view raw_text_;
};

} // namespace aiec

#endif // AIEC_LEX_TOKEN_H
