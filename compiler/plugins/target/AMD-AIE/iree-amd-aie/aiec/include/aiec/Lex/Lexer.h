// Lexer.h - tokenize a source buffer.
//
// Mirrors clang::Lexer: caller calls Lex() to get the next token.
// Comments and whitespace are skipped internally.

#ifndef AIEC_LEX_LEXER_H
#define AIEC_LEX_LEXER_H

#include "aiec/Basic/Diagnostic.h"
#include "aiec/Basic/SourceLocation.h"
#include "aiec/Lex/Token.h"

namespace aiec {

class Lexer {
public:
  Lexer(FileID f, const SourceManager &sm, DiagnosticEngine &diag);

  // Lex the next token. Returns true if a token was produced (always true
  // until eof, which is itself a token). Argument is filled in.
  void lex(Token &result);

  // Peek without consuming. Caches one lookahead token.
  const Token &peek();

private:
  void lexImpl(Token &result);
  void skipWhitespaceAndComments();
  void lexIdentifierOrKeyword(Token &result, uint32_t startOff);
  void lexNumericLiteral(Token &result, uint32_t startOff);
  bool match(char c);
  char peekChar(uint32_t lookahead = 0) const;
  char advanceChar();

  SourceLocation locAt(uint32_t off) const {
    return SourceLocation::make(file_, off);
  }

  FileID file_;
  const SourceManager &sm_;
  DiagnosticEngine &diag_;
  std::string_view buf_;
  uint32_t off_ = 0;
  bool havePeek_ = false;
  Token peekTok_;
};

} // namespace aiec

#endif // AIEC_LEX_LEXER_H
