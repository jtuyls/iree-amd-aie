#include "aiec/Lex/Lexer.h"

#include <cctype>
#include <charconv>
#include <string>

namespace aiec {

Lexer::Lexer(FileID f, const SourceManager &sm, DiagnosticEngine &diag)
    : file_(f), sm_(sm), diag_(diag), buf_(sm.getBufferText(f)) {}

char Lexer::peekChar(uint32_t lookahead) const {
  return (off_ + lookahead < buf_.size()) ? buf_[off_ + lookahead] : '\0';
}

char Lexer::advanceChar() {
  if (off_ >= buf_.size())
    return '\0';
  return buf_[off_++];
}

bool Lexer::match(char c) {
  if (peekChar() == c) {
    ++off_;
    return true;
  }
  return false;
}

void Lexer::skipWhitespaceAndComments() {
  while (off_ < buf_.size()) {
    char c = buf_[off_];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++off_;
    } else if (c == '/' && peekChar(1) == '/') {
      // Line comment.
      while (off_ < buf_.size() && buf_[off_] != '\n')
        ++off_;
    } else if (c == '/' && peekChar(1) == '*') {
      uint32_t startOff = off_;
      off_ += 2;
      while (off_ + 1 < buf_.size() && !(buf_[off_] == '*' && buf_[off_ + 1] == '/'))
        ++off_;
      if (off_ + 1 >= buf_.size()) {
        diag_.report(locAt(startOff), diag::err_unterminated_block_comment);
        off_ = static_cast<uint32_t>(buf_.size());
      } else {
        off_ += 2;
      }
    } else {
      break;
    }
  }
}

void Lexer::lexIdentifierOrKeyword(Token &result, uint32_t startOff) {
  while (off_ < buf_.size()) {
    char c = buf_[off_];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      ++off_;
    else
      break;
  }
  uint32_t len = off_ - startOff;
  std::string_view text = buf_.substr(startOff, len);
  tok::TokenKind kind = tok::keywordKind(text);
  result = Token(kind, locAt(startOff), len);
  result.setRawText(text);
}

void Lexer::lexNumericLiteral(Token &result, uint32_t startOff) {
  while (off_ < buf_.size() && std::isdigit(static_cast<unsigned char>(buf_[off_])))
    ++off_;
  uint32_t len = off_ - startOff;
  std::string_view text = buf_.substr(startOff, len);
  result = Token(tok::numeric_literal, locAt(startOff), len);
  result.setRawText(text);
  // Parse the integer value (no exceptions: the IREE build disables them).
  long long value = 0;
  auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{}) {
    diag_.report(locAt(startOff), diag::err_numeric_literal_too_big);
    value = 0;
  }
  result.setIntValue(value);
}

void Lexer::lex(Token &result) {
  if (havePeek_) {
    result = peekTok_;
    havePeek_ = false;
    return;
  }
  lexImpl(result);
}

const Token &Lexer::peek() {
  if (!havePeek_) {
    lexImpl(peekTok_);
    havePeek_ = true;
  }
  return peekTok_;
}

void Lexer::lexImpl(Token &result) {
  skipWhitespaceAndComments();
  if (off_ >= buf_.size()) {
    result = Token(tok::eof, locAt(off_), 0);
    return;
  }
  uint32_t startOff = off_;
  char c = advanceChar();

  // Identifier / keyword
  if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
    --off_; // re-include 'c'
    lexIdentifierOrKeyword(result, startOff);
    return;
  }

  // Numeric
  if (std::isdigit(static_cast<unsigned char>(c))) {
    --off_;
    lexNumericLiteral(result, startOff);
    return;
  }

  // String literal: "..."  (rawText is the content without the quotes).
  if (c == '"') {
    uint32_t contentStart = off_;
    while (off_ < buf_.size() && buf_[off_] != '"')
      ++off_;
    std::string_view content = buf_.substr(contentStart, off_ - contentStart);
    if (off_ < buf_.size())
      ++off_; // consume closing quote
    else
      diag_.report(locAt(startOff), diag::err_invalid_character) << "\"";
    result = Token(tok::string_literal, locAt(startOff), off_ - startOff);
    result.setRawText(content);
    return;
  }

  // Punctuation.
  auto make = [&](tok::TokenKind k, uint32_t len) {
    result = Token(k, locAt(startOff), len);
  };

  switch (c) {
  case '(': return make(tok::l_paren, 1);
  case ')': return make(tok::r_paren, 1);
  case '{': return make(tok::l_brace, 1);
  case '}': return make(tok::r_brace, 1);
  case '[': return make(tok::l_square, 1);
  case ']': return make(tok::r_square, 1);
  case ',': return make(tok::comma, 1);
  case ';': return make(tok::semi, 1);
  case '+': return make(tok::plus, 1);
  case '*': return make(tok::star, 1);
  case '/': return make(tok::slash, 1);
  case '%': return make(tok::percent, 1);
  case '?': return make(tok::question, 1);
  case '&':
    if (match('&')) return make(tok::amp_amp, 2);
    return make(tok::amp, 1);
  case '|':
    if (match('|')) return make(tok::pipe_pipe, 2);
    return make(tok::pipe, 1);
  case '!':
    if (match('=')) return make(tok::bangequal, 2);
    return make(tok::bang, 1);
  case '-':
    if (match('>')) return make(tok::arrow, 2);
    return make(tok::minus, 1);
  case '=':
    if (match('=')) return make(tok::equalequal, 2);
    return make(tok::equal, 1);
  case '<':
    if (match('=')) return make(tok::lessequal, 2);
    return make(tok::l_angle, 1);
  case '>':
    if (match('=')) return make(tok::greaterequal, 2);
    return make(tok::r_angle, 1);
  case ':':
    if (match('=')) return make(tok::coloneq, 2);
    return make(tok::colon, 1);
  case '.':
    if (match('.')) return make(tok::dotdot, 2);
    return make(tok::dot, 1);
  default: {
    std::string s(1, c);
    diag_.report(locAt(startOff), diag::err_invalid_character) << s;
    result = Token(tok::unknown, locAt(startOff), 1);
    return;
  }
  }
}

} // namespace aiec
