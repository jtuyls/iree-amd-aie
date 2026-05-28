// Parser.h - recursive-descent parser for .aiec.

#ifndef AIEC_PARSE_PARSER_H
#define AIEC_PARSE_PARSER_H

#include "aiec/AST/KASTContext.h"
#include "aiec/AST/KDecl.h"
#include "aiec/Basic/Diagnostic.h"
#include "aiec/Lex/Lexer.h"

#include <optional>
#include <string>
#include <vector>

namespace aiec {

class Parser {
public:
  Parser(Lexer &lex, KASTContext &ctx, DiagnosticEngine &diag);

  KModuleDecl *parseFile();

private:
  void advance();
  bool consume(tok::TokenKind k);
  bool expect(tok::TokenKind k, const char *what);
  const Token &peek() const { return tok_; }
  bool at(tok::TokenKind k) const { return tok_.is(k); }
  std::string tokText() const { return std::string(tok_.rawText()); }
  SourceLocation tokLoc() const { return tok_.getLocation(); }
  bool atIdentLike() const;

  // Decls
  KModuleDecl *parseModule();
  KDecl *parseTopLevelDecl();
  KExternFnDecl *parseExternFn();
  KKernelDecl *parseKernel();
  KCatalogDecl *parseCatalog();
  KBufferDecl *parseBuffer();
  KRouteDecl *parseRoute();
  KOnCoreDecl *parseOnCore();
  KOnControllerDecl *parseOnController();

  // Statements
  KStmt *parseStmt();
  KForallStmt *parseForall();
  KReduceStmt *parseReduce();
  KStmt *parseLetOrAssign();
  KStmt *parseFor();
  KIssueStmt *parseIssue();

  // Expressions (allowRel: see comment in Parser.cpp)
  KExpr *parseExpr(bool allowRel = true);
  KExpr *parseTernary(bool allowRel);
  KExpr *parseLogicalOr(bool allowRel);
  KExpr *parseLogicalAnd(bool allowRel);
  KExpr *parseEquality(bool allowRel);
  KExpr *parseRelational(bool allowRel);
  KExpr *parseAdditive();
  KExpr *parseMultiplicative();
  KExpr *parseUnary();
  KExpr *parsePrimary();
  KExpr *parsePostfix(KExpr *base);
  std::vector<KExpr *> parseCallArgs();
  KExpr *parseListLiteral();

  // Type / placement helpers
  bool parseMemrefType(KMemrefType &out);
  KPlacement parsePlacement();
  std::vector<KIndexBinding> parseIndexBindings();
  KViaChannel parseViaChannel();
  // Parse `target(...)` or `source(...)` clause. Returns std::nullopt if
  // the next tokens aren't a route-side clause.
  std::optional<KRouteSide> tryParseRouteSide(const std::string &keyword);

  Lexer &lex_;
  KASTContext &ctx_;
  DiagnosticEngine &diag_;
  Token tok_;
};

} // namespace aiec

#endif // AIEC_PARSE_PARSER_H
