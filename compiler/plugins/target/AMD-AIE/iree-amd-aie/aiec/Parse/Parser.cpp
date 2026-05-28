// Parser.cpp - recursive-descent parser for the generic .aiec syntax.

#include "aiec/Parse/Parser.h"

namespace aiec {

Parser::Parser(Lexer &lex, KASTContext &ctx, DiagnosticEngine &diag)
    : lex_(lex), ctx_(ctx), diag_(diag) {
  lex_.lex(tok_);
}

bool Parser::atIdentLike() const {
  // Accept any plain identifier as a name. Dim-label keywords are no
  // longer reserved by the lexer — we treat tok::identifier exclusively.
  return tok_.is(tok::identifier);
}

void Parser::advance() { lex_.lex(tok_); }

bool Parser::consume(tok::TokenKind k) {
  if (tok_.is(k)) {
    advance();
    return true;
  }
  return false;
}

bool Parser::expect(tok::TokenKind k, const char *what) {
  if (consume(k)) return true;
  diag_.report(tok_.getLocation(), diag::err_expected) << what;
  return false;
}

// ─── top-level ──────────────────────────────────────────────────────────

KModuleDecl *Parser::parseFile() { return parseModule(); }

KModuleDecl *Parser::parseModule() {
  SourceLocation loc = tokLoc();
  if (!expect(tok::kw_module, "'module'")) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "module name";
    return nullptr;
  }
  std::string name = tokText();
  advance();
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KDecl *> decls;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KDecl *d = parseTopLevelDecl();
    if (!d) return nullptr;
    decls.push_back(d);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KModuleDecl>(loc, std::move(name), std::move(decls));
}

KDecl *Parser::parseTopLevelDecl() {
  if (at(tok::kw_extern)) return parseExternFn();
  if (at(tok::kw_kernel)) return parseKernel();
  diag_.report(tokLoc(), diag::err_unexpected_token) << tokText();
  return nullptr;
}

// ─── memref<...> type ──────────────────────────────────────────────────
//
// memref<dim, dim, ..., elemType, memspace>
//   dim       : constexpr int expression (no relational, since `>` closes)
//   elemType  : one of i8 / i16 / i32 / bf16 / f32
//   memspace  : integer literal

bool Parser::parseMemrefType(KMemrefType &out) {
  if (!expect(tok::kw_memref, "'memref'")) return false;
  if (!expect(tok::l_angle, "'<'")) return false;
  while (!at(tok::r_angle)) {
    // Element type token (one of i8/i16/i32/bf16/f32) signals the
    // start of trailing (elemType, memspace).
    if (at(tok::kw_i8) || at(tok::kw_i16) || at(tok::kw_i32) ||
        at(tok::kw_bf16) || at(tok::kw_f32)) {
      out.elemType = tokText();
      advance();
      if (!expect(tok::comma, "','")) return false;
      if (!at(tok::numeric_literal)) {
        diag_.report(tokLoc(), diag::err_expected) << "memspace integer";
        return false;
      }
      out.memSpace = (int)tok_.getIntValue();
      advance();
      break;
    }
    KExpr *e = parseExpr(/*allowRel=*/false);
    if (!e) return false;
    out.dims.push_back(e);
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_angle, "'>'")) return false;
  return true;
}

// ─── extern fn ──────────────────────────────────────────────────────────

KExternFnDecl *Parser::parseExternFn() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_extern)) return nullptr;
  if (!expect(tok::kw_fn, "'fn'")) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "function name";
    return nullptr;
  }
  std::string name = tokText();
  advance();
  if (!expect(tok::l_paren, "'('")) return nullptr;
  std::vector<KExternFnArg> args;
  while (!at(tok::r_paren)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "parameter name";
      return nullptr;
    }
    KExternFnArg a;
    a.name = tokText();
    advance();
    if (!expect(tok::colon, "':'")) return nullptr;
    if (!parseMemrefType(a.type)) return nullptr;
    args.push_back(std::move(a));
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  // Optional `impl "path.mlir"` clause pointing to the function body.
  std::string implPath;
  if (atIdentLike() && tokText() == "impl") {
    advance();
    if (!at(tok::string_literal)) {
      diag_.report(tokLoc(), diag::err_expected) << "string path after 'impl'";
      return nullptr;
    }
    implPath = tokText();
    advance();
  }
  consume(tok::semi);
  return ctx_.create<KExternFnDecl>(loc, std::move(name), std::move(args),
                                    std::move(implPath));
}

// ─── kernel ─────────────────────────────────────────────────────────────

KKernelDecl *Parser::parseKernel() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_kernel)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "kernel name";
    return nullptr;
  }
  std::string name = tokText();
  advance();

  std::vector<KTemplateParam> tparams;
  if (!expect(tok::l_angle, "'<'")) return nullptr;
  while (!at(tok::r_angle)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "template parameter name";
      return nullptr;
    }
    KTemplateParam p;
    p.name = tokText();
    p.type = "int";
    advance();
    if (consume(tok::colon)) {
      // Accept ident or type keyword.
      if (!(at(tok::identifier) || at(tok::kw_i32) || at(tok::kw_i8) ||
            at(tok::kw_i16) || at(tok::kw_bf16) || at(tok::kw_f32))) {
        diag_.report(tokLoc(), diag::err_expected) << "parameter type";
        return nullptr;
      }
      p.type = tokText();
      advance();
    }
    tparams.push_back(std::move(p));
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_angle, "'>'")) return nullptr;

  // kernel args: (NAME: TYPE, ...).  Type is either memref<...> or a scalar.
  std::vector<KKernelArg> kargs;
  if (!expect(tok::l_paren, "'('")) return nullptr;
  while (!at(tok::r_paren)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "argument name";
      return nullptr;
    }
    KKernelArg ka;
    ka.name = tokText();
    advance();
    if (!expect(tok::colon, "':'")) return nullptr;
    if (at(tok::kw_memref)) {
      if (!parseMemrefType(ka.type)) return nullptr;
    } else if (at(tok::kw_i32) || at(tok::kw_i8) || at(tok::kw_i16) ||
               at(tok::kw_bf16) || at(tok::kw_f32)) {
      ka.scalarType = tokText();
      advance();
    } else {
      diag_.report(tokLoc(), diag::err_expected)
          << "'memref<...>' or scalar type";
      return nullptr;
    }
    kargs.push_back(std::move(ka));
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;

  // where clauses
  std::vector<KExpr *> whereConstraints;
  if (consume(tok::kw_where)) {
    while (true) {
      KExpr *e = parseExpr();
      if (!e) return nullptr;
      whereConstraints.push_back(e);
      if (!consume(tok::comma)) break;
    }
  }

  // Optional kernel attributes: `NAME = EXPR` pairs before the body. The
  // body opens with '{', so any identifier here starts an attribute.
  std::vector<KKernelAttr> attrs;
  while (atIdentLike()) {
    KKernelAttr a;
    a.name = tokText();
    advance();
    if (!expect(tok::equal, "'='")) return nullptr;
    a.value = parseExpr();
    if (!a.value) return nullptr;
    attrs.push_back(std::move(a));
    consume(tok::comma);
  }

  // body
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KDecl *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KDecl *d = nullptr;
    if (at(tok::kw_shim) || at(tok::kw_memtile) || at(tok::kw_core) ||
        at(tok::kw_controller)) {
      d = parseCatalog();
    } else if (at(tok::kw_on)) {
      advance();
      if (at(tok::kw_core))
        d = parseOnCore();
      else if (at(tok::kw_controller))
        d = parseOnController();
      else {
        diag_.report(tokLoc(), diag::err_expected) << "'core' or 'controller'";
        return nullptr;
      }
    } else if (at(tok::kw_buffer)) {
      d = parseBuffer();
    } else if (at(tok::kw_route)) {
      d = parseRoute();
    } else {
      diag_.report(tokLoc(), diag::err_unexpected_token) << tokText();
      return nullptr;
    }
    if (!d) return nullptr;
    body.push_back(d);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KKernelDecl>(loc, std::move(name), std::move(tparams),
                                  std::move(kargs), std::move(whereConstraints),
                                  std::move(body), std::move(attrs));
}

// ─── catalog ────────────────────────────────────────────────────────────

KCatalogDecl *Parser::parseCatalog() {
  SourceLocation loc = tokLoc();
  CatalogKind ck;
  if (consume(tok::kw_shim))
    ck = CatalogKind::Shim;
  else if (consume(tok::kw_memtile))
    ck = CatalogKind::Memtile;
  else if (consume(tok::kw_core))
    ck = CatalogKind::Core;
  else if (consume(tok::kw_controller))
    ck = CatalogKind::Controller;
  else {
    diag_.report(tokLoc(), diag::err_unexpected_token) << tokText();
    return nullptr;
  }
  std::vector<KIndexBinding> indices;
  if (at(tok::l_square)) indices = parseIndexBindings();
  if (!expect(tok::arrow, "'->'")) return nullptr;
  if (!expect(tok::kw_tile, "'tile'")) return nullptr;
  if (!expect(tok::l_paren, "'('")) return nullptr;
  KExpr *colE = parseExpr();
  if (!colE) return nullptr;
  KExpr *rowE = nullptr;
  if (consume(tok::comma)) {
    rowE = parseExpr();
    if (!rowE) return nullptr;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  return ctx_.create<KCatalogDecl>(loc, ck, std::move(indices), colE, rowE);
}

std::vector<KIndexBinding> Parser::parseIndexBindings() {
  std::vector<KIndexBinding> out;
  expect(tok::l_square, "'['");
  while (!at(tok::r_square)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "index variable";
      return out;
    }
    KIndexBinding ib;
    ib.name = tokText();
    advance();
    if (!expect(tok::kw_in, "'in'")) return out;
    KExpr *lo = parseExpr();
    if (!lo) return out;
    if (!expect(tok::dotdot, "'..'")) return out;
    KExpr *hi = parseExpr();
    if (!hi) return out;
    ib.lo = lo;
    ib.hi = hi;
    out.push_back(std::move(ib));
    if (!consume(tok::comma)) break;
  }
  expect(tok::r_square, "']'");
  return out;
}

// ─── buffer ─────────────────────────────────────────────────────────────

KBufferDecl *Parser::parseBuffer() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_buffer)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "buffer name";
    return nullptr;
  }
  std::string name = tokText();
  advance();
  std::vector<std::string> ivs;
  if (consume(tok::l_square)) {
    while (!at(tok::r_square)) {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "buffer index variable";
        return nullptr;
      }
      ivs.push_back(tokText());
      advance();
      if (!consume(tok::comma)) break;
    }
    if (!expect(tok::r_square, "']'")) return nullptr;
  }
  if (!expect(tok::colon, "':'")) return nullptr;
  KMemrefType type;
  if (!parseMemrefType(type)) return nullptr;
  // Optional `depth <constexpr>` (objectfifo ping-pong depth; default 1).
  // A constexpr (not just a literal) so it can be a template parameter.
  KExpr *depthExpr = nullptr;
  if (atIdentLike() && tokText() == "depth") {
    advance();
    depthExpr = parseExpr();
    if (!depthExpr) return nullptr;
  }
  if (!expect(tok::kw_on, "'on'")) return nullptr;
  KPlacement pl = parsePlacement();
  KExpr *whenC = nullptr;
  if (consume(tok::kw_when)) {
    whenC = parseExpr();
    if (!whenC) return nullptr;
  }
  return ctx_.create<KBufferDecl>(loc, std::move(name), std::move(ivs),
                                  std::move(type), std::move(pl), whenC,
                                  depthExpr);
}

// ─── placement ───────────────────────────────────────────────────────────

KPlacement Parser::parsePlacement() {
  KPlacement pl;
  if (at(tok::kw_shim)) {
    pl.catalog = "shim";
    advance();
  } else if (at(tok::kw_memtile)) {
    pl.catalog = "memtile";
    advance();
  } else if (at(tok::kw_core)) {
    pl.catalog = "core";
    advance();
  } else if (at(tok::kw_controller)) {
    pl.catalog = "controller";
    advance();
    return pl;
  } else if (atIdentLike()) {
    pl.catalog = tokText();
    advance();
  } else {
    diag_.report(tokLoc(), diag::err_expected) << "catalog name";
    return pl;
  }
  if (consume(tok::l_square)) {
    while (!at(tok::r_square)) {
      KExpr *e;
      if (at(tok::star)) {
        e = ctx_.create<KWildcardExpr>(tokLoc());
        advance();
      } else {
        e = parseExpr();
        if (!e) return pl;
      }
      pl.axes.push_back(e);
      if (!consume(tok::comma)) break;
    }
    expect(tok::r_square, "']'");
  }
  return pl;
}

// ─── route ──────────────────────────────────────────────────────────────

std::optional<KRouteSide> Parser::tryParseRouteSide(
    const std::string &keyword) {
  // Expecting: source(buf=NAME[idx], offsets=[...], sizes=[...], strides=[...])
  // OR: target(...)
  // Caller has already verified tok_.rawText() == keyword.
  if (!atIdentLike() || tokText() != keyword) return std::nullopt;
  advance();
  if (!expect(tok::l_paren, "'('")) return std::nullopt;
  KRouteSide rs;
  while (!at(tok::r_paren)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "field name";
      return std::nullopt;
    }
    std::string field = tokText();
    advance();
    if (!expect(tok::equal, "'='")) return std::nullopt;
    if (field == "buf") {
      // buffer name with optional [idx, ...]
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "buffer name";
        return std::nullopt;
      }
      rs.bufName = tokText();
      advance();
      if (consume(tok::l_square)) {
        while (!at(tok::r_square)) {
          KExpr *e = parseExpr();
          if (!e) return std::nullopt;
          rs.bufIndices.push_back(e);
          if (!consume(tok::comma)) break;
        }
        expect(tok::r_square, "']'");
      }
    } else if (field == "offsets" || field == "sizes" || field == "strides") {
      KExpr *list = parseListLiteral();
      if (!list) return std::nullopt;
      auto *ll = cast<KListLitExpr>(list);
      auto &target = (field == "offsets") ? rs.offsets
                     : (field == "sizes") ? rs.sizes
                                          : rs.strides;
      target = ll->getElements();
    } else {
      diag_.report(tokLoc(), diag::err_unexpected_token) << field;
      return std::nullopt;
    }
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_paren, "')'")) return std::nullopt;
  return rs;
}

KViaChannel Parser::parseViaChannel() {
  KViaChannel vc;
  vc.placement = parsePlacement();
  // Optional `dma_mm2s = EXPR` or `dma_s2mm = EXPR`.
  if (atIdentLike()) {
    std::string field = tokText();
    if (field == "dma_mm2s") {
      vc.mm2s = true;
      advance();
      if (!expect(tok::equal, "'='")) return vc;
      vc.channelId = parseExpr(/*allowRel=*/false);
    } else if (field == "dma_s2mm") {
      vc.mm2s = false;
      advance();
      if (!expect(tok::equal, "'='")) return vc;
      vc.channelId = parseExpr(/*allowRel=*/false);
    }
  }
  return vc;
}

KRouteDecl *Parser::parseRoute() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_route)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "route name";
    return nullptr;
  }
  std::string name = tokText();
  advance();
  std::vector<std::string> ivs;
  if (consume(tok::l_square)) {
    while (!at(tok::r_square)) {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "route index variable";
        return nullptr;
      }
      ivs.push_back(tokText());
      advance();
      if (!consume(tok::comma)) break;
    }
    if (!expect(tok::r_square, "']'")) return nullptr;
  }
  if (!expect(tok::colon, "':'")) return nullptr;
  RouteType rt;
  if (consume(tok::kw_packet))
    rt = RouteType::Packet;
  else if (consume(tok::kw_circuit))
    rt = RouteType::Circuit;
  else {
    diag_.report(tokLoc(), diag::err_expected) << "'packet' or 'circuit'";
    return nullptr;
  }
  // Optional parameterization marker after route type:
  //   route NAME: packet, source_parameterized  ... or  target_parameterized
  RouteParameterization param = RouteParameterization::Persistent;
  if (consume(tok::comma)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected)
          << "'source_parameterized' or 'target_parameterized'";
      return nullptr;
    }
    std::string p = tokText();
    advance();
    if (p == "source_parameterized")
      param = RouteParameterization::SourceParameterized;
    else if (p == "target_parameterized")
      param = RouteParameterization::TargetParameterized;
    else {
      diag_.report(tokLoc(), diag::err_unexpected_token) << p;
      return nullptr;
    }
  }

  // Optional source(...) and target(...) clauses (one or both, order
  // free).
  std::optional<KRouteSide> src, tgt;
  while (atIdentLike() && (tokText() == "source" || tokText() == "target")) {
    if (tokText() == "source")
      src = tryParseRouteSide("source");
    else
      tgt = tryParseRouteSide("target");
  }

  if (!expect(tok::kw_via, "'via'")) return nullptr;
  KViaChannel viaSrc = parseViaChannel();
  if (!expect(tok::arrow, "'->'")) return nullptr;
  KViaChannel viaDst = parseViaChannel();

  KExpr *whenC = nullptr;
  if (consume(tok::kw_when)) {
    whenC = parseExpr();
    if (!whenC) return nullptr;
  }
  return ctx_.create<KRouteDecl>(loc, std::move(name), std::move(ivs), rt,
                                 param, std::move(src), std::move(tgt),
                                 std::move(viaSrc), std::move(viaDst), whenC);
}

// ─── on core / on controller ───────────────────────────────────────────

KOnCoreDecl *Parser::parseOnCore() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_core)) return nullptr;
  std::vector<std::string> ivs;
  if (consume(tok::l_square)) {
    while (!at(tok::r_square)) {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "core index variable";
        return nullptr;
      }
      ivs.push_back(tokText());
      advance();
      if (!consume(tok::comma)) break;
    }
    if (!expect(tok::r_square, "']'")) return nullptr;
  }
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KStmt *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KStmt *s = parseStmt();
    if (!s) return nullptr;
    body.push_back(s);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KOnCoreDecl>(loc, std::move(ivs), std::move(body));
}

KOnControllerDecl *Parser::parseOnController() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_controller)) return nullptr;
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KStmt *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KStmt *s = parseStmt();
    if (!s) return nullptr;
    body.push_back(s);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KOnControllerDecl>(loc, std::move(body));
}

// ─── statements ─────────────────────────────────────────────────────────

KStmt *Parser::parseStmt() {
  if (at(tok::kw_barrier)) {
    SourceLocation loc = tokLoc();
    advance();
    return ctx_.create<KBarrierStmt>(loc);
  }
  if (at(tok::kw_forall)) return parseForall();
  if (at(tok::kw_reduce)) return parseReduce();
  if (at(tok::kw_let)) return parseLetOrAssign();
  if (at(tok::kw_for)) return parseFor();
  if (at(tok::kw_issue)) return parseIssue();
  if (at(tok::kw_wait)) {
    SourceLocation loc = tokLoc();
    advance();
    // Optional target: wait(route), wait(route[i]) or wait(route[lo..hi]).
    std::string routeName;
    KExpr *idxLo = nullptr, *idxHi = nullptr;
    if (consume(tok::l_paren)) {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "route name";
        return nullptr;
      }
      routeName = tokText();
      advance();
      if (consume(tok::l_square)) {
        idxLo = parseExpr();
        if (!idxLo) return nullptr;
        if (consume(tok::dotdot)) {
          idxHi = parseExpr();
          if (!idxHi) return nullptr;
        }
        if (!expect(tok::r_square, "']'")) return nullptr;
      }
      if (!expect(tok::r_paren, "')'")) return nullptr;
    }
    return ctx_.create<KWaitStmt>(loc, std::move(routeName), idxLo, idxHi);
  }
  if (at(tok::kw_zero)) {
    SourceLocation loc = tokLoc();
    advance();
    if (!expect(tok::l_paren, "'('")) return nullptr;
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "handle";
      return nullptr;
    }
    std::string h = tokText();
    advance();
    if (!expect(tok::r_paren, "')'")) return nullptr;
    return ctx_.create<KZeroStmt>(loc, std::move(h));
  }
  if (at(tok::kw_release)) {
    SourceLocation loc = tokLoc();
    advance();
    if (!expect(tok::l_paren, "'('")) return nullptr;
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "handle";
      return nullptr;
    }
    std::string h = tokText();
    advance();
    if (!expect(tok::r_paren, "')'")) return nullptr;
    return ctx_.create<KReleaseStmt>(loc, std::move(h));
  }
  // Either `NAME(args...)` (call) or `NAME := expr` (assign).
  if (atIdentLike()) {
    SourceLocation loc = tokLoc();
    std::string name = tokText();
    advance();
    if (consume(tok::l_paren)) {
      std::vector<std::string> args;
      while (!at(tok::r_paren)) {
        if (!atIdentLike()) {
          diag_.report(tokLoc(), diag::err_expected) << "argument";
          return nullptr;
        }
        args.push_back(tokText());
        advance();
        if (!consume(tok::comma)) break;
      }
      if (!expect(tok::r_paren, "')'")) return nullptr;
      return ctx_.create<KCallStmt>(loc, std::move(name), std::move(args));
    }
    if (consume(tok::coloneq)) {
      // NAME := NAME '*' expr
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "handle on RHS";
        return nullptr;
      }
      std::string lhsHandle = tokText();
      advance();
      if (lhsHandle != name) {
        diag_.report(loc, diag::err_unexpected_token)
            << "expected self-mul pattern";
        return nullptr;
      }
      if (!expect(tok::star, "'*'")) return nullptr;
      KExpr *rhs = parseExpr();
      if (!rhs) return nullptr;
      return ctx_.create<KAssignMulStmt>(loc, std::move(name), rhs);
    }
    diag_.report(loc, diag::err_unexpected_token) << name;
    return nullptr;
  }
  diag_.report(tokLoc(), diag::err_unexpected_token) << tokText();
  return nullptr;
}

KForallStmt *Parser::parseForall() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_forall)) return nullptr;
  if (!expect(tok::l_paren, "'('")) return nullptr;
  std::vector<std::string> ivs;
  while (!at(tok::r_paren)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "forall var";
      return nullptr;
    }
    ivs.push_back(tokText());
    advance();
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  if (!expect(tok::kw_in, "'in'")) return nullptr;
  if (!expect(tok::l_paren, "'('")) return nullptr;
  std::vector<KExpr *> dims;
  while (!at(tok::r_paren)) {
    KExpr *e = parseExpr();
    if (!e) return nullptr;
    dims.push_back(e);
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KStmt *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KStmt *s = parseStmt();
    if (!s) return nullptr;
    body.push_back(s);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KForallStmt>(loc, std::move(ivs), std::move(dims),
                                  std::move(body));
}

KReduceStmt *Parser::parseReduce() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_reduce)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "reduce var";
    return nullptr;
  }
  std::string iv = tokText();
  advance();
  if (!expect(tok::kw_in, "'in'")) return nullptr;
  KExpr *lo = parseExpr();
  if (!lo) return nullptr;
  if (!expect(tok::dotdot, "'..'")) return nullptr;
  KExpr *hi = parseExpr();
  if (!hi) return nullptr;
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KStmt *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KStmt *s = parseStmt();
    if (!s) return nullptr;
    body.push_back(s);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KReduceStmt>(loc, std::move(iv), lo, hi, std::move(body));
}

KStmt *Parser::parseLetOrAssign() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_let)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "binding name";
    return nullptr;
  }
  std::string name = tokText();
  advance();
  if (!expect(tok::equal, "'='")) return nullptr;
  // `let name = base[indices]` — a sub-tile view of an already-acquired buffer
  // handle (no `acquire`). Lets the 4 micro-tile matmuls write into one
  // acquired output buffer instead of four separate fifo acquires.
  if (!at(tok::kw_acquire)) {
    KExpr *sv = parsePrimary();
    if (!sv) return nullptr;
    sv = parsePostfix(sv);
    if (!sv) return nullptr;
    return ctx_.create<KLetStmt>(loc, std::move(name), sv, AcquireRole::Produce,
                                 /*subview=*/true);
  }
  if (!consume(tok::kw_acquire)) {
    diag_.report(tokLoc(), diag::err_expected) << "'acquire'";
    return nullptr;
  }
  if (!expect(tok::l_paren, "'('")) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "buffer name";
    return nullptr;
  }
  KExpr *base = parsePrimary();
  if (!base) return nullptr;
  base = parsePostfix(base);
  if (!base) return nullptr;
  if (!expect(tok::comma, "','")) return nullptr;
  AcquireRole role;
  if (consume(tok::kw_Consume))
    role = AcquireRole::Consume;
  else if (consume(tok::kw_Produce))
    role = AcquireRole::Produce;
  else {
    diag_.report(tokLoc(), diag::err_expected) << "'Consume' or 'Produce'";
    return nullptr;
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  return ctx_.create<KLetStmt>(loc, std::move(name), base, role);
}

KStmt *Parser::parseFor() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_for)) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "for var";
    return nullptr;
  }
  std::string iv = tokText();
  advance();
  if (!expect(tok::kw_in, "'in'")) return nullptr;
  KExpr *lo = parseExpr();
  if (!lo) return nullptr;
  if (!expect(tok::dotdot, "'..'")) return nullptr;
  KExpr *hi = parseExpr();
  if (!hi) return nullptr;
  if (!expect(tok::l_brace, "'{'")) return nullptr;
  std::vector<KStmt *> body;
  while (!at(tok::r_brace) && !at(tok::eof)) {
    KStmt *s = parseStmt();
    if (!s) return nullptr;
    body.push_back(s);
  }
  if (!expect(tok::r_brace, "'}'")) return nullptr;
  return ctx_.create<KForStmt>(loc, std::move(iv), lo, hi, std::move(body));
}

KIssueStmt *Parser::parseIssue() {
  SourceLocation loc = tokLoc();
  if (!consume(tok::kw_issue)) return nullptr;
  if (!expect(tok::l_paren, "'('")) return nullptr;
  if (!atIdentLike()) {
    diag_.report(tokLoc(), diag::err_expected) << "route name";
    return nullptr;
  }
  std::string rname = tokText();
  advance();
  std::vector<KExpr *> idx;
  if (consume(tok::l_square)) {
    while (!at(tok::r_square)) {
      KExpr *e = parseExpr();
      if (!e) return nullptr;
      idx.push_back(e);
      if (!consume(tok::comma)) break;
    }
    if (!expect(tok::r_square, "']'")) return nullptr;
  }
  std::string srcBind, dstBind;
  std::vector<KExpr *> offsets, sizes, strides;
  KExpr *bdId = nullptr;
  while (consume(tok::comma)) {
    if (!atIdentLike()) {
      diag_.report(tokLoc(), diag::err_expected) << "field name";
      return nullptr;
    }
    std::string field = tokText();
    advance();
    if (!expect(tok::equal, "'='")) return nullptr;
    if (field == "source" || field == "src") {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "binding name";
        return nullptr;
      }
      srcBind = tokText();
      advance();
    } else if (field == "target" || field == "dest") {
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "binding name";
        return nullptr;
      }
      dstBind = tokText();
      advance();
    } else if (field == "offsets" || field == "sizes" || field == "strides") {
      KExpr *list = parseListLiteral();
      if (!list) return nullptr;
      auto *ll = cast<KListLitExpr>(list);
      auto &dst = (field == "offsets") ? offsets
                  : (field == "sizes") ? sizes
                                       : strides;
      dst = ll->getElements();
    } else if (field == "bd") {
      bdId = parseExpr();
      if (!bdId) return nullptr;
    } else {
      diag_.report(tokLoc(), diag::err_unexpected_token) << field;
      return nullptr;
    }
  }
  if (!expect(tok::r_paren, "')'")) return nullptr;
  return ctx_.create<KIssueStmt>(loc, std::move(rname), std::move(idx),
                                 std::move(srcBind), std::move(dstBind),
                                 std::move(offsets), std::move(sizes),
                                 std::move(strides), bdId);
}

// ─── expressions ────────────────────────────────────────────────────────

KExpr *Parser::parseExpr(bool allowRel) { return parseTernary(allowRel); }

KExpr *Parser::parseTernary(bool allowRel) {
  KExpr *cond = parseLogicalOr(allowRel);
  if (!cond) return nullptr;
  if (consume(tok::question)) {
    SourceLocation loc = cond->getLocation();
    KExpr *thenE = parseExpr(allowRel);
    if (!thenE) return nullptr;
    if (!expect(tok::colon, "':'")) return nullptr;
    KExpr *elseE = parseExpr(allowRel);
    if (!elseE) return nullptr;
    return ctx_.create<KTernaryExpr>(loc, cond, thenE, elseE);
  }
  return cond;
}

KExpr *Parser::parseLogicalOr(bool allowRel) {
  KExpr *lhs = parseLogicalAnd(allowRel);
  while (lhs && at(tok::pipe_pipe)) {
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseLogicalAnd(allowRel);
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, BinOpKind::Or, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseLogicalAnd(bool allowRel) {
  KExpr *lhs = parseEquality(allowRel);
  while (lhs && at(tok::amp_amp)) {
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseEquality(allowRel);
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, BinOpKind::And, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseEquality(bool allowRel) {
  KExpr *lhs = parseRelational(allowRel);
  while (lhs && (at(tok::equalequal) || at(tok::bangequal))) {
    BinOpKind op = at(tok::equalequal) ? BinOpKind::Eq : BinOpKind::Ne;
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseRelational(allowRel);
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, op, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseRelational(bool allowRel) {
  KExpr *lhs = parseAdditive();
  if (!allowRel) return lhs;
  while (lhs && (at(tok::l_angle) || at(tok::r_angle) || at(tok::lessequal) ||
                 at(tok::greaterequal))) {
    BinOpKind op;
    if (at(tok::l_angle))
      op = BinOpKind::Lt;
    else if (at(tok::r_angle))
      op = BinOpKind::Gt;
    else if (at(tok::lessequal))
      op = BinOpKind::Le;
    else
      op = BinOpKind::Ge;
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseAdditive();
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, op, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseAdditive() {
  KExpr *lhs = parseMultiplicative();
  while (lhs && (at(tok::plus) || at(tok::minus))) {
    BinOpKind op = at(tok::plus) ? BinOpKind::Add : BinOpKind::Sub;
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseMultiplicative();
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, op, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseMultiplicative() {
  KExpr *lhs = parseUnary();
  while (lhs && (at(tok::star) || at(tok::slash) || at(tok::percent))) {
    BinOpKind op;
    if (at(tok::star))
      op = BinOpKind::Mul;
    else if (at(tok::slash))
      op = BinOpKind::Div;
    else
      op = BinOpKind::Mod;
    SourceLocation loc = tokLoc();
    advance();
    KExpr *rhs = parseUnary();
    if (!rhs) return nullptr;
    lhs = ctx_.create<KBinOpExpr>(loc, op, lhs, rhs);
  }
  return lhs;
}

KExpr *Parser::parseUnary() {
  if (at(tok::minus)) {
    SourceLocation loc = tokLoc();
    advance();
    KExpr *o = parseUnary();
    if (!o) return nullptr;
    return ctx_.create<KUnaryOpExpr>(loc, UnaryOpKind::Neg, o);
  }
  if (at(tok::bang)) {
    SourceLocation loc = tokLoc();
    advance();
    KExpr *o = parseUnary();
    if (!o) return nullptr;
    return ctx_.create<KUnaryOpExpr>(loc, UnaryOpKind::Not, o);
  }
  KExpr *p = parsePrimary();
  if (!p) return nullptr;
  return parsePostfix(p);
}

KExpr *Parser::parsePrimary() {
  SourceLocation loc = tokLoc();
  if (at(tok::numeric_literal)) {
    int64_t v = tok_.getIntValue();
    advance();
    return ctx_.create<KIntLitExpr>(loc, v);
  }
  if (at(tok::kw_true) || at(tok::kw_false)) {
    int64_t v = at(tok::kw_true) ? 1 : 0;
    advance();
    return ctx_.create<KIntLitExpr>(loc, v);
  }
  if (at(tok::l_paren)) {
    advance();
    KExpr *e = parseExpr();
    if (!e) return nullptr;
    if (!expect(tok::r_paren, "')'")) return nullptr;
    return e;
  }
  if (at(tok::l_square)) {
    return parseListLiteral();
  }
  if (atIdentLike()) {
    std::string name = tokText();
    advance();
    return ctx_.create<KIdentExpr>(loc, std::move(name));
  }
  diag_.report(tokLoc(), diag::err_unexpected_token) << tokText();
  return nullptr;
}

KExpr *Parser::parseListLiteral() {
  SourceLocation loc = tokLoc();
  if (!expect(tok::l_square, "'['")) return nullptr;
  std::vector<KExpr *> elements;
  while (!at(tok::r_square)) {
    KExpr *e = parseExpr();
    if (!e) return nullptr;
    elements.push_back(e);
    if (!consume(tok::comma)) break;
  }
  if (!expect(tok::r_square, "']'")) return nullptr;
  return ctx_.create<KListLitExpr>(loc, std::move(elements));
}

KExpr *Parser::parsePostfix(KExpr *base) {
  for (;;) {
    if (at(tok::l_square)) {
      SourceLocation loc = tokLoc();
      advance();
      std::vector<KExpr *> idx;
      while (!at(tok::r_square)) {
        KExpr *e = parseExpr();
        if (!e) return nullptr;
        idx.push_back(e);
        if (!consume(tok::comma)) break;
      }
      if (!expect(tok::r_square, "']'")) return nullptr;
      base = ctx_.create<KIndexExpr>(loc, base, std::move(idx));
      continue;
    }
    if (at(tok::dot)) {
      SourceLocation loc = tokLoc();
      advance();
      if (!atIdentLike()) {
        diag_.report(tokLoc(), diag::err_expected) << "method name";
        return nullptr;
      }
      std::string method = tokText();
      advance();
      if (!expect(tok::l_paren, "'('")) return nullptr;
      std::vector<KExpr *> args = parseCallArgs();
      expect(tok::r_paren, "')'");
      base =
          ctx_.create<KCallExpr>(loc, base, std::move(method), std::move(args));
      continue;
    }
    break;
  }
  return base;
}

std::vector<KExpr *> Parser::parseCallArgs() {
  std::vector<KExpr *> args;
  while (!at(tok::r_paren)) {
    KExpr *e = parseExpr();
    if (!e) return args;
    args.push_back(e);
    if (!consume(tok::comma)) break;
  }
  return args;
}

}  // namespace aiec
