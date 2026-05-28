// KExpr.h - AST expressions.
//
// Expression class hierarchy with LLVM-style RTTI (classof + a Kind
// discriminator). Mirrors clang/include/clang/AST/Expr.h.
//
// Expressions appear in:
//   - buffer dim labels: `m = 32`, `kblock = K_mm1 / 64`
//   - slice constructors: `chunks = 2`, `offset = c * 32`
//   - tile coordinates: `tile(c, r + 2)`
//   - where clauses: `K_mm1 % 64 == 0`
//   - range expressions: `0 .. COLS`
//   - statement expressions: `out := out * act_scale`

#ifndef AIEC_AST_K_EXPR_H
#define AIEC_AST_K_EXPR_H

#include "aiec/Basic/SourceLocation.h"

#include <cstdint>
#include <string>
#include <vector>

namespace aiec {

class KExpr {
public:
  enum class Kind {
    IntLit,
    Ident,
    BinOp,
    UnaryOp,
    Ternary,            // `cond ? a : b`
    Call,
    Wildcard,           // `*` in placements
    Range,              // `lo .. hi`
    NamedArg,           // `name = expr`
    MemberAccess,
    Index,              // `arr[i, j]`
    ListLit,            // `[e1, e2, ...]` for offsets/sizes/strides
  };

  KExpr(Kind k, SourceLocation loc) : kind_(k), loc_(loc) {}
  virtual ~KExpr() = default;

  Kind getKind() const { return kind_; }
  SourceLocation getLocation() const { return loc_; }

private:
  Kind kind_;
  SourceLocation loc_;
};

class KIntLitExpr : public KExpr {
public:
  KIntLitExpr(SourceLocation loc, int64_t v)
      : KExpr(Kind::IntLit, loc), value_(v) {}
  int64_t getValue() const { return value_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::IntLit; }

private:
  int64_t value_;
};

class KIdentExpr : public KExpr {
public:
  KIdentExpr(SourceLocation loc, std::string name)
      : KExpr(Kind::Ident, loc), name_(std::move(name)) {}
  const std::string &getName() const { return name_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::Ident; }

private:
  std::string name_;
};

enum class BinOpKind {
  Add, Sub, Mul, Div, Mod,
  Eq, Ne, Lt, Le, Gt, Ge,
  And, Or,
};

class KBinOpExpr : public KExpr {
public:
  KBinOpExpr(SourceLocation loc, BinOpKind op, KExpr *lhs, KExpr *rhs)
      : KExpr(Kind::BinOp, loc), op_(op), lhs_(lhs), rhs_(rhs) {}
  BinOpKind getOp() const { return op_; }
  KExpr *getLHS() const { return lhs_; }
  KExpr *getRHS() const { return rhs_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::BinOp; }

private:
  BinOpKind op_;
  KExpr *lhs_;
  KExpr *rhs_;
};

enum class UnaryOpKind { Neg, Not };

class KUnaryOpExpr : public KExpr {
public:
  KUnaryOpExpr(SourceLocation loc, UnaryOpKind op, KExpr *operand)
      : KExpr(Kind::UnaryOp, loc), op_(op), operand_(operand) {}
  UnaryOpKind getOp() const { return op_; }
  KExpr *getOperand() const { return operand_; }
  static bool classof(const KExpr *e) {
    return e->getKind() == Kind::UnaryOp;
  }

private:
  UnaryOpKind op_;
  KExpr *operand_;
};

// `name = expr` used inside slice constructor / route arg lists.
class KNamedArgExpr : public KExpr {
public:
  KNamedArgExpr(SourceLocation loc, std::string name, KExpr *value)
      : KExpr(Kind::NamedArg, loc), name_(std::move(name)), value_(value) {}
  const std::string &getName() const { return name_; }
  KExpr *getValue() const { return value_; }
  static bool classof(const KExpr *e) {
    return e->getKind() == Kind::NamedArg;
  }

private:
  std::string name_;
  KExpr *value_;
};

// Function/method call: name(args...) or receiver.method(args...).
// receiver may be null for plain calls.
class KCallExpr : public KExpr {
public:
  KCallExpr(SourceLocation loc, KExpr *receiver, std::string callee,
            std::vector<KExpr *> args)
      : KExpr(Kind::Call, loc), receiver_(receiver),
        callee_(std::move(callee)), args_(std::move(args)) {}
  KExpr *getReceiver() const { return receiver_; } // may be null
  const std::string &getCallee() const { return callee_; }
  const std::vector<KExpr *> &getArgs() const { return args_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::Call; }

private:
  KExpr *receiver_;
  std::string callee_;
  std::vector<KExpr *> args_;
};

// `*` placeholder (used in placement expressions like core[*, r]).
class KWildcardExpr : public KExpr {
public:
  KWildcardExpr(SourceLocation loc) : KExpr(Kind::Wildcard, loc) {}
  static bool classof(const KExpr *e) {
    return e->getKind() == Kind::Wildcard;
  }
};

// `lo .. hi` half-open range.
class KRangeExpr : public KExpr {
public:
  KRangeExpr(SourceLocation loc, KExpr *lo, KExpr *hi)
      : KExpr(Kind::Range, loc), lo_(lo), hi_(hi) {}
  KExpr *getLo() const { return lo_; }
  KExpr *getHi() const { return hi_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::Range; }

private:
  KExpr *lo_;
  KExpr *hi_;
};

// `arr[i, j, ...]` indexing. Used for buffer / catalog references.
class KIndexExpr : public KExpr {
public:
  KIndexExpr(SourceLocation loc, KExpr *base, std::vector<KExpr *> indices)
      : KExpr(Kind::Index, loc), base_(base), indices_(std::move(indices)) {}
  KExpr *getBase() const { return base_; }
  const std::vector<KExpr *> &getIndices() const { return indices_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::Index; }

private:
  KExpr *base_;
  std::vector<KExpr *> indices_;
};

// `cond ? then : else`
class KTernaryExpr : public KExpr {
public:
  KTernaryExpr(SourceLocation loc, KExpr *cond, KExpr *thenE, KExpr *elseE)
      : KExpr(Kind::Ternary, loc), cond_(cond), then_(thenE), else_(elseE) {}
  KExpr *getCond() const { return cond_; }
  KExpr *getThen() const { return then_; }
  KExpr *getElse() const { return else_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::Ternary; }

private:
  KExpr *cond_, *then_, *else_;
};

// `[e1, e2, e3]` literal list (used for offsets/sizes/strides).
class KListLitExpr : public KExpr {
public:
  KListLitExpr(SourceLocation loc, std::vector<KExpr *> elements)
      : KExpr(Kind::ListLit, loc), elements_(std::move(elements)) {}
  const std::vector<KExpr *> &getElements() const { return elements_; }
  static bool classof(const KExpr *e) { return e->getKind() == Kind::ListLit; }

private:
  std::vector<KExpr *> elements_;
};

} // namespace aiec

#endif // AIEC_AST_K_EXPR_H
