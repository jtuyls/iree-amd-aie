// KStmt.h - AST statements (compute-body and controller-body ops).

#ifndef AIEC_AST_K_STMT_H
#define AIEC_AST_K_STMT_H

#include "aiec/AST/KExpr.h"
#include "aiec/Basic/SourceLocation.h"

#include <string>
#include <vector>

namespace aiec {

class KStmt {
public:
  enum class Kind {
    Barrier,
    Forall,
    Reduce,
    Let,
    Zero,
    Release,
    AssignMul,
    CallStmt,
    For,
    Issue,
    Wait,
  };

  KStmt(Kind k, SourceLocation loc) : kind_(k), loc_(loc) {}
  virtual ~KStmt() = default;
  Kind getKind() const { return kind_; }
  SourceLocation getLocation() const { return loc_; }

private:
  Kind kind_;
  SourceLocation loc_;
};

class KBarrierStmt : public KStmt {
public:
  KBarrierStmt(SourceLocation loc) : KStmt(Kind::Barrier, loc) {}
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Barrier; }
};

class KForallStmt : public KStmt {
public:
  KForallStmt(SourceLocation loc, std::vector<std::string> indVars,
              std::vector<KExpr *> dims, std::vector<KStmt *> body)
      : KStmt(Kind::Forall, loc), indVars_(std::move(indVars)),
        dims_(std::move(dims)), body_(std::move(body)) {}
  const std::vector<std::string> &getIndVars() const { return indVars_; }
  const std::vector<KExpr *> &getDims() const { return dims_; }
  const std::vector<KStmt *> &getBody() const { return body_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Forall; }

private:
  std::vector<std::string> indVars_;
  std::vector<KExpr *> dims_;
  std::vector<KStmt *> body_;
};

class KReduceStmt : public KStmt {
public:
  KReduceStmt(SourceLocation loc, std::string indVar, KExpr *lo, KExpr *hi,
              std::vector<KStmt *> body)
      : KStmt(Kind::Reduce, loc), indVar_(std::move(indVar)), lo_(lo), hi_(hi),
        body_(std::move(body)) {}
  const std::string &getIndVar() const { return indVar_; }
  KExpr *getLo() const { return lo_; }
  KExpr *getHi() const { return hi_; }
  const std::vector<KStmt *> &getBody() const { return body_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Reduce; }

private:
  std::string indVar_;
  KExpr *lo_;
  KExpr *hi_;
  std::vector<KStmt *> body_;
};

enum class AcquireRole { Consume, Produce };
class KLetStmt : public KStmt {
public:
  KLetStmt(SourceLocation loc, std::string name, KExpr *bufRef,
           AcquireRole role)
      : KStmt(Kind::Let, loc), name_(std::move(name)), bufRef_(bufRef),
        role_(role) {}
  const std::string &getName() const { return name_; }
  KExpr *getBufRef() const { return bufRef_; }
  AcquireRole getRole() const { return role_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Let; }

private:
  std::string name_;
  KExpr *bufRef_;
  AcquireRole role_;
};

class KZeroStmt : public KStmt {
public:
  KZeroStmt(SourceLocation loc, std::string handle)
      : KStmt(Kind::Zero, loc), handle_(std::move(handle)) {}
  const std::string &getHandle() const { return handle_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Zero; }

private:
  std::string handle_;
};

class KReleaseStmt : public KStmt {
public:
  KReleaseStmt(SourceLocation loc, std::string handle)
      : KStmt(Kind::Release, loc), handle_(std::move(handle)) {}
  const std::string &getHandle() const { return handle_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Release; }

private:
  std::string handle_;
};

class KAssignMulStmt : public KStmt {
public:
  KAssignMulStmt(SourceLocation loc, std::string handle, KExpr *rhs)
      : KStmt(Kind::AssignMul, loc), handle_(std::move(handle)), rhs_(rhs) {}
  const std::string &getHandle() const { return handle_; }
  KExpr *getRHS() const { return rhs_; }
  static bool classof(const KStmt *s) {
    return s->getKind() == Kind::AssignMul;
  }

private:
  std::string handle_;
  KExpr *rhs_;
};

class KCallStmt : public KStmt {
public:
  KCallStmt(SourceLocation loc, std::string callee,
            std::vector<std::string> args)
      : KStmt(Kind::CallStmt, loc), callee_(std::move(callee)),
        args_(std::move(args)) {}
  const std::string &getCallee() const { return callee_; }
  const std::vector<std::string> &getArgs() const { return args_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::CallStmt; }

private:
  std::string callee_;
  std::vector<std::string> args_;
};

class KForStmt : public KStmt {
public:
  KForStmt(SourceLocation loc, std::string indVar, KExpr *lo, KExpr *hi,
           std::vector<KStmt *> body)
      : KStmt(Kind::For, loc), indVar_(std::move(indVar)), lo_(lo), hi_(hi),
        body_(std::move(body)) {}
  const std::string &getIndVar() const { return indVar_; }
  KExpr *getLo() const { return lo_; }
  KExpr *getHi() const { return hi_; }
  const std::vector<KStmt *> &getBody() const { return body_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::For; }

private:
  std::string indVar_;
  KExpr *lo_;
  KExpr *hi_;
  std::vector<KStmt *> body_;
};

// `issue(route[idx], source/target=BIND, offsets=[...], sizes=[...],
//        strides=[...])`. The slice describes the parameterized side of
// the route (the L3 side for L3↔L2 routes).
class KIssueStmt : public KStmt {
public:
  KIssueStmt(SourceLocation loc, std::string routeName,
             std::vector<KExpr *> indices,
             std::string srcBinding, std::string dstBinding,
             std::vector<KExpr *> offsets, std::vector<KExpr *> sizes,
             std::vector<KExpr *> strides, KExpr *bdId = nullptr)
      : KStmt(Kind::Issue, loc), routeName_(std::move(routeName)),
        indices_(std::move(indices)), srcBinding_(std::move(srcBinding)),
        dstBinding_(std::move(dstBinding)), offsets_(std::move(offsets)),
        sizes_(std::move(sizes)), strides_(std::move(strides)), bdId_(bdId) {}
  const std::string &getRouteName() const { return routeName_; }
  const std::vector<KExpr *> &getIndices() const { return indices_; }
  const std::string &getSrcBinding() const { return srcBinding_; }
  const std::string &getDstBinding() const { return dstBinding_; }
  const std::vector<KExpr *> &getOffsets() const { return offsets_; }
  const std::vector<KExpr *> &getSizes() const { return sizes_; }
  const std::vector<KExpr *> &getStrides() const { return strides_; }
  // Logical buffer-descriptor id for this transfer (constexpr). Null if the
  // source omits it. The value is logical; the device-aware backend maps it
  // onto the tile's physical BD pool.
  KExpr *getBdId() const { return bdId_; }
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Issue; }

private:
  std::string routeName_;
  std::vector<KExpr *> indices_;
  std::string srcBinding_;
  std::string dstBinding_;
  std::vector<KExpr *> offsets_;
  std::vector<KExpr *> sizes_;
  std::vector<KExpr *> strides_;
  KExpr *bdId_;
};

// `wait` blocks on outstanding issued transfers. Forms:
//   wait                     — all outstanding transfers (waitAll)
//   wait(route)              — all outstanding instances of `route`
//   wait(route[i])           — the instance at index i
//   wait(route[lo..hi])      — instances in [lo, hi)  (one HW sync)
// A single emitted dma_wait can block on several tokens at once.
class KWaitStmt : public KStmt {
public:
  KWaitStmt(SourceLocation loc, std::string routeName = {},
            KExpr *idxLo = nullptr, KExpr *idxHi = nullptr)
      : KStmt(Kind::Wait, loc), routeName_(std::move(routeName)), idxLo_(idxLo),
        idxHi_(idxHi) {}
  const std::string &getRouteName() const { return routeName_; }  // "" = all
  KExpr *getIdxLo() const { return idxLo_; }   // null = all instances
  KExpr *getIdxHi() const { return idxHi_; }   // non-null = range [lo, hi)
  static bool classof(const KStmt *s) { return s->getKind() == Kind::Wait; }

private:
  std::string routeName_;
  KExpr *idxLo_;
  KExpr *idxHi_;
};

} // namespace aiec

#endif // AIEC_AST_K_STMT_H
