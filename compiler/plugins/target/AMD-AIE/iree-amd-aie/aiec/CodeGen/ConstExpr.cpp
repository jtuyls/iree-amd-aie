#include "aiec/CodeGen/ConstExpr.h"

#include "aiec/AST/KDecl.h"

namespace aiec {

int64_t ConstExprEvaluator::evalInt(const KExpr *e) {
  if (!e) return 0;
  if (auto *lit = dyn_cast<KIntLitExpr>(e))   return lit->getValue();
  if (auto *id = dyn_cast<KIdentExpr>(e))     return evalIdent(id);
  if (auto *bin = dyn_cast<KBinOpExpr>(e))    return evalBinOp(bin);
  if (auto *un = dyn_cast<KUnaryOpExpr>(e))   return evalUnary(un);
  if (auto *tern = dyn_cast<KTernaryExpr>(e))
    return evalInt(tern->getCond()) != 0 ? evalInt(tern->getThen())
                                          : evalInt(tern->getElse());
  diag_.report(e->getLocation(), diag::err_unexpected_token)
      << "<non-constant expression>";
  return 0;
}

bool ConstExprEvaluator::evalBool(const KExpr *e) { return evalInt(e) != 0; }

int64_t ConstExprEvaluator::evalIdent(const KIdentExpr *e) {
  int64_t v;
  if (lookup(e->getName(), v)) return v;
  auto it = globals_.find(e->getName());
  if (it != globals_.end()) return it->second;
  diag_.report(e->getLocation(), diag::err_undefined_identifier)
      << e->getName();
  return 0;
}

int64_t ConstExprEvaluator::evalBinOp(const KBinOpExpr *e) {
  int64_t l = evalInt(e->getLHS());
  int64_t r = evalInt(e->getRHS());
  switch (e->getOp()) {
  case BinOpKind::Add: return l + r;
  case BinOpKind::Sub: return l - r;
  case BinOpKind::Mul: return l * r;
  case BinOpKind::Div:
    if (r == 0) {
      diag_.report(e->getLocation(), diag::err_unexpected_token) << "/0";
      return 0;
    }
    return l / r;
  case BinOpKind::Mod:
    if (r == 0) {
      diag_.report(e->getLocation(), diag::err_unexpected_token) << "%0";
      return 0;
    }
    return l % r;
  case BinOpKind::Eq: return l == r ? 1 : 0;
  case BinOpKind::Ne: return l != r ? 1 : 0;
  case BinOpKind::Lt: return l <  r ? 1 : 0;
  case BinOpKind::Le: return l <= r ? 1 : 0;
  case BinOpKind::Gt: return l >  r ? 1 : 0;
  case BinOpKind::Ge: return l >= r ? 1 : 0;
  case BinOpKind::And: return (l && r) ? 1 : 0;
  case BinOpKind::Or:  return (l || r) ? 1 : 0;
  }
  return 0;
}

int64_t ConstExprEvaluator::evalUnary(const KUnaryOpExpr *e) {
  int64_t v = evalInt(e->getOperand());
  switch (e->getOp()) {
  case UnaryOpKind::Neg: return -v;
  case UnaryOpKind::Not: return v == 0 ? 1 : 0;
  }
  return 0;
}

} // namespace aiec
