// ConstExpr.h - constant-expression evaluator.
//
// Walks a KExpr tree under a scope (name → int) and returns the integer
// result, or reports an error and returns 0.
//
// Used for:
//   - kernel template parameters (`COLS=2`, etc.)
//   - buffer dim labels (`kblock = K_mm1 / 64` → 2 when K_mm1=128)
//   - slice constructor args (`offset = c * 32` → 32 when c=1)
//   - `where` clauses (`K_mm1 % 64 == 0`)
//   - `for` / `reduce` ranges
//   - `forall` dims

#ifndef AIEC_CODEGEN_CONST_EXPR_H
#define AIEC_CODEGEN_CONST_EXPR_H

#include "aiec/AST/KExpr.h"
#include "aiec/Basic/Diagnostic.h"

#include <map>
#include <string>

namespace aiec {

class ConstExprEvaluator {
public:
  ConstExprEvaluator(DiagnosticEngine &diag) : diag_(diag) {}

  // Push/pop a scope frame. Use RAII helper Scope below.
  void push() { stack_.emplace_back(); }
  void pop() { stack_.pop_back(); }
  void bind(const std::string &name, int64_t value) {
    stack_.back()[name] = value;
  }
  bool lookup(const std::string &name, int64_t &out) const {
    for (auto it = stack_.rbegin(); it != stack_.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end()) { out = f->second; return true; }
    }
    return false;
  }

  // Initialize global params (template params + derived M, N).
  void bindGlobal(const std::string &name, int64_t value) {
    globals_[name] = value;
  }

  // Evaluate. Returns 0 + reports diagnostic if expr is non-constant.
  int64_t evalInt(const KExpr *e);
  bool evalBool(const KExpr *e);

  // True if name has a binding in any scope (incl. globals).
  bool inScope(const std::string &name) const {
    int64_t dummy;
    return lookup(name, dummy) || globals_.find(name) != globals_.end();
  }

private:
  int64_t evalIdent(const KIdentExpr *e);
  int64_t evalBinOp(const KBinOpExpr *e);
  int64_t evalUnary(const KUnaryOpExpr *e);

  DiagnosticEngine &diag_;
  std::vector<std::map<std::string, int64_t>> stack_;
  std::map<std::string, int64_t> globals_;
};

// RAII scope helper.
class ConstExprScope {
public:
  explicit ConstExprScope(ConstExprEvaluator &e) : eval_(e) { e.push(); }
  ~ConstExprScope() { eval_.pop(); }

private:
  ConstExprEvaluator &eval_;
};

} // namespace aiec

#endif // AIEC_CODEGEN_CONST_EXPR_H
