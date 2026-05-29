// KDecl.h - top-level AST declarations.
//
// Mirrors clang/include/clang/AST/Decl.h: class hierarchy with
// LLVM-style RTTI.

#ifndef AIEC_AST_K_DECL_H
#define AIEC_AST_K_DECL_H

#include "aiec/AST/KExpr.h"
#include "aiec/AST/KStmt.h"
#include "aiec/Basic/SourceLocation.h"

#include <optional>
#include <string>
#include <vector>

namespace aiec {

class KDecl {
public:
  enum class Kind {
    Module,
    Kernel,
    ExternFn,
    Buffer,
    Route,
    Catalog,
    OnCore,
    OnController,
  };

  KDecl(Kind k, SourceLocation loc) : kind_(k), loc_(loc) {}
  virtual ~KDecl() = default;
  Kind getKind() const { return kind_; }
  SourceLocation getLocation() const { return loc_; }

private:
  Kind kind_;
  SourceLocation loc_;
};

// A memref type: positional dim sizes + element type + memory space.
//   memref<dim1, dim2, ..., elemType, memspace>
// All dims are constexpr ints; the last two items in `< >` are the elem
// type keyword and memory space integer.
struct KMemrefType {
  std::vector<KExpr *> dims;        // dim size expressions
  std::string elemType;             // "i32" / "i8" / etc.
  int memSpace = 0;                 // 0 = DDR, 1 = L2, 2 = L1
};

// extern fn NAME(arg: memref<...>, ...)
struct KExternFnArg {
  std::string name;
  KMemrefType type;
};
class KExternFnDecl : public KDecl {
public:
  KExternFnDecl(SourceLocation loc, std::string name,
                std::vector<KExternFnArg> args, std::string implPath = {},
                std::string linkWith = {})
      : KDecl(Kind::ExternFn, loc), name_(std::move(name)),
        args_(std::move(args)), implPath_(std::move(implPath)),
        linkWith_(std::move(linkWith)) {}
  const std::string &getName() const { return name_; }
  const std::vector<KExternFnArg> &getArgs() const { return args_; }
  // Path (relative to the .aiec) to the MLIR template that implements this
  // function, or empty if none. The frontend emits the template verbatim
  // with placeholder substitution — it has no knowledge of the body.
  const std::string &getImplPath() const { return implPath_; }
  // Object file (e.g. "matmul.o") for a precompiled bareptr ukernel, or empty.
  // When set, the fn is emitted as a `func.func private` declaration with the
  // bareptr ABI (each arg -> (memref<elem, space>, index); link_with attr) and
  // call sites pass extract_strided_metadata (base, offset) pairs. Used for
  // bf16/i8 matmul ukernels, which cannot be pure-vectorized.
  const std::string &getLinkWith() const { return linkWith_; }
  bool isBareptr() const { return !linkWith_.empty(); }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::ExternFn; }

private:
  std::string name_;
  std::vector<KExternFnArg> args_;
  std::string implPath_;
  std::string linkWith_;
};

// Catalog entries: shim [c in 0..COLS] -> tile(c, 0)
enum class CatalogKind { Shim, Memtile, Core, Controller };
struct KIndexBinding {
  std::string name;                 // e.g. "c"
  KExpr *lo;
  KExpr *hi;
};
class KCatalogDecl : public KDecl {
public:
  KCatalogDecl(SourceLocation loc, CatalogKind ck,
               std::vector<KIndexBinding> indices,
               KExpr *colExpr, KExpr *rowExpr)
      : KDecl(Kind::Catalog, loc), ck_(ck), indices_(std::move(indices)),
        colExpr_(colExpr), rowExpr_(rowExpr) {}
  CatalogKind getCatalogKind() const { return ck_; }
  const std::vector<KIndexBinding> &getIndices() const { return indices_; }
  KExpr *getColExpr() const { return colExpr_; }
  KExpr *getRowExpr() const { return rowExpr_; }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::Catalog; }

private:
  CatalogKind ck_;
  std::vector<KIndexBinding> indices_;
  KExpr *colExpr_;
  KExpr *rowExpr_;
};

// Placement (used in `on PLACEMENT` and `via PLACEMENT`).
struct KPlacement {
  std::string catalog;              // "memtile", "core", "shim", "controller"
  // Per-axis: each KExpr is an index expression, a wildcard (`*`), or
  // a constexpr int. The number of axes matches the catalog's arity.
  std::vector<KExpr *> axes;
};

// buffer NAME[indices]: memref<...> on PLACEMENT [when COND]
class KBufferDecl : public KDecl {
public:
  KBufferDecl(SourceLocation loc, std::string name,
              std::vector<std::string> indexVars, KMemrefType type,
              KPlacement placement, KExpr *whenCond = nullptr,
              KExpr *depthExpr = nullptr)
      : KDecl(Kind::Buffer, loc), name_(std::move(name)),
        indexVars_(std::move(indexVars)), type_(std::move(type)),
        placement_(std::move(placement)), whenCond_(whenCond),
        depthExpr_(depthExpr) {}
  const std::string &getName() const { return name_; }
  const std::vector<std::string> &getIndexVars() const { return indexVars_; }
  const KMemrefType &getType() const { return type_; }
  const KPlacement &getPlacement() const { return placement_; }
  KExpr *getWhenCondition() const { return whenCond_; }
  // Objectfifo depth (ping-pong buffering) as a constexpr. Null = default 1.
  // It is a constexpr so the depth can be a template parameter, keeping the
  // single/double-buffering decision at instantiation, not in the frontend.
  KExpr *getDepthExpr() const { return depthExpr_; }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::Buffer; }

private:
  std::string name_;
  std::vector<std::string> indexVars_;
  KMemrefType type_;
  KPlacement placement_;
  KExpr *whenCond_;
  KExpr *depthExpr_;
};

// One side of a route — describes a strided view of either a tensor
// (kernel binding) or a buffer. Used in route declarations and in
// `issue` statements (for the parameterized side).
struct KRouteSide {
  // What this side refers to. Exactly one of bufName/binding will be
  // non-empty depending on the call site.
  std::string bufName;                    // buffer name (e.g. "a_l2") or empty
  std::vector<KExpr *> bufIndices;        // [c, r] indices on the buffer
  std::vector<KExpr *> offsets;
  std::vector<KExpr *> sizes;
  std::vector<KExpr *> strides;
};

// Channel spec on one side of `via`.
struct KViaChannel {
  KPlacement placement;
  bool mm2s = true;                       // true: MM2S, false: S2MM
  KExpr *channelId = nullptr;             // optional explicit channel ID
};

enum class RouteType { Packet, Circuit };
enum class RouteParameterization {
  Persistent,             // both sides persistent (no issue)
  SourceParameterized,    // source side comes from issue
  TargetParameterized,    // target side comes from issue
};

class KRouteDecl : public KDecl {
public:
  KRouteDecl(SourceLocation loc, std::string name,
             std::vector<std::string> indexVars, RouteType rt,
             RouteParameterization param,
             std::optional<KRouteSide> source,
             std::optional<KRouteSide> target,
             KViaChannel viaSrc, KViaChannel viaDst,
             KExpr *whenCond = nullptr)
      : KDecl(Kind::Route, loc), name_(std::move(name)),
        indexVars_(std::move(indexVars)), routeType_(rt), param_(param),
        source_(std::move(source)), target_(std::move(target)),
        viaSrc_(std::move(viaSrc)), viaDst_(std::move(viaDst)),
        whenCond_(whenCond) {}
  const std::string &getName() const { return name_; }
  const std::vector<std::string> &getIndexVars() const { return indexVars_; }
  RouteType getRouteType() const { return routeType_; }
  RouteParameterization getParameterization() const { return param_; }
  const std::optional<KRouteSide> &getSource() const { return source_; }
  const std::optional<KRouteSide> &getTarget() const { return target_; }
  const KViaChannel &getViaSource() const { return viaSrc_; }
  const KViaChannel &getViaTarget() const { return viaDst_; }
  KExpr *getWhenCondition() const { return whenCond_; }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::Route; }

private:
  std::string name_;
  std::vector<std::string> indexVars_;
  RouteType routeType_;
  RouteParameterization param_;
  std::optional<KRouteSide> source_;
  std::optional<KRouteSide> target_;
  KViaChannel viaSrc_;
  KViaChannel viaDst_;
  KExpr *whenCond_;
};

// on core[c, r] [stack_size=<expr>] { ... }
class KOnCoreDecl : public KDecl {
public:
  KOnCoreDecl(SourceLocation loc, std::vector<std::string> indexVars,
              KExpr *stackSize, std::vector<KStmt *> body)
      : KDecl(Kind::OnCore, loc), indexVars_(std::move(indexVars)),
        stackSize_(stackSize), body_(std::move(body)) {}
  const std::vector<std::string> &getIndexVars() const { return indexVars_; }
  KExpr *getStackSize() const { return stackSize_; }
  const std::vector<KStmt *> &getBody() const { return body_; }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::OnCore; }

private:
  std::vector<std::string> indexVars_;
  KExpr *stackSize_;
  std::vector<KStmt *> body_;
};

// on controller { ... }
class KOnControllerDecl : public KDecl {
public:
  KOnControllerDecl(SourceLocation loc, std::vector<KStmt *> body)
      : KDecl(Kind::OnController, loc), body_(std::move(body)) {}
  const std::vector<KStmt *> &getBody() const { return body_; }
  static bool classof(const KDecl *d) {
    return d->getKind() == Kind::OnController;
  }

private:
  std::vector<KStmt *> body_;
};

// kernel NAME<params>(args) where ... { decls... }
struct KTemplateParam {
  std::string name;
  std::string type;                 // "int"
};
struct KKernelArg {
  std::string name;
  KMemrefType type;                 // for tensor args
  std::string scalarType;           // non-empty for scalar args (e.g. "i32")
  bool isScalar() const { return !scalarType.empty(); }
};

// A kernel attribute: `name = expr` declared in the kernel header (before
// the body). E.g. `control_packets = true`.
struct KKernelAttr {
  std::string name;
  KExpr *value;
};

class KKernelDecl : public KDecl {
public:
  KKernelDecl(SourceLocation loc, std::string name,
              std::vector<KTemplateParam> templateParams,
              std::vector<KKernelArg> kernelArgs,
              std::vector<KExpr *> whereConstraints,
              std::vector<KDecl *> body,
              std::vector<KKernelAttr> attrs = {})
      : KDecl(Kind::Kernel, loc), name_(std::move(name)),
        templateParams_(std::move(templateParams)),
        kernelArgs_(std::move(kernelArgs)),
        whereConstraints_(std::move(whereConstraints)), body_(std::move(body)),
        attrs_(std::move(attrs)) {}
  const std::string &getName() const { return name_; }
  const std::vector<KTemplateParam> &getTemplateParams() const {
    return templateParams_;
  }
  const std::vector<KKernelArg> &getKernelArgs() const { return kernelArgs_; }
  const std::vector<KExpr *> &getWhereConstraints() const {
    return whereConstraints_;
  }
  const std::vector<KDecl *> &getBody() const { return body_; }
  const std::vector<KKernelAttr> &getAttrs() const { return attrs_; }
  // Returns the attribute's value expr, or nullptr if not declared.
  KExpr *getAttr(const std::string &name) const {
    for (auto &a : attrs_)
      if (a.name == name) return a.value;
    return nullptr;
  }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::Kernel; }

private:
  std::string name_;
  std::vector<KTemplateParam> templateParams_;
  std::vector<KKernelArg> kernelArgs_;
  std::vector<KExpr *> whereConstraints_;
  std::vector<KDecl *> body_;
  std::vector<KKernelAttr> attrs_;
};

class KModuleDecl : public KDecl {
public:
  KModuleDecl(SourceLocation loc, std::string name, std::vector<KDecl *> decls)
      : KDecl(Kind::Module, loc), name_(std::move(name)),
        decls_(std::move(decls)) {}
  const std::string &getName() const { return name_; }
  const std::vector<KDecl *> &getDecls() const { return decls_; }
  static bool classof(const KDecl *d) { return d->getKind() == Kind::Module; }

private:
  std::string name_;
  std::vector<KDecl *> decls_;
};

// LLVM-style cast/dyn_cast helpers
template <typename T> bool isa(const KDecl *d) { return d && T::classof(d); }
template <typename T> bool isa(const KStmt *s) { return s && T::classof(s); }
template <typename T> bool isa(const KExpr *e) { return e && T::classof(e); }
template <typename T> T *cast(KDecl *d) { return static_cast<T *>(d); }
template <typename T> const T *cast(const KDecl *d) {
  return static_cast<const T *>(d);
}
template <typename T> T *cast(KStmt *s) { return static_cast<T *>(s); }
template <typename T> const T *cast(const KStmt *s) {
  return static_cast<const T *>(s);
}
template <typename T> T *cast(KExpr *e) { return static_cast<T *>(e); }
template <typename T> const T *cast(const KExpr *e) {
  return static_cast<const T *>(e);
}
template <typename T> T *dyn_cast(KDecl *d) {
  return isa<T>(d) ? cast<T>(d) : nullptr;
}
template <typename T> const T *dyn_cast(const KDecl *d) {
  return isa<T>(d) ? cast<T>(d) : nullptr;
}
template <typename T> T *dyn_cast(KStmt *s) {
  return isa<T>(s) ? cast<T>(s) : nullptr;
}
template <typename T> const T *dyn_cast(const KStmt *s) {
  return isa<T>(s) ? cast<T>(s) : nullptr;
}
template <typename T> T *dyn_cast(KExpr *e) {
  return isa<T>(e) ? cast<T>(e) : nullptr;
}
template <typename T> const T *dyn_cast(const KExpr *e) {
  return isa<T>(e) ? cast<T>(e) : nullptr;
}

} // namespace aiec

#endif // AIEC_AST_K_DECL_H
