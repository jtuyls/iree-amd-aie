// Emitter.cpp - in-tree OpBuilder-based AST -> AMDAIE IR codegen.
//
// Replaces the standalone frontend's textual string emitter: walks the
// parsed .aiec AST and constructs AMDAIE/MLIR ops in memory via OpBuilder,
// so the result is MLIR-verified (no string assembly). The resolution
// layer (catalog/buffers/routes) is dialect-independent and shared with the
// original design; only the emission constructs ops instead of text.
//
// Ported incrementally; each landed phase keeps the in-tree build green.
// The IREE HAL/Stream wrapper around the kernel remains textual plumbing.

#include "aiec/CodeGen/Emitter.h"

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "aiec/AST/KDecl.h"
#include "aiec/CodeGen/ConstExpr.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/IR/AMDAIETypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::iree_compiler;  // brings AMDAIE and IREE::HAL into scope

namespace aiec {
namespace {

std::string itos(int64_t v) { return std::to_string(v); }

// ─── Loop induction-variable use analysis ──────────────────────────────
//
// A loop lowers to a rolled `scf.for` unless its body references the
// induction variable. A loop whose iv appears in a hardware-instance
// subscript (`a_l3_l2[r]`, `core[c, r]`) selects a distinct hardware object
// per iteration and *must* be unrolled to produce valid IR; a loop whose iv
// only feeds offsets/sizes could roll, but materializing the iv there needs a
// runtime (SSA) value the emitter doesn't yet produce — so for now any iv use
// keeps the loop unrolled. Compute (core) loops are iv-independent (data is
// cycled by the object-fifos, not indexed by the iv), so they roll — which is
// what keeps the core program within instruction memory at scale.
//
// Unknown expression kinds conservatively report a reference, so an
// unrecognized iv use never causes an incorrect roll.
bool exprRefsName(const KExpr *e, const std::string &name) {
  if (!e) return false;
  if (auto *id = dyn_cast<KIdentExpr>(e)) return id->getName() == name;
  if (auto *bo = dyn_cast<KBinOpExpr>(e))
    return exprRefsName(bo->getLHS(), name) || exprRefsName(bo->getRHS(), name);
  if (auto *u = dyn_cast<KUnaryOpExpr>(e))
    return exprRefsName(u->getOperand(), name);
  if (auto *t = dyn_cast<KTernaryExpr>(e))
    return exprRefsName(t->getCond(), name) ||
           exprRefsName(t->getThen(), name) || exprRefsName(t->getElse(), name);
  if (auto *c = dyn_cast<KCallExpr>(e)) {
    if (exprRefsName(c->getReceiver(), name)) return true;
    for (auto *a : c->getArgs())
      if (exprRefsName(a, name)) return true;
    return false;
  }
  if (auto *na = dyn_cast<KNamedArgExpr>(e))
    return exprRefsName(na->getValue(), name);
  if (auto *r = dyn_cast<KRangeExpr>(e))
    return exprRefsName(r->getLo(), name) || exprRefsName(r->getHi(), name);
  if (auto *idx = dyn_cast<KIndexExpr>(e)) {
    if (exprRefsName(idx->getBase(), name)) return true;
    for (auto *i : idx->getIndices())
      if (exprRefsName(i, name)) return true;
    return false;
  }
  if (auto *l = dyn_cast<KListLitExpr>(e)) {
    for (auto *el : l->getElements())
      if (exprRefsName(el, name)) return true;
    return false;
  }
  if (isa<KIntLitExpr>(e) || isa<KWildcardExpr>(e)) return false;
  return true;  // unrecognized kind: assume it references the name
}

bool bodyRefsName(const std::vector<KStmt *> &body, const std::string &name);

bool stmtRefsName(const KStmt *s, const std::string &name) {
  if (auto *l = dyn_cast<KLetStmt>(s))
    return exprRefsName(l->getBufRef(), name);
  if (auto *am = dyn_cast<KAssignMulStmt>(s))
    return exprRefsName(am->getRHS(), name);
  if (auto *red = dyn_cast<KReduceStmt>(s))
    return exprRefsName(red->getLo(), name) ||
           exprRefsName(red->getHi(), name) ||
           bodyRefsName(red->getBody(), name);
  if (auto *f = dyn_cast<KForStmt>(s))
    return exprRefsName(f->getLo(), name) || exprRefsName(f->getHi(), name) ||
           bodyRefsName(f->getBody(), name);
  if (auto *fa = dyn_cast<KForallStmt>(s)) {
    for (auto *d : fa->getDims())
      if (exprRefsName(d, name)) return true;
    return bodyRefsName(fa->getBody(), name);
  }
  if (auto *iss = dyn_cast<KIssueStmt>(s)) {
    for (auto *i : iss->getIndices())
      if (exprRefsName(i, name)) return true;
    for (auto *o : iss->getOffsets())
      if (exprRefsName(o, name)) return true;
    for (auto *sz : iss->getSizes())
      if (exprRefsName(sz, name)) return true;
    for (auto *st : iss->getStrides())
      if (exprRefsName(st, name)) return true;
    return exprRefsName(iss->getBdId(), name);
  }
  if (auto *w = dyn_cast<KWaitStmt>(s))
    return exprRefsName(w->getIdxLo(), name) ||
           exprRefsName(w->getIdxHi(), name);
  // Zero / Release / Call (args are handle names) / Barrier: no iv-bearing
  // expr.
  return false;
}

bool bodyRefsName(const std::vector<KStmt *> &body, const std::string &name) {
  for (auto *s : body)
    if (stmtRefsName(s, name)) return true;
  return false;
}

// A loop is *spatial* iff its induction variable selects a distinct hardware
// instance — i.e. it appears in a route/instance subscript (`issue(route[iv])`,
// `wait(route[iv])`). Such a loop must be unrolled at instantiation (each
// iteration is a different `amdaie.connection`, which can't be a runtime loop).
// Every other loop is *temporal* and rolls to `scf.for` (iv-dependent
// offsets/sizes are materialized as SSA). Only spatial loops unroll.
bool loopIsSpatial(const std::string &iv, const std::vector<KStmt *> &body) {
  for (auto *s : body) {
    if (auto *iss = dyn_cast<KIssueStmt>(s)) {
      for (auto *i : iss->getIndices())
        if (exprRefsName(i, iv)) return true;
    } else if (auto *w = dyn_cast<KWaitStmt>(s)) {
      if (exprRefsName(w->getIdxLo(), iv) || exprRefsName(w->getIdxHi(), iv))
        return true;
    } else if (auto *f = dyn_cast<KForStmt>(s)) {
      if (loopIsSpatial(iv, f->getBody())) return true;
    }
  }
  return false;
}

// Build an `scf.for` carrying the `llvm.loop.unroll.disable` annotation so
// Peano keeps the core loop rolled (otherwise LLVM re-unrolls it and the core
// program overflows instruction memory). Mirrors
// AMDAIE::createForOpWithUnrollingDisabled, kept local so the frontend tool
// need not link the Transforms pass library.
scf::ForOp createRolledForOp(OpBuilder &b, Location loc, int64_t lo,
                             int64_t hi) {
  Value lb = arith::ConstantIndexOp::create(b, loc, lo);
  Value ub = arith::ConstantIndexOp::create(b, loc, hi);
  Value step = arith::ConstantIndexOp::create(b, loc, 1);
  auto forOp = scf::ForOp::create(b, loc, lb, ub, step);
  auto unroll = LLVM::LoopUnrollAttr::get(
      b.getContext(), /*disable=*/b.getBoolAttr(true), /*count=*/{},
      /*runtimeDisable=*/{}, /*full=*/{}, /*followupUnrolled=*/{},
      /*followupRemainder=*/{}, /*followupAll=*/{});
  auto ann = LLVM::LoopAnnotationAttr::get(
      b.getContext(), /*disableNonforced=*/{}, /*vectorize=*/{},
      /*interleave=*/{}, /*unroll=*/unroll, /*unrollAndJam=*/{}, /*licm=*/{},
      /*distribute=*/{}, /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
      /*mustProgress=*/{}, /*isVectorized=*/{}, /*startLoc=*/{}, /*endLoc=*/{},
      /*parallelAccesses=*/{});
  forOp->setAttr("loop_annotation", ann);
  return forOp;
}

// ─── Resolution structs (dialect-independent) ──────────────────────────

struct ResolvedCatalogEntry {
  CatalogKind kind;
  std::vector<std::string> indexVars;
  std::vector<std::pair<int64_t, int64_t>> ranges;
  KExpr *colExpr = nullptr;
  KExpr *rowExpr = nullptr;
};

struct Tile {
  int col, row;
};

Tile resolveTile(const ResolvedCatalogEntry &e,
                 const std::vector<int64_t> &idxVals,
                 ConstExprEvaluator &eval) {
  ConstExprScope s(eval);
  for (size_t i = 0; i < e.indexVars.size() && i < idxVals.size(); ++i)
    eval.bind(e.indexVars[i], idxVals[i]);
  int col = (int)eval.evalInt(e.colExpr);
  int row = e.rowExpr ? (int)eval.evalInt(e.rowExpr) : 0;
  return {col, row};
}

struct PlacementInst {
  Tile tile;
  std::vector<int64_t> axisValues;
};

std::vector<PlacementInst> resolvePlacement(
    const KPlacement &pl,
    const std::map<CatalogKind, ResolvedCatalogEntry> &catalog,
    ConstExprEvaluator &eval) {
  CatalogKind ck;
  if (pl.catalog == "shim")
    ck = CatalogKind::Shim;
  else if (pl.catalog == "memtile")
    ck = CatalogKind::Memtile;
  else if (pl.catalog == "core")
    ck = CatalogKind::Core;
  else if (pl.catalog == "controller")
    ck = CatalogKind::Controller;
  else
    return {};
  auto cit = catalog.find(ck);
  if (cit == catalog.end()) return {};
  const auto &e = cit->second;

  std::vector<std::vector<int64_t>> axisValues(e.indexVars.size());
  for (size_t i = 0; i < e.indexVars.size(); ++i) {
    if (i >= pl.axes.size() || isa<KWildcardExpr>(pl.axes[i])) {
      for (int64_t v = e.ranges[i].first; v < e.ranges[i].second; ++v)
        axisValues[i].push_back(v);
    } else {
      axisValues[i].push_back(eval.evalInt(pl.axes[i]));
    }
  }
  std::vector<PlacementInst> insts;
  if (axisValues.empty()) {
    insts.push_back({resolveTile(e, {}, eval), {}});
    return insts;
  }
  std::vector<size_t> pos(axisValues.size(), 0);
  while (true) {
    std::vector<int64_t> vals(axisValues.size());
    for (size_t i = 0; i < pos.size(); ++i) vals[i] = axisValues[i][pos[i]];
    insts.push_back({resolveTile(e, vals, eval), vals});
    size_t i = axisValues.size();
    while (i > 0) {
      --i;
      if (++pos[i] < axisValues[i].size()) break;
      pos[i] = 0;
      if (i == 0) return insts;
    }
  }
}

int64_t memrefFlat(const KMemrefType &t, ConstExprEvaluator &eval) {
  int64_t p = 1;
  for (auto *e : t.dims) p *= eval.evalInt(e);
  return p;
}

// Buffer/route instances, carrying the built SSA Values.
struct BufferInst {
  const KBufferDecl *decl;
  std::vector<int64_t> idxVals;
  std::vector<PlacementInst> placement;
  Value lof;  // logicalobjectfifo.from_memref result
};
struct RouteInst {
  const KRouteDecl *decl;
  std::vector<int64_t> idxVals;
  PlacementInst viaSrc;
  PlacementInst viaDst;
  std::vector<PlacementInst> viaDstAll;
  int srcChannel = 0;
  int dstChannel = 0;
  Value conn;  // amdaie.connection result
};

// ─── Emitter ────────────────────────────────────────────────────────────

class OpBuilderEmitter {
 public:
  OpBuilderEmitter(const std::map<std::string, int64_t> &params,
                   DiagnosticEngine &diag, std::string sourceDir)
      : diag_(diag),
        eval_(diag),
        builder_(&ctx_),
        sourceDir_(std::move(sourceDir)) {
    ctx_.loadDialect<AMDAIE::AMDAIEDialect, func::FuncDialect,
                     arith::ArithDialect, memref::MemRefDialect,
                     scf::SCFDialect, vector::VectorDialect,
                     linalg::LinalgDialect, IREE::HAL::HALDialect,
                     IREE::Codegen::IREECodegenDialect, LLVM::LLVMDialect>();
    for (auto &kv : params) {
      params_[kv.first] = kv.second;
      eval_.bindGlobal(kv.first, kv.second);
    }
  }

  std::string run(const KModuleDecl *mod) {
    const KKernelDecl *k = nullptr;
    for (auto *d : mod->getDecls())
      if (auto *kd = dyn_cast<KKernelDecl>(d)) k = kd;
    if (!k) return {};
    k_ = k;

    // Enforce the kernel's `where` constraints against the bound template
    // params before emitting — a violated constraint must be a hard error,
    // not silently-wrong IR (e.g. K_mm1 not a multiple of 64).
    for (KExpr *c : k->getWhereConstraints()) {
      if (!eval_.evalBool(c)) {
        diag_.report(c->getLocation(), diag::err_where_violated)
            << "constraint is false for these parameters";
        return {};
      }
    }

    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    module_ = module;
    b.setInsertionPointToEnd(module.getBody());

    resolveCatalog(k);
    resolveBuffers(k);
    resolveRoutes(k);

    // Matmul ukernel (extern fn impl template) parsed in at module scope.
    emitExternFns(mod);

    b.setInsertionPointToEnd(module.getBody());
    // Dispatch export name the splice/wrapper expects: derived from the
    // output element count (M*N) and the OUTPUT binding dtype, so the minted
    // wrapper's ref (a dummy `@fused` elementwise) matches for any dtype.
    int64_t mn = (params_.count("M") ? params_["M"] : 0) *
                 (params_.count("N") ? params_["N"] : 0);
    auto targs = tensorArgs(k);
    std::string outDtype =
        targs.empty() ? "i32" : targs.back()->type.elemType;
    auto fn = func::FuncOp::create(
        b, loc, "fused_dispatch_0_elementwise_" + itos(mn) + "_" + outDtype,
        b.getFunctionType({}, {}));
    // Mark as a pre-placed (Custom) dispatch so the lof pipeline runs only
    // the mechanical tail. Parsed (avoids a hard dep on the codegen dialect).
    if (Attribute ti = parseAttribute(
            "#iree_codegen.translation_info<pipeline = Custom>", &ctx_))
      fn->setAttr("translation_info", ti);
    b.setInsertionPointToEnd(fn.addEntryBlock());
    auto wg = AMDAIE::WorkgroupOp::create(b, loc);
    b.setInsertionPoint(wg.getControlCode());

    buildBuffersAndLofs();
    buildL2L1Connections();
    buildL3L2Connections();
    buildDrainsAndCores(k);
    emitDeallocs();
    emitBindingsAndDDRLofs(k);
    if (controlPacketsEnabled(k)) emitCtrlOverlay(k);
    emitControlcode(k, wg);

    b.setInsertionPointToEnd(&fn.getBody().front());
    func::ReturnOp::create(b, loc);

    if (failed(verify(module))) {
      diag_.report(SourceLocation{}, diag::err_unexpected_token)
          << "OpBuilder module verification failed";
      module.print(llvm::errs());
      return {};
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    // Print without attribute/type aliases (#map, etc.): the placed module
    // is spliced into a wrapper that won't carry top-level alias defs.
    module.print(os, OpPrintingFlags().useLocalScope());
    return os.str();
  }

 private:
  // ── Resolution (ported verbatim from the textual emitter) ──
  void resolveCatalog(const KKernelDecl *k) {
    catalog_.clear();
    indexRanges_.clear();
    for (auto *d : k->getBody()) {
      if (auto *cd = dyn_cast<KCatalogDecl>(d)) {
        ResolvedCatalogEntry e;
        e.kind = cd->getCatalogKind();
        e.colExpr = cd->getColExpr();
        e.rowExpr = cd->getRowExpr();
        for (auto &ib : cd->getIndices()) {
          e.indexVars.push_back(ib.name);
          int64_t lo = eval_.evalInt(ib.lo), hi = eval_.evalInt(ib.hi);
          e.ranges.push_back({lo, hi});
          indexRanges_[ib.name] = {lo, hi};
        }
        catalog_[cd->getCatalogKind()] = e;
      }
    }
    auto core = catalog_.find(CatalogKind::Core);
    if (core != catalog_.end()) {
      cols_ =
          (int)(core->second.ranges[0].second - core->second.ranges[0].first);
      rows_ =
          (int)(core->second.ranges[1].second - core->second.ranges[1].first);
    }
  }

  void rangeOf(const std::vector<std::string> &ivars, std::vector<int64_t> &lo,
               std::vector<int64_t> &hi) {
    for (auto &iv : ivars) {
      auto it = indexRanges_.find(iv);
      if (it != indexRanges_.end()) {
        lo.push_back(it->second.first);
        hi.push_back(it->second.second);
      } else {
        lo.push_back(0);
        hi.push_back(1);
      }
    }
  }

  void resolveBuffers(const KKernelDecl *k) {
    bufferInsts_.clear();
    for (auto *d : k->getBody()) {
      auto *b = dyn_cast<KBufferDecl>(d);
      if (!b) continue;
      std::vector<int64_t> lo, hi;
      rangeOf(b->getIndexVars(), lo, hi);
      std::vector<int64_t> idx = lo;
      auto step = [&]() -> bool {
        if (idx.empty()) return false;
        for (int i = (int)idx.size() - 1; i >= 0; --i) {
          if (++idx[i] < hi[i]) return true;
          idx[i] = lo[i];
        }
        return false;
      };
      auto doInst = [&]() {
        ConstExprScope s(eval_);
        for (size_t i = 0; i < b->getIndexVars().size(); ++i)
          eval_.bind(b->getIndexVars()[i], idx[i]);
        if (b->getWhenCondition() && !eval_.evalBool(b->getWhenCondition()))
          return;
        BufferInst bi;
        bi.decl = b;
        bi.idxVals = idx;
        bi.placement = resolvePlacement(b->getPlacement(), catalog_, eval_);
        bufferInsts_.push_back(std::move(bi));
      };
      if (idx.empty())
        doInst();
      else {
        doInst();
        while (step()) doInst();
      }
    }
  }

  void resolveRoutes(const KKernelDecl *k) {
    routeInsts_.clear();
    for (auto *d : k->getBody()) {
      auto *r = dyn_cast<KRouteDecl>(d);
      if (!r) continue;
      std::vector<int64_t> lo, hi;
      rangeOf(r->getIndexVars(), lo, hi);
      std::vector<int64_t> idx = lo;
      auto step = [&]() -> bool {
        if (idx.empty()) return false;
        for (int i = (int)idx.size() - 1; i >= 0; --i) {
          if (++idx[i] < hi[i]) return true;
          idx[i] = lo[i];
        }
        return false;
      };
      auto doInst = [&]() {
        ConstExprScope s(eval_);
        for (size_t i = 0; i < r->getIndexVars().size(); ++i)
          eval_.bind(r->getIndexVars()[i], idx[i]);
        if (r->getWhenCondition() && !eval_.evalBool(r->getWhenCondition()))
          return;
        RouteInst ri;
        ri.decl = r;
        ri.idxVals = idx;
        auto srcInsts =
            resolvePlacement(r->getViaSource().placement, catalog_, eval_);
        if (!srcInsts.empty()) ri.viaSrc = srcInsts[0];
        ri.viaDstAll =
            resolvePlacement(r->getViaTarget().placement, catalog_, eval_);
        if (!ri.viaDstAll.empty()) ri.viaDst = ri.viaDstAll[0];
        if (r->getViaSource().channelId)
          ri.srcChannel = (int)eval_.evalInt(r->getViaSource().channelId);
        if (r->getViaTarget().channelId)
          ri.dstChannel = (int)eval_.evalInt(r->getViaTarget().channelId);
        routeInsts_.push_back(std::move(ri));
      };
      if (idx.empty())
        doInst();
      else {
        doInst();
        while (step()) doInst();
      }
    }
  }

  // ── Emission ──

  // Parse each extern fn's `impl "..."` MLIR template (placeholder
  // substituted) and splice its function into the module at top level.
  void emitExternFns(const KModuleDecl *mod) {
    for (auto *d : mod->getDecls()) {
      auto *fn = dyn_cast<KExternFnDecl>(d);
      if (!fn) continue;
      externFns_.insert(fn->getName());
      externFnDecls_[fn->getName()] = fn;
      if (fn->isBareptr()) {
        // Precompiled bareptr ukernel (bf16/i8 matmul, zero_fill, ...): emit a
        // `func.func private` declaration with the bareptr ABI — each arg
        // becomes (memref<elem, space>, index); link_with names the object.
        OpBuilder &b = builder_;
        SmallVector<Type> inTys;
        for (auto &a : fn->getArgs()) {
          Attribute msAttr = a.type.memSpace
                                 ? b.getI32IntegerAttr(a.type.memSpace)
                                 : Attribute{};
          inTys.push_back(MemRefType::get({}, mlirElemType(a.type.elemType),
                                          MemRefLayoutAttrInterface{}, msAttr));
          inTys.push_back(b.getIndexType());
        }
        b.setInsertionPointToEnd(module_.getBody());
        auto f = func::FuncOp::create(b, b.getUnknownLoc(), fn->getName(),
                                      b.getFunctionType(inTys, {}));
        f.setPrivate();
        f->setAttr("link_with", b.getStringAttr(fn->getLinkWith()));
        f->setAttr("llvm.bareptr", b.getBoolAttr(true));
        continue;
      }
      if (fn->getImplPath().empty()) continue;
      std::string path = sourceDir_.empty()
                             ? fn->getImplPath()
                             : sourceDir_ + "/" + fn->getImplPath();
      std::ifstream in(path);
      if (!in) {
        diag_.report(fn->getLocation(), diag::err_expected)
            << ("readable impl file for extern fn '" + fn->getName() +
                "' (cannot open '" + path + "')");
        continue;
      }
      std::stringstream ss;
      ss << in.rdbuf();
      std::string tmpl = ss.str();
      std::map<std::string, std::string> subs;
      subs["NAME"] = fn->getName();
      for (size_t i = 0; i < fn->getArgs().size(); ++i) {
        std::string shape;
        for (auto *e : fn->getArgs()[i].type.dims)
          shape += itos(eval_.evalInt(e)) + "x";
        shape += fn->getArgs()[i].type.elemType;
        subs["ARG" + itos(i) + "SHAPE"] = shape;
        subs["ARG" + itos(i) + "SPACE"] = itos(fn->getArgs()[i].type.memSpace);
      }
      std::string text;
      for (size_t i = 0; i < tmpl.size();) {
        if (tmpl[i] == '$' && i + 1 < tmpl.size() && tmpl[i + 1] == '{') {
          size_t end = tmpl.find('}', i + 2);
          std::string key = tmpl.substr(i + 2, end - (i + 2));
          text += subs.count(key) ? subs[key] : "";
          i = end + 1;
        } else
          text += tmpl[i++];
      }
      // Parse `func.func ...` (wrap in a module to parse, then move the fn).
      auto parsed = parseSourceString<ModuleOp>(text, &ctx_);
      if (!parsed) {
        diag_.report(fn->getLocation(), diag::err_expected)
            << ("parsable MLIR in impl template for extern fn '" +
                fn->getName() + "' ('" + path + "')");
        continue;
      }
      OpBuilder &b = builder_;
      b.setInsertionPointToEnd(module_.getBody());
      for (auto &op :
           llvm::make_early_inc_range(parsed->getBody()->getOperations()))
        op.moveBefore(module_.getBody(), module_.getBody()->end());
    }
  }

  // Get (creating + memoizing) the amdaie.tile op Value for a coordinate.
  Value getTile(int col, int row) {
    auto key = std::make_pair(col, row);
    auto it = tiles_.find(key);
    if (it != tiles_.end()) return it->second;
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    Value c = arith::ConstantIndexOp::create(b, loc, col);
    Value r = arith::ConstantIndexOp::create(b, loc, row);
    Value t = AMDAIE::TileOp::create(b, loc, b.getIndexType(), c, r);
    tiles_[key] = t;
    return t;
  }

  // Map a .aiec element-type string to an MLIR type. The frontend is
  // dtype-generic: buffers/routes carry their own elemType, threaded through
  // every memref/vector the emitter synthesizes. (i32 is special only in that
  // its matmul vectorizes without a ukernel; bf16/i8 need the bareptr ukernel.)
  Type mlirElemType(StringRef et) {
    OpBuilder &b = builder_;
    if (et == "i8") return b.getI8Type();
    if (et == "i16") return b.getIntegerType(16);
    if (et == "i32") return b.getI32Type();
    if (et == "bf16") return b.getBF16Type();
    if (et == "f32") return b.getF32Type();
    return b.getI32Type();  // fallback (parser restricts to the set above)
  }

  // The logicalobjectfifo type for a buffer: a FLAT 1-D memref (the DMA
  // slice strides in the .aiec are written against the flattened layout;
  // the compute body reinterpret_casts back to the packed shape on access).
  // The objectfifo depth comes from the buffer's `depth N` attribute.
  AMDAIE::LogicalObjectFifoType lofTypeFor(const KMemrefType &ty,
                                           unsigned depth = 1) {
    OpBuilder &b = builder_;
    int64_t flat = memrefFlat(ty, eval_);
    Attribute msAttr =
        ty.memSpace ? b.getI32IntegerAttr(ty.memSpace) : Attribute{};
    auto memrefTy = MemRefType::get({flat}, mlirElemType(ty.elemType),
                                    MemRefLayoutAttrInterface{}, msAttr);
    return AMDAIE::LogicalObjectFifoType::get(memrefTy, depth);
  }

  // Evaluate a buffer's `depth` constexpr (default 1 if unspecified).
  unsigned bufferDepth(const KBufferDecl *b) {
    KExpr *e = b->getDepthExpr();
    return e ? (unsigned)eval_.evalInt(e) : 1u;
  }

  // The packed (declared-shape) memref type for a buffer.
  MemRefType packedType(const KMemrefType &ty) {
    OpBuilder &b = builder_;
    SmallVector<int64_t> shape;
    for (auto *e : ty.dims) shape.push_back(eval_.evalInt(e));
    Attribute msAttr =
        ty.memSpace ? b.getI32IntegerAttr(ty.memSpace) : Attribute{};
    return MemRefType::get(shape, mlirElemType(ty.elemType),
                           MemRefLayoutAttrInterface{}, msAttr);
  }

  // Build, per buffer instance: an allocation + a logicalobjectfifo view on
  // its placement tile(s). L1 (memory space 2) buffers share one allocation
  // per buffer name; L2 (memory space 1) buffers allocate per instance.
  void buildBuffersAndLofs() {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    std::map<std::string, Value> l1Alloc;  // shared L1 alloc by buffer name
    for (auto &bi : bufferInsts_) {
      const KMemrefType &ty = bi.decl->getType();
      SmallVector<int64_t> shape;
      for (auto *e : ty.dims) shape.push_back(eval_.evalInt(e));
      int ms = ty.memSpace;
      Attribute msAttr = ms ? b.getI32IntegerAttr(ms) : Attribute{};
      auto memrefTy = MemRefType::get(shape, mlirElemType(ty.elemType),
                                      MemRefLayoutAttrInterface{}, msAttr);
      Value alloc;
      if (ms == 2) {
        auto it = l1Alloc.find(bi.decl->getName());
        if (it != l1Alloc.end())
          alloc = it->second;
        else {
          alloc = memref::AllocOp::create(b, loc, memrefTy);
          l1Alloc[bi.decl->getName()] = alloc;
          l1AllocByName_[bi.decl->getName()] = alloc;
          l1Allocs_.push_back(alloc);
        }
      } else {
        alloc = memref::AllocOp::create(b, loc, memrefTy);
        l2Allocs_.push_back(alloc);
      }
      // Tile Values from the placement (created/memoized).
      SmallVector<Value> tileVals;
      for (auto &p : bi.placement)
        tileVals.push_back(getTile(p.tile.col, p.tile.row));
      bi.lof = AMDAIE::LogicalObjectFifoFromMemrefOp::create(
          b, loc, lofTypeFor(ty, bufferDepth(bi.decl)), alloc, tileVals);
    }
  }

  const BufferInst *findBufferInst(const std::string &name,
                                   const std::vector<int64_t> &idxs) {
    for (auto &bi : bufferInsts_)
      if (bi.decl->getName() == name && bi.idxVals == idxs) return &bi;
    return nullptr;
  }

  const KBufferDecl *findBufferDecl(const std::string &name) {
    for (auto *d : k_->getBody())
      if (auto *bd = dyn_cast<KBufferDecl>(d))
        if (bd->getName() == name) return bd;
    return nullptr;
  }

  // L2<->L1 routes: memtile (source) -> core[*] (targets). Build the MM2S
  // channel on the memtile, the S2MM channels on each compute tile, and the
  // amdaie.connection joining the (persistent) L2 and L1 logicalobjectfifos.
  void buildL2L1Connections() {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    for (auto &ri : routeInsts_) {
      const auto &rd = ri.decl;
      if (!(rd->getViaSource().placement.catalog == "memtile" &&
            rd->getViaTarget().placement.catalog == "core"))
        continue;
      if (!rd->getSource().has_value() || !rd->getTarget().has_value())
        continue;
      const BufferInst *srcBi =
          findBufferInst(rd->getSource()->bufName, ri.idxVals);
      if (!srcBi) srcBi = findBufferInst(rd->getSource()->bufName, {});
      const BufferInst *tgtBi =
          findBufferInst(rd->getTarget()->bufName, ri.idxVals);
      if (!tgtBi) tgtBi = findBufferInst(rd->getTarget()->bufName, {});
      if (!srcBi || !tgtBi || !srcBi->lof || !tgtBi->lof) continue;

      Value mtTile = getTile(ri.viaSrc.tile.col, ri.viaSrc.tile.row);
      Value srcCh = AMDAIE::ChannelOp::create(b, loc, mtTile, ri.srcChannel,
                                              AMDAIE::StrmSwPortType::DMA,
                                              AMDAIE::DMAChannelDir::MM2S);
      SmallVector<Value> dstChs;
      for (auto &dt : ri.viaDstAll) {
        Value coTile = getTile(dt.tile.col, dt.tile.row);
        dstChs.push_back(AMDAIE::ChannelOp::create(
            b, loc, coTile, ri.dstChannel, AMDAIE::StrmSwPortType::DMA,
            AMDAIE::DMAChannelDir::S2MM));
      }
      ri.conn = AMDAIE::ConnectionOp::create(
          b, loc, tgtBi->lof, dstChs, srcBi->lof, ValueRange(srcCh),
          AMDAIE::ConnectionTypeAttr::get(&ctx_,
                                          AMDAIE::ConnectionType::Circuit),
          /*flow=*/nullptr);
    }
  }

  // Find the kernel binding (DDR tensor) a shim route carries, by scanning
  // the controlcode issues for the first issue on that route.
  const KKernelArg *findRouteBinding(const std::string &routeName) {
    std::string bindName;
    std::function<void(const std::vector<KStmt *> &)> walk =
        [&](const std::vector<KStmt *> &stmts) {
          for (auto *s : stmts) {
            if (auto *iss = dyn_cast<KIssueStmt>(s)) {
              if (iss->getRouteName() == routeName && bindName.empty()) {
                if (!iss->getSrcBinding().empty())
                  bindName = iss->getSrcBinding();
                else if (!iss->getDstBinding().empty())
                  bindName = iss->getDstBinding();
              }
            } else if (auto *f = dyn_cast<KForStmt>(s)) {
              walk(f->getBody());
            }
          }
        };
    for (auto *d : k_->getBody())
      if (auto *ctl = dyn_cast<KOnControllerDecl>(d)) walk(ctl->getBody());
    for (auto &a : k_->getKernelArgs())
      if (a.name == bindName && !a.isScalar()) return &a;
    return nullptr;
  }

  // L3<->L2 routes (shim boundary). The L3 side is a logicalobjectfifo
  // placeholder (its slice is parameterized at issue time); the L2 side is
  // the persistent buffer LOF. Source-parameterized = push (shim->memtile);
  // target-parameterized = drain (memtile->shim).
  void buildL3L2Connections() {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    for (auto &ri : routeInsts_) {
      const auto &rd = ri.decl;
      bool isShim = rd->getViaSource().placement.catalog == "shim" ||
                    rd->getViaTarget().placement.catalog == "shim";
      if (!isShim) continue;
      const KKernelArg *bind = findRouteBinding(rd->getName());
      if (!bind) continue;
      SmallVector<int64_t> shape;
      for (auto *e : bind->type.dims) shape.push_back(eval_.evalInt(e));
      auto phMemref = MemRefType::get(shape, mlirElemType(bind->type.elemType));
      auto phLofTy = AMDAIE::LogicalObjectFifoType::get(phMemref);
      auto connTy = AMDAIE::ConnectionTypeAttr::get(
          &ctx_, rd->getRouteType() == RouteType::Packet
                     ? AMDAIE::ConnectionType::Packet
                     : AMDAIE::ConnectionType::Circuit);

      if (rd->getParameterization() ==
          RouteParameterization::SourceParameterized) {
        // Push: target = L2 buffer (memtile S2MM), source = placeholder (shim
        // MM2S).
        if (!rd->getTarget().has_value()) continue;
        const BufferInst *bi =
            findBufferInst(rd->getTarget()->bufName, ri.idxVals);
        if (!bi) bi = findBufferInst(rd->getTarget()->bufName, {});
        if (!bi || !bi->lof) continue;
        Value shTile = getTile(ri.viaSrc.tile.col, ri.viaSrc.tile.row);
        Value mtTile = getTile(ri.viaDst.tile.col, ri.viaDst.tile.row);
        Value ph = AMDAIE::LogicalObjectFifoPlaceholderOp::create(
            b, loc, phLofTy, ValueRange(shTile));
        Value shCh = AMDAIE::ChannelOp::create(b, loc, shTile, ri.srcChannel,
                                               AMDAIE::StrmSwPortType::DMA,
                                               AMDAIE::DMAChannelDir::MM2S);
        Value mtCh = AMDAIE::ChannelOp::create(b, loc, mtTile, ri.dstChannel,
                                               AMDAIE::StrmSwPortType::DMA,
                                               AMDAIE::DMAChannelDir::S2MM);
        // The control-packet overlay reuses these shim-source DMA channels as
        // its packet feeds (keyed by column + channel index).
        shimDmaCh_[{ri.viaSrc.tile.col, ri.srcChannel}] = shCh;
        ri.conn = AMDAIE::ConnectionOp::create(
            b, loc, bi->lof, ValueRange(mtCh), ph, ValueRange(shCh), connTy,
            /*flow=*/nullptr);
      } else if (rd->getParameterization() ==
                 RouteParameterization::TargetParameterized) {
        // Drain: source = L2 buffer (memtile MM2S), target = placeholder (shim
        // S2MM).
        if (!rd->getSource().has_value()) continue;
        const BufferInst *bi =
            findBufferInst(rd->getSource()->bufName, ri.idxVals);
        if (!bi) bi = findBufferInst(rd->getSource()->bufName, {});
        if (!bi || !bi->lof) continue;
        Value mtTile = getTile(ri.viaSrc.tile.col, ri.viaSrc.tile.row);
        Value shTile = getTile(ri.viaDst.tile.col, ri.viaDst.tile.row);
        Value ph = AMDAIE::LogicalObjectFifoPlaceholderOp::create(
            b, loc, phLofTy, ValueRange(shTile));
        Value mtCh = AMDAIE::ChannelOp::create(b, loc, mtTile, ri.srcChannel,
                                               AMDAIE::StrmSwPortType::DMA,
                                               AMDAIE::DMAChannelDir::MM2S);
        Value shCh = AMDAIE::ChannelOp::create(b, loc, shTile, ri.dstChannel,
                                               AMDAIE::StrmSwPortType::DMA,
                                               AMDAIE::DMAChannelDir::S2MM);
        ri.conn = AMDAIE::ConnectionOp::create(
            b, loc, ph, ValueRange(shCh), bi->lof, ValueRange(mtCh), connTy,
            /*flow=*/nullptr);
      }
    }
  }

  Tile coreTile(int c, int r) {
    auto it = catalog_.find(CatalogKind::Core);
    if (it == catalog_.end()) return {c, r};
    return resolveTile(it->second, {(int64_t)c, (int64_t)r}, eval_);
  }

  // The persistent route connection feeding/draining a buffer reference in
  // a core body: Produce buffer = a route source, Consume buffer = a route
  // target. Matched by the route's index vars in the current (c,r) scope.
  Value resolveBufferConn(const std::string &bufName,
                          AMDAIE::LogicalObjectFifoPort port) {
    bool produce = port == AMDAIE::LogicalObjectFifoPort::Produce;
    for (auto &ri : routeInsts_) {
      const auto &side = produce ? ri.decl->getSource() : ri.decl->getTarget();
      if (!side.has_value() || side->bufName != bufName) continue;
      bool match = true;
      const auto &ivars = ri.decl->getIndexVars();
      for (size_t i = 0; i < ivars.size() && i < ri.idxVals.size(); ++i) {
        int64_t v;
        if (!eval_.lookup(ivars[i], v) || v != ri.idxVals[i]) {
          match = false;
          break;
        }
      }
      if (match) return ri.conn;
    }
    return Value();
  }

  static std::string bufRef(const KExpr *e) {
    if (auto *idx = dyn_cast<KIndexExpr>(e))
      if (auto *id = dyn_cast<KIdentExpr>(idx->getBase())) return id->getName();
    if (auto *id = dyn_cast<KIdentExpr>(e)) return id->getName();
    return "";
  }

  // Per (c,r): build the L1->L2 drain connection, then the amdaie.core op
  // with its in/out DMA lists and the compute body (scf.forall + acquire/
  // access/matmul/activation/release), mechanically walking the on-core AST.
  void buildDrainsAndCores(const KKernelDecl *k) {
    const KOnCoreDecl *onCore = nullptr;
    for (auto *d : k->getBody())
      if (auto *o = dyn_cast<KOnCoreDecl>(d)) onCore = o;
    if (!onCore) return;
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();

    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        ConstExprScope sc(eval_);
        eval_.bind("c", c);
        eval_.bind("r", r);

        // Drain connection (out_l1[c,r] -> c_l2[c]) via core MM2S, memtile
        // S2MM.
        for (auto &ri : routeInsts_) {
          const auto &rd = ri.decl;
          if (!(rd->getViaSource().placement.catalog == "core" &&
                rd->getViaTarget().placement.catalog == "memtile"))
            continue;
          if (ri.idxVals.size() != 2 || ri.idxVals[0] != c ||
              ri.idxVals[1] != r)
            continue;
          if (!rd->getSource().has_value() || !rd->getTarget().has_value())
            continue;
          const BufferInst *src =
              findBufferInst(rd->getSource()->bufName, ri.idxVals);
          const BufferInst *tgt =
              findBufferInst(rd->getTarget()->bufName, {(int64_t)c});
          if (!src || !tgt || !src->lof || !tgt->lof) continue;
          Tile ct = coreTile(c, r);
          Value coTile = getTile(ct.col, ct.row);
          Value mtTile = getTile(ri.viaDst.tile.col, ri.viaDst.tile.row);
          Value coCh = AMDAIE::ChannelOp::create(b, loc, coTile, ri.srcChannel,
                                                 AMDAIE::StrmSwPortType::DMA,
                                                 AMDAIE::DMAChannelDir::MM2S);
          Value mtCh = AMDAIE::ChannelOp::create(b, loc, mtTile, ri.dstChannel,
                                                 AMDAIE::StrmSwPortType::DMA,
                                                 AMDAIE::DMAChannelDir::S2MM);
          ri.conn = AMDAIE::ConnectionOp::create(
              b, loc, tgt->lof, ValueRange(mtCh), src->lof, ValueRange(coCh),
              AMDAIE::ConnectionTypeAttr::get(&ctx_,
                                              AMDAIE::ConnectionType::Circuit),
              /*flow=*/nullptr);
        }

        buildCore(c, r, onCore);
      }
    }
  }

  void buildCore(int c, int r, const KOnCoreDecl *onCore) {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();

    // Split the body into barrier-delimited segments.
    std::vector<std::vector<const KStmt *>> segs;
    std::vector<const KStmt *> cur;
    for (auto *s : onCore->getBody()) {
      if (isa<KBarrierStmt>(s)) {
        segs.push_back(cur);
        cur.clear();
      } else
        cur.push_back(s);
    }
    segs.push_back(cur);

    // in = per-segment-distinct Consume conns; out = body-distinct Produce.
    SmallVector<Value> inList, outList;
    std::set<void *> outSeen;
    const KBufferDecl *produceBuf = nullptr;
    std::function<void(const std::vector<const KStmt *> &, std::set<void *> &)>
        scan = [&](const std::vector<const KStmt *> &body,
                   std::set<void *> &segIn) {
          for (auto *s : body) {
            if (auto *l = dyn_cast<KLetStmt>(s)) {
              if (l->isSubview()) continue;
              bool produce = l->getRole() == AcquireRole::Produce;
              auto port = produce ? AMDAIE::LogicalObjectFifoPort::Produce
                                  : AMDAIE::LogicalObjectFifoPort::Consume;
              // The first Produce buffer (incl. a routeless local accumulator)
              // shapes the zero constant — it's what `zero()` writes.
              if (produce && !produceBuf)
                produceBuf = findBufferDecl(bufRef(l->getBufRef()));
              Value conn = resolveBufferConn(bufRef(l->getBufRef()), port);
              if (!conn) continue;
              if (produce) {
                if (outSeen.insert(conn.getAsOpaquePointer()).second)
                  outList.push_back(conn);
              } else if (segIn.insert(conn.getAsOpaquePointer()).second) {
                inList.push_back(conn);
              }
            } else if (auto *red = dyn_cast<KReduceStmt>(s)) {
              std::vector<const KStmt *> bb(red->getBody().begin(),
                                            red->getBody().end());
              scan(bb, segIn);
            } else if (auto *f = dyn_cast<KForallStmt>(s)) {
              std::vector<const KStmt *> bb(f->getBody().begin(),
                                            f->getBody().end());
              scan(bb, segIn);
            } else if (auto *fr = dyn_cast<KForStmt>(s)) {
              std::vector<const KStmt *> bb(fr->getBody().begin(),
                                            fr->getBody().end());
              scan(bb, segIn);
            }
          }
        };
    for (auto &seg : segs) {
      std::set<void *> segIn;
      scan(seg, segIn);
    }

    Tile ct = coreTile(c, r);
    auto tileOp = cast<AMDAIE::TileOp>(getTile(ct.col, ct.row).getDefiningOp());
    auto core = AMDAIE::CoreOp::create(b, loc, tileOp, inList, outList);
    // Build the core body in its own region, then restore the workgroup-body
    // insertion point (so the next core/drain is a sibling, not nested).
    OpBuilder::InsertionGuard coreGuard(b);
    Block *blk = b.createBlock(&core.getRegion());
    b.setInsertionPointToEnd(blk);

    // One zeroing vector constant at core-block scope (dominates all forall
    // segments), shaped to the produced buffer.
    Value zeroCst;
    if (produceBuf) {
      SmallVector<int64_t> shape;
      for (auto *e : produceBuf->getType().dims)
        shape.push_back(eval_.evalInt(e));
      Type et = mlirElemType(produceBuf->getType().elemType);
      auto vecTy = VectorType::get(shape, et);
      Attribute zeroAttr = isa<FloatType>(et)
                               ? (Attribute)b.getFloatAttr(et, 0.0)
                               : (Attribute)b.getIntegerAttr(et, 0);
      zeroCst = arith::ConstantOp::create(
          b, loc, vecTy, DenseElementsAttr::get(vecTy, zeroAttr));
    }

    struct Handle {
      Value mem;
      Value conn;
      AMDAIE::LogicalObjectFifoPort port;
      const KMemrefType *type;
    };
    std::map<std::string, Handle> handles;
    // Objects to link into this core's ELF, aggregated from the bareptr
    // ukernels it calls. In the placed/lof path the core is hand-emitted (not
    // built by AMDAIEInsertCores), so the emitter sets link_with directly;
    // AMDAIELowerToAIE then copies it onto the aie.core.
    std::set<std::string> coreLinkWith;

    std::function<void(const std::vector<KStmt *> &)> walk =
        [&](const std::vector<KStmt *> &body) {
          for (auto *s : body) {
            if (auto *l = dyn_cast<KLetStmt>(s)) {
              if (l->isSubview()) {
                // `let name = base[indices]`: a sub-tile view of an already
                // acquired buffer handle. Reinterpret the base buffer at the
                // sub-tile's flat offset so the 4 micro-tile matmuls write into
                // one acquired output buffer (depth 2) instead of 4 separate
                // fifo acquires (which forced out_l1 depth-4 → q/p unroll-4).
                auto *idx = dyn_cast<KIndexExpr>(l->getBufRef());
                auto *baseId =
                    idx ? dyn_cast<KIdentExpr>(idx->getBase()) : nullptr;
                if (!idx || !baseId) continue;
                auto bit = handles.find(baseId->getName());
                if (bit == handles.end()) continue;
                auto baseTy = cast<MemRefType>(bit->second.mem.getType());
                ArrayRef<int64_t> shape = baseTy.getShape();
                int nIdx = (int)idx->getIndices().size();
                SmallVector<int64_t> subSizes(shape.begin() + nIdx,
                                              shape.end());
                int64_t subElems = 1;
                for (int64_t d : subSizes) subElems *= d;
                int64_t flatIdx = 0;
                for (int i = 0; i < nIdx; ++i) {
                  int64_t iv = eval_.evalInt(idx->getIndices()[i]);
                  int64_t prod = 1;
                  for (int j = i + 1; j < nIdx; ++j) prod *= shape[j];
                  flatIdx += iv * prod;
                }
                int64_t offset = flatIdx * subElems;
                SmallVector<int64_t> subStrides(subSizes.size(), 1);
                for (int i = (int)subSizes.size() - 2; i >= 0; --i)
                  subStrides[i] = subStrides[i + 1] * subSizes[i + 1];
                auto subTy = MemRefType::get(
                    subSizes, baseTy.getElementType(),
                    StridedLayoutAttr::get(&ctx_, offset, subStrides),
                    baseTy.getMemorySpace());
                Value sub = memref::ReinterpretCastOp::create(
                    b, loc, subTy, bit->second.mem, offset, subSizes,
                    subStrides);
                handles[l->getName()] = {sub, bit->second.conn,
                                         bit->second.port, bit->second.type};
                continue;
              }
              const KBufferDecl *bd = findBufferDecl(bufRef(l->getBufRef()));
              if (!bd) continue;
              bool produce = l->getRole() == AcquireRole::Produce;
              auto port = produce ? AMDAIE::LogicalObjectFifoPort::Produce
                                  : AMDAIE::LogicalObjectFifoPort::Consume;
              Value conn = resolveBufferConn(bufRef(l->getBufRef()), port);
              if (!conn) {
                // Routeless buffer = a LOCAL accumulator (no DMA fifo): use its
                // L1 alloc directly. The matmul writes it and a `:=` cast reads
                // it; acquire/release are no-ops. (Used for the bf16 fused
                // chain's f32 accumulator, which is cast to bf16 before drain.)
                auto ait = l1AllocByName_.find(bufRef(l->getBufRef()));
                if (ait == l1AllocByName_.end()) continue;
                handles[l->getName()] = {ait->second, Value{}, port,
                                         &bd->getType()};
                continue;
              }
              Value acq = AMDAIE::LogicalObjectFifoAcquire::create(
                  b, loc, lofTypeFor(bd->getType(), bufferDepth(bd)), conn,
                  port);
              Value flat = AMDAIE::LogicalObjectFifoAccessOp::create(
                  b, loc, acq,
                  produce ? AMDAIE::MemoryAccess::Write
                          : AMDAIE::MemoryAccess::Read);
              // The LOF/access is flat; reinterpret to the declared packed
              // shape (contiguous strides) for the compute ops.
              MemRefType packed = packedType(bd->getType());
              SmallVector<int64_t> sizes(packed.getShape().begin(),
                                         packed.getShape().end());
              SmallVector<int64_t> strides(sizes.size(), 1);
              for (int i = (int)sizes.size() - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * sizes[i + 1];
              Value mem = memref::ReinterpretCastOp::create(
                  b, loc, packed, flat, /*offset=*/0, sizes, strides);
              handles[l->getName()] = {mem, conn, port, &bd->getType()};
            } else if (auto *z = dyn_cast<KZeroStmt>(s)) {
              auto it = handles.find(z->getHandle());
              if (it == handles.end()) continue;
              int nd = (int)it->second.type->dims.size();
              Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
              SmallVector<Value> idx(nd, c0);
              SmallVector<bool> inb(nd, true);
              if (zeroCst)
                vector::TransferWriteOp::create(b, loc, zeroCst, it->second.mem,
                                                idx, inb);
            } else if (auto *red = dyn_cast<KReduceStmt>(s)) {
              int64_t lo = eval_.evalInt(red->getLo());
              int64_t hi = eval_.evalInt(red->getHi());
              if (bodyRefsName(red->getBody(), red->getIndVar())) {
                // iv used in the body (e.g. an instance subscript) → unroll.
                for (int64_t kv = lo; kv < hi; ++kv) {
                  ConstExprScope ks(eval_);
                  eval_.bind(red->getIndVar(), kv);
                  walk(red->getBody());
                }
              } else {
                // iv-independent → roll to scf.for (data cycled by fifos).
                // Disable LLVM unrolling so Peano keeps the loop rolled and the
                // core program stays within instruction memory.
                auto forOp = createRolledForOp(b, loc, lo, hi);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToStart(forOp.getBody());
                walk(red->getBody());
              }
            } else if (auto *call = dyn_cast<KCallStmt>(s)) {
              if (!externFns_.count(call->getCallee())) {
                diag_.report(call->getLocation(),
                             diag::err_undefined_identifier)
                    << call->getCallee();
                continue;
              }
              auto fdIt = externFnDecls_.find(call->getCallee());
              bool bareptr =
                  fdIt != externFnDecls_.end() && fdIt->second->isBareptr();
              SmallVector<Value> args;
              bool ok = true;
              for (auto &a : call->getArgs()) {
                auto h = handles.find(a);
                if (h == handles.end()) {
                  ok = false;
                  break;
                }
                if (bareptr) {
                  // bareptr ABI: pass (base_buffer, offset) per memref arg.
                  coreLinkWith.insert(fdIt->second->getLinkWith());
                  auto md = memref::ExtractStridedMetadataOp::create(
                      b, loc, h->second.mem);
                  args.push_back(md.getBaseBuffer());
                  args.push_back(md.getOffset());
                } else {
                  args.push_back(h->second.mem);
                }
              }
              if (!ok) continue;
              func::CallOp::create(b, loc, TypeRange{},
                                   SymbolRefAttr::get(&ctx_, call->getCallee()),
                                   args);
            } else if (auto *am = dyn_cast<KAssignMulStmt>(s)) {
              auto it = handles.find(am->getHandle());
              if (it == handles.end()) continue;
              int64_t scale = eval_.evalInt(am->getRHS());
              int nd = (int)it->second.type->dims.size();
              auto idMap = AffineMap::getMultiDimIdentityMap(nd, &ctx_);
              SmallVector<utils::IteratorType> iters(
                  nd, utils::IteratorType::parallel);
              Type et = mlirElemType(it->second.type->elemType);
              linalg::GenericOp::create(
                  b, loc, TypeRange{}, ValueRange{}, ValueRange{it->second.mem},
                  ArrayRef<AffineMap>{idMap}, iters,
                  [&](OpBuilder &nb, Location nloc, ValueRange bbargs) {
                    Value m;
                    if (isa<FloatType>(et)) {
                      Value sc = arith::ConstantOp::create(
                          nb, nloc, nb.getFloatAttr(et, (double)scale));
                      m = arith::MulFOp::create(nb, nloc, bbargs[0], sc);
                    } else {
                      Value sc = arith::ConstantOp::create(
                          nb, nloc, nb.getIntegerAttr(et, scale));
                      m = arith::MulIOp::create(nb, nloc, bbargs[0], sc);
                    }
                    linalg::YieldOp::create(nb, nloc, m);
                  });
            } else if (auto *cp = dyn_cast<KAssignCopyStmt>(s)) {
              // `dst := src`: elementwise copy with dtype conversion
              // (f32->bf16 truncf, bf16->f32 extf, else identity copy).
              auto dit = handles.find(cp->getDst());
              auto sit = handles.find(cp->getSrc());
              if (dit == handles.end() || sit == handles.end()) continue;
              int nd = (int)dit->second.type->dims.size();
              auto idMap = AffineMap::getMultiDimIdentityMap(nd, &ctx_);
              SmallVector<utils::IteratorType> iters(
                  nd, utils::IteratorType::parallel);
              Type dstEt = mlirElemType(dit->second.type->elemType);
              Type srcEt = mlirElemType(sit->second.type->elemType);
              linalg::GenericOp::create(
                  b, loc, TypeRange{}, ValueRange{sit->second.mem},
                  ValueRange{dit->second.mem}, ArrayRef<AffineMap>{idMap, idMap},
                  iters, [&](OpBuilder &nb, Location nloc, ValueRange bb) {
                    Value v = bb[0];
                    if (srcEt != dstEt) {
                      if (isa<FloatType>(srcEt) && isa<FloatType>(dstEt)) {
                        auto sw = cast<FloatType>(srcEt).getWidth();
                        auto dw = cast<FloatType>(dstEt).getWidth();
                        v = sw > dw
                                ? (Value)arith::TruncFOp::create(nb, nloc, dstEt, v)
                                : (Value)arith::ExtFOp::create(nb, nloc, dstEt, v);
                      }
                    }
                    linalg::YieldOp::create(nb, nloc, v);
                  });
            } else if (auto *rel = dyn_cast<KReleaseStmt>(s)) {
              auto it = handles.find(rel->getHandle());
              if (it == handles.end()) continue;
              if (!it->second.conn) continue;  // local accumulator: no fifo
              AMDAIE::LogicalObjectFifoRelease::create(b, loc, it->second.conn,
                                                       it->second.port);
            } else if (auto *f = dyn_cast<KForallStmt>(s)) {
              // `forall` → scf.forall (parallel output-tile iteration).
              auto &dims = f->getDims();
              SmallVector<OpFoldResult> ubs;
              for (auto *e : dims)
                ubs.push_back(b.getIndexAttr(eval_.evalInt(e)));
              auto forall = scf::ForallOp::create(b, loc, ubs, ValueRange{},
                                                  std::nullopt);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPointToStart(forall.getBody());
              walk(f->getBody());
            } else if (auto *fr = dyn_cast<KForStmt>(s)) {
              // `for` (output-pass / tile loop). Rolled to scf.for when the
              // body is iv-independent (data per pass sequenced by controlcode
              // + objectfifo cycling); unrolled when the iv is referenced
              // (e.g. an instance subscript). barrier is a phase delimiter.
              int64_t lo = eval_.evalInt(fr->getLo());
              int64_t hi = eval_.evalInt(fr->getHi());
              if (bodyRefsName(fr->getBody(), fr->getIndVar())) {
                for (int64_t v = lo; v < hi; ++v) {
                  ConstExprScope sc(eval_);
                  eval_.bind(fr->getIndVar(), v);
                  walk(fr->getBody());
                }
              } else {
                auto forOp = createRolledForOp(b, loc, lo, hi);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToStart(forOp.getBody());
                walk(fr->getBody());
              }
            }
          }
        };
    // The core body is emitted by a single walk: bare statements (the
    // output-stationary form holds accumulators directly, no forall), foralls,
    // and for-loops are all handled uniformly.
    walk(onCore->getBody());
    if (!coreLinkWith.empty()) {
      SmallVector<std::string> objs(coreLinkWith.begin(), coreLinkWith.end());
      core.setLinkWith(b.getStringAttr(llvm::join(objs, ",")));
    }
    AMDAIE::EndOp::create(b, loc);
  }

  // Tensor (non-scalar) kernel args, in source order.
  std::vector<const KKernelArg *> tensorArgs(const KKernelDecl *k) {
    std::vector<const KKernelArg *> t;
    for (auto &a : k->getKernelArgs())
      if (!a.isScalar()) t.push_back(&a);
    return t;
  }
  // Binding names a drain writes to (issue target=) — read-write.
  std::set<std::string> rwBindings(const KKernelDecl *k) {
    std::set<std::string> rw;
    std::function<void(const std::vector<KStmt *> &)> walk =
        [&](const std::vector<KStmt *> &stmts) {
          for (auto *s : stmts) {
            if (auto *iss = dyn_cast<KIssueStmt>(s)) {
              if (!iss->getDstBinding().empty())
                rw.insert(iss->getDstBinding());
            } else if (auto *f = dyn_cast<KForStmt>(s))
              walk(f->getBody());
          }
        };
    for (auto *d : k->getBody())
      if (auto *ctl = dyn_cast<KOnControllerDecl>(d)) walk(ctl->getBody());
    return rw;
  }
  Value shimTileVal(int col) {
    auto it = catalog_.find(CatalogKind::Shim);
    Tile t = (it != catalog_.end())
                 ? resolveTile(it->second, {(int64_t)col}, eval_)
                 : Tile{col, 0};
    return getTile(t.col, t.row);
  }
  Tile memtileTileFor(int col) {
    auto it = catalog_.find(CatalogKind::Memtile);
    return (it != catalog_.end())
               ? resolveTile(it->second, {(int64_t)col}, eval_)
               : Tile{col, 1};
  }

  // HAL bindings (subspans) + per-column DDR (flattened) logicalobjectfifos.
  void emitBindingsAndDDRLofs(const KKernelDecl *k) {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    auto tensors = tensorArgs(k);
    std::set<std::string> rw = rwBindings(k);

    // Pipeline layout: one binding per tensor arg; read-write iff drained to.
    SmallVector<IREE::HAL::PipelineBindingAttr> binds;
    for (auto *a : tensors) {
      auto flags = rw.count(a->name) ? IREE::HAL::DescriptorFlags::Indirect
                                     : (IREE::HAL::DescriptorFlags::ReadOnly |
                                        IREE::HAL::DescriptorFlags::Indirect);
      binds.push_back(IREE::HAL::PipelineBindingAttr::get(
          &ctx_, IREE::HAL::DescriptorType::StorageBuffer, flags));
    }
    auto layout = IREE::HAL::PipelineLayoutAttr::get(
        &ctx_, binds, /*constants=*/0,
        IREE::HAL::PipelineLayoutFlags::Indirect);

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    int bIdx = 0;
    for (auto *a : tensors) {
      SmallVector<int64_t> shape;
      for (auto *e : a->type.dims) shape.push_back(eval_.evalInt(e));
      int64_t flat = 1;
      for (auto s : shape) flat *= s;
      Type elemTy = mlirElemType(a->type.elemType);
      auto memTy = MemRefType::get(shape, elemTy);
      auto descFlags = rw.count(a->name)
                           ? IREE::HAL::DescriptorFlags::Indirect
                           : (IREE::HAL::DescriptorFlags::ReadOnly |
                              IREE::HAL::DescriptorFlags::Indirect);
      auto subspan = IREE::HAL::InterfaceBindingSubspanOp::create(
          b, loc, memTy, layout, b.getIndexAttr(bIdx), c0,
          /*dynamic_dims=*/ValueRange{}, b.getIndexAttr(64),
          IREE::HAL::DescriptorFlagsAttr::get(&ctx_, descFlags));
      Value aligned = memref::AssumeAlignmentOp::create(b, loc, subspan, 64);
      auto flatLofTy = AMDAIE::LogicalObjectFifoType::get(
          MemRefType::get({flat}, elemTy));
      for (int c = 0; c < cols_; ++c) {
        SmallVector<Value> tv{shimTileVal(c)};
        Value lof = AMDAIE::LogicalObjectFifoFromMemrefOp::create(
            b, loc, flatLofTy, aligned, tv);
        ddrLofs_[a->name + "_" + itos(c)] = lof;
      }
      ++bIdx;
    }
  }

  Value findRouteConn(const std::string &name,
                      const std::vector<int64_t> &idx) {
    for (auto &ri : routeInsts_)
      if (ri.decl->getName() == name && ri.idxVals == idx) return ri.conn;
    return Value();
  }
  const RouteInst *findRouteInst(const std::string &name,
                                 const std::vector<int64_t> &idx) {
    for (auto &ri : routeInsts_)
      if (ri.decl->getName() == name && ri.idxVals == idx) return &ri;
    return nullptr;
  }
  SmallVector<int64_t> evalList(const std::vector<KExpr *> &es) {
    SmallVector<int64_t> v;
    for (auto *e : es) v.push_back(eval_.evalInt(e));
    return v;
  }

  // True if `e` references any loop induction variable currently bound to an
  // SSA value (i.e. a rolled controlcode loop). Such expressions must be
  // emitted as runtime arithmetic, not folded to a constant.
  bool exprUsesSsaIv(const KExpr *e) {
    for (auto &kv : ssaIvs_)
      if (exprRefsName(e, kv.first)) return true;
    return false;
  }

  // Emit `e` as an `index` SSA value: a rolled loop's iv resolves to its
  // induction variable; everything else is constexpr-evaluated to a constant.
  Value emitExprValue(OpBuilder &b, Location loc, const KExpr *e) {
    if (auto *il = dyn_cast<KIntLitExpr>(e))
      return arith::ConstantIndexOp::create(b, loc, il->getValue());
    if (auto *id = dyn_cast<KIdentExpr>(e)) {
      auto it = ssaIvs_.find(id->getName());
      if (it != ssaIvs_.end()) return it->second;
      return arith::ConstantIndexOp::create(b, loc, eval_.evalInt(e));
    }
    if (auto *bo = dyn_cast<KBinOpExpr>(e)) {
      Value l = emitExprValue(b, loc, bo->getLHS());
      Value r = emitExprValue(b, loc, bo->getRHS());
      switch (bo->getOp()) {
        case BinOpKind::Add:
          return arith::AddIOp::create(b, loc, l, r);
        case BinOpKind::Sub:
          return arith::SubIOp::create(b, loc, l, r);
        case BinOpKind::Mul:
          return arith::MulIOp::create(b, loc, l, r);
        case BinOpKind::Div:
          return arith::DivSIOp::create(b, loc, l, r);
        case BinOpKind::Mod:
          return arith::RemSIOp::create(b, loc, l, r);
        default:
          break;
      }
    }
    if (auto *u = dyn_cast<KUnaryOpExpr>(e)) {
      if (u->getOp() == UnaryOpKind::Neg) {
        Value z = arith::ConstantIndexOp::create(b, loc, 0);
        return arith::SubIOp::create(b, loc, z,
                                     emitExprValue(b, loc, u->getOperand()));
      }
    }
    // Fallback: anything else must be constexpr in this context.
    return arith::ConstantIndexOp::create(b, loc, eval_.evalInt(e));
  }

  // An offset/size/stride entry: a static attr when constexpr, or a dynamic
  // SSA value when it depends on a rolled loop's induction variable.
  OpFoldResult emitExprOFR(OpBuilder &b, Location loc, const KExpr *e) {
    if (exprUsesSsaIv(e)) return emitExprValue(b, loc, e);
    return b.getIndexAttr(eval_.evalInt(e));
  }

  // Free the buffer allocations at the end of the workgroup body (L1 shared
  // allocs first, then L2 per-instance allocs), mirroring the textual emitter.
  void emitDeallocs() {
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    for (Value a : l1Allocs_) memref::DeallocOp::create(b, loc, a);
    for (Value a : l2Allocs_) memref::DeallocOp::create(b, loc, a);
  }

  // Whether the kernel uses the control-packet overlay (kernel attribute
  // `control_packets = true`). Absent ⇒ false (no overlay emitted).
  bool controlPacketsEnabled(const KKernelDecl *k) {
    KExpr *v = k->getAttr("control_packets");
    return v && eval_.evalBool(v);
  }

  // A control-packet placeholder logicalobjectfifo on the given tiles: a
  // dynamic `memref<?xi32>` LOF (its concrete shape is filled in by the
  // downstream control-packet generation pass).
  Value ctrlPlaceholder(ArrayRef<Value> tileVals) {
    OpBuilder &b = builder_;
    auto dynMemref = MemRefType::get({ShapedType::kDynamic}, b.getI32Type());
    auto lofTy = AMDAIE::LogicalObjectFifoType::get(dynMemref);
    return AMDAIE::LogicalObjectFifoPlaceholderOp::create(b, b.getUnknownLoc(),
                                                          lofTy, tileVals);
  }
  AMDAIE::ChannelOp ctrlChannel(Value tile, AMDAIE::StrmSwPortType port,
                                AMDAIE::DMAChannelDir dir) {
    OpBuilder &b = builder_;
    return AMDAIE::ChannelOp::create(b, b.getUnknownLoc(), tile, 0, port, dir);
  }
  Value ctrlConn(Value tgtLof, ArrayRef<Value> tgtChs, Value srcLof,
                 ArrayRef<Value> srcChs, AMDAIE::ConnectionType ct) {
    OpBuilder &b = builder_;
    return AMDAIE::ConnectionOp::create(
        b, b.getUnknownLoc(), tgtLof, tgtChs, srcLof, srcChs,
        AMDAIE::ConnectionTypeAttr::get(&ctx_, ct), /*flow=*/nullptr);
  }

  // The control-packet overlay: per-column shim/memtile CTRL packet feeds
  // (reusing the L3→L2 shim DMA channels), a single broadcast feed to all
  // compute tiles' CTRL ports, and per-column SOUTH circuit returns. This is
  // HW network configuration (structurally fixed), not kernel semantics, and
  // mirrors the textual emitter's overlay exactly.
  // The shim DMA MM2S channel (col, idx) that feeds a ctrl-packet route.
  // Reuses a data-push channel if one exists on that column/index; otherwise
  // creates a dedicated feed channel (needed on columns a route doesn't push
  // to, e.g. non-square grids where A only pushes to the first ROWS columns).
  Value ctrlFeedCh(int col, int idx) {
    auto it = shimDmaCh_.find({col, idx});
    if (it != shimDmaCh_.end()) return it->second;
    Value ch = AMDAIE::ChannelOp::create(
        builder_, builder_.getUnknownLoc(), shimTileVal(col), idx,
        AMDAIE::StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
    shimDmaCh_[{col, idx}] = ch;
    return ch;
  }

  void emitCtrlOverlay(const KKernelDecl *k) {
    ctrlShConns_.assign(cols_, Value());
    ctrlMtConns_.assign(cols_, Value());
    southConns_.assign(cols_, Value());

    std::vector<Value> shPh(cols_);
    for (int c = 0; c < cols_; ++c) {
      Value shTile = shimTileVal(c);
      Tile mt = memtileTileFor(c);
      Value mtTile = getTile(mt.col, mt.row);
      Value ch0 = ctrlFeedCh(c, 0);
      Value ch1 = ctrlFeedCh(c, 1);

      // Shim CTRL feed (fed by shim DMA channel 0).
      AMDAIE::ChannelOp shCtrlS2mm = ctrlChannel(
          shTile, AMDAIE::StrmSwPortType::CTRL, AMDAIE::DMAChannelDir::S2MM);
      shPh[c] = ctrlPlaceholder({shTile});
      ctrlShConns_[c] = ctrlConn(shPh[c], {shCtrlS2mm}, shPh[c], {ch0},
                                 AMDAIE::ConnectionType::Packet);
      // Memtile CTRL feed (fed by shim DMA channel 1).
      AMDAIE::ChannelOp mtCtrlS2mm = ctrlChannel(
          mtTile, AMDAIE::StrmSwPortType::CTRL, AMDAIE::DMAChannelDir::S2MM);
      Value mtPh = ctrlPlaceholder({mtTile});
      ctrlMtConns_[c] = ctrlConn(mtPh, {mtCtrlS2mm}, shPh[c], {ch1},
                                 AMDAIE::ConnectionType::Packet);
    }

    // Compute-tile CTRL ports: one broadcast packet feed (col-major) fed by
    // column-0 shim DMA channel 0.
    SmallVector<Value> coChs, coTiles;
    for (int c = 0; c < cols_; ++c) {
      for (int r = 0; r < rows_; ++r) {
        Tile ct = coreTile(c, r);
        Value coTile = getTile(ct.col, ct.row);
        coChs.push_back(ctrlChannel(coTile, AMDAIE::StrmSwPortType::CTRL,
                                    AMDAIE::DMAChannelDir::S2MM));
        coTiles.push_back(coTile);
      }
    }
    Value coPh = ctrlPlaceholder(coTiles);
    Value feed0 = ctrlFeedCh(0, 0);
    ctrlCoConn_ = ctrlConn(coPh, coChs, shPh.empty() ? Value() : shPh[0],
                           {feed0}, AMDAIE::ConnectionType::Packet);

    // Per-column SOUTH circuit return (CTRL MM2S -> SOUTH S2MM on the shim).
    for (int c = 0; c < cols_; ++c) {
      Value shTile = shimTileVal(c);
      AMDAIE::ChannelOp shCtrlMm2s = ctrlChannel(
          shTile, AMDAIE::StrmSwPortType::CTRL, AMDAIE::DMAChannelDir::MM2S);
      AMDAIE::ChannelOp southS2mm = ctrlChannel(
          shTile, AMDAIE::StrmSwPortType::SOUTH, AMDAIE::DMAChannelDir::S2MM);
      southConns_[c] = ctrlConn(shPh[c], {southS2mm}, shPh[c], {shCtrlMm2s},
                                AMDAIE::ConnectionType::Circuit);
    }
  }

  // Controlcode: persistent circular_dma setup for every route, then a
  // mechanical walk of the controller body (for/issue/wait/barrier).
  void emitControlcode(const KKernelDecl *k, AMDAIE::WorkgroupOp wg) {
    const KOnControllerDecl *ctl = nullptr;
    for (auto *d : k->getBody())
      if (auto *o = dyn_cast<KOnControllerDecl>(d)) ctl = o;
    if (!ctl) return;
    OpBuilder &b = builder_;
    Location loc = b.getUnknownLoc();
    b.setInsertionPoint(wg.getControlCode().getBody()->getTerminator());

    // Persistent-side circular_dma_cpy_nd for each route instance.
    for (auto &ri : routeInsts_) {
      if (!ri.conn) continue;
      ConstExprScope s(eval_);
      for (size_t i = 0; i < ri.decl->getIndexVars().size(); ++i)
        eval_.bind(ri.decl->getIndexVars()[i], ri.idxVals[i]);
      SmallVector<int64_t> to, ts, tst, so, ss, sst;
      if (ri.decl->getTarget().has_value()) {
        to = evalList(ri.decl->getTarget()->offsets);
        ts = evalList(ri.decl->getTarget()->sizes);
        tst = evalList(ri.decl->getTarget()->strides);
      }
      if (ri.decl->getSource().has_value()) {
        so = evalList(ri.decl->getSource()->offsets);
        ss = evalList(ri.decl->getSource()->sizes);
        sst = evalList(ri.decl->getSource()->strides);
      }
      AMDAIE::NpuCircularDmaCpyNdOp::create(b, loc, ri.conn, to, ts, tst, so,
                                            ss, sst);
    }

    // Outstanding issued transfers, for `wait` resolution.
    struct Out {
      Value token;
      bool isSource;
      std::string route;
      int64_t idx;
    };
    std::vector<Out> outstanding;
    std::map<std::string, int> bdNext;

    std::function<void(const std::vector<KStmt *> &)> walk =
        [&](const std::vector<KStmt *> &stmts) {
          for (auto *s : stmts) {
            if (auto *forS = dyn_cast<KForStmt>(s)) {
              int64_t lo = eval_.evalInt(forS->getLo());
              int64_t hi = eval_.evalInt(forS->getHi());
              if (loopIsSpatial(forS->getIndVar(), forS->getBody())) {
                // Spatial loop: each iteration selects a distinct connection;
                // must unroll (constexpr-bind the iv, re-emit per iteration).
                for (int64_t v = lo; v < hi; ++v) {
                  ConstExprScope sc(eval_);
                  eval_.bind(forS->getIndVar(), v);
                  walk(forS->getBody());
                }
              } else {
                // Temporal loop: roll to scf.for so AssignNpuDmaBdIds cycles
                // the BD ids. The iv is bound to the SSA induction variable;
                // iv-dependent offsets become runtime arithmetic.
                Value lb = arith::ConstantIndexOp::create(b, loc, lo);
                Value ub = arith::ConstantIndexOp::create(b, loc, hi);
                Value step = arith::ConstantIndexOp::create(b, loc, 1);
                auto forOp = scf::ForOp::create(b, loc, lb, ub, step);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToStart(forOp.getBody());
                ssaIvs_[forS->getIndVar()] = forOp.getInductionVar();
                walk(forS->getBody());
                ssaIvs_.erase(forS->getIndVar());
              }
            } else if (auto *iss = dyn_cast<KIssueStmt>(s)) {
              std::vector<int64_t> idx;
              for (auto *e : iss->getIndices()) idx.push_back(eval_.evalInt(e));
              const RouteInst *ri = findRouteInst(iss->getRouteName(), idx);
              if (!ri || !ri->conn) continue;
              bool isSource = !iss->getSrcBinding().empty();
              std::string bind =
                  isSource ? iss->getSrcBinding() : iss->getDstBinding();
              int col = idx.empty() ? 0 : (int)idx[0];
              auto lofIt = ddrLofs_.find(bind + "_" + itos(col));
              if (lofIt == ddrLofs_.end()) continue;
              Value ddr = lofIt->second;
              // BD id on the L3 (shim) tile.
              Tile bdt = isSource ? ri->viaSrc.tile : ri->viaDst.tile;
              int bdVal = iss->getBdId() ? (int)eval_.evalInt(iss->getBdId())
                                         : bdNext[itos(bdt.col)]++;
              Value bdConst = arith::ConstantIndexOp::create(b, loc, bdVal);
              Value bdId = AMDAIE::BdIdOp::create(
                  b, loc, getTile(bdt.col, bdt.row), bdConst);
              auto toOFR = [&](const std::vector<KExpr *> &es) {
                SmallVector<OpFoldResult> v;
                for (auto *e : es) v.push_back(emitExprOFR(b, loc, e));
                return v;
              };
              auto offs = toOFR(iss->getOffsets());
              auto szs = toOFR(iss->getSizes());
              auto sts = toOFR(iss->getStrides());
              Value tok;
              if (isSource) {
                auto tokTy = AMDAIE::AsyncSourceTokenType::get(&ctx_);
                tok = AMDAIE::NpuDmaCpyNdOp::create(
                          b, loc, TypeRange{tokTy}, ri->conn,
                          /*target=*/Value{}, ArrayRef<OpFoldResult>{},
                          ArrayRef<OpFoldResult>{}, ArrayRef<OpFoldResult>{},
                          /*target_bd_id=*/Value{}, ddr, offs, szs, sts, bdId)
                          .getResult(0);
              } else {
                auto tokTy = AMDAIE::AsyncTargetTokenType::get(&ctx_);
                tok =
                    AMDAIE::NpuDmaCpyNdOp::create(
                        b, loc, TypeRange{tokTy}, ri->conn, ddr, offs, szs, sts,
                        bdId, /*source=*/Value{}, ArrayRef<OpFoldResult>{},
                        ArrayRef<OpFoldResult>{}, ArrayRef<OpFoldResult>{},
                        /*source_bd_id=*/Value{})
                        .getResult(0);
              }
              outstanding.push_back({tok, isSource, iss->getRouteName(),
                                     idx.empty() ? 0 : idx[0]});
            } else if (auto *w = dyn_cast<KWaitStmt>(s)) {
              bool haveLo = w->getIdxLo() != nullptr;
              bool haveHi = w->getIdxHi() != nullptr;
              int64_t lo = haveLo ? eval_.evalInt(w->getIdxLo()) : 0;
              int64_t hi = haveHi ? eval_.evalInt(w->getIdxHi()) : 0;
              auto matches = [&](const Out &o) {
                if (!w->getRouteName().empty() && o.route != w->getRouteName())
                  return false;
                if (haveHi) return o.idx >= lo && o.idx < hi;
                if (haveLo) return o.idx == lo;
                return true;
              };
              SmallVector<Value> toks;
              std::vector<Out> rest;
              for (auto &o : outstanding) {
                if (matches(o))
                  toks.push_back(o.token);
                else
                  rest.push_back(o);
              }
              outstanding = std::move(rest);
              if (!toks.empty()) AMDAIE::NpuDmaWaitOp::create(b, loc, toks);
            } else if (isa<KBarrierStmt>(s)) {
              AMDAIE::NpuBarrierOp::create(b, loc);
            }
          }
        };
    walk(ctl->getBody());

    // dma_placeholders keep the control-packet overlay connections alive
    // (they have no other controlcode user). Same order as the textual
    // emitter: shim feeds, memtile feeds, compute broadcast, south returns.
    if (controlPacketsEnabled(k)) {
      auto ph = [&](Value conn) {
        if (conn) AMDAIE::NpuDmaPlaceHolderOp::create(b, loc, conn);
      };
      for (auto &c : ctrlShConns_) ph(c);
      for (auto &c : ctrlMtConns_) ph(c);
      ph(ctrlCoConn_);
      for (auto &c : southConns_) ph(c);
    }
  }

  MLIRContext ctx_;
  DiagnosticEngine &diag_;
  ConstExprEvaluator eval_;
  OpBuilder builder_;
  std::map<std::string, int64_t> params_;
  // Loop induction variables currently bound to an SSA value (rolled scf.for
  // in the controlcode), as opposed to constexpr values bound in `eval_`.
  std::map<std::string, Value> ssaIvs_;

  const KKernelDecl *k_ = nullptr;
  std::map<CatalogKind, ResolvedCatalogEntry> catalog_;
  std::map<std::string, std::pair<int64_t, int64_t>> indexRanges_;
  int cols_ = 0, rows_ = 0;
  std::vector<BufferInst> bufferInsts_;
  std::vector<RouteInst> routeInsts_;
  std::map<std::pair<int, int>, Value> tiles_;  // amdaie.tile by (col,row)
  std::map<std::string, Value> ddrLofs_;        // DDR LOF by "<binding>_<col>"
  // Shim-source DMA channels (col,channel) -> Value, reused as ctrl feeds.
  std::map<std::pair<int, int>, Value> shimDmaCh_;
  // Control-packet overlay connections, kept for the controlcode placeholders.
  std::vector<Value> ctrlShConns_, ctrlMtConns_, southConns_;
  Value ctrlCoConn_;
  // Buffer allocations to free at workgroup end (L1 shared, then L2 per inst).
  std::vector<Value> l1Allocs_, l2Allocs_;
  std::string sourceDir_;
  ModuleOp module_;
  std::set<std::string> externFns_;  // declared extern fn names (call targets)
  std::map<std::string, const KExternFnDecl *> externFnDecls_;  // for ABI lookup
  std::map<std::string, Value> l1AllocByName_;  // L1 allocs (routeless locals)
};

}  // namespace

std::string emitModule(const KModuleDecl *mod,
                       const std::map<std::string, int64_t> &params,
                       DiagnosticEngine &diag, const std::string &sourceDir,
                       bool wrap) {
  if (wrap) {
    // Self-contained hal.executable-sources wrapper minting is not implemented
    // in the in-tree OpBuilder emitter. Fail loudly rather than silently emit
    // a bare placed module that the caller will misuse. Use the splice flow
    // (aie_compile.py --ref) for the wrapped output.
    diag.report(SourceLocation{}, diag::err_expected)
        << "--emit-wrapper is not implemented in the in-tree emitter; emit the "
           "placed module (no --emit-wrapper) and wrap via aie_compile.py "
           "--ref";
    return {};
  }
  OpBuilderEmitter e(params, diag, sourceDir);
  return e.run(mod);
}

}  // namespace aiec
