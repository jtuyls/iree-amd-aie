// KIR.h - Kernel Intermediate Representation, post-instantiation.
//
// KIR is what the emitter consumes: every parameter substituted, every
// `for`/`when` unrolled, every constexpr evaluated. It mirrors the LOF
// MLIR structure 1:1.
//
// In the MVP, KIR is hand-constructed by the driver. Later, the Sema +
// Instantiator layers produce KIR from a KAST.

#ifndef AIEC_CODEGEN_KIR_H
#define AIEC_CODEGEN_KIR_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace aiec {
namespace kir {

// ─── Common types ──────────────────────────────────────────────────────

// A physical tile coordinate.
struct Tile {
  int col;
  int row;
};

// A labeled buffer dim (e.g. "ping=2"). Labels are from the closed
// vocabulary defined in the design doc § 3.3.
struct DimLabel {
  std::string name; // ping | kblock | sub | om | on | ok | m | n | k
  int64_t size;
};

// Element type — narrow set for the MVP.
enum class ElemType {
  i32,
  // bf16, f32 later.
};

// Multi-target placement: which physical tiles this buffer lives on.
// For single-target: tiles.size() == 1.
struct Placement {
  std::vector<Tile> tiles;
};

// ─── Buffer ────────────────────────────────────────────────────────────

struct Buffer {
  std::string name;             // base name (e.g. "a_l2")
  std::optional<int> col;       // [c] index if any
  std::optional<int> row;       // [r] index if any (can have both)
  ElemType elemType;
  std::vector<DimLabel> shape;  // ordered labels + sizes
  Placement placement;          // tiles this buffer lives on

  // Cached flat size in elements (product of shape sizes).
  int64_t flatSize() const {
    int64_t p = 1;
    for (auto &d : shape) p *= d.size;
    return p;
  }
};

// ─── Route ─────────────────────────────────────────────────────────────

enum class RouteType { Packet, Circuit };

// A stride pattern: source/target offsets + sizes + strides describing
// a strided view (matches LOF's dma_cpy_nd offset/size/stride lists).
struct StridePattern {
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

// A circular_dma_cpy_nd setup: target view + source view.
struct CircularDmaSetup {
  StridePattern targetView;
  StridePattern sourceView;
};

// One physical channel allocation on a tile.
struct Channel {
  Tile tile;
  int id;                       // channel index within the tile
  bool isMM2S;                  // true = MM2S, false = S2MM
  std::string portType;         // "DMA" | "CTRL" | "SOUTH"
};

// A route connects logical objectfifos via channels. The CircularDmaSetup
// fires per-segment in the controlcode.
struct Route {
  std::string name;             // base name
  std::optional<int> col;       // [c] index if any
  std::optional<int> row;       // [r] index if any
  RouteType type;

  // Source endpoint: either a binding name (e.g. "A") with a slice, or
  // a buffer reference.
  enum class EndpointKind { Binding, Buffer };
  EndpointKind sourceKind;
  std::string sourceName;       // binding name or buffer name
  std::optional<int> sourceCol;
  std::optional<int> sourceRow;

  // Target endpoint (mirror).
  EndpointKind targetKind;
  std::string targetName;
  std::optional<int> targetCol;
  std::optional<int> targetRow;

  // The two via tiles (single-hop).
  Tile viaSource;
  Tile viaTarget;
  Placement viaTargets;         // for multi-target circuit routes (e.g. fan-out)

  // Channels assigned by the channel allocator.
  Channel sourceChannel;
  Channel targetChannel;
  std::vector<Channel> extraTargetChannels; // multi-target fan-out

  // Circular DMA setup (filled by stride deriver).
  CircularDmaSetup circular;
};

// ─── Compute body ───────────────────────────────────────────────────────

// Statements appearing inside `on core[c, r] { ... }`. Each segment
// (barrier-delimited) is itself a vector of these.
struct ZeroStmt { std::string bufName; };
struct AcquireStmt {
  std::string handle;           // user-facing handle name
  std::string bufName;
  bool produce;                 // true = Produce, false = Consume
};
struct ReleaseStmt { std::string handle; };
struct MacStmt { std::string lhs, rhs, out; };
struct CallUkernelStmt {
  std::string ukernel;
  std::vector<std::string> args;
};
struct ActScaleStmt { std::string buf; int64_t scale; };  // out := out * scale
struct ForallStmt;
struct ReduceStmt;
using ComputeStmt = std::variant<ZeroStmt, AcquireStmt, ReleaseStmt,
                                  CallUkernelStmt, ActScaleStmt,
                                  std::shared_ptr<ForallStmt>,
                                  std::shared_ptr<ReduceStmt>>;
struct ForallStmt {
  int dimM, dimN;                 // forall (mi, ni) in (M, N)
  std::vector<ComputeStmt> body;
};
struct ReduceStmt {
  std::string indVar;
  int64_t lo, hi;
  std::vector<ComputeStmt> body;
};

// Per-tile compute segments, delimited by barriers.
struct CoreBody {
  std::vector<std::vector<ComputeStmt>> segments;
};

// ─── Controlcode ────────────────────────────────────────────────────────

struct IssueStmt {
  std::string routeName;
  std::optional<int> col;       // [c] if any
  std::optional<int> row;       // [r] if any
  std::string srcBinding;       // L3 src override (e.g. "A", "T")
  std::string dstBinding;       // L3 dst override
};
struct WaitStmt {};
struct ControlBarrierStmt {};
using ControlStmt = std::variant<IssueStmt, WaitStmt, ControlBarrierStmt>;

struct ControllerBody {
  std::vector<std::vector<ControlStmt>> segments;
};

// ─── Tile catalog ───────────────────────────────────────────────────────

struct TileCatalog {
  std::vector<Tile> shims;       // shims[c] = tile(c, 0)
  std::vector<Tile> memtiles;    // memtiles[c] = tile(c, 1)
  std::vector<std::vector<Tile>> cores; // cores[c][r] = tile(c, r+2)
  Tile controller;
};

// ─── Bindings ───────────────────────────────────────────────────────────

struct Binding {
  std::string name;             // "A", "B1", "T", "B2", "C"
  int index;                    // hal binding index
  int64_t dim0, dim1;           // 2-D tensor shape
  ElemType elemType;
  bool readOnly;
};

// ─── Kernel ─────────────────────────────────────────────────────────────

struct Kernel {
  std::string name;             // mangled instantiation name
  int dispatchSize;             // = N*M typically (used in @fused_dispatch_0_elementwise_<SIZE>_i32)
  std::vector<Binding> bindings;
  TileCatalog catalog;
  std::vector<Buffer> buffers;
  std::vector<Route> routes;
  CoreBody perCoreBody;         // same body executes on each core tile
  ControllerBody controller;
  int actScale = 3;
};

// ─── Module ─────────────────────────────────────────────────────────────

struct Module {
  std::string name;
  std::vector<Kernel> kernels;
};

} // namespace kir
} // namespace aiec

#endif // AIEC_CODEGEN_KIR_H
