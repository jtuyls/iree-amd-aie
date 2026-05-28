// KASTContext.h - bump-allocator-owned AST node lifetime.
//
// Mirrors clang::ASTContext: AST nodes are allocated from a
// per-compilation pool and live as long as the context. No
// per-node free; predictable lifetime.

#ifndef AIEC_AST_K_AST_CONTEXT_H
#define AIEC_AST_K_AST_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <new>
#include <vector>

namespace aiec {

// Simple bump allocator: hands out chunks from large slabs.
class BumpAllocator {
public:
  BumpAllocator() = default;
  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;

  void *allocate(size_t size, size_t align) {
    if (slabs_.empty() || space_ < size + align) {
      growSlab(std::max(size + align, kSlabSize));
    }
    // Align up.
    void *p = cur_;
    size_t pad = (align - (reinterpret_cast<uintptr_t>(p) % align)) % align;
    cur_ = static_cast<char *>(cur_) + pad + size;
    space_ -= pad + size;
    return static_cast<char *>(p) + pad;
  }

private:
  static constexpr size_t kSlabSize = 4096;
  std::list<std::vector<char>> slabs_;
  void *cur_ = nullptr;
  size_t space_ = 0;

  void growSlab(size_t bytes) {
    slabs_.emplace_back(bytes);
    cur_ = slabs_.back().data();
    space_ = bytes;
  }
};

class KASTContext {
public:
  KASTContext() = default;
  KASTContext(const KASTContext &) = delete;

  template <typename T, typename... Args>
  T *create(Args &&...args) {
    void *mem = alloc_.allocate(sizeof(T), alignof(T));
    T *t = new (mem) T(std::forward<Args>(args)...);
    // Register for explicit destruction so non-trivial fields (e.g.
    // std::string members) clean up correctly.
    cleanups_.push_back([t] { t->~T(); });
    return t;
  }

  ~KASTContext() {
    // Destroy in reverse order.
    for (auto it = cleanups_.rbegin(); it != cleanups_.rend(); ++it)
      (*it)();
  }

private:
  BumpAllocator alloc_;
  std::vector<std::function<void()>> cleanups_;
};

} // namespace aiec

#endif // AIEC_AST_K_AST_CONTEXT_H
