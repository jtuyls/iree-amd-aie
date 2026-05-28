// SourceLocation.h - source position tracking.
//
// Mirrors clang::SourceLocation in spirit (opaque handle + manager for
// expansion), but simplified: we don't need macro expansion or include
// chains, so a SourceLocation is just (fileID, offset). The
// SourceManager owns the buffer text and resolves (fileID, offset) to
// (line, column) on demand.
//
// See clang/include/clang/Basic/SourceLocation.h for the clang version.

#ifndef AIEC_BASIC_SOURCE_LOCATION_H
#define AIEC_BASIC_SOURCE_LOCATION_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aiec {

// FileID identifies one input buffer. Index into SourceManager's buffers.
class FileID {
public:
  FileID() = default;
  static FileID get(uint32_t id) {
    FileID f;
    f.id_ = id;
    return f;
  }
  bool isValid() const { return id_ != kInvalid; }
  uint32_t getOpaque() const { return id_; }
  bool operator==(FileID o) const { return id_ == o.id_; }
  bool operator!=(FileID o) const { return id_ != o.id_; }

private:
  static constexpr uint32_t kInvalid = 0xFFFFFFFF;
  uint32_t id_ = kInvalid;
};

// SourceLocation is a (FileID, byteOffset) pair, packed into 64 bits.
// Opaque outside the SourceManager.
class SourceLocation {
public:
  SourceLocation() = default;
  static SourceLocation make(FileID f, uint32_t offset) {
    SourceLocation s;
    s.fileID_ = f;
    s.offset_ = offset;
    return s;
  }
  bool isValid() const { return fileID_.isValid(); }
  FileID getFileID() const { return fileID_; }
  uint32_t getOffset() const { return offset_; }

private:
  FileID fileID_;
  uint32_t offset_ = 0;
};

// SourceRange is [begin, end) — both inclusive in spirit, end is the
// past-the-end byte of the last character.
struct SourceRange {
  SourceLocation begin;
  SourceLocation end;
};

// SourceManager owns input buffers and resolves SourceLocations to
// human-readable (line, column) tuples. One SourceManager per
// compilation; passed by reference to Lexer, Parser, etc.
class SourceManager {
public:
  // Add a buffer to the manager. Returns its FileID.
  // `path` is shown in diagnostics; `contents` is the source text.
  FileID addBuffer(std::string path, std::string contents);

  // Resolve a SourceLocation to (line, column), both 1-indexed.
  struct LineCol {
    uint32_t line;
    uint32_t column;
  };
  LineCol getLineCol(SourceLocation loc) const;

  // Get the buffer text for a FileID.
  std::string_view getBufferText(FileID f) const;
  std::string_view getBufferPath(FileID f) const;

  // Get the line of text containing loc (no trailing newline).
  std::string_view getLineText(SourceLocation loc) const;

private:
  struct Buffer {
    std::string path;
    std::string contents;
    // Byte offsets of the start of each line (lazily computed).
    mutable std::vector<uint32_t> lineStarts;
    mutable bool lineStartsComputed = false;
    void ensureLineStarts() const;
  };
  std::vector<Buffer> buffers_;
};

} // namespace aiec

#endif // AIEC_BASIC_SOURCE_LOCATION_H
