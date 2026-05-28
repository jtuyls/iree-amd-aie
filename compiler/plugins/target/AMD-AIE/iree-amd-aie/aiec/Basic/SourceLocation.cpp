#include "aiec/Basic/SourceLocation.h"

namespace aiec {

void SourceManager::Buffer::ensureLineStarts() const {
  if (lineStartsComputed)
    return;
  lineStarts.clear();
  lineStarts.push_back(0);
  for (uint32_t i = 0; i < contents.size(); ++i) {
    if (contents[i] == '\n')
      lineStarts.push_back(i + 1);
  }
  lineStartsComputed = true;
}

FileID SourceManager::addBuffer(std::string path, std::string contents) {
  buffers_.push_back({std::move(path), std::move(contents), {}, false});
  return FileID::get(static_cast<uint32_t>(buffers_.size() - 1));
}

SourceManager::LineCol SourceManager::getLineCol(SourceLocation loc) const {
  if (!loc.isValid())
    return {0, 0};
  const auto &buf = buffers_[loc.getFileID().getOpaque()];
  buf.ensureLineStarts();
  uint32_t off = loc.getOffset();
  // Binary search for the line.
  uint32_t lo = 0, hi = buf.lineStarts.size();
  while (lo + 1 < hi) {
    uint32_t mid = (lo + hi) / 2;
    if (buf.lineStarts[mid] <= off)
      lo = mid;
    else
      hi = mid;
  }
  return {lo + 1, off - buf.lineStarts[lo] + 1};
}

std::string_view SourceManager::getBufferText(FileID f) const {
  return buffers_[f.getOpaque()].contents;
}

std::string_view SourceManager::getBufferPath(FileID f) const {
  return buffers_[f.getOpaque()].path;
}

std::string_view SourceManager::getLineText(SourceLocation loc) const {
  if (!loc.isValid())
    return {};
  const auto &buf = buffers_[loc.getFileID().getOpaque()];
  buf.ensureLineStarts();
  auto lc = getLineCol(loc);
  uint32_t lineStart = buf.lineStarts[lc.line - 1];
  uint32_t lineEnd = (lc.line < buf.lineStarts.size())
                         ? buf.lineStarts[lc.line] - 1 // exclude '\n'
                         : static_cast<uint32_t>(buf.contents.size());
  // Strip a trailing '\r' if present.
  if (lineEnd > lineStart && buf.contents[lineEnd - 1] == '\r')
    --lineEnd;
  return std::string_view(buf.contents).substr(lineStart, lineEnd - lineStart);
}

} // namespace aiec
