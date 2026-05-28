#include "aiec/Frontend/CompilerInstance.h"

#include "aiec/AST/KASTContext.h"
#include "aiec/AST/KDecl.h"
#include "aiec/CodeGen/Emitter.h"
#include "aiec/Lex/Lexer.h"
#include "aiec/Parse/Parser.h"

#include <cstdint>
#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aiec {

CompilerInstance::CompilerInstance(CompilerInvocation invocation,
                                   std::ostream &diagOut)
    : invocation_(std::move(invocation)), diagOut_(diagOut), sm_(),
      diag_(sm_, diagOut) {}

int CompilerInstance::run() {
  switch (invocation_.action) {
  case FrontendActionKind::EmitMLIR:
    return runEmitMLIR();
  case FrontendActionKind::ParseOnly:
    return runParseOnly();
  }
  return 1;
}

namespace {

bool slurpFile(const std::string &path, std::string &contents) {
  std::ifstream f(path);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  contents = ss.str();
  return true;
}

} // namespace

int CompilerInstance::runParseOnly() {
  std::string contents;
  if (!slurpFile(invocation_.inputPath, contents)) {
    diagOut_ << "aiec-compile: cannot read input '" << invocation_.inputPath
             << "'\n";
    return 1;
  }
  FileID file = sm_.addBuffer(invocation_.inputPath, contents);
  Lexer lex(file, sm_, diag_);
  KASTContext ctx;
  Parser parser(lex, ctx, diag_);
  KModuleDecl *mod = parser.parseFile();
  if (!mod || diag_.hasErrors()) {
    diagOut_ << "aiec-compile: parse failed\n";
    return 1;
  }
  diagOut_ << "aiec-compile: parse OK (" << mod->getDecls().size()
           << " top-level decls)\n";
  return 0;
}

int CompilerInstance::runEmitMLIR() {
  std::string srcContents;
  if (!slurpFile(invocation_.inputPath, srcContents)) {
    diagOut_ << "aiec-compile: cannot read input '" << invocation_.inputPath
             << "'\n";
    return 1;
  }
  FileID file = sm_.addBuffer(invocation_.inputPath, srcContents);

  Lexer lex(file, sm_, diag_);
  KASTContext ctx;
  Parser parser(lex, ctx, diag_);
  KModuleDecl *mod = parser.parseFile();
  if (!mod || diag_.hasErrors()) {
    diagOut_ << "aiec-compile: parse failed\n";
    return 1;
  }

  // Convert string template params to int64 (no exceptions: the IREE build
  // disables them).
  std::map<std::string, int64_t> params;
  for (auto &kv : invocation_.templateParams) {
    long long value = 0;
    const char *first = kv.second.c_str();
    const char *last = first + kv.second.size();
    auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr != last) {
      diagOut_ << "aiec-compile: parameter '" << kv.first
               << "' is not an integer ('" << kv.second << "')\n";
      return 1;
    }
    params[kv.first] = static_cast<int64_t>(value);
  }

  // Directory of the source file, so the emitter can resolve relative
  // `extern fn ... impl "path"` templates.
  std::string sourceDir;
  {
    auto slash = invocation_.inputPath.find_last_of('/');
    if (slash != std::string::npos)
      sourceDir = invocation_.inputPath.substr(0, slash);
  }
  std::string mlir =
      emitModule(mod, params, diag_, sourceDir, invocation_.emitWrapper);
  if (mlir.empty() || diag_.hasErrors()) {
    diagOut_ << "aiec-compile: codegen failed\n";
    return 1;
  }

  if (invocation_.outputPath.empty()) {
    std::cout << mlir;
  } else {
    std::ofstream f(invocation_.outputPath);
    if (!f) {
      diagOut_ << "aiec-compile: cannot write '" << invocation_.outputPath
               << "'\n";
      return 1;
    }
    f << mlir;
  }
  return 0;
}

} // namespace aiec
