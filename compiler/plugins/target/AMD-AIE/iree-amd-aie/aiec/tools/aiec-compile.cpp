// aiec-compile.cpp - driver entrypoint.
//
// Thin wrapper around CompilerInvocation + CompilerInstance. Mirrors
// clang/tools/driver/cc1_main.cpp's split: argv parsing here, all
// compilation logic in CompilerInstance.
//
// To add a multi-process driver later (clang's `clang` → `clang -cc1`
// model): introduce a `tools/aiec/aiec.cpp` that drives jobs and forks
// `aiec-compile -cc1` per TU. The `-cc1` flag is reserved for that
// future split; in this MVP every invocation is implicitly `-cc1` mode.

#include "aiec/Frontend/CompilerInstance.h"
#include "aiec/Frontend/CompilerInvocation.h"

#include <iostream>

int main(int argc, char **argv) {
  aiec::CompilerInvocation invocation;
  if (!aiec::CompilerInvocation::createFromArgs(argc, argv, invocation,
                                                std::cerr)) {
    return 2;
  }
  aiec::CompilerInstance instance(std::move(invocation), std::cerr);
  return instance.run();
}
