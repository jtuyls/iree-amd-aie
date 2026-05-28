#include "aiec/Frontend/CompilerInvocation.h"

#include <cstring>
#include <iostream>
#include <string>

namespace aiec {

namespace {

bool startsWith(const char *s, const char *prefix) {
  return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

// Parse "KEY=VALUE" into the map. Returns false on malformed input.
bool parseParam(const std::string &kv, std::map<std::string, std::string> &out,
                std::ostream &err) {
  auto eq = kv.find('=');
  if (eq == std::string::npos) {
    err << "aiec-compile: expected KEY=VALUE in template param, got '"
        << kv << "'\n";
    return false;
  }
  out[kv.substr(0, eq)] = kv.substr(eq + 1);
  return true;
}

void usage(std::ostream &os) {
  os <<
R"(usage: aiec-compile [options] INPUT.aiec
options:
  -o, --output PATH    write output MLIR to PATH (default: stdout)
  -p, --param KEY=VAL  set a template parameter (repeatable)
  -k, --kernel NAME    instantiate the named kernel from the module
                       (default: only kernel in the module)
  --target DEVICE      target device: npu4 (default: npu4)
  --parse-only         parse + sema only; do not emit MLIR
  -Werror              treat warnings as errors
  -h, --help           print this message

example:
  aiec-compile fused_chain.aiec \
      --param COLS=2 --param ROWS=2 \
      --param M=128 --param N=128 \
      --param K_mm1=128 --param K_mm2=128 \
      --param act_scale=3 \
      -o fused_2x2.lof.mlir
)";
}

} // namespace

bool CompilerInvocation::createFromArgs(int argc, char **argv,
                                        CompilerInvocation &out,
                                        std::ostream &err) {
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        err << "aiec-compile: missing argument after " << what << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      usage(std::cout);
      std::exit(0);
    } else if (std::strcmp(a, "-o") == 0 || std::strcmp(a, "--output") == 0) {
      const char *v = need(a);
      if (!v) return false;
      out.outputPath = v;
    } else if (std::strcmp(a, "-p") == 0 || std::strcmp(a, "--param") == 0) {
      const char *v = need(a);
      if (!v) return false;
      if (!parseParam(v, out.templateParams, err)) return false;
    } else if (std::strcmp(a, "-k") == 0 || std::strcmp(a, "--kernel") == 0) {
      const char *v = need(a);
      if (!v) return false;
      out.kernelName = v;
    } else if (std::strcmp(a, "--target") == 0) {
      const char *v = need(a);
      if (!v) return false;
      if (std::strcmp(v, "npu4") == 0) out.target = AIEDevice::npu4;
      else {
        err << "aiec-compile: unknown target '" << v << "'\n";
        return false;
      }
    } else if (std::strcmp(a, "--parse-only") == 0) {
      out.action = FrontendActionKind::ParseOnly;
    } else if (std::strcmp(a, "--emit-wrapper") == 0) {
      out.emitWrapper = true;
    } else if (std::strcmp(a, "-Werror") == 0) {
      out.warningsAsErrors = true;
    } else if (startsWith(a, "-")) {
      err << "aiec-compile: unknown option '" << a << "'\n";
      return false;
    } else {
      if (!out.inputPath.empty()) {
        err << "aiec-compile: multiple input files not supported yet\n";
        return false;
      }
      out.inputPath = a;
    }
  }

  if (out.inputPath.empty()) {
    err << "aiec-compile: no input file (use -h for usage)\n";
    return false;
  }
  return true;
}

void CompilerInvocation::dump(std::ostream &os) const {
  os << "CompilerInvocation:\n"
     << "  inputPath=" << inputPath << "\n"
     << "  outputPath=" << (outputPath.empty() ? "<stdout>" : outputPath) << "\n"
     << "  kernel=" << (kernelName.empty() ? "<auto>" : kernelName) << "\n"
     << "  action="
     << (action == FrontendActionKind::ParseOnly ? "parse-only" : "emit-mlir")
     << "\n  target=npu4\n  params:\n";
  for (auto &kv : templateParams)
    os << "    " << kv.first << " = " << kv.second << "\n";
}

} // namespace aiec
