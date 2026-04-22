//===- salmon_tablegen.cpp -------------------------------------*- C++ -*-===//
//
// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level entry point for `salmon-tblgen`. The tool is a thin wrapper
// around `llvm::TableGenMain`: it reads an LLVM TableGen file (in practice
// the AMDGPU target's `AMDGPU.td`), then dispatches to whichever generator
// was selected on the command line.
//
// Generators are registered as global initializers via
// `llvm::TableGen::Emitter::Opt` in their own translation units (see
// `mcinst_wrapper_gen.cpp` for the AMDGCN MCInst-wrapper backend). To add
// a new generator, drop a new `*.cpp` file into this directory, register
// its callback with `TableGen::Emitter::Opt`, and add it to the
// `salmon-tblgen` source list in `CMakeLists.txt`.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"

int main(int argc, char **argv) {
  llvm::InitLLVM x(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // `TableGenMain` with a null callback dispatches to whichever
  // `TableGen::Emitter::Opt` was selected on the command line (e.g.
  // `--gen-amdgcn-mcinst-wrappers`). The `MultiFileTableGenMainFn`
  // overload is picked explicitly to disambiguate the two `nullptr`
  // overloads of `TableGenMain`.
  return llvm::TableGenMain(argv[0], llvm::MultiFileTableGenMainFn(nullptr));
}
