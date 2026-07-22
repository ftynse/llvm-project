//===- raise_cli.cpp - Hotswap raiser test driver -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line front end for the hotswap raiser, used by the lit tests under
// test-lit/hotswap-raise. It loads a code object, enumerates its kernels, and
// dumps the raised LLVM IR for FileCheck.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> [--isa=<arch>] [--emit-ir[=<kernel>]]
//
// The source ISA is taken from --isa, else the filename (a `gfx<digits>`
// substring), else the ELF e_flags. --emit-ir selects which kernels to dump:
// bare (or absent) dumps every kernel in code-object order; --emit-ir=<k> or
// --emit-ir=<k1>,<k2> dumps the listed kernels in the given order. A
// multi-kernel dump separates each kernel's IR with a `; === raise_cli
// kernel: <name> ===` line so a single FileCheck pass can anchor per-kernel
// checks. Diagnostics go to stderr and the IR to stdout, so a refuse test can
// FileCheck stderr under `%not ... 2>&1` while a raise test checks stdout.
// The exit code is 0 iff every selected kernel raised.
//
//===----------------------------------------------------------------------===//

#include "comgr-metadata.h"
#include "comgr.h"
#include "hotswap/canonical-op.h"
#include "hotswap/code-object-utils.h"
#include "hotswap/decode.h"
#include "hotswap/mc-state.h"
#include "hotswap/opcode-map.h"
#include "hotswap/raiser.h"

// raiser.h forward-declares llvm::LLVMContext and llvm::Module, but
// RaiseResult holds them by unique_ptr, so the destructor synthesized in
// main() needs the complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>

namespace {

namespace cl = llvm::cl;

// Look for a `gfx<digits>[a-z]?` substring anywhere in the path so an ISA can
// be inferred from a filename when --isa is not given.
std::string autoDetectIsa(llvm::StringRef Path) {
  for (size_t I = 0; I + 3 < Path.size(); ++I) {
    if (Path[I] != 'g' || Path[I + 1] != 'f' || Path[I + 2] != 'x')
      continue;
    size_t J = I + 3;
    while (J < Path.size() && std::isdigit(static_cast<unsigned char>(Path[J])))
      ++J;
    if (J == I + 3)
      continue;
    if (J < Path.size() && Path[J] >= 'a' && Path[J] <= 'z')
      ++J;
    return Path.substr(I, J - I).str();
  }
  return {};
}

cl::opt<std::string> CoPathOpt(cl::Positional, cl::Required,
                               cl::desc("<code-object.co|.hsaco>"));

cl::opt<std::string> IsaOpt("isa", cl::value_desc("arch"),
                            cl::desc("Source ISA; inferred from the filename "
                                     "or ELF e_flags when not given."));

cl::opt<std::string>
    EmitIrOpt("emit-ir", cl::ValueOptional,
              cl::value_desc("kernel[,kernel...]"),
              cl::desc("Dump raised LLVM IR on stdout. Bare or absent = all "
                       "kernels; =<k>[,<k>...] selects a subset in order."));

cl::opt<std::string> DumpDecodedOpt(
    "dump-decoded", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
    cl::desc("Dump the decoded instruction listing (offset, canonical op, "
             "disassembly) instead of raising. Same kernel selection as "
             "--emit-ir."));

// Resolve a --emit-ir / --dump-decoded value into the ordered list of kernels
// to process: empty selects every kernel in code-object order; a comma list
// selects the named kernels in order. Reports unknown names on stderr.
bool resolveTargets(llvm::StringRef Requested,
                    llvm::ArrayRef<std::string> KernelNames,
                    llvm::StringRef CoPath,
                    llvm::SmallVectorImpl<std::string> &Targets) {
  if (Requested.empty()) {
    Targets.assign(KernelNames.begin(), KernelNames.end());
    return true;
  }
  llvm::SmallVector<llvm::StringRef> RequestedNames;
  Requested.split(RequestedNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef Name : RequestedNames) {
    Name = Name.trim();
    if (!llvm::is_contained(KernelNames, Name)) {
      llvm::errs() << "raise_cli: kernel '" << Name << "' not found in "
                   << CoPath << "\n";
      return false;
    }
    Targets.push_back(Name.str());
  }
  return true;
}

} // namespace

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv,
                              "Hotswap raiser test driver: dumps the raised "
                              "LLVM IR for a code object's kernels.\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> CoBufOrErr =
      llvm::MemoryBuffer::getFile(CoPathOpt, /*IsText=*/false);
  if (!CoBufOrErr) {
    llvm::errs() << "raise_cli: cannot read " << CoPathOpt << ": "
                 << CoBufOrErr.getError().message() << "\n";
    return 2;
  }
  llvm::MemoryBufferRef CoData = (*CoBufOrErr)->getMemBufferRef();

  std::string Isa = IsaOpt;
  if (Isa.empty()) {
    Isa = autoDetectIsa(CoPathOpt);
    if (Isa.empty()) {
      std::string ElfIsa;
      if (COMGR::metadata::getElfIsaName(CoData, ElfIsa) ==
          AMD_COMGR_STATUS_SUCCESS)
        Isa = std::move(ElfIsa);
    }
    if (Isa.empty()) {
      llvm::errs() << "raise_cli: could not infer ISA from " << CoPathOpt
                   << "; pass --isa=<arch>\n";
      return 2;
    }
  }

  llvm::Expected<llvm::SmallVector<std::string>> KernelNamesOrErr =
      COMGR::hotswap::listKernelNames(CoData);
  if (!KernelNamesOrErr) {
    llvm::errs() << "raise_cli: no kernels in " << CoPathOpt << ": "
                 << llvm::toString(KernelNamesOrErr.takeError()) << "\n";
    return 2;
  }
  llvm::SmallVector<std::string> KernelNames = std::move(*KernelNamesOrErr);
  if (KernelNames.empty()) {
    llvm::errs() << "raise_cli: no kernels in " << CoPathOpt << "\n";
    return 2;
  }

  bool DumpDecoded = DumpDecodedOpt.getNumOccurrences() > 0;
  llvm::SmallVector<std::string> Targets;
  if (!resolveTargets(DumpDecoded ? llvm::StringRef(DumpDecodedOpt)
                                  : llvm::StringRef(EmitIrOpt),
                      KernelNames, CoPathOpt, Targets))
    return 2;

  bool Multi = Targets.size() > 1;
  bool AnyFailed = false;

  // Both modes work over the kernel .text and need AMDGPU registered into this
  // binary's own LLVM (see standalone-init.cpp for why not the amd_comgr copy).
  COMGR::ensureLLVMInitialized();
  llvm::Expected<COMGR::hotswap::TextSection> TextOrErr =
      COMGR::hotswap::extractTextSection(CoData);
  if (!TextOrErr) {
    llvm::errs() << "raise_cli: could not extract .text from " << CoPathOpt
                 << ": " << llvm::toString(TextOrErr.takeError()) << "\n";
    return 2;
  }
  COMGR::hotswap::TextSection Text = std::move(*TextOrErr);

  // --dump-decoded path: decode each kernel's .text to a canonical instruction
  // listing without raising. Exercises the MC stack, opcode map, and decoder.
  if (DumpDecoded) {
    // initMCState wants the bare AMDGPU processor (e.g. gfx942); the --isa /
    // ELF form may be a full target id like "amdgcn-amd-amdhsa--gfx942:xnack-".
    llvm::StringRef Cpu = llvm::StringRef(Isa).rsplit('-').second;
    if (Cpu.empty())
      Cpu = Isa;
    Cpu = Cpu.take_until([](char C) { return C == ':'; });

    llvm::Expected<COMGR::hotswap::MCState> McOrErr =
        COMGR::hotswap::initMCState(Cpu);
    if (!McOrErr) {
      llvm::errs() << "raise_cli: MC init failed for ISA '" << Isa
                   << "': " << llvm::toString(McOrErr.takeError()) << "\n";
      return 2;
    }
    COMGR::hotswap::MCState Mc = std::move(*McOrErr);
    COMGR::hotswap::OpcodeMap OpcMap;
    OpcMap.build(*Mc.InstrInfo);

    for (const std::string &Target : Targets) {
      llvm::Expected<COMGR::hotswap::KernelSymbolExtent> ExtentOrErr =
          COMGR::hotswap::findKernelSymbolExtent(CoData, Target);
      if (!ExtentOrErr) {
        llvm::errs() << "raise_cli: kernel '" << Target
                     << "' extent: " << llvm::toString(ExtentOrErr.takeError())
                     << "\n";
        AnyFailed = true;
        continue;
      }
      llvm::Expected<COMGR::hotswap::DecodeResult> DecodedOrErr =
          COMGR::hotswap::decodeKernel(Mc, OpcMap, Text.Bytes,
                                       ExtentOrErr->Offset,
                                       ExtentOrErr->Offset + ExtentOrErr->Size);
      if (!DecodedOrErr) {
        llvm::errs() << "raise_cli: kernel '" << Target
                     << "' decode: " << llvm::toString(DecodedOrErr.takeError())
                     << "\n";
        AnyFailed = true;
        continue;
      }
      if (Multi)
        llvm::outs() << "; === raise_cli kernel: " << Target << " ===\n";
      for (const COMGR::hotswap::DecodedInst &Di : DecodedOrErr->Insts) {
        llvm::outs() << "0x";
        llvm::outs().write_hex(Di.Offset);
        llvm::outs() << "  " << COMGR::hotswap::canonicalOpName(Di.CanonOp)
                     << "  " << Di.FullText << "\n";
      }
    }
    return AnyFailed ? 1 : 0;
  }

  // --emit-ir path (default): raise each kernel and dump its LLVM IR.
  for (const std::string &Target : Targets) {
    llvm::Expected<COMGR::hotswap::KernelMeta> MetaOrErr =
        COMGR::hotswap::extractKernelMeta(CoData, Target);
    if (!MetaOrErr) {
      llvm::errs() << "raise_cli: kernel '" << Target
                   << "' metadata: " << llvm::toString(MetaOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }
    COMGR::hotswap::KernelMeta Meta = std::move(*MetaOrErr);

    llvm::Expected<COMGR::hotswap::KernelSymbolExtent> ExtentOrErr =
        COMGR::hotswap::findKernelSymbolExtent(CoData, Target);
    if (!ExtentOrErr) {
      llvm::errs() << "raise_cli: kernel '" << Target
                   << "' extent: " << llvm::toString(ExtentOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }

    llvm::Expected<COMGR::hotswap::RaiseResult> RaisedOrErr =
        COMGR::hotswap::raiseToIR(Text.Bytes, Isa, Target, Meta,
                                  ExtentOrErr->Offset, ExtentOrErr->Size,
                                  /*CompilationTargetIsa=*/"",
                                  /*EnableWritelaneRewrite=*/true,
                                  /*EnableWaveNative=*/true,
                                  /*AssumeHipGlobalOffsetZero=*/false,
                                  /*ForceModrepDoubled=*/false, Text.Address,
                                  Text.ImageSections);
    if (!RaisedOrErr) {
      // The raiser only returns a module on success, so a failure has no
      // partial IR to dump; report the structured reason on stderr.
      llvm::errs() << "raise_cli: kernel '" << Target << "' failed to raise: "
                   << llvm::toString(RaisedOrErr.takeError()) << "\n";
      AnyFailed = true;
      continue;
    }
    COMGR::hotswap::RaiseResult Raised = std::move(*RaisedOrErr);

    if (Multi)
      llvm::outs() << "; === raise_cli kernel: " << Target << " ===\n";
    Raised.Module->print(llvm::outs(), nullptr);
  }

  return AnyFailed ? 1 : 0;
}
