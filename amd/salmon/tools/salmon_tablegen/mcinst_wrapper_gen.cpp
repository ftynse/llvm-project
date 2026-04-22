//===- MCInstWrapperGen.cpp ----------------------------------------------===//
//
// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a TableGen backend that emits one C++ wrapper class
// per AMDGCN instruction *family* (VOP1, VOP2, VOP3, VOP3P, VOPC, SOP1, SOP2,
// SOPK, SOPC, SOPP, MUBUF, MTBUF, SMRD, MIMG, VIMAGE, VSAMPLE, EXP, FLAT, DS,
// VINTRP, SDWA, DPP, TRANS, LDSDIR, VINTERP, ...).
//
// Each wrapper:
//   * holds a reference to a `llvm::MCInst` plus a `llvm::MCInstrInfo` for
//     `TSFlags` lookups,
//   * provides a `classof` based on `SIInstrFlags::<Family>` so that
//     `llvm::isa<VOP2Wrapper>(...)` / `llvm::dyn_cast<VOP2Wrapper>(...)` work,
//   * exposes one named accessor per operand observed in any record belonging
//     to the family, returning `const llvm::MCOperand *` (nullptr if the
//     concrete opcode does not have that operand).
//
// Accessor lookup strategy
// ------------------------
// The fast path uses the `AMDGPU::OpName` enum and `getNamedOperandIdx` table
// that LLVM's AMDGPU backend emits for any record that opts in via
// `UseNamedOperandTable = 1` (which is the vast majority).
//
// A handful of pseudos do *not* set that bit, and therefore have operand
// names (e.g. `vsrc`, `subreg`) that are absent from `AMDGPU::OpName`. For
// those we cannot call `getNamedOperandIdx`, so this generator emits a
// per-opcode positional fallback `switch` statement covering exactly the
// affected `(opcode, MCInst-operand-index)` pairs, computed at TableGen
// time from the flattened `(outs, ins)` dag. The two paths are disjoint by
// construction (an opcode either is in LLVM's named table or is in the
// fallback switch), so combining them is safe.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <algorithm>
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static cl::OptionCategory mcWrapCat("MCInst Wrapper Generator Options");

static cl::opt<bool>
    skipPseudos("amdgcn-mcwrap-skip-pseudos",
                cl::desc("Exclude `isPseudo` records when collecting operand "
                         "names for each family wrapper"),
                cl::cat(mcWrapCat), cl::init(false));

static cl::opt<bool> skipReals(
    "amdgcn-mcwrap-skip-reals",
    cl::desc("Exclude non-pseudo (`isPseudo == 0`, hardware-encodable) "
             "records when collecting operand names for each family "
             "wrapper"),
    cl::cat(mcWrapCat), cl::init(false));

static cl::opt<std::string> wrapperNamespace(
    "amdgcn-mcwrap-namespace",
    cl::desc("C++ namespace for generated wrapper classes (default: "
             "transpiler::amdgcn::mcwrap)"),
    cl::cat(mcWrapCat), cl::init("transpiler::amdgcn::mcwrap"));

//===----------------------------------------------------------------------===//
// Family table
//
// Each entry maps a TableGen boolean field on an `InstSI`-derived record to
// the matching `llvm::SIInstrFlags` enumerator and the wrapper class name we
// emit. Names match `SIInstrFormats.td` and `SIDefines.h` exactly.
//===----------------------------------------------------------------------===//

namespace {
struct Family {
  /// TableGen boolean field name on InstSI (e.g. "VOP2").
  StringRef tdField;
  /// llvm::SIInstrFlags enumerator name (e.g. "VOP2").
  StringRef siFlag;
  /// Wrapper class name to emit (e.g. "VOP2Wrapper").
  StringRef className;
};
} // namespace

// Order matters only for output stability; keep it grouped by category.
static constexpr Family kFamilies[] = {
    // Scalar ALU encoding formats.
    {"SOP1", "SOP1", "SOP1Wrapper"},
    {"SOP2", "SOP2", "SOP2Wrapper"},
    {"SOPC", "SOPC", "SOPCWrapper"},
    {"SOPK", "SOPK", "SOPKWrapper"},
    {"SOPP", "SOPP", "SOPPWrapper"},
    // Vector ALU encoding formats.
    {"VOP1", "VOP1", "VOP1Wrapper"},
    {"VOP2", "VOP2", "VOP2Wrapper"},
    {"VOPC", "VOPC", "VOPCWrapper"},
    {"VOP3", "VOP3", "VOP3Wrapper"},
    {"VOP3P", "VOP3P", "VOP3PWrapper"},
    {"VINTRP", "VINTRP", "VINTRPWrapper"},
    {"SDWA", "SDWA", "SDWAWrapper"},
    {"DPP", "DPP", "DPPWrapper"},
    {"TRANS", "TRANS", "TRANSWrapper"},
    {"VINTERP", "VINTERP", "VINTERPWrapper"},
    {"LDSDIR", "LDSDIR", "LDSDIRWrapper"},
    // Memory encoding formats.
    {"MUBUF", "MUBUF", "MUBUFWrapper"},
    {"MTBUF", "MTBUF", "MTBUFWrapper"},
    {"SMRD", "SMRD", "SMRDWrapper"},
    {"MIMG", "MIMG", "MIMGWrapper"},
    {"VIMAGE", "VIMAGE", "VIMAGEWrapper"},
    {"VSAMPLE", "VSAMPLE", "VSAMPLEWrapper"},
    {"EXP", "EXP", "EXPWrapper"},
    {"FLAT", "FLAT", "FLATWrapper"},
    {"DS", "DS", "DSWrapper"},
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
/// Per-operand bookkeeping inside a family.
///
/// Record names and operand names are stored as `StringRef` because they
/// alias storage owned by the `RecordKeeper`, which outlives this generator.
struct OperandInfo {
  /// True iff at least one record (anywhere in AMDGPU.td, not just the ones
  /// matching the user's pseudo/real filter) carries this operand name and
  /// sets `UseNamedOperandTable = 1`. When true, the operand name *is*
  /// available as a `::llvm::AMDGPU::OpName::<name>` enumerator and the
  /// generated accessor can call `AMDGPU::getNamedOperandIdx` for it.
  bool inLLVMTable = false;
  /// `(opcode-record-name, MCInst-operand-index)` pairs for records that
  /// carry this operand but do *not* set `UseNamedOperandTable = 1`. These
  /// are the records that drop out of LLVM's named-operand lookup table and
  /// that we therefore need to handle through a positional `switch` in the
  /// generated accessor. Sorted by record name for deterministic output.
  SmallVector<std::pair<StringRef, unsigned>, 2> fallbacks;
};

/// Aggregated information for one family.
struct FamilyInfo {
  /// Per-operand-name info. `StringMap` is hash-based; emit code sorts the
  /// keys before iterating so output remains deterministic.
  StringMap<OperandInfo> operands;
  /// Number of records contributing to this family (for the comment).
  unsigned numContributingRecords = 0;
};

/// One named operand within a record, paired with its position in the
/// flattened `(outs..., ins...)` dag. The position is the operand's index
/// inside `MCInst::getOperand()`.
struct NamedArgPos {
  StringRef name;
  unsigned position;
};
} // namespace

/// Read a boolean InstSI field; returns false if the record does not define
/// the field at all (i.e. is not an InstSI subclass).
static bool readBitOrFalse(const Record *rec, StringRef field) {
  const RecordVal *rv = rec->getValue(field);
  if (!rv)
    return false;
  if (auto *bi = dyn_cast_or_null<BitInit>(rv->getValue()))
    return bi->getValue();
  return false;
}

/// Returns true if the given record looks like an instruction we should
/// inspect: subclass of `Instruction` and of `InstSI` with the operand-list
/// dags. We deliberately do *not* filter on `UseNamedOperandTable` here;
/// records that do not set that bit are picked up so that their operands
/// can be served via the per-opcode positional fallback path.
static bool isCandidateInstruction(const Record *rec, const Record *instClass,
                                   const Record *instSIClass) {
  if (!rec->isSubClassOf(instClass))
    return false;
  if (instSIClass && !rec->isSubClassOf(instSIClass))
    return false;
  if (!rec->getValue("InOperandList") || !rec->getValue("OutOperandList"))
    return false;
  return true;
}

/// Walk the `OutOperandList` and `InOperandList` dags of `rec`, returning
/// the named arguments together with their MCInst operand index. The index
/// is the position inside the flattened `(outs..., ins...)` arg sequence
/// (anonymous args still occupy a slot but are skipped from the result).
static SmallVector<NamedArgPos, 8> collectNamedArgs(const Record *rec) {
  SmallVector<NamedArgPos, 8> result;
  unsigned pos = 0;
  auto walk = [&](StringRef field) {
    const RecordVal *rv = rec->getValue(field);
    if (!rv)
      return;
    const auto *dag = dyn_cast_or_null<DagInit>(rv->getValue());
    if (!dag)
      return;
    for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
      StringRef name = dag->getArgNameStr(i);
      unsigned thisPos = pos++;
      if (name.empty())
        continue;
      result.push_back({name, thisPos});
    }
  };
  walk("OutOperandList");
  walk("InOperandList");
  return result;
}

/// Return a valid C++ identifier suitable as a method name. AMDGPU operand
/// names already are lowercase identifiers, but we keep this defensive in
/// case future records introduce something exotic.
static SmallString<32> sanitizeIdent(StringRef name) {
  SmallString<32> out;
  if (!name.empty() && std::isdigit(static_cast<unsigned char>(name.front())))
    out.push_back('_');
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Code emission
//===----------------------------------------------------------------------===//

/// Emit the file header, includes, and the shared `InstWrapper` base class.
static void emitPrologue(raw_ostream &os, ArrayRef<StringRef> namespaces) {
  os << "//===- AMDGCN MCInst family wrappers (auto-generated) -*- C++ "
        "-*-===//\n"
     << "//\n"
     << "// Generated by `amdgcn-tblgen --gen-amdgcn-mcinst-wrappers`.\n"
     << "// DO NOT EDIT.\n"
     << "//\n"
     << "//===---------------------------------------------------------------"
        "------===//\n\n"
     << "#ifndef AMDGCN_GEN_MCINST_WRAPPERS\n"
     << "#define AMDGCN_GEN_MCINST_WRAPPERS\n\n"
     << "// The following includes pull in the LLVM-side helpers we depend "
        "on:\n"
     << "//   * `AMDGPU::OpName` enum and `AMDGPU::getNamedOperandIdx`,\n"
     << "//     emitted by the AMDGPU backend tablegen because every "
        "instruction\n"
     << "//     class sets `UseNamedOperandTable = 1`,\n"
     << "//   * the AMDGPU opcode enum (e.g. `AMDGPU::V_ADD_F32_e32_gfx11`),\n"
     << "//     used by the per-opcode positional fallback `switch` we emit\n"
     << "//     for operands that are not in `AMDGPU::OpName`,\n"
     << "//   * `SIInstrFlags::*` constants for fast family checks,\n"
     << "//   * the standard `llvm::MCInst` / `llvm::MCInstrInfo` API.\n"
     << "#include \"MCTargetDesc/AMDGPUMCTargetDesc.h\"\n"
     << "#include \"SIDefines.h\"\n"
     << "#include \"Utils/AMDGPUBaseInfo.h\"\n"
     << "#include \"llvm/MC/MCInst.h\"\n"
     << "#include \"llvm/MC/MCInstrDesc.h\"\n"
     << "#include \"llvm/MC/MCInstrInfo.h\"\n"
     << "#include \"llvm/Support/Casting.h\"\n\n";

  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";
  os << "\n";

  os << R"cpp(/// Lightweight base wrapper around a `llvm::MCInst` for a disassembled AMDGCN
/// instruction. Concrete subclasses (one per encoding family, e.g.
/// `VOP2Wrapper`, `SOPPWrapper`) add typed accessors and a `classof` that
/// inspects the AMDGPU `TSFlags`.
///
/// Usage:
/// \code
///   InstWrapper base(mcInst, mcInstrInfo);
///   if (auto *vop2 = llvm::dyn_cast<VOP2Wrapper>(&base)) {
///     const llvm::MCOperand *dst  = vop2->vdst();
///     const llvm::MCOperand *src0 = vop2->src0();
///     ...
///   }
/// \endcode
///
/// The wrapper is non-owning: callers must keep the underlying `MCInst` and
/// `MCInstrInfo` alive for the duration of any access.
class InstWrapper {
public:
  InstWrapper(const ::llvm::MCInst &inst, const ::llvm::MCInstrInfo &mcii)
      : inst(&inst), mcii(&mcii) {}

  /// Returns the underlying `MCInst`.
  const ::llvm::MCInst &getMCInst() const { return *inst; }

  /// Returns the LLVM AMDGPU opcode of the wrapped instruction.
  unsigned getOpcode() const { return inst->getOpcode(); }

  /// Returns the AMDGPU `TSFlags` for the wrapped opcode.
  uint64_t getTSFlags() const {
    return mcii->get(inst->getOpcode()).TSFlags;
  }

  /// Returns the `MCInstrInfo` used for descriptor lookups.
  const ::llvm::MCInstrInfo &getMCInstrInfo() const { return *mcii; }

  /// Look up an operand by its AMDGPU `OpName`; returns `nullptr` if the
  /// concrete opcode does not have that operand or if the named index is
  /// past the MCInst's operand count. Public so cross-family callers
  /// (e.g. anything that reads `cpol` regardless of MUBUF/FLAT/MTBUF/...
  /// dispatch) can use it without first `dyn_cast`-ing to a specific
  /// family wrapper.
  const ::llvm::MCOperand *
  getNamedOperand(::llvm::AMDGPU::OpName name) const {
    int idx = ::llvm::AMDGPU::getNamedOperandIdx(inst->getOpcode(), name);
    if (idx < 0 || static_cast<unsigned>(idx) >= inst->getNumOperands())
      return nullptr;
    return &inst->getOperand(static_cast<unsigned>(idx));
  }

  /// Look up the MCInst-operand index of an AMDGPU `OpName`, or -1 if
  /// the opcode does not expose that operand. Provided alongside
  /// `getNamedOperand` for callers (e.g. raiser handlers that drive
  /// `RaiseContext::readOp32` / `readOp64`) that need the index, not the
  /// `MCOperand`. Bounds against `getNumOperands()` are NOT applied
  /// here — that is the caller's responsibility, mirroring the raw
  /// `AMDGPU::getNamedOperandIdx` return.
  int getNamedOperandIdx(::llvm::AMDGPU::OpName name) const {
    return ::llvm::AMDGPU::getNamedOperandIdx(inst->getOpcode(), name);
  }

private:
  const ::llvm::MCInst *inst;
  const ::llvm::MCInstrInfo *mcii;
};

)cpp";
}

/// Emit one family wrapper class.
static void emitFamilyClass(raw_ostream &os, const Family &fam,
                            const FamilyInfo &info) {
  os << "//===-- " << fam.className
     << " ---------------------------------------------===//\n";
  os << "// `" << fam.tdField << "` family. Aggregated from "
     << info.numContributingRecords << " AMDGPU TableGen record"
     << (info.numContributingRecords == 1 ? "" : "s")
     << ".\n//===-----------------------------------------------------------"
        "----------===//\n";

  os << "class " << fam.className << " : public InstWrapper {\n"
     << "public:\n"
     << "  using InstWrapper::InstWrapper;\n\n"
     << "  /// LLVM-style RTTI hook: matches any opcode whose `TSFlags` has "
        "the\n"
     << "  /// `" << fam.siFlag << "` bit set.\n"
     << "  static bool classof(const InstWrapper *w) {\n"
     << "    return (w->getTSFlags() & ::llvm::SIInstrFlags::" << fam.siFlag
     << ") != 0;\n"
     << "  }\n";

  if (info.operands.empty()) {
    os << "\n  // No named operands observed in any " << fam.tdField
       << " record.\n";
  } else {
    os << "\n";
    // `StringMap` iterates in hash order; sort keys for stable output.
    SmallVector<StringRef, 16> sortedNames;
    sortedNames.reserve(info.operands.size());
    for (const auto &kv : info.operands)
      sortedNames.push_back(kv.first());
    llvm::sort(sortedNames);

    for (StringRef name : sortedNames) {
      const OperandInfo &oi = info.operands.find(name)->second;
      SmallString<32> ident = sanitizeIdent(name);
      os << "  /// Returns the `" << name
         << "` operand or `nullptr` if absent for this opcode.\n"
         << "  const ::llvm::MCOperand *" << ident << "() const {\n";

      // Fast path through LLVM's AMDGPU::OpName lookup table.
      if (oi.inLLVMTable) {
        os << "    if (auto *op = getNamedOperand(::llvm::AMDGPU::OpName::"
           << name << "))\n"
           << "      return op;\n";
      }

      // Positional fallback for opcodes whose record does not set
      // `UseNamedOperandTable = 1` and that therefore are absent from
      // LLVM's lookup table.
      if (!oi.fallbacks.empty()) {
        if (oi.inLLVMTable) {
          os << "    // Fallback for opcodes whose record omits "
                "`UseNamedOperandTable = 1`\n"
                "    // and is therefore not present in "
                "`AMDGPU::getNamedOperandIdx`'s table.\n";
        } else {
          os << "    // The `" << name
             << "` operand name is not in `::llvm::AMDGPU::OpName` because no\n"
                "    // record carrying it sets `UseNamedOperandTable = 1`. "
                "Fall back to a per-opcode\n"
                "    // positional table computed from the flattened "
                "`(outs..., ins...)` dag.\n";
        }
        // Group by MCInst index so opcodes that share the same operand
        // position collapse into a single fall-through body. There are
        // typically only a handful of distinct positions per operand, so
        // the inline-storage hint of 4 covers the common case.
        SmallDenseMap<unsigned, SmallVector<StringRef, 8>, 4> byPosition;
        for (const auto &[opcode, idx] : oi.fallbacks)
          byPosition[idx].push_back(opcode);

        // Sort positions ascending so case bodies appear in increasing
        // operand-index order.
        SmallVector<unsigned, 4> positions;
        positions.reserve(byPosition.size());
        for (const auto &kv : byPosition)
          positions.push_back(kv.first);
        llvm::sort(positions);

        os << "    switch (getOpcode()) {\n";
        for (unsigned idx : positions) {
          // Opcodes within a position bucket are inserted in `oi.fallbacks`
          // order (already sorted alphabetically) but re-sort defensively
          // to guarantee stability if `fallbacks` grows out-of-order keys.
          SmallVectorImpl<StringRef> &opcodes = byPosition[idx];
          llvm::sort(opcodes);
          for (StringRef opcode : opcodes)
            os << "    case ::llvm::AMDGPU::" << opcode << ":\n";
          os << "      if (" << idx << "u < getMCInst().getNumOperands())\n"
             << "        return &getMCInst().getOperand(" << idx << "u);\n"
             << "      break;\n";
        }
        os << "    default:\n"
           << "      break;\n"
           << "    }\n";
      }

      os << "    return nullptr;\n"
         << "  }\n";
    }
  }
  os << "};\n\n";
}

/// Emit closing namespaces and include guard.
static void emitEpilogue(raw_ostream &os, ArrayRef<StringRef> namespaces) {
  for (auto it = namespaces.rbegin(), e = namespaces.rend(); it != e; ++it)
    os << "} // namespace " << *it << "\n";
  os << "\n#endif // AMDGCN_GEN_MCINST_WRAPPERS\n";
}

//===----------------------------------------------------------------------===//
// Main generator
//===----------------------------------------------------------------------===//

static void generateMCInstWrappers(const RecordKeeper &records,
                                   raw_ostream &os) {
  const Record *instClass = records.getClass("Instruction");
  if (!instClass)
    PrintFatalError("could not find the `Instruction` TableGen class");
  // `InstSI` carries the family bits. We do not hard-fail if it is missing
  // so the tool can still be exercised against non-AMDGPU targets, but in
  // practice every AMDGPU run will have it.
  const Record *instSIClass = records.getClass("InstSI");

  // Pass 1: discover which operand names are exposed via LLVM's
  // `AMDGPU::OpName` enum. A name is in that enum iff at least one record
  // has it *and* sets `UseNamedOperandTable = 1`. We deliberately ignore
  // the user's pseudo/real filter here: `AMDGPU::OpName` is generated from
  // the entire AMDGPU.td universe, and what we care about is whether the
  // C++ symbol exists, not whether the contributing record survives our
  // own filter.
  StringSet<> llvmKnownNames;
  for (const auto &[name, def] : records.getDefs()) {
    const Record *rec = def.get();
    if (!isCandidateInstruction(rec, instClass, instSIClass))
      continue;
    if (!readBitOrFalse(rec, "UseNamedOperandTable"))
      continue;
    for (const NamedArgPos &arg : collectNamedArgs(rec))
      llvmKnownNames.insert(arg.name);
  }

  // Pass 2: walk every instruction record, classify it, and accumulate
  // per-family operand info. Records that do not appear in LLVM's named
  // table contribute their `(opcode, MCInst position)` to the per-operand
  // fallback list so the generated accessor can still serve them via a
  // positional `switch`.
  //
  // `infos` is keyed by `tdField` (a `StringRef` into the immutable
  // `kFamilies` table); we always iterate `kFamilies` in declaration order
  // for emission, so the unordered nature of `DenseMap` is fine here.
  DenseMap<StringRef, FamilyInfo> infos;
  for (const Family &f : kFamilies)
    (void)infos[f.tdField];

  for (const auto &[name, def] : records.getDefs()) {
    const Record *rec = def.get();
    if (!isCandidateInstruction(rec, instClass, instSIClass))
      continue;

    bool isPseudo = readBitOrFalse(rec, "isPseudo");
    if (isPseudo && skipPseudos)
      continue;
    if (!isPseudo && skipReals)
      continue;

    bool recIsNamed = readBitOrFalse(rec, "UseNamedOperandTable");
    SmallVector<NamedArgPos, 8> args = collectNamedArgs(rec);
    StringRef recName = rec->getName();

    for (const Family &f : kFamilies) {
      if (!readBitOrFalse(rec, f.tdField))
        continue;
      FamilyInfo &info = infos[f.tdField];
      ++info.numContributingRecords;
      for (const NamedArgPos &arg : args) {
        OperandInfo &oi = info.operands[arg.name];
        if (!recIsNamed)
          oi.fallbacks.emplace_back(recName, arg.position);
      }
    }
  }

  // Resolve per-operand `inLLVMTable` flags and dedupe fallback lists.
  // (Different family bits on the same record can map a single `(record,
  // position)` pair to the same per-family operand, which would create
  // duplicate switch cases without this pass.)
  for (auto &kv : infos) {
    FamilyInfo &info = kv.second;
    for (auto &opKV : info.operands) {
      OperandInfo &oi = opKV.second;
      oi.inLLVMTable = llvmKnownNames.contains(opKV.first());
      llvm::sort(oi.fallbacks);
      oi.fallbacks.erase(std::unique(oi.fallbacks.begin(), oi.fallbacks.end()),
                         oi.fallbacks.end());
    }
  }

  // Split the configurable `transpiler::amdgcn::mcwrap` namespace into pieces.
  SmallVector<StringRef, 4> nsParts;
  StringRef nsRef = wrapperNamespace.getValue();
  while (!nsRef.empty()) {
    auto [head, tail] = nsRef.split("::");
    if (!head.empty())
      nsParts.push_back(head);
    nsRef = tail;
  }

  emitPrologue(os, nsParts);
  for (const Family &f : kFamilies) {
    auto it = infos.find(f.tdField);
    if (it == infos.end())
      continue;
    emitFamilyClass(os, f, it->second);
  }
  emitEpilogue(os, nsParts);
}

//===----------------------------------------------------------------------===//
// TableGen registration
//===----------------------------------------------------------------------===//

static TableGen::Emitter::Opt generateMCInstWrappersReg(
    "gen-amdgcn-mcinst-wrappers", generateMCInstWrappers,
    "Generate C++ wrapper classes (one per AMDGCN encoding family) around "
    "llvm::MCInst");
