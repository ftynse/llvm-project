#include "opcode_map.hpp"

#include <cstdint>
#include <optional>
#include <string>

// AMDGPU target-private headers. They expose:
//   AMDGPU::getMCOpcode           (declared in Utils/AMDGPUBaseInfo.h)
//   AMDGPU::getVOPe64 / getVOPe32 / getDPPOp32 / getDPPOp64 /
//   getSDWAOp / getBasicFromSDWAOp / getGlobalVaddrOp
//                                  (declared in SIInstrInfo.h, implemented
//                                   in the TableGen-generated
//                                   AMDGPUGenInstrInfo.inc under
//                                   `#define GET_INSTRMAP_INFO`, linked from
//                                   libLLVMAMDGPUUtils.a).
//
// SIInstrInfo.h drags in the CodeGen TargetInstrInfo base, which we do not
// use at runtime, but pulling it in is preferable to hand-rolling forward
// declarations that would silently go stale if LLVM changes a signature.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Canonical AMDGPU pseudo -> Salmon SemOp table
// (`transpiler::amdgcn::canon::{Entry, kEntries}`).
//
// `Entry`/`kEntries` come from `td/SalmonAMDGCN.td` via
// `salmon-tblgen --gen-amdgcn-canon-table`. The .td file owns every
// editorial decision (renames, semantic collapses, family expansions);
// the generator walks `SalmonOp` records and emits this table while
// preserving source order so consumers that build a `DenseMap` via
// `try_emplace` get "first-binding-wins" semantics.
//
// `SemOp` enumerator drift fails this translation unit's C++ compile;
// missing AMDGPU `Instruction` records fail at TableGen time with a
// source location.
//
// The include sits outside `namespace transpiler` because the generated
// header opens that namespace itself.
#include "salmon/amdgcn_canon_table.h.inc"

namespace transpiler {

namespace {

// Iteration bound for SIEncodingFamily: the enum in SIDefines.h is a closed
// numeric set with GFX13 as the current maximum, so we scan [0, GFX13] when
// inverting the pseudo -> MC map.  If LLVM adds a new family the next
// enumerator value appears here automatically and the build still compiles;
// the static_assert keeps us honest if LLVM ever renames the sentinel we use.
static_assert(SIEncodingFamily::GFX13 >= SIEncodingFamily::SI,
              "SIEncodingFamily enum layout changed unexpectedly");
constexpr unsigned kNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Build a reverse map MC-opcode -> canonical pseudo by scanning every pseudo
// opcode across all subtarget generations. This is ~O(N * 15) work at init
// time (N ~= 70k AMDGPU opcodes on recent LLVM), which is well under a
// millisecond on modern hardware and done once per raiser.
DenseMap<unsigned, unsigned>
buildMcToPseudoMap(unsigned numOpc) {
  DenseMap<unsigned, unsigned> result;
  for (unsigned p = 0; p < numOpc; ++p) {
    for (unsigned gen = 0; gen < kNumEncodingFamilies; ++gen) {
      int mc = AMDGPU::getMCOpcode(p, gen);
      if (mc > 0 && static_cast<unsigned>(mc) != p)
        result.try_emplace(static_cast<unsigned>(mc), p);
    }
  }
  return result;
}

// Rule predicates: an optional semantic invariant the alias must preserve.
// Every time an alias step is committed (source pseudo S collapses onto
// target pseudo T), the firing rule's predicate is evaluated against
// MCInstrDesc(S) and MCInstrDesc(T). A violation means LLVM renamed or
// repurposed a pseudo in a way that breaks our naming contract, and is
// reported as a fatal error at init time rather than silently producing
// wrong IR at runtime.
using RulePredicate = bool (*)(const MCInstrDesc &src, const MCInstrDesc &tgt);

// `_RTN` collapse: source must be an atomic with a return value; target must
// be the same atomic without one. The raiser uses `numDefs` as the
// "publishes old value" signal, so that invariant must also hold.
static bool atomicRetToNoRet(const MCInstrDesc &src, const MCInstrDesc &tgt) {
  constexpr uint64_t kRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t kNoRet = SIInstrFlags::IsAtomicNoRet;
  return (src.TSFlags & kRet) && (tgt.TSFlags & kNoRet) &&
         src.getNumDefs() > 0 && tgt.getNumDefs() == 0;
}

// `_vgprcd_` / `_mac_` collapse: both source and target must be MFMA
// (matrix-accumulate) pseudos.
static bool bothAreMAI(const MCInstrDesc &src, const MCInstrDesc &tgt) {
  constexpr uint64_t kMAI = SIInstrFlags::IsMAI;
  return (src.TSFlags & kMAI) && (tgt.TSFlags & kMAI);
}

// `_nosdst_` collapse: starting with GFX11, VOPC CMPX instructions no longer
// write a scalar destination register (EXEC receives the mask directly) and
// LLVM represents this as a `_nosdst_` variant. The non-`_nosdst_` target
// form keeps the scalar dst (for older subtargets). Both forms share
// dispatch-relevant TSFlags; the raiser's CMPX handler only writes EXEC and
// ignores the optional sdst, so collapsing the variant onto the base is
// safe. The source has one fewer def when the base includes sdst (e64
// forms) and the same number of defs otherwise (e32, where both lack sdst).

// Bits we require to be identical between source and target for an alias
// collapse to be considered semantically safe. Deliberately excludes encoding
// variation flags like `VOP3_OPSEL` (set on `_t16_` op-sel encodings but not
// on the base `_e64`) and `renamedInGFX9` (set only on the subtarget-specific
// pseudo). Everything listed below represents *what the handler dispatches
// on*: instruction family (SOP/VOP/FLAT/DS/...), atomic kind, and MAI.
static constexpr uint64_t kSemanticShapeMask =
    // Instruction families.
    SIInstrFlags::SOP1 | SIInstrFlags::SOP2 | SIInstrFlags::SOPC |
    SIInstrFlags::SOPK | SIInstrFlags::SOPP | SIInstrFlags::VOP1 |
    SIInstrFlags::VOP2 | SIInstrFlags::VOPC | SIInstrFlags::VOP3 |
    SIInstrFlags::VOP3P | SIInstrFlags::SDWA | SIInstrFlags::DPP |
    SIInstrFlags::MUBUF | SIInstrFlags::MTBUF | SIInstrFlags::SMRD |
    SIInstrFlags::FLAT | SIInstrFlags::DS | SIInstrFlags::MIMG |
    // Semantic classification (atomic kind, MAI).
    SIInstrFlags::IsAtomicRet | SIInstrFlags::IsAtomicNoRet |
    SIInstrFlags::IsMAI;

// Subtarget-/operand-class variants (`_gfx9`, `_t16_`, `_fake16_`, `_agpr`,
// etc.) may legitimately toggle encoding flags such as `VOP3_OPSEL` or
// `renamedInGFX9` between source and target, but they must preserve the
// instruction's dispatch identity: same family, same atomic kind, same MAI
// classification, same def arity. A violation means LLVM renamed or
// repurposed a pseudo in a way our alias map cannot safely collapse.
static bool sameSemanticShape(const MCInstrDesc &src,
                              const MCInstrDesc &tgt) {
  return (src.TSFlags & kSemanticShapeMask) ==
             (tgt.TSFlags & kSemanticShapeMask) &&
         src.getNumDefs() == tgt.getNumDefs();
}

// `_nosdst_` collapse: same dispatch identity as the base, and the source
// never has more defs than the target (the scalar dst is either dropped
// entirely or added back on the target's e64 form).
static bool nosdstDropsScalarDef(const MCInstrDesc &src,
                                 const MCInstrDesc &tgt) {
  return (src.TSFlags & kSemanticShapeMask) ==
             (tgt.TSFlags & kSemanticShapeMask) &&
         tgt.getNumDefs() >= src.getNumDefs() &&
         tgt.getNumDefs() - src.getNumDefs() <= 1;
}

// Build an alias map that collapses "parallel" pseudos LLVM generates for the
// same semantic instruction into a single canonical pseudo. Examples:
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// LLVM does not expose a helper for this collapse, so we match on pseudo name
// at init time. Name lookups are confined to this one-shot scan over
// `MCII.getNumOpcodes()`; runtime lookups remain pure DenseMap hits.
DenseMap<unsigned, unsigned>
buildPseudoAliasMap(const MCInstrInfo &MCII) {
  unsigned numOpc = MCII.getNumOpcodes();

  llvm::StringMap<unsigned> byName;
  for (unsigned p = 0; p < numOpc; ++p)
    byName.try_emplace(MCII.getName(p), p);

  struct Rule {
    llvm::StringRef needle;
    bool isSuffix;
    // Optional semantic check on (source, target) MCInstrDesc. A null
    // predicate means "no validation yet" (see the older subtarget/operand
    // markers below).
    RulePredicate pred;
  };
  // Subtarget-specific markers ("_gfx9", "_gfx1250", ...) and operand-size
  // markers ("_t16_", "_fake16_") that LLVM injects into the pseudo name.
  // A single pseudo can carry multiple markers (e.g.
  // `V_BITOP3_B16_gfx1250_fake16_e64`), so the outer loop below applies these
  // rules iteratively until the name stops shrinking.
  static const Rule rules[] = {
      // Subtarget-specific markers. LLVM emits a dedicated pseudo per
      // subtarget (e.g. `_gfx9`, `_gfx1250`, `_vi_gfx9`) with the same
      // TableGen class as the base; collapsing them is sound as long as
      // TSFlags and def arity match.
      {"_vi_gfx9", true, sameSemanticShape},
      {"_gfx9", true, sameSemanticShape},
      {"_gfx1250", true, sameSemanticShape},
      {"_gfx1250_", false, sameSemanticShape},
      {"_pseudo_", false, sameSemanticShape},
      // True16 / Fake16 mark the 16-bit operand encoding variant; LLVM has
      // no dedicated TSFlag bit for this (the distinction lives in
      // True16Predicate on the TableGen side), and the t16 encoding toggles
      // `VOP3_OPSEL`. We cross-check that dispatch-relevant TSFlags and def
      // arity are preserved, but tolerate encoding-bit drift.
      {"_t16_", false, sameSemanticShape},
      {"_fake16_", false, sameSemanticShape},
      // `_OP_SEL_` infix marks the gfx11+ encoding variant for VOP1 cvt
      // instructions (e.g. v_cvt_f32_{fp8,bf8}, v_cvt_pk_f32_{fp8,bf8}).
      // The OP_SEL pseudo carries an extra `byte_sel` immediate operand and
      // toggles `VOP3_OPSEL`/`maybeAtomic`/`ASYNC_CNT` bits relative to the
      // base e64 pseudo, but dispatch identity (instruction family + def
      // arity + atomic/MAI classification) is preserved. Collapsing onto
      // the base pseudo lets a single SemOp handler service both the
      // pre-gfx11 SDWA/byte_sel-via-disassembly form and the gfx11+ encoded
      // byte_sel form; the handler reads the byte_sel from the disassembly
      // text (`op_sel:`), which is identical for both.
      {"_OP_SEL_", false, sameSemanticShape},
      // GFX11+ VOPC CMPX family drops the scalar destination register; the
      // raiser's CMPX handler only touches EXEC so the `_nosdst_` form
      // collapses cleanly onto the base pseudo of the same encoding width.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA register-class modifiers.  `_vgprcd_` marks a VGPR destination
      // variant; `_mac_` marks a multiply-accumulate (tied dst/src2) variant.
      // Both keep the same TableGen intrinsic and semantic shape, so they
      // collapse onto the base `_e64` pseudo.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // Atomic return-value variants: LLVM emits distinct `_RTN` pseudos for
      // the forms that return the pre-modification value, plus `_agpr`
      // variants that just pick an AGPR destination register class. These
      // pseudos carry the same TableGen intrinsic and identical semantics;
      // the only difference is whether the handler should write the result
      // back, which the raiser already derives from `di.numDefs`
      // (MCInstrDesc::getNumDefs()). Collapse them onto the non-RTN pseudo
      // so both forms share a single SemOp.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule or -1 if no rule applies.
  auto stripOnce = [&](llvm::StringRef name, std::string &out) -> int {
    for (size_t i = 0; i < std::size(rules); ++i) {
      const Rule &r = rules[i];
      if (r.isSuffix) {
        if (!name.ends_with(r.needle))
          continue;
        out = name.drop_back(r.needle.size()).str();
        return static_cast<int>(i);
      }
      size_t pos = name.find(r.needle);
      if (pos == llvm::StringRef::npos)
        continue;
      out = (name.substr(0, pos).str() + std::string("_") +
             name.substr(pos + r.needle.size()).str());
      return static_cast<int>(i);
    }
    return -1;
  };

  DenseMap<unsigned, unsigned> alias;
  for (const auto &kv : byName) {
    std::string cur = kv.first().str();
    unsigned curOpc = kv.second;
    unsigned finalOpc = kv.second;
    while (true) {
      std::string next;
      int ruleIdx = stripOnce(cur, next);
      if (ruleIdx < 0)
        break;
      auto it = byName.find(next);
      if (it != byName.end() && it->second != kv.second) {
        const Rule &r = rules[ruleIdx];
        if (r.pred && !r.pred(MCII.get(curOpc), MCII.get(it->second))) {
          report_fatal_error(
              Twine("opcode_map: alias rule '") + r.needle +
              "' broke its semantic invariant while collapsing '" + cur +
              "' -> '" + next +
              "'. LLVM likely renamed or repurposed a pseudo; update the "
              "alias rules or the predicate.");
        }
        curOpc = it->second;
        finalOpc = curOpc;
      }
      cur = std::move(next);
    }
    if (finalOpc != kv.second)
      alias.try_emplace(kv.second, finalOpc);
  }
  return alias;
}

// Build a reverse DPP map: DPP opcode -> base VOP opcode. LLVM only provides
// forward mappings (base -> DPP32 / DPP64), so we invert by scanning.
DenseMap<unsigned, unsigned>
buildDppToBaseMap(unsigned numOpc) {
  DenseMap<unsigned, unsigned> result;
  for (unsigned p = 0; p < numOpc; ++p) {
    int d32 = AMDGPU::getDPPOp32(p);
    if (d32 > 0)
      result.try_emplace(static_cast<unsigned>(d32), p);
    int d64 = AMDGPU::getDPPOp64(p);
    if (d64 > 0)
      result.try_emplace(static_cast<unsigned>(d64), p);
  }
  return result;
}

// Canonicalize any MC opcode `mc` to the pseudo form matched in kCanonTable.
// The chain is:
//   MC -> pseudo                (TableGen Subtarget map)
//   pseudo -> base VOP          (strip DPP / SDWA)
//   e32 -> e64                  (collapse VOP encoding variants)
//   SADDR -> VADDR              (FLAT/GLOBAL global-saddr table)
unsigned canonicalize(unsigned mc,
                      const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &mcToPseudo,
                      const DenseMap<unsigned, unsigned> &pseudoAlias,
                      const DenseMap<unsigned, unsigned> &dppToBase) {
  unsigned p = mc;

  // MC (subtarget-specific real) -> pseudo.
  if (auto it = mcToPseudo.find(p); it != mcToPseudo.end())
    p = it->second;

  // Parallel-pseudo alias -> base pseudo (strips _gfx9, _t16_, _fake16_).
  if (auto it = pseudoAlias.find(p); it != pseudoAlias.end())
    p = it->second;

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both getDPPOp32 and
  // getDPPOp64.
  if (auto it = dppToBase.find(p); it != dppToBase.end())
    p = it->second;

  // SDWA -> base. LLVM provides a forward helper for this direction.
  int base = AMDGPU::getBasicFromSDWAOp(p);
  if (base > 0)
    p = static_cast<unsigned>(base);

  // e32 -> e64.
  int e64 = AMDGPU::getVOPe64(p);
  if (e64 > 0)
    p = static_cast<unsigned>(e64);

  // Re-apply the pseudo-alias step. `getVOPe64` can resolve an `_e32` pseudo
  // (e.g. `V_LSHLREV_B64_pseudo_e32`) to an `_e64` pseudo with a parallel
  // variant marker (`V_LSHLREV_B64_pseudo_e64`) that only collapses once
  // both the `_pseudo_` infix and `_e64` suffix are visible together.
  if (auto it = pseudoAlias.find(p); it != pseudoAlias.end())
    p = it->second;

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag; the helper returns -1 for non-FLAT opcodes but
  // checking the flag first avoids the lookup for every non-FLAT opcode.
  if (p < MCII.getNumOpcodes() &&
      (MCII.get(p).TSFlags & SIInstrFlags::FLAT) != 0) {
    int vaddr = AMDGPU::getGlobalVaddrOp(p);
    if (vaddr > 0)
      p = static_cast<unsigned>(vaddr);
  }

  return p;
}

// Parse a canonical vector-compare pseudo name into (predicate, bits, kind).
// Accepted shape: `V_CMP_<PRED>_<TYPE><BITS>_e64` where
//   PRED  ∈ {EQ, NE, GT, GE, LT, LE, LG, NEQ, NLT, NLE, NGT, NGE, NLG, U, O,
//            CLASS}
//   TYPE  ∈ {U, I, F} (CLASS only ever appears with TYPE=F)
//   BITS  ∈ {16, 32, 64}
// and an optional `V_CMPX_` prefix plays the role of `V_CMP_`. Returns
// `std::nullopt` for anything else; caller is responsible for only passing
// compare-family pseudos.
//
// Rationale: LLVM exposes `AMDGPU::getVCMPXOpFromVCMP` as a V_CMP → V_CMPX
// mapping, but no public helper that hands back a CmpInst::Predicate or
// element width. Rather than hand-list 100 opcode→metadata pairs we parse
// the pseudo name once at init time; the same token grammar is already
// hard-coded in LLVM's TableGen for these instructions.
//
// CLASS is special: `V_CMP_CLASS_F<bits>` is *not* a predicate compare. src1
// is an i32 mask of FP classes (signaling NaN, quiet NaN, ±inf, ±normal,
// ±subnormal, ±0), and the result lane bit is set iff src0's IEEE class
// matches any enabled bit in the mask. We collapse it onto the same
// `V_CMP` / `V_CMPX` SemOps and signal the special-case lift via
// `VCmpMeta::isClass`; the dispatch in handle_valu_vcmp.cpp branches on
// that flag and emits `llvm.amdgcn.class.f<bits>` instead of an FCmp.
// This keeps the parser surface narrow (one extra grammar branch, no new
// SemOps) and matches the wave-mask write-back path used by the other
// V_CMP forms.
std::optional<VCmpMeta> parseVCmpPseudoName(llvm::StringRef name) {
  llvm::StringRef rest = name;
  if (!rest.consume_front("V_CMPX_") && !rest.consume_front("V_CMP_"))
    return std::nullopt;
  if (!rest.consume_back("_e64"))
    return std::nullopt;

  auto [predTok, typeTok] = rest.rsplit('_');
  if (predTok.empty() || typeTok.size() < 2)
    return std::nullopt;

  const char typeCh = typeTok[0];
  unsigned bits = 0;
  if (typeTok.drop_front().getAsInteger(10, bits))
    return std::nullopt;
  if (bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;

  using llvm::CmpInst;
  VCmpMeta m{CmpInst::BAD_ICMP_PREDICATE, static_cast<uint8_t>(bits), false};

  // V_CMP_CLASS_F<bits> / V_CMPX_CLASS_F<bits>: floating-point classification
  // mask, not a predicate compare. The handler takes the `isClass` branch and
  // ignores `pred`; we leave `pred` as BAD_ICMP_PREDICATE so any accidental
  // FCmp/ICmp use would assert loudly rather than silently miscompile.
  if (predTok == "CLASS") {
    if (typeCh != 'F')
      return std::nullopt;
    m.isFloat = true;
    m.isClass = true;
    return m;
  }

  if (typeCh == 'F') {
    m.isFloat = true;
    // Float predicates: ordered variants set the O-prefix predicates;
    // N-prefixed AMDGPU names select the "unordered-or-..." complements.
    if (predTok == "EQ")        m.pred = CmpInst::FCMP_OEQ;
    else if (predTok == "GT")   m.pred = CmpInst::FCMP_OGT;
    else if (predTok == "GE")   m.pred = CmpInst::FCMP_OGE;
    else if (predTok == "LT")   m.pred = CmpInst::FCMP_OLT;
    else if (predTok == "LE")   m.pred = CmpInst::FCMP_OLE;
    // LG ("less or greater"), NE, and NEQ all mean "ordered and !=" in
    // AMDGPU's model and all lower to FCMP_ONE.
    else if (predTok == "LG" || predTok == "NE" || predTok == "NEQ")
                                m.pred = CmpInst::FCMP_ONE;
    else if (predTok == "NLT")  m.pred = CmpInst::FCMP_UGE;
    else if (predTok == "NLE")  m.pred = CmpInst::FCMP_UGT;
    else if (predTok == "NGT")  m.pred = CmpInst::FCMP_ULE;
    else if (predTok == "NGE")  m.pred = CmpInst::FCMP_ULT;
    // NLG ("not (less or greater)") is the unordered-or-equal complement.
    else if (predTok == "NLG")  m.pred = CmpInst::FCMP_UEQ;
    else if (predTok == "U")    m.pred = CmpInst::FCMP_UNO;
    else if (predTok == "O")    m.pred = CmpInst::FCMP_ORD;
    else return std::nullopt;
  } else if (typeCh == 'U' || typeCh == 'I') {
    const bool isSigned = typeCh == 'I';
    if (predTok == "EQ")        m.pred = CmpInst::ICMP_EQ;
    else if (predTok == "NE")   m.pred = CmpInst::ICMP_NE;
    else if (predTok == "GT")   m.pred = isSigned ? CmpInst::ICMP_SGT
                                                   : CmpInst::ICMP_UGT;
    else if (predTok == "GE")   m.pred = isSigned ? CmpInst::ICMP_SGE
                                                   : CmpInst::ICMP_UGE;
    else if (predTok == "LT")   m.pred = isSigned ? CmpInst::ICMP_SLT
                                                   : CmpInst::ICMP_ULT;
    else if (predTok == "LE")   m.pred = isSigned ? CmpInst::ICMP_SLE
                                                   : CmpInst::ICMP_ULE;
    else return std::nullopt;
  } else {
    return std::nullopt;
  }

  return m;
}

} // namespace

SemOp OpcodeMap::lookup(unsigned opcode) const {
  auto it = map_.find(opcode);
  return it != map_.end() ? it->second : SemOp::Unknown;
}

const VCmpMeta *OpcodeMap::lookupVCmp(unsigned opcode) const {
  auto it = vcmp_.find(opcode);
  return it != vcmp_.end() ? &it->second : nullptr;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // Flatten the auto-generated canon table into a DenseMap for O(1) lookups
  // during the subsequent scan over every MC opcode. The generator emits
  // entries in source order, so `try_emplace` mirrors the historical
  // "first-binding-wins" semantics of the previous hand-rolled table.
  DenseMap<unsigned, SemOp> canonToSem;
  canonToSem.reserve(std::size(amdgcn::canon::kEntries));
  for (const amdgcn::canon::Entry &e : amdgcn::canon::kEntries)
    canonToSem.try_emplace(e.opc, e.sem);

  const unsigned numOpc = MCII.getNumOpcodes();
  const auto mcToPseudo  = buildMcToPseudoMap(numOpc);
  const auto pseudoAlias = buildPseudoAliasMap(MCII);
  const auto dppToBase   = buildDppToBaseMap(numOpc);

  map_.clear();
  vcmp_.clear();
  // Heuristic: roughly a quarter of MC opcodes carry a SemOp in practice;
  // resizing a few times is fine for a one-shot init.
  map_.reserve(numOpc / 4);
  for (unsigned mc = 0; mc < numOpc; ++mc) {
    const unsigned canon =
        canonicalize(mc, MCII, mcToPseudo, pseudoAlias, dppToBase);
    if (auto it = canonToSem.find(canon); it != canonToSem.end()) {
      map_[mc] = it->second;
      continue;
    }
    // The canonical pseudo was not enumerated in `kCanonTable`. Check if it
    // belongs to the V_CMP / V_CMPX family (which is handled via metadata
    // side-table rather than per-opcode enumeration). Use the canonical
    // pseudo's name so we don't have to re-canonicalize any DPP/SDWA
    // variants (those have already been folded by `canonicalize`).
    if (canon >= numOpc)
      continue;
    llvm::StringRef canonName = MCII.getName(canon);
    const bool isCmp  = canonName.starts_with("V_CMP_");
    const bool isCmpX = canonName.starts_with("V_CMPX_");
    if (!isCmp && !isCmpX)
      continue;
    if (auto meta = parseVCmpPseudoName(canonName)) {
      map_[mc] = isCmpX ? SemOp::V_CMPX : SemOp::V_CMP;
      vcmp_.try_emplace(mc, *meta);
    }
    // Names that start with V_CMP_ but don't parse (e.g. a hypothetical
    // future family) are left as SemOp::Unknown so the raiser reports them
    // loudly rather than silently producing wrong IR.
  }
}

} // namespace transpiler
