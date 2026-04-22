//===-- SalmonAMDGCN.td - Salmon-side AMDGCN annotations --*- tablegen -*-===//
//
// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Salmon-side TableGen layer that imports LLVM's AMDGPU target description
// and decorates a curated subset of `Instruction` records with Salmon's
// dispatch metadata.
//
// The single concept introduced here is `SalmonOp`, which ties an LLVM
// `Instruction` record (e.g. `V_ADD_F32_e64`) to Salmon's chosen `SemOp`
// tag (e.g. `"V_ADD_F32"`). The `salmon-tblgen --gen-amdgcn-canon-table`
// backend walks every `SalmonOp` record and emits the static
// `kCanonTable[]` array that `OpcodeMap::build` consumes.
//
// Compared with the previous hand-rolled C++ array this gives us:
//
//   * Type-checked family expansion via `multiclass`. `!cast<Instruction>`
//     fails at TableGen time with a record name and source location if a
//     base record vanishes upstream, instead of bottoming out in an
//     obscure C++ macro / enum error.
//   * A natural place to grow per-instruction Salmon metadata (handler
//     hints, gfx-rev gating, conformance tags) as additional fields on
//     `SalmonOp` rather than as parallel C++ tables.
//   * A single source of truth all `salmon-tblgen` backends can iterate.
//
// The data the tables carry is *still* 100% Salmon-curated — what's
// supported, the editorial renames (`S_LOAD_DWORD` -> `S_LOAD_B32`),
// the deliberate semantic collapses (`S_AND_SAVEEXEC_B64` ->
// `S_AND_SAVEEXEC_B32`). Nothing here is auto-derived from AMDGPU.td.
//
//===----------------------------------------------------------------------===//

// Pulls in the entire AMDGPU TableGen universe (`Instruction` records,
// `SIInstrFlags`, the per-subtarget reals, ...). The same `-I` paths
// that `salmon-tblgen` uses for the MCInst-wrapper backend resolve this
// include.
include "AMDGPU.td"

//===----------------------------------------------------------------------===//
// Salmon decorator class.
//===----------------------------------------------------------------------===//

/// `SalmonOp` ties an AMDGPU `Instruction` record to a Salmon `SemOp` tag.
///
/// `Sem` is a string rather than an enum because the C++ `SemOp` enum
/// lives outside TableGen; the generated `kCanonTable[]` emits
/// `SemOp::<Sem>`, so any drift between this file and `semop.hpp` fails
/// the C++ compile of `opcode_map.cpp` (missing enumerator) — same error
/// surface as today's `E(OP, SEM)` macro.
class SalmonOp<Instruction inst, string sem> {
  Instruction Inst = inst;
  string Sem = sem;
}

//===----------------------------------------------------------------------===//
// Family multiclasses.
//
// AMDGPU.td synthesises a small, fixed set of addressing-mode variants
// per memory base opcode. Salmon supports all of them with the same
// `SemOp`, so we expand each base name once into the matching set of
// `SalmonOp` records using `!cast<Instruction>` to type-check that the
// underlying record actually exists.
//===----------------------------------------------------------------------===//

/// MUBUF buffer ops: four addressing modes (OFFSET / OFFEN / IDXEN /
/// BOTHEN) per base opcode.
multiclass SalmonMUBUF4<string base, string sem> {
  def _OFFSET : SalmonOp<!cast<Instruction>(base # "_OFFSET"), sem>;
  def _OFFEN  : SalmonOp<!cast<Instruction>(base # "_OFFEN"),  sem>;
  def _IDXEN  : SalmonOp<!cast<Instruction>(base # "_IDXEN"),  sem>;
  def _BOTHEN : SalmonOp<!cast<Instruction>(base # "_BOTHEN"), sem>;
}

/// gfx12+ VBUFFER fork of MUBUF: same four addressing modes but with the
/// `_VBUFFER_` infix that distinguishes the new buffer-descriptor source.
multiclass SalmonVBUF4<string base, string sem> {
  def _VBUFFER_OFFSET : SalmonOp<!cast<Instruction>(base # "_VBUFFER_OFFSET"), sem>;
  def _VBUFFER_OFFEN  : SalmonOp<!cast<Instruction>(base # "_VBUFFER_OFFEN"),  sem>;
  def _VBUFFER_IDXEN  : SalmonOp<!cast<Instruction>(base # "_VBUFFER_IDXEN"),  sem>;
  def _VBUFFER_BOTHEN : SalmonOp<!cast<Instruction>(base # "_VBUFFER_BOTHEN"), sem>;
}

/// SMEM scalar loads: three operand-source forms (IMM / SGPR / SGPR_IMM)
/// per base opcode.
multiclass SalmonSMEM3<string base, string sem> {
  def _IMM      : SalmonOp<!cast<Instruction>(base # "_IMM"),      sem>;
  def _SGPR     : SalmonOp<!cast<Instruction>(base # "_SGPR"),     sem>;
  def _SGPR_IMM : SalmonOp<!cast<Instruction>(base # "_SGPR_IMM"), sem>;
}

//===----------------------------------------------------------------------===//
// Salmon canonical opcode -> SemOp table.
//
// The block below mirrors the order and grouping of the previous
// `kCanonTable[]` literal in `opcode_map.cpp`. Comments explaining
// editorial decisions (renames, collapses, gfx gating) are preserved
// verbatim.
//
// IMPORTANT: when the same `Instruction` is `def`-d twice with different
// `SemOp` values, the first occurrence wins (matches the runtime's
// `try_emplace` semantics on `kCanonTable`). The generator preserves
// source order to keep this behavior stable.
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------
// SOPP
// ---------------------------------------------------------------------
def : SalmonOp<S_ENDPGM, "S_ENDPGM">;
def : SalmonOp<S_NOP, "S_NOP">;
def : SalmonOp<S_BRANCH, "S_BRANCH">;
def : SalmonOp<S_CODE_END, "S_CODE_END">;
// `s_cbranch_<cond>`: scalar conditional branches keyed off SCC / VCC /
// EXEC, both Z (zero) and NZ (non-zero) polarities for the latter two.
foreach cond = ["SCC0", "SCC1", "VCCZ", "VCCNZ", "EXECZ", "EXECNZ"] in
  def : SalmonOp<!cast<Instruction>("S_CBRANCH_" # cond),
                 "S_CBRANCH_" # cond>;
def : SalmonOp<S_WAITCNT, "S_WAITCNT">;
// `s_wait_<counter>`: gfx12+ split waitcnt counters. The combined
// `LOADCNT_DSCNT` form is the gfx12 fused wait — kept in the same list
// because the lookup table treats it as just another opcode.
foreach cnt = ["LOADCNT", "KMCNT", "DSCNT", "XCNT", "LOADCNT_DSCNT"] in
  def : SalmonOp<!cast<Instruction>("S_WAIT_" # cnt),
                 "S_WAIT_" # cnt>;
def : SalmonOp<S_WAITCNT_DEPCTR, "S_WAIT_ALU">;
def : SalmonOp<S_CLAUSE, "S_CLAUSE">;
def : SalmonOp<S_DELAY_ALU, "S_DELAY_ALU">;
def : SalmonOp<S_SET_GPR_IDX_ON, "S_SET_GPR_IDX_ON">;
def : SalmonOp<S_SET_GPR_IDX_OFF, "S_SET_GPR_IDX_OFF">;
def : SalmonOp<S_SETVSKIP, "S_SETVSKIP">;

// Barriers. GFX < 12 has a single SOPP `S_BARRIER` pseudo; GFX12+ splits
// it into a SOPP wait and a SOP1 signal (both IMM and M0 forms). All
// `_ISFIRST`/`_INIT`/`_JOIN`/`_LEAVE` variants are intentionally left
// unmapped — the raiser does not model them yet.
def : SalmonOp<S_BARRIER, "S_BARRIER">;
def : SalmonOp<S_BARRIER_WAIT, "S_BARRIER_WAIT">;
def : SalmonOp<S_BARRIER_SIGNAL_IMM, "S_BARRIER_SIGNAL">;
def : SalmonOp<S_BARRIER_SIGNAL_M0, "S_BARRIER_SIGNAL">;

// ---------------------------------------------------------------------
// SMEM scalar loads
// ---------------------------------------------------------------------
defm : SalmonSMEM3<"S_LOAD_DWORD", "S_LOAD_B32">;
defm : SalmonSMEM3<"S_LOAD_DWORDX2", "S_LOAD_B64">;
defm : SalmonSMEM3<"S_LOAD_DWORDX3", "S_LOAD_B96">;
defm : SalmonSMEM3<"S_LOAD_DWORDX4", "S_LOAD_B128">;
defm : SalmonSMEM3<"S_LOAD_DWORDX8", "S_LOAD_B256">;
defm : SalmonSMEM3<"S_LOAD_DWORDX16", "S_LOAD_B512">;
// gfx12+ scalar narrow loads. All four types expose the full
// IMM/SGPR/SGPR_IMM triad so the existing SMEM3 macro applies
// unchanged. Handler in handle_smem.cpp.
defm : SalmonSMEM3<"S_LOAD_U8", "S_LOAD_U8">;
defm : SalmonSMEM3<"S_LOAD_I8", "S_LOAD_I8">;
defm : SalmonSMEM3<"S_LOAD_U16", "S_LOAD_U16">;
defm : SalmonSMEM3<"S_LOAD_I16", "S_LOAD_I16">;
defm : SalmonSMEM3<"S_STORE_DWORD", "S_STORE_B32">;
defm : SalmonSMEM3<"S_STORE_DWORDX2", "S_STORE_B64">;
defm : SalmonSMEM3<"S_STORE_DWORDX4", "S_STORE_B128">;

// ---------------------------------------------------------------------
// SOPC
//
// Scalar `S_CMP_<pred>_<ty>` opcodes. Each type only ships a subset of
// the predicate space, so we list predicates and types separately and
// let `foreach` materialise the correct (pred, ty) pairs:
//
//   * U32 / I32 ship the six ordered predicates
//     (EQ / LG / GT / GE / LT / LE).
//   * U64 ships only the equality pair (EQ / LG).
//   * F32 / F16 add the six unordered predicates
//     (NEQ / NGT / NGE / NLT / NLE / NLG) on top of the ordered set.
//
// The per-section ordering matches the previous hand-rolled table so the
// `try_emplace` "first-wins" semantics observed downstream are stable.
// ---------------------------------------------------------------------
defvar SalmonIntOrderedCmpPreds =
    ["EQ", "LG", "GT", "GE", "LT", "LE"];
defvar SalmonFloatCmpPreds =
    !listconcat(SalmonIntOrderedCmpPreds,
                ["NEQ", "NGT", "NGE", "NLT", "NLE", "NLG"]);

foreach pred = SalmonIntOrderedCmpPreds in
  def : SalmonOp<!cast<Instruction>("S_CMP_" # pred # "_U32"),
                 "S_CMP_" # pred # "_U32">;

foreach pred = ["EQ", "LG"] in
  def : SalmonOp<!cast<Instruction>("S_CMP_" # pred # "_U64"),
                 "S_CMP_" # pred # "_U64">;

foreach pred = SalmonIntOrderedCmpPreds in
  def : SalmonOp<!cast<Instruction>("S_CMP_" # pred # "_I32"),
                 "S_CMP_" # pred # "_I32">;

foreach ty = ["F32", "F16"] in
  foreach pred = SalmonFloatCmpPreds in
    def : SalmonOp<!cast<Instruction>("S_CMP_" # pred # "_" # ty),
                   "S_CMP_" # pred # "_" # ty>;

// ---------------------------------------------------------------------
// SOPK
// ---------------------------------------------------------------------
def : SalmonOp<S_MOVK_I32, "S_MOVK_I32">;
def : SalmonOp<S_ADDK_I32, "S_ADDK_I32">;
def : SalmonOp<S_MULK_I32, "S_MULK_I32">;

// `S_CMPK_<pred>_<ty>`: SOPK encodes a 16-bit immediate compare, available
// for the same six ordered integer predicates as SOPC's `S_CMP_*` on
// both signed (I32) and unsigned (U32) operands. The hand-written table
// emitted ordered comparisons grouped by type first, then the equality
// pair grouped by predicate; the two `foreach`es preserve that ordering
// so the generated `kEntries[]` stays byte-identical.
foreach ty = ["I32", "U32"] in
  foreach pred = ["GE", "GT", "LE", "LT"] in
    def : SalmonOp<!cast<Instruction>("S_CMPK_" # pred # "_" # ty),
                   "S_CMPK_" # pred # "_" # ty>;

foreach pred = ["EQ", "LG"] in
  foreach ty = ["I32", "U32"] in
    def : SalmonOp<!cast<Instruction>("S_CMPK_" # pred # "_" # ty),
                   "S_CMPK_" # pred # "_" # ty>;

def : SalmonOp<S_GETREG_B32, "S_GETREG_B32">;
def : SalmonOp<S_SETREG_B32, "S_SETREG_B32">;
def : SalmonOp<S_SETREG_IMM32_B32, "S_SETREG_IMM32_B32">;

// ---------------------------------------------------------------------
// SOP1
// ---------------------------------------------------------------------
def : SalmonOp<S_MOV_B32, "S_MOV_B32">;
def : SalmonOp<S_MOV_B64, "S_MOV_B64">;
def : SalmonOp<S_NOT_B32, "S_NOT_B32">;
def : SalmonOp<S_NOT_B64, "S_NOT_B64">;
def : SalmonOp<S_BREV_B32, "S_BREV_B32">;
def : SalmonOp<S_FF0_I32_B32, "S_FF0_I32_B32">;
def : SalmonOp<S_FF0_I32_B64, "S_FF0_I32_B64">;
def : SalmonOp<S_FF1_I32_B32, "S_FF1_I32_B32">;
def : SalmonOp<S_FF1_I32_B64, "S_FF1_I32_B64">;
def : SalmonOp<S_FLBIT_I32_B32, "S_FLBIT_I32_B32">;
def : SalmonOp<S_FLBIT_I32_B64, "S_FLBIT_I32_B64">;
def : SalmonOp<S_FLBIT_I32, "S_FLBIT_I32">;
def : SalmonOp<S_FLBIT_I32_I64, "S_FLBIT_I32_I64">;
def : SalmonOp<S_SEXT_I32_I8, "S_SEXT_I32_I8">;
def : SalmonOp<S_SEXT_I32_I16, "S_SEXT_I32_I16">;
def : SalmonOp<S_CVT_F32_U32, "S_CVT_F32_U32">;
def : SalmonOp<S_CVT_F32_I32, "S_CVT_F32_I32">;
def : SalmonOp<S_CVT_U32_F32, "S_CVT_U32_F32">;
def : SalmonOp<S_CVT_I32_F32, "S_CVT_I32_F32">;
// `s_<op>_saveexec_<width>`: each logical op has both EXEC widths.
// B32 stays on its own SemOp; the GFX9 B64 form aliases onto the B32
// SemOp because the handler is EXEC-width-aware and shares semantics.
defvar SalmonSaveExecOps = ["AND", "OR", "XOR", "ANDN2", "ORN2"];
foreach op = SalmonSaveExecOps in
  def : SalmonOp<!cast<Instruction>("S_" # op # "_SAVEEXEC_B32"),
                 "S_" # op # "_SAVEEXEC_B32">;
foreach op = SalmonSaveExecOps in
  def : SalmonOp<!cast<Instruction>("S_" # op # "_SAVEEXEC_B64"),
                 "S_" # op # "_SAVEEXEC_B32">;
def : SalmonOp<S_GETPC_B64, "S_GETPC_B64">;
// gfx1250 asm renames `S_SETPC_B64` to `s_set_pc_i64`
// (SOPInstructions.td:2208 declares the rename via
// `SOP1_Real_gfx1250<0x048, "s_set_pc_i64">`); the LLVM MC opcode
// remains `S_SETPC_B64`. We canonicalise both names onto
// SemOp::S_SET_PC_I64 so the handler dispatches uniformly across
// gfx versions; the new name is the SemOp because that is how the
// corpus surfaces it (gfx1250 disasm) and what the SemOp enum
// comment in semop.hpp documents.
def : SalmonOp<S_SETPC_B64, "S_SET_PC_I64">;
// gfx1250 asm renames `S_SWAPPC_B64` to `s_swap_pc_i64`
// (SOPInstructions.td:336 declares the SOP1_64 with isCall=1;
// line 2311 renames the asm string via
// `SOP1_Real_gfx1250<0x049, "s_swap_pc_i64">`); the LLVM MC
// opcode remains `S_SWAPPC_B64`. We canonicalise both names
// onto SemOp::S_SWAP_PC_I64 so the handler dispatches uniformly
// across gfx versions; the new name is the SemOp because that
// is how the corpus surfaces it (gfx1250 disasm) and what the
// SemOp enum comment in semop.hpp documents.
def : SalmonOp<S_SWAPPC_B64, "S_SWAP_PC_I64">;
def : SalmonOp<S_ABS_I32, "S_ABS_I32">;
def : SalmonOp<S_SET_VGPR_MSB, "S_SET_VGPR_MSB">;
// `s_bitset{0,1}_b{32,64}`: both polarities at both EXEC widths.
foreach width = ["B32", "B64"] in
  foreach polarity = ["0", "1"] in
    def : SalmonOp<!cast<Instruction>("S_BITSET" # polarity # "_" # width),
                   "S_BITSET" # polarity # "_" # width>;
// SOPC bit-test family (SOPInstructions.td:1411-1414).  Each LLVM
// MC opcode is a pseudo that every real-subtarget form
// (S_BITCMP0_B32_gfx{10,11,12,13}, etc.) is canonicalised onto
// during the MC->pseudo step in `OpcodeMap::canonicalize`, so a
// single entry per variant covers every AMDGPU generation back to
// gfx6.  Writes SCC only — no scalar destination register — which
// matches the SOPC compare shape already implemented alongside in
// handle_sopc.cpp.
def : SalmonOp<S_BITCMP0_B32, "S_BITCMP0_B32">;
def : SalmonOp<S_BITCMP1_B32, "S_BITCMP1_B32">;
def : SalmonOp<S_BITCMP0_B64, "S_BITCMP0_B64">;
def : SalmonOp<S_BITCMP1_B64, "S_BITCMP1_B64">;
// S_CMOV uses [SCC] but does not write SCC, and has no _e64 form
// (SOP1 has only one encoding). The B64 alias is included even
// though the corpus first surfaced only the B32 form, because
// the B64 form ships in the same gfx1250 SOP1 table at adjacent
// opcode (.td: 0x002 / 0x003) and the handler is symmetric.
def : SalmonOp<S_CMOV_B32, "S_CMOV_B32">;
def : SalmonOp<S_CMOV_B64, "S_CMOV_B64">;

// ---------------------------------------------------------------------
// SOP2
// ---------------------------------------------------------------------
def : SalmonOp<S_ADD_U32, "S_ADD_U32">;
def : SalmonOp<S_ADD_I32, "S_ADD_U32">;
def : SalmonOp<S_ADDC_U32, "S_ADDC_U32">;
def : SalmonOp<S_SUB_U32, "S_SUB_U32">;
def : SalmonOp<S_SUB_I32, "S_SUB_U32">;
def : SalmonOp<S_SUBB_U32, "S_SUBB_U32">;
def : SalmonOp<S_ADD_U64, "S_ADD_U64">;
def : SalmonOp<S_AND_B32, "S_AND_B32">;
def : SalmonOp<S_AND_B64, "S_AND_B64">;
def : SalmonOp<S_OR_B32, "S_OR_B32">;
def : SalmonOp<S_OR_B64, "S_OR_B64">;
def : SalmonOp<S_XOR_B32, "S_XOR_B32">;
def : SalmonOp<S_XOR_B64, "S_XOR_B64">;
def : SalmonOp<S_ANDN2_B32, "S_ANDN2_B32">;
def : SalmonOp<S_ANDN2_B64, "S_ANDN2_B64">;
def : SalmonOp<S_ORN2_B32, "S_ORN2_B32">;
def : SalmonOp<S_ORN2_B64, "S_ORN2_B64">;
def : SalmonOp<S_NAND_B32, "S_NAND_B32">;
def : SalmonOp<S_NAND_B64, "S_NAND_B64">;
def : SalmonOp<S_NOR_B32, "S_NOR_B32">;
def : SalmonOp<S_NOR_B64, "S_NOR_B64">;
def : SalmonOp<S_XNOR_B32, "S_XNOR_B32">;
def : SalmonOp<S_XNOR_B64, "S_XNOR_B64">;
def : SalmonOp<S_ABSDIFF_I32, "S_ABSDIFF_I32">;
def : SalmonOp<S_LSHL_B32, "S_LSHL_B32">;
def : SalmonOp<S_LSHL_B64, "S_LSHL_B64">;
def : SalmonOp<S_LSHR_B32, "S_LSHR_B32">;
def : SalmonOp<S_LSHR_B64, "S_LSHR_B64">;
def : SalmonOp<S_ASHR_I32, "S_ASHR_I32">;
def : SalmonOp<S_ASHR_I64, "S_ASHR_I64">;
def : SalmonOp<S_MUL_I32, "S_MUL_I32">;
def : SalmonOp<S_MUL_HI_U32, "S_MUL_HI_U32">;
def : SalmonOp<S_MUL_HI_I32, "S_MUL_HI_I32">;
def : SalmonOp<S_MUL_U64, "S_MUL_U64">;
def : SalmonOp<S_MUL_F32, "S_MUL_F32">;
def : SalmonOp<S_ADD_F32, "S_ADD_F32">;
def : SalmonOp<S_SUB_F32, "S_SUB_F32">;
def : SalmonOp<S_BFE_U32, "S_BFE_U32">;
def : SalmonOp<S_BFE_I32, "S_BFE_I32">;
def : SalmonOp<S_BFM_B32, "S_BFM_B32">;
def : SalmonOp<S_BFM_B64, "S_BFM_B64">;
def : SalmonOp<S_CSELECT_B32, "S_CSELECT_B32">;
def : SalmonOp<S_CSELECT_B64, "S_CSELECT_B64">;
def : SalmonOp<S_MIN_I32, "S_MIN_I32">;
def : SalmonOp<S_MIN_U32, "S_MIN_U32">;
def : SalmonOp<S_MAX_I32, "S_MAX_I32">;
def : SalmonOp<S_MAX_U32, "S_MAX_U32">;
def : SalmonOp<S_PACK_LL_B32_B16, "S_PACK_LL_B32_B16">;
def : SalmonOp<S_PACK_LH_B32_B16, "S_PACK_LH_B32_B16">;
def : SalmonOp<S_LSHL1_ADD_U32, "S_LSHL1_ADD_U32">;
def : SalmonOp<S_LSHL2_ADD_U32, "S_LSHL2_ADD_U32">;
def : SalmonOp<S_LSHL3_ADD_U32, "S_LSHL3_ADD_U32">;
def : SalmonOp<S_LSHL4_ADD_U32, "S_LSHL4_ADD_U32">;
// LLVM's pseudo is `S_ADD_U64`; we still surface it as `SemOp::S_ADD_NC_U64`
// to keep handler parity with the no-carry mnemonic used downstream.
def : SalmonOp<S_ADD_U64, "S_ADD_NC_U64">;
// gfx12 `s_sub_nc_u64` (renamed from `s_sub_u64` in the
// assembler — see SOPInstructions.td 2311
// `S_SUB_U64 ... "s_sub_nc_u64"`). LLVM's pseudo is still
// `S_SUB_U64`; surface it as `SemOp::S_SUB_NC_U64` to mirror
// the S_ADD_NC_U64 convention.
def : SalmonOp<S_SUB_U64, "S_SUB_NC_U64">;

// ---------------------------------------------------------------------
// VOP1 (canonical form is `_e64` after getVOPe64 collapses e32)
// ---------------------------------------------------------------------
def : SalmonOp<V_MOV_B32_e64, "V_MOV_B32">;
def : SalmonOp<V_MOV_B64_e64, "V_MOV_B64">;
def : SalmonOp<V_MOV_B64_PSEUDO, "V_MOV_B64">;
def : SalmonOp<V_SWAP_B32, "V_SWAP_B32">;
def : SalmonOp<V_NOP_e64, "V_NOP">;
def : SalmonOp<V_NOT_B32_e64, "V_NOT_B32">;
def : SalmonOp<V_BFREV_B32_e64, "V_BFREV_B32">;
def : SalmonOp<V_FFBH_U32_e64, "V_FFBH_U32">;
def : SalmonOp<V_FFBL_B32_e64, "V_FFBL_B32">;
def : SalmonOp<V_FFBH_I32_e64, "V_FFBH_I32">;
def : SalmonOp<V_CVT_F32_I32_e64, "V_CVT_F32_I32">;
def : SalmonOp<V_CVT_F32_U32_e64, "V_CVT_F32_U32">;
def : SalmonOp<V_CVT_I32_F32_e64, "V_CVT_I32_F32">;
def : SalmonOp<V_CVT_U32_F32_e64, "V_CVT_U32_F32">;
def : SalmonOp<V_CVT_F16_F32_e64, "V_CVT_F16_F32">;
def : SalmonOp<V_CVT_F32_F16_e64, "V_CVT_F32_F16">;
def : SalmonOp<V_CVT_F32_BF16_e64, "V_CVT_F32_BF16">;
def : SalmonOp<V_CVT_PK_F32_FP8_e64, "V_CVT_PK_F32_FP8">;
def : SalmonOp<V_CVT_PK_F32_BF8_e64, "V_CVT_PK_F32_BF8">;
def : SalmonOp<V_CVT_F32_FP8_e64, "V_CVT_F32_FP8">;
def : SalmonOp<V_CVT_F32_BF8_e64, "V_CVT_F32_BF8">;
// gfx1250-only packed-8 FP4 -> BF16 scaled convert
// (VOP3Instructions.td:1788). The real form
// `V_CVT_SCALE_PK8_BF16_FP4_e64_gfx1250` canonicalises onto the
// `_e64` pseudo via the MC->pseudo step in
// `OpcodeMap::canonicalize` (getMCOpcode-derived table); a single
// entry therefore covers every real subtarget slot.
def : SalmonOp<V_CVT_SCALE_PK8_BF16_FP4_e64, "V_CVT_SCALE_PK8_BF16_FP4">;
def : SalmonOp<V_CVT_F32_UBYTE0_e64, "V_CVT_F32_UBYTE0">;
def : SalmonOp<V_CVT_F32_UBYTE1_e64, "V_CVT_F32_UBYTE1">;
def : SalmonOp<V_CVT_F32_UBYTE2_e64, "V_CVT_F32_UBYTE2">;
def : SalmonOp<V_CVT_F32_UBYTE3_e64, "V_CVT_F32_UBYTE3">;
def : SalmonOp<V_CVT_F64_U32_e64, "V_CVT_F64_U32">;
def : SalmonOp<V_CVT_F64_I32_e64, "V_CVT_F64_I32">;
def : SalmonOp<V_CVT_U32_F64_e64, "V_CVT_U32_F64">;
def : SalmonOp<V_RCP_IFLAG_F32_e64, "V_RCP_IFLAG_F32">;
def : SalmonOp<V_RCP_F32_e64, "V_RCP_F32">;
def : SalmonOp<V_RSQ_F32_e64, "V_RSQ_F32">;
def : SalmonOp<V_SQRT_F32_e64, "V_SQRT_F32">;
def : SalmonOp<V_EXP_F32_e64, "V_EXP_F32">;
def : SalmonOp<V_LOG_F32_e64, "V_LOG_F32">;
def : SalmonOp<V_FLOOR_F32_e64, "V_FLOOR_F32">;
def : SalmonOp<V_CEIL_F32_e64, "V_CEIL_F32">;
def : SalmonOp<V_TRUNC_F32_e64, "V_TRUNC_F32">;
def : SalmonOp<V_FRACT_F32_e64, "V_FRACT_F32">;
def : SalmonOp<V_READFIRSTLANE_B32, "V_READFIRSTLANE_B32">;
def : SalmonOp<V_FLOOR_F16_e64, "V_FLOOR_F16">;
def : SalmonOp<V_CVT_F16_U16_e64, "V_CVT_F16_U16">;
def : SalmonOp<V_CVT_U16_F16_e64, "V_CVT_U16_F16">;

// ---------------------------------------------------------------------
// VOP2 / VOP3
// ---------------------------------------------------------------------
def : SalmonOp<V_ADD_F32_e64, "V_ADD_F32">;
def : SalmonOp<V_SUB_F32_e64, "V_SUB_F32">;
def : SalmonOp<V_SUBREV_F32_e64, "V_SUBREV_F32">;
def : SalmonOp<V_MUL_F32_e64, "V_MUL_F32">;
def : SalmonOp<V_FMAC_F32_e64, "V_FMAC_F32">;
def : SalmonOp<V_FMA_F32_e64, "V_FMA_F32">;
def : SalmonOp<V_FMAMK_F32, "V_FMAMK_F32">;
def : SalmonOp<V_FMAAK_F32, "V_FMAAK_F32">;
def : SalmonOp<V_MAX_F32_e64, "V_MAX_F32">;
def : SalmonOp<V_MIN_F32_e64, "V_MIN_F32">;
// gfx11+ IEEE-754 2019 maximum/minimum. These differ from V_MAX_NUM_F32 /
// V_MIN_NUM_F32 in their NaN semantics: maximum/minimum propagate NaN,
// while maxnum/minnum return the non-NaN operand.
def : SalmonOp<V_MAXIMUM_F32_e64, "V_MAXIMUM_F32">;
def : SalmonOp<V_MINIMUM_F32_e64, "V_MINIMUM_F32">;
def : SalmonOp<V_DIV_FIXUP_F32_e64, "V_DIV_FIXUP_F32">;
def : SalmonOp<V_DIV_FMAS_F32_e64, "V_DIV_FMAS_F32">;
def : SalmonOp<V_DIV_SCALE_F32_e64, "V_DIV_SCALE_F32">;
// LLVM's no-carry 32-bit add/sub pseudos are just `V_ADD_U32` etc.; the
// carry-in-carry-out form is the older `V_ADDC_U32`/`V_SUBB_U32` family.
def : SalmonOp<V_ADD_U32_e64, "V_ADD_NC_U32">;
def : SalmonOp<V_SUB_U32_e64, "V_SUB_NC_U32">;
def : SalmonOp<V_SUBREV_U32_e64, "V_SUBREV_NC_U32">;
// GFX9 VOP3-only signed add/sub (saddsat/ssubsat when clamp is set,
// plain add/sub otherwise). Distinct from the legacy GFX10+ NC variants.
def : SalmonOp<V_ADD_I32_e64, "V_ADD_I32">;
def : SalmonOp<V_SUB_I32_e64, "V_SUB_I32">;
def : SalmonOp<V_ADD_CO_U32_e64, "V_ADD_CO_U32">;
def : SalmonOp<V_ADDC_U32_e64, "V_ADD_CO_CI_U32">;
def : SalmonOp<V_SUB_CO_U32_e64, "V_SUB_CO_U32">;
def : SalmonOp<V_SUBREV_CO_U32_e64, "V_SUBREV_CO_U32">;
def : SalmonOp<V_SUBB_U32_e64, "V_SUB_CO_CI_U32">;
def : SalmonOp<V_SUBBREV_U32_e64, "V_SUBREV_CO_CI_U32">;
def : SalmonOp<V_AND_B32_e64, "V_AND_B32">;
def : SalmonOp<V_OR_B32_e64, "V_OR_B32">;
def : SalmonOp<V_XOR_B32_e64, "V_XOR_B32">;
def : SalmonOp<V_XNOR_B32_e64, "V_XNOR_B32">;
def : SalmonOp<V_LSHLREV_B32_e64, "V_LSHLREV_B32">;
def : SalmonOp<V_LSHRREV_B32_e64, "V_LSHRREV_B32">;
def : SalmonOp<V_ASHRREV_I32_e64, "V_ASHRREV_I32">;
def : SalmonOp<V_CNDMASK_B32_e64, "V_CNDMASK_B32">;
def : SalmonOp<V_MUL_LO_U32_e64, "V_MUL_LO_U32">;
def : SalmonOp<V_MUL_HI_U32_e64, "V_MUL_HI_U32">;
def : SalmonOp<V_MUL_HI_I32_e64, "V_MUL_HI_I32">;
def : SalmonOp<V_MUL_I32_I24_e64, "V_MUL_I32_I24">;
def : SalmonOp<V_MUL_U32_U24_e64, "V_MUL_U32_U24">;
def : SalmonOp<V_MUL_HI_U32_U24_e64, "V_MUL_HI_U32_U24">;
def : SalmonOp<V_MUL_HI_I32_I24_e64, "V_MUL_HI_I32_I24">;
def : SalmonOp<V_MAD_U32_U24_e64, "V_MAD_U32_U24">;
def : SalmonOp<V_MAD_U32_e64, "V_MAD_U32">;
def : SalmonOp<V_ADD3_U32_e64, "V_ADD3_U32">;
def : SalmonOp<V_LSHL_ADD_U32_e64, "V_LSHL_ADD_U32">;
def : SalmonOp<V_ADD_LSHL_U32_e64, "V_ADD_LSHL_U32">;
def : SalmonOp<V_LSHL_OR_B32_e64, "V_LSHL_OR_B32">;
def : SalmonOp<V_AND_OR_B32_e64, "V_AND_OR_B32">;
def : SalmonOp<V_OR3_B32_e64, "V_OR3_B32">;
// gfx9+ ternary xor+add (VOP3Instructions.td:684, iselect
// pattern at :831 — `add(xor(a, b), c)`).
def : SalmonOp<V_XAD_U32_e64, "V_XAD_U32">;
// gfx6+ funnel-shift right (VOP3Instructions.td:218,
// gfx11/gfx12 opcode 0x216 at :1787). The `_t16` and
// `_fake16` variants collapse onto V_ALIGNBIT_B32_e64 via
// sameSemanticShape.
def : SalmonOp<V_ALIGNBIT_B32_e64, "V_ALIGNBIT_B32">;
// gfx10+ ternary xor (VOP3Instructions.td:1348). VOP3
// ternaries only have the e64 form.
def : SalmonOp<V_XOR3_B32_e64, "V_XOR3_B32">;
// gfx10+ VOP3 16-bit no-carry add (VOP3Instructions.td:1362,
// real opcode 0x303 at :1852/2016). The opcode_map's
// sameSemanticShape canonicalizer collapses
// V_ADD_NC_U16_t16_e64 and V_ADD_NC_U16_fake16_e64 onto this
// base entry automatically, so a single mapping suffices.
def : SalmonOp<V_ADD_NC_U16_e64, "V_ADD_NC_U16">;
def : SalmonOp<V_BFE_U32_e64, "V_BFE_U32">;
def : SalmonOp<V_BFE_I32_e64, "V_BFE_I32">;
// gfx6+ VOP3 bit-field insert. Ternary, e64-only (no VOP1/VOP2
// form). Hardware semantic: `dst = (src0 & src1) | (~src0 & src2)`
// — src0 is the mask, src1 provides bits where mask=1, src2
// provides bits where mask=0. See VOP3Instructions.td
// `AMDGPUbfiPattern`. Same canonical opcode across gfx11/gfx12;
// libdevice's asin lowering emits this as its sign-stitching
// step, which was the real-world coverage gap that motivated
// the handler.
def : SalmonOp<V_BFI_B32_e64, "V_BFI_B32">;
def : SalmonOp<V_PERM_B32_e64, "V_PERM_B32">;
def : SalmonOp<V_MBCNT_LO_U32_B32_e64, "V_MBCNT_LO_U32_B32">;
def : SalmonOp<V_MBCNT_HI_U32_B32_e64, "V_MBCNT_HI_U32_B32">;
def : SalmonOp<V_READLANE_B32, "V_READLANE_B32">;
def : SalmonOp<V_WRITELANE_B32, "V_WRITELANE_B32">;
def : SalmonOp<V_MED3_F32_e64, "V_MED3_F32">;
def : SalmonOp<V_MAX3_F32_e64, "V_MAX3_F32">;
def : SalmonOp<V_MIN3_F32_e64, "V_MIN3_F32">;
// gfx11 V_MINMAX_F32 (opcode 0x25f) and the gfx12-renamed
// V_MINMAX_NUM_F32 (opcode 0x268) share the same minnum-of-
// maxnum semantics; both LLVM pseudos canonicalize onto
// V_MINMAX_F32_e64 so a single mapping suffices.
def : SalmonOp<V_MINMAX_F32_e64, "V_MINMAX_NUM_F32">;
// gfx11/gfx12 VOP3 integer 3-way max (opcode 0x21e, see
// VOP3Instructions.td:1795). LLVM only emits the e64 form for
// VOP3 ternaries; no DPP variant exists for v_max3_*.
def : SalmonOp<V_MAX3_U32_e64, "V_MAX3_U32">;
def : SalmonOp<V_MED3_I32_e64, "V_MED3_I32">;
// LLVM does not yet expose a `V_MAX3_MAXIMUM_F32` pseudo; leave the
// `V_MAX3_NUM_F32` SemOp unmapped until it does.
def : SalmonOp<V_BITOP3_B32_e64, "V_BITOP3_B32">;
def : SalmonOp<V_BITOP3_B16_e64, "V_BITOP3_B16">;
def : SalmonOp<V_FMA_MIX_F32, "V_FMA_MIX_F32">;
// gfx9.5+/gfx1250 BF16-narrow sibling (VOP3PInstructions.td:109).
// Same VOP3P operand shape (3 srcs + op_sel / op_sel_hi modifiers)
// as V_FMA_MIX_F32; only the narrow-half element type changes
// (bf16 vs f16). The real form V_FMA_MIX_F32_BF16_gfx1250 is
// canonicalized onto this pseudo via the MC->pseudo step in
// `canonicalize` (getMCOpcode-derived table), mirroring the
// V_FMA_MIX_F32 entry above.
def : SalmonOp<V_FMA_MIX_F32_BF16, "V_FMA_MIX_F32_BF16">;
def : SalmonOp<V_ADD_F16_e64, "V_ADD_F16">;
def : SalmonOp<V_MUL_F16_e64, "V_MUL_F16">;
def : SalmonOp<V_SUB_F16_e64, "V_SUB_F16">;
def : SalmonOp<V_SUBREV_F16_e64, "V_SUBREV_F16">;
def : SalmonOp<V_MAC_F16_e64, "V_MAC_F16">;
def : SalmonOp<V_MADMK_F16, "V_MADMK_F16">;
def : SalmonOp<V_MADAK_F16, "V_MADAK_F16">;
def : SalmonOp<V_FMAC_F16_e64, "V_FMAC_F16">;
def : SalmonOp<V_MAX_F16_e64, "V_MAX_F16">;
def : SalmonOp<V_MIN_F16_e64, "V_MIN_F16">;
def : SalmonOp<V_LDEXP_F16_e64, "V_LDEXP_F16">;
def : SalmonOp<V_LDEXP_F32_e64, "V_LDEXP_F32">;
def : SalmonOp<V_DOT2C_I32_I16_e64, "V_DOT2C_I32_I16">;
def : SalmonOp<V_DOT4C_I32_I8_e64, "V_DOT4C_I32_I8">;
def : SalmonOp<V_DOT8C_I32_I4_e64, "V_DOT8C_I32_I4">;
def : SalmonOp<V_PK_FMAC_F16_e64, "V_PK_FMAC_F16">;
def : SalmonOp<V_MAX_U16_e64, "V_MAX_U16">;
def : SalmonOp<V_MIN_U16_e64, "V_MIN_U16">;
def : SalmonOp<V_MAX_I16_e64, "V_MAX_I16">;
def : SalmonOp<V_MIN_I16_e64, "V_MIN_I16">;
def : SalmonOp<V_ADD_U16_e64, "V_ADD_U16">;
def : SalmonOp<V_SUB_U16_e64, "V_SUB_U16">;
def : SalmonOp<V_SUBREV_U16_e64, "V_SUBREV_U16">;
def : SalmonOp<V_MUL_LO_U16_e64, "V_MUL_LO_U16">;
def : SalmonOp<V_ASHRREV_I16_e64, "V_ASHRREV_I16">;
def : SalmonOp<V_LSHRREV_B16_e64, "V_LSHRREV_B16">;
def : SalmonOp<V_LSHLREV_B16_e64, "V_LSHLREV_B16">;
def : SalmonOp<V_PACK_B32_F16_e64, "V_PACK_B32_F16">;
def : SalmonOp<V_CVT_PK_BF16_F32_e64, "V_CVT_PK_BF16_F32">;
def : SalmonOp<V_CVT_PK_BF8_F32_e64, "V_CVT_PK_BF8_F32">;
def : SalmonOp<V_CVT_PK_FP8_F32_e64, "V_CVT_PK_FP8_F32">;
def : SalmonOp<V_CVT_PKRTZ_F16_F32_e64, "V_CVT_PKRTZ_F16_F32">;
def : SalmonOp<V_CVT_PK_F16_F32_e64, "V_CVT_PK_F16_F32">;
def : SalmonOp<V_CVT_SCALEF32_PK_FP4_F32_e64, "V_CVT_SCALEF32_PK_FP4_F32">;
def : SalmonOp<V_BFM_B32_e64, "V_BFM_B32">;

// ---------------------------------------------------------------------
// FP64
// ---------------------------------------------------------------------
def : SalmonOp<V_ADD_F64_e64, "V_ADD_F64">;
def : SalmonOp<V_MUL_F64_e64, "V_MUL_F64">;
def : SalmonOp<V_FMA_F64_e64, "V_FMA_F64">;
def : SalmonOp<V_FMAC_F64_e64, "V_FMAC_F64">;
// V_RCP_F64 has no DPP (`VOP1_Real_..._NO_DPP_...`); the e32
// form is collapsed to e64 by getVOPe64 before lookup, so a
// single e64 entry covers both encodings.
def : SalmonOp<V_RCP_F64_e64, "V_RCP_F64">;

// ---------------------------------------------------------------------
// More VOP3 32-bit int min/max and lane perms
// ---------------------------------------------------------------------
def : SalmonOp<V_MAX_U32_e64, "V_MAX_U32">;
def : SalmonOp<V_MIN_U32_e64, "V_MIN_U32">;
def : SalmonOp<V_MAX_I32_e64, "V_MAX_I32">;
def : SalmonOp<V_MIN_I32_e64, "V_MIN_I32">;
def : SalmonOp<V_PERMLANE16_B32_e64, "V_PERMLANE16_B32">;
def : SalmonOp<V_PERMLANEX16_B32_e64, "V_PERMLANEX16_B32">;
def : SalmonOp<V_PERMLANE64_B32, "V_PERMLANE64_B32">;
def : SalmonOp<V_PERMLANE16_SWAP_B32_e32, "V_PERMLANE16_SWAP_B32">;
def : SalmonOp<V_PERMLANE16_SWAP_B32_e64, "V_PERMLANE16_SWAP_B32">;
def : SalmonOp<V_PERMLANE32_SWAP_B32_e32, "V_PERMLANE32_SWAP_B32">;
def : SalmonOp<V_PERMLANE32_SWAP_B32_e64, "V_PERMLANE32_SWAP_B32">;

// ---------------------------------------------------------------------
// VOPC (V_CMP_* and V_CMPX_*) are NOT enumerated here. They're picked up
// by scanning canonical pseudo names in `OpcodeMap::build` and routed to
// the two collapsed SemOps (V_CMP / V_CMPX) with metadata (predicate,
// element width, int/float) in the `vcmp_` side table.
//
// ---------------------------------------------------------------------
// VOP3P (canonical is bare pseudo name, no _e64)
// ---------------------------------------------------------------------
def : SalmonOp<V_PK_ADD_F32, "V_PK_ADD_F32">;
def : SalmonOp<V_PK_MUL_F32, "V_PK_MUL_F32">;
def : SalmonOp<V_PK_FMA_F32, "V_PK_FMA_F32">;
// LLVM has no `V_PK_MAX_F32`/`V_PK_MIN_F32` pseudo (only F16 variants);
// leave the matching SemOps unmapped until one appears.
def : SalmonOp<V_PK_MOV_B32, "V_PK_MOV_B32">;
// Packed `<2 x i16>` int family. LLVM emits the bare TableGen pseudo
// (`V_PK_ADD_U16` / `V_PK_LSHLREV_B16`); the gfx10/gfx11/gfx12/vi
// realtriples (`_gfx10`, `_vi`, etc.) all canonicalize back to it
// through the pseudo-alias step. Two opcodes is the entire VOP3P
// packed-int surface the kerneldex corpus exercises today; siblings
// V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16 / V_PK_MUL_LO_U16
// / V_PK_MAX_{I,U}16 / V_PK_MIN_{I,U}16 share the same handler shape
// and are intentionally NOT pre-enumerated — the no-fallback policy
// says we wait for a corpus producer rather than ship dead lift code.
def : SalmonOp<V_PK_ADD_U16, "V_PK_ADD_U16">;
def : SalmonOp<V_PK_LSHLREV_B16, "V_PK_LSHLREV_B16">;

// ---------------------------------------------------------------------
// 64-bit vector
// ---------------------------------------------------------------------
def : SalmonOp<V_LSHLREV_B64_e64, "V_LSHLREV_B64">;
// V_LSHRREV_B64 / V_ASHRREV_I64 — gfx8+ 64-bit logical / arithmetic
// right shift with reversed operands (`dst = src1 >> shamt`). Only
// the plain `_e64` pseudo exists; per-ISA realtriples (gfx10 / gfx11
// / gfx1250) all collapse back to it through the disassembler's
// pseudo-alias step (TableGen treats the gfx1250 form as a `_e64`
// realtriple, not a `_t16_e64` half-precision variant — see
// VOP3Instructions.td:2341 vs the b16 family at 2336-2340).
def : SalmonOp<V_LSHRREV_B64_e64, "V_LSHRREV_B64">;
def : SalmonOp<V_ASHRREV_I64_e64, "V_ASHRREV_I64">;
def : SalmonOp<V_LSHL_ADD_U64_e64, "V_LSHL_ADD_U64">;
// LLVM's 64-bit no-carry add pseudo is simply `V_ADD_U64_e64`.
def : SalmonOp<V_ADD_U64_e64, "V_ADD_NC_U64">;
// gfx1250 VOP2 V_MUL_U64; canonicalizer collapses `_gfx1250` suffix
// onto the bare `V_MUL_U64_e64` pseudo.
def : SalmonOp<V_MUL_U64_e64, "V_MUL_U64">;
def : SalmonOp<V_MAD_U64_U32_e64, "V_MAD_U64_U32">;
// LLVM no longer exposes a distinct carry-out variant; `V_MAD_CO_U64_U32`
// SemOp stays unmapped until one reappears.

// ---------------------------------------------------------------------
// FLAT
// ---------------------------------------------------------------------
// FLAT / GLOBAL untyped load/store size suffixes. The DWORDX{1..4} tail
// is shared across all four families; the byte/short prefixes differ:
// loads carry an explicit U/S sign on byte/short, stores use unsigned
// byte/short, and only GLOBAL_LOAD ships the `_SHORT_D16_HI` variant.
defvar SalmonDwordWidths = ["DWORD", "DWORDX2", "DWORDX3", "DWORDX4"];

foreach suf = !listconcat(["UBYTE", "SBYTE", "USHORT", "SSHORT"],
                          SalmonDwordWidths) in
  def : SalmonOp<!cast<Instruction>("FLAT_LOAD_" # suf),
                 "FLAT_LOAD_" # suf>;
foreach suf = !listconcat(["BYTE", "SHORT", "SHORT_D16_HI"],
                          SalmonDwordWidths) in
  def : SalmonOp<!cast<Instruction>("FLAT_STORE_" # suf),
                 "FLAT_STORE_" # suf>;
foreach suf = !listconcat(["UBYTE", "SBYTE", "USHORT", "SSHORT",
                           "SHORT_D16_HI"],
                          SalmonDwordWidths) in
  def : SalmonOp<!cast<Instruction>("GLOBAL_LOAD_" # suf),
                 "GLOBAL_LOAD_" # suf>;
foreach suf = !listconcat(["BYTE", "SHORT", "SHORT_D16_HI"],
                          SalmonDwordWidths) in
  def : SalmonOp<!cast<Instruction>("GLOBAL_STORE_" # suf),
                 "GLOBAL_STORE_" # suf>;

// ---------------------------------------------------------------------
// FLAT atomics (canonicalized to non-SADDR via getGlobalVaddrOp; no-return form)
// ---------------------------------------------------------------------
// FLAT atomic ops the raiser handles. GLOBAL atomics extend this same
// list with two PK_ADD packed variants (see below).
defvar SalmonFlatAtomicOps = [
    "ADD", "SUB", "AND", "OR", "XOR",
    "SMIN", "SMAX", "UMIN", "UMAX",
    "SWAP", "CMPSWAP", "ADD_F32"];

foreach op = SalmonFlatAtomicOps in
  def : SalmonOp<!cast<Instruction>("FLAT_ATOMIC_" # op),
                 "FLAT_ATOMIC_" # op>;

// ---------------------------------------------------------------------
// GLOBAL atomics
// ---------------------------------------------------------------------
// GLOBAL atomics: same as the FLAT atomic set plus the gfx9+ packed
// half-precision adds, which the FLAT path doesn't ship.
foreach op = !listconcat(SalmonFlatAtomicOps,
                         ["PK_ADD_BF16", "PK_ADD_F16"]) in
  def : SalmonOp<!cast<Instruction>("GLOBAL_ATOMIC_" # op),
                 "GLOBAL_ATOMIC_" # op>;

// ---------------------------------------------------------------------
// SMEM atomics (enumerate addressing forms: IMM / SGPR / SGPR_IMM)
// ---------------------------------------------------------------------
defm : SalmonSMEM3<"S_ATOMIC_SWAP", "S_ATOMIC_SWAP">;
// S_ATOMIC_DEC (scalar wrap-decrement) — the three addressing forms
// (IMM / SGPR / SGPR_IMM) share a SemOp.  The `_RTN` variants the
// disassembler emits for the GLC=1 returned-old-value form collapse
// onto these non-RTN pseudos via the `_RTN` alias rule below; the
// handler keys off `di.numDefs` (== MCInstrDesc::getNumDefs()) for
// the write-back decision, so a single lowering covers both halves
// of the contract.
defm : SalmonSMEM3<"S_ATOMIC_DEC", "S_ATOMIC_DEC">;

// ---------------------------------------------------------------------
// DS
// ---------------------------------------------------------------------
def : SalmonOp<DS_LOAD_TR16_B128, "DS_LOAD_TR16_B128">;
def : SalmonOp<DS_READ_B64_TR_B16, "DS_READ_B64_TR_B16">;
def : SalmonOp<DS_READ_B64_TR_B8, "DS_READ_B64_TR_B8">;
// gfx1250 64-bit transposed LDS load (i8 element, 8 elements per
// lane post-transpose). Distinct LLVM MC pseudo from the gfx950
// `DS_READ_B64_TR_B8` sibling — same semantics, different isel
// gate (isGFX1250Plus vs HasGFX950Insts). The handler folds both
// paths into a shared hand-rolled emulation; canonicalising the
// SemOp on the gfx1250 spelling matches the disassembly the
// raise_cli operator sees.
def : SalmonOp<DS_LOAD_TR8_B64, "DS_LOAD_TR8_B64">;
def : SalmonOp<DS_READ_B32, "DS_READ_B32">;
def : SalmonOp<DS_READ_B64, "DS_READ_B64">;
// 96-bit LDS load. LLVM MC keeps the legacy `DS_READ_B96`
// pseudo name for what gfx11+ disassembles as `ds_load_b96`
// (DSInstructions.td:1578); we canonicalise on the gfx11+
// spelling, mirroring the s_set_pc_i64 / DS_LOAD_TR8_B64
// precedent. The handler dispatches via the existing generic
// DS read range — see dsClassify in handle_ds.cpp for the
// {dwords=3, loadBits=96} entry.
def : SalmonOp<DS_READ_B96, "DS_READ_B96">;
def : SalmonOp<DS_READ_B128, "DS_READ_B128">;
def : SalmonOp<DS_READ2_B32, "DS_READ2_B32">;
def : SalmonOp<DS_READ2_B64, "DS_READ2_B64">;
// gfx11+ stride-64 two-address LDS loads (DSInstructions.td:1529,
// 1542 — `ds_load_2addr_stride64_b{32,64}`). Same two-offset MC
// shape as the non-ST64 siblings; the canonicalization chain
// collapses the _gfx9/_gfx11/_gfx12 reals onto the pseudo
// forms here, mirroring the DS_READ2_B{32,64} entries above.
def : SalmonOp<DS_READ2ST64_B32, "DS_READ2ST64_B32">;
def : SalmonOp<DS_READ2ST64_B64, "DS_READ2ST64_B64">;
// DS sub-dword loads: signed/unsigned x {16, 8}-bit. Source order is
// width-major (16 first, then 8) within each sign group to match the
// hand-rolled table.
foreach width = ["16", "8"] in
  foreach sign = ["U", "I"] in
    def : SalmonOp<!cast<Instruction>("DS_READ_" # sign # width),
                   "DS_READ_" # sign # width>;
def : SalmonOp<DS_WRITE_B32, "DS_WRITE_B32">;
def : SalmonOp<DS_WRITE_B64, "DS_WRITE_B64">;
// Symmetric 96-bit LDS store. gfx11+ asm spelling is
// `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
// remains `DS_WRITE_B96`.
def : SalmonOp<DS_WRITE_B96, "DS_WRITE_B96">;
def : SalmonOp<DS_WRITE_B128, "DS_WRITE_B128">;
def : SalmonOp<DS_WRITE2_B32, "DS_WRITE2_B32">;
def : SalmonOp<DS_WRITE2_B64, "DS_WRITE2_B64">;
// gfx11+ stride-64 two-address LDS stores — mirror of the
// DS_READ2ST64 read-side entries above.
def : SalmonOp<DS_WRITE2ST64_B32, "DS_WRITE2ST64_B32">;
def : SalmonOp<DS_WRITE2ST64_B64, "DS_WRITE2ST64_B64">;
def : SalmonOp<DS_WRITE_B16, "DS_WRITE_B16">;
def : SalmonOp<DS_WRITE_B8, "DS_WRITE_B8">;
// gfx8+ HasD16LoadStore D16_HI store family (DSInstructions.td
// §604-606). Stores bits [31:16] (B16_HI) or bits [23:16] (B8_HI)
// of the source VGPR to LDS — same VGPR/i32 source operand
// shape as their non-_HI siblings, so the canonical-table macro
// routes both encoding forks (gfx10 m0-based vs gfx11+ no-m0)
// through the same SemOp.
def : SalmonOp<DS_WRITE_B16_D16_HI, "DS_WRITE_B16_D16_HI">;
def : SalmonOp<DS_WRITE_B8_D16_HI, "DS_WRITE_B8_D16_HI">;
def : SalmonOp<DS_BPERMUTE_B32, "DS_BPERMUTE_B32">;
// ds_swizzle_b32 — wave-width-specific cross-lane shuffle. The
// handler refuses with `unsupportedShape` until the P6 rewrite
// lands (see the ds_swizzle_b32 row of hotswap/docs/wave-size-
// translation.md §5.3); the wave-size classifier (Phase 1.4.5)
// flags it as a Class 2 obstruction (wave-size-translation.md
// §6) in the cross-wave case.
def : SalmonOp<DS_SWIZZLE_B32, "DS_SWIZZLE_B32">;

// ---------------------------------------------------------------------
// MUBUF direct-to-LDS loads (distinct semantics from VGPR-dest loads)
// ---------------------------------------------------------------------
// LLVM only ships DWORD and DWORDX4 LDS pseudos — the DWORDX2_LDS SemOp
// stays unmapped until an LLVM pseudo exists for it.
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORD_LDS", "BUFFER_LOAD_DWORD_LDS">;
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORDX4_LDS", "BUFFER_LOAD_DWORDX4_LDS">;

// ---------------------------------------------------------------------
// MUBUF loads
// ---------------------------------------------------------------------
// gfx10- uses `BUFFER_LOAD_DWORD_*`; gfx11+ (vbuffer) uses the same
// pseudo name with a `_VBUFFER_` infix rather than a `B32` rename.
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORD", "BUFFER_LOAD_DWORD">;
defm : SalmonVBUF4<"BUFFER_LOAD_DWORD", "BUFFER_LOAD_DWORD">;
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORDX2", "BUFFER_LOAD_DWORDX2">;
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORDX3", "BUFFER_LOAD_DWORDX3">;
defm : SalmonMUBUF4<"BUFFER_LOAD_DWORDX4", "BUFFER_LOAD_DWORDX4">;
defm : SalmonMUBUF4<"BUFFER_LOAD_UBYTE", "BUFFER_LOAD_UBYTE">;
defm : SalmonMUBUF4<"BUFFER_LOAD_SBYTE", "BUFFER_LOAD_SBYTE">;
defm : SalmonMUBUF4<"BUFFER_LOAD_USHORT", "BUFFER_LOAD_USHORT">;
defm : SalmonMUBUF4<"BUFFER_LOAD_SSHORT", "BUFFER_LOAD_SSHORT">;
// D16 partial-write loads: the 16-bit (or 8-bit, sign/zero-extended
// to i16) datum lands in the lo half (`_D16`) or hi half
// (`_D16_HI`) of the destination VGPR, preserving the other 16
// bits (BUFInstructions.td:1155-1177, predicate
// `D16PreservesUnusedBits`). gfx10- uses the legacy MUBUF
// encodings; gfx11+/gfx1250 fork to VBUFFER (BUFInstructions.td
// 2610-2620 maps `_d16_*` mnemonics to the VBUFFER pseudos). Both
// encodings dispatch to the same partial-write handler in
// handle_mubuf.cpp via the SemOp; the `D16PreservesUnusedBits`
// semantics is enforced there by reading the prior dst and
// merging.
defm : SalmonMUBUF4<"BUFFER_LOAD_SHORT_D16", "BUFFER_LOAD_SHORT_D16">;
defm : SalmonVBUF4<"BUFFER_LOAD_SHORT_D16", "BUFFER_LOAD_SHORT_D16">;
defm : SalmonMUBUF4<"BUFFER_LOAD_SHORT_D16_HI", "BUFFER_LOAD_SHORT_D16_HI">;
defm : SalmonVBUF4<"BUFFER_LOAD_SHORT_D16_HI", "BUFFER_LOAD_SHORT_D16_HI">;
defm : SalmonMUBUF4<"BUFFER_LOAD_UBYTE_D16", "BUFFER_LOAD_UBYTE_D16">;
defm : SalmonVBUF4<"BUFFER_LOAD_UBYTE_D16", "BUFFER_LOAD_UBYTE_D16">;
defm : SalmonMUBUF4<"BUFFER_LOAD_UBYTE_D16_HI", "BUFFER_LOAD_UBYTE_D16_HI">;
defm : SalmonVBUF4<"BUFFER_LOAD_UBYTE_D16_HI", "BUFFER_LOAD_UBYTE_D16_HI">;
defm : SalmonMUBUF4<"BUFFER_LOAD_SBYTE_D16", "BUFFER_LOAD_SBYTE_D16">;
defm : SalmonVBUF4<"BUFFER_LOAD_SBYTE_D16", "BUFFER_LOAD_SBYTE_D16">;
defm : SalmonMUBUF4<"BUFFER_LOAD_SBYTE_D16_HI", "BUFFER_LOAD_SBYTE_D16_HI">;
defm : SalmonVBUF4<"BUFFER_LOAD_SBYTE_D16_HI", "BUFFER_LOAD_SBYTE_D16_HI">;

// ---------------------------------------------------------------------
// MUBUF stores
// ---------------------------------------------------------------------
defm : SalmonMUBUF4<"BUFFER_STORE_DWORD", "BUFFER_STORE_DWORD">;
defm : SalmonVBUF4<"BUFFER_STORE_DWORD", "BUFFER_STORE_DWORD">;
defm : SalmonMUBUF4<"BUFFER_STORE_DWORDX2", "BUFFER_STORE_DWORDX2">;
defm : SalmonMUBUF4<"BUFFER_STORE_DWORDX3", "BUFFER_STORE_DWORDX3">;
defm : SalmonMUBUF4<"BUFFER_STORE_DWORDX4", "BUFFER_STORE_DWORDX4">;
defm : SalmonMUBUF4<"BUFFER_STORE_BYTE", "BUFFER_STORE_BYTE">;
defm : SalmonMUBUF4<"BUFFER_STORE_SHORT", "BUFFER_STORE_SHORT">;

// ---------------------------------------------------------------------
// MUBUF atomics (non-return; handlers don't distinguish RTN yet)
// ---------------------------------------------------------------------
defm : SalmonMUBUF4<"BUFFER_ATOMIC_ADD", "BUFFER_ATOMIC_ADD">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_SUB", "BUFFER_ATOMIC_SUB">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_AND", "BUFFER_ATOMIC_AND">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_OR", "BUFFER_ATOMIC_OR">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_XOR", "BUFFER_ATOMIC_XOR">;
// Class 3 non-commutative atomics (hotswap/docs/wave-size-
// translation.md §6). The wave-size classifier flags them in
// the cross-wave case before
// dispatch ever reaches handle_mubuf.cpp's switch (which would
// otherwise refuse them via the default branch since they don't
// fit the AtomicRMW commutative model).
defm : SalmonMUBUF4<"BUFFER_ATOMIC_SWAP", "BUFFER_ATOMIC_SWAP">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_CMPSWAP", "BUFFER_ATOMIC_CMPSWAP">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_ADD_F32", "BUFFER_ATOMIC_ADD_F32">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_PK_ADD_BF16", "BUFFER_ATOMIC_PK_ADD_BF16">;
defm : SalmonMUBUF4<"BUFFER_ATOMIC_PK_ADD_F16", "BUFFER_ATOMIC_PK_ADD_F16">;
// gfx11+/gfx12 VBUFFER fork for the buffer atomics. The asm
// spelling on gfx11+/gfx1250 renames `BUFFER_ATOMIC_ADD` to
// `buffer_atomic_add_u32` (BUFInstructions.td:2789 declares
// `defm BUFFER_ATOMIC_ADD : MUBUF_Real_Atomic_gfx11_gfx12<0x035,
// "buffer_atomic_add_u32">`); LLVM MC keeps the legacy
// `BUFFER_ATOMIC_*` pseudo names but suffixes them with
// `_VBUFFER_<addressing>` to distinguish the gfx12 buffer-
// descriptor encoding from the legacy MUBUF one. The decoder in
// mubuf_addr.cpp explicitly recognises both encodings (keys on
// ParsedReg::Kind rather than operand position) so the existing
// BUFFER_ATOMIC_* SemOps and the AtomicRMW lowering in
// handle_mubuf.cpp's `sop >= BUFFER_ATOMIC_ADD &&
// sop <= BUFFER_ATOMIC_PK_ADD_F16` branch work unchanged for
// VBUFFER atomics. Without these entries the gfx1250 corpus
// (e.g. scope_discovery___sum_bitmatrix_rows refused on
// `buffer_atomic_add_u32 v0, v1, s[4:7], null offen`) would
// refuse with `unsupportedOpcode` despite the underlying SemOp
// being present.
//
// PK_ADD_BF16 / PK_ADD_F16 are intentionally omitted — they have
// no VBUFFER Real form in BUFInstructions.td (the gfx12 buffer
// packed-add fork uses a different mnemonic family); a stray
// VBUF4 entry would expand to AMDGPU enum values that don't
// exist and fail to compile.
defm : SalmonVBUF4<"BUFFER_ATOMIC_ADD", "BUFFER_ATOMIC_ADD">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_SUB", "BUFFER_ATOMIC_SUB">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_AND", "BUFFER_ATOMIC_AND">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_OR", "BUFFER_ATOMIC_OR">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_XOR", "BUFFER_ATOMIC_XOR">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_SWAP", "BUFFER_ATOMIC_SWAP">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_CMPSWAP", "BUFFER_ATOMIC_CMPSWAP">;
defm : SalmonVBUF4<"BUFFER_ATOMIC_ADD_F32", "BUFFER_ATOMIC_ADD_F32">;

// ---------------------------------------------------------------------
// AGPR moves
// ---------------------------------------------------------------------
def : SalmonOp<V_ACCVGPR_READ_B32_e64, "V_ACCVGPR_READ_B32">;
def : SalmonOp<V_ACCVGPR_WRITE_B32_e64, "V_ACCVGPR_WRITE_B32">;

// ---------------------------------------------------------------------
// MFMA shapes with a single TableGen pseudo per intrinsic.
// Register-class variants (`_vgprcd_`, `_mac_`) collapse to the base
// `_e64` via pseudoAlias, so we only list the base pseudo here.
// ---------------------------------------------------------------------
def : SalmonOp<V_MFMA_F32_16X16X16F16_e64, "V_MFMA_F32_16x16x16_F16">;
def : SalmonOp<V_MFMA_F32_32X32X8F16_e64, "V_MFMA_F32_32x32x8_F16">;
def : SalmonOp<V_MFMA_F32_16X16X4F32_e64, "V_MFMA_F32_16x16x4_F32">;
def : SalmonOp<V_MFMA_F32_32X32X1F32_e64, "V_MFMA_F32_32x32x1_F32">;
def : SalmonOp<V_MFMA_F32_32X32X2F32_e64, "V_MFMA_F32_32x32x2_F32">;
def : SalmonOp<V_MFMA_F32_4X4X1F32_e64, "V_MFMA_F32_4x4x1_F32">;
def : SalmonOp<V_MFMA_F32_16X16X1F32_e64, "V_MFMA_F32_16x16x1_F32">;
def : SalmonOp<V_MFMA_F32_32X32X4F16_e64, "V_MFMA_F32_32x32x4_F16">;
def : SalmonOp<V_MFMA_F32_16X16X4F16_e64, "V_MFMA_F32_16x16x4_F16">;
def : SalmonOp<V_MFMA_F32_4X4X4F16_e64, "V_MFMA_F32_4x4x4_F16">;
def : SalmonOp<V_MFMA_I32_16X16X32I8_e64, "V_MFMA_I32_16x16x32_I8">;
def : SalmonOp<V_MFMA_I32_32X32X16I8_e64, "V_MFMA_I32_32x32x16_I8">;
def : SalmonOp<V_MFMA_I32_32X32X4I8_e64, "V_MFMA_I32_32x32x4_I8">;
def : SalmonOp<V_MFMA_I32_16X16X4I8_e64, "V_MFMA_I32_16x16x4_I8">;
def : SalmonOp<V_MFMA_I32_4X4X4I8_e64, "V_MFMA_I32_4x4x4_I8">;
def : SalmonOp<V_MFMA_F32_16X16X8XF32_e64, "V_MFMA_F32_16x16x8_XF32">;
def : SalmonOp<V_MFMA_F32_32X32X4XF32_e64, "V_MFMA_F32_32x32x4_XF32">;
def : SalmonOp<V_MFMA_F32_32X32X2BF16_e64, "V_MFMA_F32_32x32x2_BF16">;
def : SalmonOp<V_MFMA_F32_16X16X2BF16_e64, "V_MFMA_F32_16x16x2_BF16">;
def : SalmonOp<V_MFMA_F32_4X4X2BF16_e64, "V_MFMA_F32_4x4x2_BF16">;
// Only the 16x16x16 and 32x32x8 1K shapes were in the legacy mnemonic
// table; the 4x4x4/16x16x4/32x32x4 1K variants are distinct intrinsics
// that the raiser does not (yet) model and therefore stay unmapped.
def : SalmonOp<V_MFMA_F32_16X16X16BF16_1K_e64, "V_MFMA_F32_16x16x16_BF16_1K">;
def : SalmonOp<V_MFMA_F32_32X32X8BF16_1K_e64, "V_MFMA_F32_32x32x8_BF16_1K">;
def : SalmonOp<V_MFMA_F32_16X16X32_BF16_e64, "V_MFMA_F32_16x16x32_BF16">;
def : SalmonOp<V_MFMA_F32_32X32X16_BF16_e64, "V_MFMA_F32_32x32x16_BF16">;
def : SalmonOp<V_MFMA_F32_16X16X32_F16_e64, "V_MFMA_F32_16x16x32_F16">;
def : SalmonOp<V_MFMA_F32_16X16X32_FP8_FP8_e64, "V_MFMA_F32_16x16x32_FP8_FP8">;
def : SalmonOp<V_MFMA_F32_16X16X32_FP8_BF8_e64, "V_MFMA_F32_16x16x32_FP8_BF8">;
def : SalmonOp<V_MFMA_F32_16X16X32_BF8_FP8_e64, "V_MFMA_F32_16x16x32_BF8_FP8">;
def : SalmonOp<V_MFMA_F32_16X16X32_BF8_BF8_e64, "V_MFMA_F32_16x16x32_BF8_BF8">;
def : SalmonOp<V_MFMA_F32_32X32X16_FP8_FP8_e64, "V_MFMA_F32_32x32x16_FP8_FP8">;
def : SalmonOp<V_MFMA_F32_32X32X16_FP8_BF8_e64, "V_MFMA_F32_32x32x16_FP8_BF8">;
def : SalmonOp<V_MFMA_F32_32X32X16_BF8_FP8_e64, "V_MFMA_F32_32x32x16_BF8_FP8">;
def : SalmonOp<V_MFMA_F32_32X32X16_BF8_BF8_e64, "V_MFMA_F32_32x32x16_BF8_BF8">;

// ---------------------------------------------------------------------
// MFMA F8F6F4: LLVM enumerates all 9 src0/src1 mantissa-type pairs as
// separate pseudos; the raiser handles them identically.
// ---------------------------------------------------------------------
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64, "V_MFMA_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64, "V_MFMA_F32_16x16x128_F8F6F4">;

def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64, "V_MFMA_SCALE_F32_16x16x128_F8F6F4">;

def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64, "V_MFMA_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64, "V_MFMA_F32_32x32x64_F8F6F4">;

def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;
def : SalmonOp<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64, "V_MFMA_SCALE_F32_32x32x64_F8F6F4">;

// ---------------------------------------------------------------------
// WMMA (gfx1250): twoaddr/threeaddr pseudo variants both mean the same
// semantic op to our raiser.
// ---------------------------------------------------------------------
def : SalmonOp<V_WMMA_F32_16X16X32_F16_w32_twoaddr, "V_WMMA_F32_16x16x32_F16">;
def : SalmonOp<V_WMMA_F32_16X16X32_F16_w32_threeaddr, "V_WMMA_F32_16x16x32_F16">;
// 16x16x4 WMMA, 32-bit f32 inputs and accumulator (gfx1250 RDNA4
// VOP3P opcode 0x05D, K=4 across 32 lanes -> per-lane <2 x f32>
// for A and B). Lifts to the gfx1250-only intrinsic
// amdgcn_wmma_f32_16x16x4_f32 in the same-target path; cross-
// target (gfx942) refuses loudly because no K=4 f32 -> MFMA
// decomposition exists in `wmma_lowering.cpp` yet.
def : SalmonOp<V_WMMA_F32_16X16X4_F32_w32_twoaddr, "V_WMMA_F32_16x16x4_F32">;
def : SalmonOp<V_WMMA_F32_16X16X4_F32_w32_threeaddr, "V_WMMA_F32_16x16x4_F32">;
// 16x16x32 WMMA, BF16 inputs, F32 accumulator (gfx1250 RDNA4 VOP3P
// opcode 0x062). Same fragment shape as the F16 variant — the only
// delta is the input element type, which routes to the bf16 MFMA
// intrinsic on gfx942 (mfma_f32_16x16x16bf16_1k) and to
// amdgcn_wmma_f32_16x16x32_bf16 when the target supports WMMA12.
def : SalmonOp<V_WMMA_F32_16X16X32_BF16_w32_twoaddr, "V_WMMA_F32_16x16x32_BF16">;
def : SalmonOp<V_WMMA_F32_16X16X32_BF16_w32_threeaddr, "V_WMMA_F32_16x16x32_BF16">;
// 16x16x64 WMMA, 8-bit element types (gfx1250 RDNA4 VOP3P opcodes
// 0x06a..0x06d). All four AB combinations share the per-lane
// fragment shape (A,B: <8 x i32>, C/D: <8 x f32>) and route through
// the gfx942 MFMA lowering with a per-variant intrinsic dispatch
// (see `runGroupPass` in wmma_lowering.cpp). Both `_twoaddr` and
// `_threeaddr` MC pseudo variants represent the same semantic op,
// mirroring the F16/BF16 mapping above.
def : SalmonOp<V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr, "V_WMMA_F32_16x16x64_FP8_FP8">;
def : SalmonOp<V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr, "V_WMMA_F32_16x16x64_FP8_FP8">;
def : SalmonOp<V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr, "V_WMMA_F32_16x16x64_FP8_BF8">;
def : SalmonOp<V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr, "V_WMMA_F32_16x16x64_FP8_BF8">;
def : SalmonOp<V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr, "V_WMMA_F32_16x16x64_BF8_FP8">;
def : SalmonOp<V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr, "V_WMMA_F32_16x16x64_BF8_FP8">;
def : SalmonOp<V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr, "V_WMMA_F32_16x16x64_BF8_BF8">;
def : SalmonOp<V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr, "V_WMMA_F32_16x16x64_BF8_BF8">;
// 16x16x64 WMMA, IU8 (signed/unsigned 8-bit integer inputs, i32
// accumulator; gfx1250 RDNA4 VOP3P opcode 0x072). The MC opcode
// sign-extension knobs travel through the `neg_lo` operand bits
// rather than the opcode itself, so there's a single SemOp for
// both signed and unsigned interpretations. On gfx942 this
// routes to mfma_i32_16x16x32_i8 (i64 packed A/B, <4 x i32>
// accumulator). Both `_twoaddr` and `_threeaddr` MC pseudo
// variants represent the same semantic op, mirroring the
// F16/BF16 / FP8/BF8 mappings above.
def : SalmonOp<V_WMMA_I32_16X16X64_IU8_w32_twoaddr, "V_WMMA_I32_16x16x64_IU8">;
def : SalmonOp<V_WMMA_I32_16X16X64_IU8_w32_threeaddr, "V_WMMA_I32_16x16x64_IU8">;
// ---------------------------------------------------------------------
// Scaled WMMA F8F6F4 (gfx1250 RDNA4 — VOP3PX2 paired form, real opcode
// 0x033 + 0x35 / 0x3a, FLATInstructions / VOP3PInstructions.td:1987).
// The 9 mantissa-pair pseudos (`f4_f4`, `f4_f6`, …, `f8_f8`) all share
// the same SemOp because the per-format element distinction (BF8 vs
// FP8 within f8, etc.) travels as the `matrix_a_fmt` /
// `matrix_b_fmt` named-immediate operands rather than the opcode
// suffix. Each mantissa-pair has both a `_twoaddr` (src2 tied to
// vdst) and `_threeaddr` (independent vdst) MC pseudo because the
// `WMMAInstGFX12` multiclass emits both — the canonicalization
// chain treats them as the same semantic op (matching the
// F8F6F4-MFMA collapse rule above and the `_twoaddr / _threeaddr`
// collapse rule used for the K=32 / K=64 WMMA family). The
// companion `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` intrinsic
// is gfx1250-only (`isGFX125xOnly`); cross-target gfx942 lift is a
// loud refusal in `handle_valu_vop3p.cpp` because the K=128 +
// matrix-fmt + scale-exponent decomposition into MFMA is not
// modelled in `wmma_lowering.cpp`. The non-paired
// `V_WMMA_F32_16X16X128_F8F6F4` family (no SCALE prefix) is not
// currently observed in the kerneldex corpus; it can be added
// alongside this entry when a kernel surfaces it.
// ---------------------------------------------------------------------
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
def : SalmonOp<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr, "V_WMMA_SCALE_F32_16x16x128_F8F6F4">;
// ---------------------------------------------------------------------
// VIMAGE TENSOR (gfx1250 RDNA4 — VIMAGE 0xc4 / 0xc5).
// The disassembler's MC opcodes are the `_gfx1250` reals
// (`TENSOR_LOAD_TO_LDS_d{2,4}_gfx1250`,
// `TENSOR_STORE_FROM_LDS_d{2,4}_gfx1250`,
// `MIMGInstructions.td:2115`); the canonicalization chain in
// `OpcodeMap::canonicalize` collapses each real onto its pseudo
// (`TENSOR_LOAD_TO_LDS_d{2,4}` / `TENSOR_STORE_FROM_LDS_d{2,4}`)
// via `buildMcToPseudoMap`. Both the `_d2` (up-to-2D) and `_d4`
// (up-to-4D) operand-shape variants share a SemOp because their
// semantic intent and their cross-target refusal contract are
// identical; `handleVIMAGE` uses `di.mnemonic` directly when it
// needs to discriminate (e.g., a future native-target intrinsic
// lowering that zero-fills D# group 2/3 for the `_d2` form).
// ---------------------------------------------------------------------
def : SalmonOp<TENSOR_LOAD_TO_LDS_d2, "TENSOR_LOAD_TO_LDS">;
def : SalmonOp<TENSOR_LOAD_TO_LDS_d4, "TENSOR_LOAD_TO_LDS">;
def : SalmonOp<TENSOR_STORE_FROM_LDS_d2, "TENSOR_STORE_FROM_LDS">;
def : SalmonOp<TENSOR_STORE_FROM_LDS_d4, "TENSOR_STORE_FROM_LDS">;
// ---------------------------------------------------------------------
// FLAT async global → LDS (gfx1250 — VFLAT 0x60-0x62 reals).
// The disassembler's MC opcodes are the `_gfx1250` (or
// `_SADDR_gfx1250`) reals declared by
// `VFLAT_Real_AllAddr_gfx1250` (FLATInstructions.td:2003-2018)
// off the `FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>` family
// (FLATInstructions.td:391-417). The canonicalization chain in
// `OpcodeMap::canonicalize` collapses each real onto its pseudo
// (`GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` / `_SADDR`
// sibling) via `buildMcToPseudoMap`; both the plain and SADDR
// variants per width share a SemOp because the lift differs
// only in the source-bank wiring (vaddr-only VGPR_64 vs
// saddr+voff SGPR_64+VGPR_32). `handleFLAT` uses `op.nSrcs()`
// (4 for plain, 5 for SADDR) to pick the operand-decoder path,
// exactly mirroring how `handleVIMAGE` discriminates
// `tensor_load_to_lds_d2` vs `_d4`.
// ---------------------------------------------------------------------
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B8, "GLOBAL_LOAD_ASYNC_TO_LDS_B8">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B8_SADDR, "GLOBAL_LOAD_ASYNC_TO_LDS_B8">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B32, "GLOBAL_LOAD_ASYNC_TO_LDS_B32">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR, "GLOBAL_LOAD_ASYNC_TO_LDS_B32">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B64, "GLOBAL_LOAD_ASYNC_TO_LDS_B64">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B64_SADDR, "GLOBAL_LOAD_ASYNC_TO_LDS_B64">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B128, "GLOBAL_LOAD_ASYNC_TO_LDS_B128">;
def : SalmonOp<GLOBAL_LOAD_ASYNC_TO_LDS_B128_SADDR, "GLOBAL_LOAD_ASYNC_TO_LDS_B128">;

// ---------------------------------------------------------------------
// FLAT VMEM prefetch (gfx1250 RDNA4 — VFLAT 0x05D, hint-class).
//
// The disassembler returns the gfx1250 reals
// `GLOBAL_PREFETCH_B8_gfx1250` (plain VGPR_64 vaddr) and
// `GLOBAL_PREFETCH_B8_SADDR_gfx1250` (SReg_64 + VGPR_32 SADDR;
// FLATInstructions.td:545-553 + FLATInstructions.td:3677-3678,
// `VFLAT_Real_AllAddr_gfx1250<0x05d>`). The canonicalization chain
// collapses each `_gfx1250` real onto its pseudo via
// `buildMcToPseudoMap`. The two addressing-mode pseudos share a
// single SemOp because they map onto the same intrinsic
// (`int_amdgcn_global_prefetch`, IntrinsicsAMDGPU.td:3211); the
// handler discriminates on `op.nSrcs()` (3 vs 4) the same way the
// GLOBAL_LOAD_ASYNC_TO_LDS_B* family above does.
//
// The matching `flat_prefetch_b8` (FLAT, not GLOBAL) is omitted —
// no kernel in the current corpus surfaces it. If one does, add a
// `FLAT_PREFETCH_B8{,_SADDR}` pair here (mapping to a separate
// SemOp, since the address space differs and the lift would use
// `int_amdgcn_flat_prefetch`).
// ---------------------------------------------------------------------
def : SalmonOp<GLOBAL_PREFETCH_B8, "GLOBAL_PREFETCH_B8">;
def : SalmonOp<GLOBAL_PREFETCH_B8_SADDR, "GLOBAL_PREFETCH_B8">;
