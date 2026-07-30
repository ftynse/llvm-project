//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Disassembles a kernel's ELF text section into a typed `DecodedInst` stream
// and builds an `llvm::Module` with a kernel function whose body is `ret void`.
// See `raiser.h` for the full raise pipeline (ELF ingestion -> decode ->
// per-format handlers -> post-raise analyses).
//
//===----------------------------------------------------------------------===//

#include "raiser.h"
#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/common/kernel-meta.h"
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/decoded-inst.h"
#include "raise_failure.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/parsed-reg.h"

#include "comgr.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "handlers.h"
#include "kernarg-layout.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/opcode-map.h"
#include "raise-context.h"
#include "reg-file.h"
#include "setpc-analysis.h"
#include "source-hidden-args.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Hardware threads-per-block maximum for the gfx9/CDNA wave64 targets the
// doubled dispatch scales up to. A source block that would exceed this once
// scaled by W_t / W_s cannot be doubled.
constexpr unsigned kTargetMaxThreadsPerBlock = 1024;

} // namespace

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise-context.h/cpp
// instructionWritesEXEC and the cross-wave gate live in wave-projection.h/cpp
// RaiseFailure + reasonString are in raise-failure.h/cpp

// ============================================================================
// Main raising function
// ============================================================================

static Expected<RaiseResult> raiseToIRImpl(
    llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
    llvm::StringRef KernelName, const KernelMeta &Meta, uint64_t KernelOffset,
    uint64_t KernelSize, uint64_t TextBaseAddress,
    llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
    llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
    bool EnableWaveNative, bool ForceThreadLoopProjection,
    bool SuppressC5ForThreadLoopRoute, bool ForceModrepDoubled,
    bool AssumeHipGlobalOffsetZero,
    llvm::ArrayRef<KernelSymbolExtent> FunctionExtents, RaiseStats *Stats) {
  RaiseResult Result;

  // Reject obviously-bad ISA inputs before reaching the MC stack -- an
  // empty or non-AMDGPU ISA string slips past `createMCSubtargetInfo`
  // (it returns a subtarget with no features) and only blows up later
  // in `createMCDisassembler` with an `llvm_unreachable`-flavoured
  // `LLVM ERROR: disassembly not yet supported for subtarget` that
  // aborts the process. Surface a structured failure instead.
  //
  // Callers may pass either the bare processor name (`gfx942`) or the
  // canonical AMDGPU ISA string (`amdgcn-amd-amdhsa--gfx942[:feat...]`).
  // Defer to Comgr's `parseTargetIdentifier` for the canonical form (it
  // handles the dash-separated Arch/Vendor/OS/Environ/Processor split
  // and the `:sramecc+/-:xnack+/-` feature suffix in one place);
  // `MCSubtargetInfo` only accepts the bare processor name, so we
  // forward `Ident.Processor` to the MC stack below.
  auto NormalizeIsa = [](StringRef Iso) -> StringRef {
    TargetIdentifier Ident;
    if (parseTargetIdentifier(Iso, Ident) == AMD_COMGR_STATUS_SUCCESS)
      return Ident.Processor;
    // Bare processor name (e.g. `gfx942`) -- not a 5-component canonical
    // ISA string. Return as-is and let the AMDGPU validator below decide.
    return Iso;
  };
  StringRef SourceCpu = NormalizeIsa(SourceIsa);
  if (SourceIsa.empty() ||
      AMDGPU::parseArchAMDGCN(SourceCpu) == AMDGPU::GK_NONE) {
    return RaiseFailure::badInput("source ISA '" + SourceIsa +
                                  "' does not name an AMDGPU GPU");
  }

  // Same normalisation for the target-side override (--target-isa on
  // raise_cli, or programmatic CompilationTargetIsa). Empty means
  // "translate in place -- reuse the source profile".
  StringRef TargetCpu = CompilationTargetIsa.empty()
                            ? CompilationTargetIsa
                            : NormalizeIsa(CompilationTargetIsa);

  // NOTE. The `HSA_HOTSWAP_WAVE_NATIVE=1` process-environment override
  // that lived here through the empirical graduation sweep (pre-
  // 2026-04-21) has been removed now that `enableWaveNative`
  // defaults to `true`. The override served one purpose -- flipping
  // every call-site's projection without editing each caller --
  // which is no longer needed. Keeping it around would subtly
  // break the opt-OUT path: `--disable-wave-native` on
  // `raise_cli` (and `enableWaveNative=false` on programmatic
  // callers) are how lit fixtures and operators pin MODREP for
  // projection-specific debugging, and a silent env-var that
  // unconditionally flips to WaveNative would defeat that. If
  // future evidence needs a global toggle, add a proper
  // `PipelineConfig` field rather than re-introducing the env var.

  Expected<MCState> MCStateOrErr = initMCState(SourceCpu);
  if (!MCStateOrErr) {
    return MCStateOrErr.takeError();
  }

  MCState Mc = std::move(*MCStateOrErr);
  ISAProfile Isa = ISAProfile::fromSubtarget(*Mc.SubtargetInfo);
  // When the caller does not specify a distinct compilation target we raise
  // in place and reuse the source profile; otherwise we spin up a throwaway
  // MCSubtargetInfo just to snapshot the target's feature bits.
  ISAProfile TargetIsa = Isa;
  std::unique_ptr<MCSubtargetInfo> TargetSti;
  if (!TargetCpu.empty()) {
    Expected<std::unique_ptr<MCSubtargetInfo>> StiOrErr =
        buildSubtargetInfo(*Mc.Target, TargetCpu);
    if (!StiOrErr)
      return StiOrErr.takeError();

    TargetSti = std::move(*StiOrErr);
    TargetIsa = ISAProfile::fromSubtarget(*TargetSti);
  }
  if (!Isa.hasValidWaveSize())
    return RaiseFailure::internalFailure(
        "transpiler: source ISA profile has unsupported wave size " +
        Twine(Isa.waveSize()));
  if (!TargetIsa.hasValidWaveSize())
    return RaiseFailure::internalFailure(
        "transpiler: target ISA profile has unsupported wave size " +
        Twine(TargetIsa.waveSize()));

  // LLVMContext + common IR types are created here (earlier than they used
  // to be) so the WaveProjection has access to i32/i64 before the cross-
  // wave gate runs. The module is still created lazily in Phase 2 so
  // early-return paths (pre-translation aborts) don't leave behind a
  // half-built module.
  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  auto *I32Ty = Type::getInt32Ty(C);
  auto *I64Ty = Type::getInt64Ty(C);

  // Projection choice.
  //
  // `ModuloReplicationProjection` is the long-standing default: it fans
  // each target lane onto `lane_id mod W_src` of the source EXEC mask
  // and truncates cross-wave ballots to source width. Correct under
  // the wave-size-obliviousness theorem (hotswap/docs/wave-size-
  // translation.md sec. 6); insufficient for kernels whose WMMA -> MFMA
  // redistribute / collect pipeline needs hardware EXEC = -1 on the
  // upper half of the Wave64 target (lanes 32..63 would otherwise
  // never update their MFMA destination VGPRs -- see the file-header
  // comment in `wmma-lowering.cpp`).
  //
  // `WaveNativeProjection` is the opt-in alternative for wave32
  // source -> wave64 target. Its `emitInitialExec` calls
  // `@llvm.amdgcn.init_whole_wave` at kernel entry to force hardware
  // EXEC = -1 for the whole kernel body while saving the original
  // per-lane active mask into the (widened) EXEC alloca; every VGPR
  // write / memory store / LDS op already routes through
  // `emitUnderExec`, which rematerialises the per-lane predicate at
  // each side-effect site. The direction gate inside the
  // `WaveNativeProjection` constructor enforces that this projection
  // is only instantiated when `isa.isWave32() && !targetIsa.isWave32()`
  // -- other directions fatal-error loudly to prevent a decider bug
  // from silently picking an unsupported shape.
  //
  // Phantom-lane fallback to MODREP.  WaveNative's `init_whole_wave`
  // sets hardware EXEC = -1 and relies on SPE `emitUnderExec`
  // diamonds (gated by `saved_exec`) to keep inactive source lanes
  // from committing side effects.  That model is correct when every
  // target-wavefront lane has a source-kernel workitem -- i.e. when
  // the HSACO's `max_flat_workgroup_size` is at least
  // `targetWaveSize` so every launch fills the target wave.  When
  // `max_flat_workgroup_size < targetWaveSize` (the phantom-lane
  // regime, e.g. Triton's `num_warps=1` kernels whose source WG is
  // 32 on wave32 compiled for a wave64 target), the "extra" target
  // lanes have no source workitem: their `workitem.id.x()` is their
  // hardware lane index (e.g. 32..63 for a 32-thread block on
  // wave64), their VGPRs hold undef / dispatcher state, and their
  // cross-lane ops (`ds_bpermute`, `ds_swizzle`, `permlane*`) read
  // from / contribute to actively-masked source lanes with
  // undef-derived values -- producing addresses that fault on
  // subsequent SPE-gated loads (the active lane's pointer
  // arithmetic picks up undef data through a cross-lane op, then
  // the gated load fires with that poisoned address).  Empirically
  // surfaced by `compare_correctness`'s `matmul_fp16` /
  // `matmul_fp16_16x16` Triton recipes (HIP error 700 on every
  // shape under WaveNative; bumping `num_warps` to 2 fills the
  // target wavefront and eliminates the fault, confirming the
  // phantom-lane attribution).
  //
  // `ModuloReplicationProjection` leaves hardware EXEC at the
  // dispatcher's boot state (the source-wave-sized active mask,
  // with the target wave's upper lanes inactive) and uses
  // `lane_id mod W_src` to project the target mask onto the source
  // EXEC alloca.  Under MODREP, phantom lanes are hardware-inactive
  // for the entire kernel body -- every ISA instruction (VALU,
  // cross-lane, memory, control flow) is HW-EXEC-masked -- so
  // undef-VGPR contamination can't escape into active lanes.  The
  // trade-off is that MODREP cannot express WMMA -> MFMA layout
  // transposes that need all 64 target lanes active (see
  // `wmma-lowering.cpp`); those kernels will refuse at lift time
  // rather than silently running wrong.  That's the principled
  // outcome for the phantom-lane regime.
  const bool PhantomLaneRegime =
      Meta.MaxFlatWorkgroupSize > 0 &&
      static_cast<unsigned>(Meta.MaxFlatWorkgroupSize) < TargetIsa.waveSize();
  const bool UseThreadLoop = ForceThreadLoopProjection;
  const bool CrossWidenWave32To64 = Isa.isWave32() && !TargetIsa.isWave32();
  // ModRepDoubledDispatchProjection is selected only via the C5 y/z-refusal
  // upgrade retry (or an explicit --force-modrep-doubled), so it is a forced
  // route just like ThreadLoop; it takes precedence over WaveNative.
  const bool UseModrepDoubled =
      !UseThreadLoop && ForceModrepDoubled && CrossWidenWave32To64;
  const bool UseWaveNative = !UseThreadLoop && !UseModrepDoubled &&
                             EnableWaveNative && CrossWidenWave32To64 &&
                             !PhantomLaneRegime;

  // Size gate for the doubled dispatch: the runtime scales the block by
  // W_t / W_s along x, so the scaled flat size must not exceed the target's
  // hardware threads-per-block maximum.
  if (UseModrepDoubled) {
    const unsigned Factor = TargetIsa.waveSize() / Isa.waveSize();
    const unsigned SourceFlat =
        Meta.MaxFlatWorkgroupSize > 0 ? Meta.MaxFlatWorkgroupSize : 1024;
    if (SourceFlat * Factor > kTargetMaxThreadsPerBlock) {
      std::string Detail =
          (Twine("ModRepDoubledDispatchProjection needs to launch ") +
           Twine(SourceFlat * Factor) +
           " threads/block (source max_flat_workgroup_size " +
           Twine(SourceFlat) + " scaled by " + Twine(Factor) +
           ") but the target hardware limit is " +
           Twine(kTargetMaxThreadsPerBlock) +
           "; refuse rather than truncate the block. See "
           "hotswap/docs/modrep-predicate-chain.md sec. 10.")
              .str();
      errs() << "transpiler: pre-translation abort: " << Detail << "\n";
      return RaiseFailure::crossWavePredicateChain(KernelName, Detail);
    }
  }

  std::unique_ptr<WaveProjection> ProjectionPtr;
  if (UseThreadLoop) {
    ProjectionPtr =
        std::make_unique<ThreadLoopProjection>(Isa, TargetIsa, I32Ty, I64Ty);
    errs() << "transpiler: kernel '" << KernelName
           << "' selected ThreadLoopProjection (analysis-triggered "
              "cross-widen route; writelane/readlane rewrite may be "
              "disabled by the retry caller)\n";
  } else if (UseModrepDoubled) {
    ProjectionPtr = std::make_unique<ModRepDoubledDispatchProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
    errs() << "transpiler: kernel '" << KernelName
           << "' selected ModRepDoubledDispatchProjection (doubled dispatch "
              "along x; each target wave hosts one source wave with replica "
              "upper lanes)\n";
  } else if (UseWaveNative) {
    ProjectionPtr =
        std::make_unique<WaveNativeProjection>(Isa, TargetIsa, I32Ty, I64Ty);
  } else {
    ProjectionPtr = std::make_unique<ModuloReplicationProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
  }
  ProjectionPtr->setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);
  WaveProjection &Projection = *ProjectionPtr;

  // Record the doubled-dispatch requirement so the launch runtime scales
  // exactly this kernel's dispatch (threaded through the transpile result and
  // the loader). Non-doubled projections leave dim=-1 / factor=1.
  if (Projection.usesDoubledDispatch()) {
    Result.DoubledDispatchDim =
        static_cast<int>(Projection.doubledDispatchDim());
    Result.DoubledDispatchFactor = Projection.doubledDispatchFactor();
  }

  if (!UseThreadLoop && !UseModrepDoubled && EnableWaveNative &&
      PhantomLaneRegime && Isa.isWave32() && !TargetIsa.isWave32()) {
    // Log the fallback so operators can trace which kernels moved to
    // MODREP and why.  A regression that silently flips WaveNative's
    // selection on a phantom-lane kernel would then (re-)produce the
    // HIP-700 miscompile this fallback guards against.
    errs() << "transpiler: kernel '" << KernelName
           << "' is in phantom-lane regime (max_flat_workgroup_size="
           << Meta.MaxFlatWorkgroupSize
           << " < target wavefront width=" << TargetIsa.waveSize()
           << "); falling back to ModuloReplicationProjection even "
              "though enableWaveNative=true, so phantom target lanes "
              "stay hardware-inactive and their undef-VGPR state "
              "cannot contaminate active-lane pointer arithmetic via "
              "cross-lane ops. See the block comment above in "
              "`raiser.cpp` for the full rationale.\n";
  }

  // Build opcode -> CanonicalOp map from MCInstrInfo
  OpcodeMap OpcMap;
  OpcMap.build(*Mc.InstrInfo);

  // The startup EXEC-attribute coverage invariant (every MC opcode that
  // implicitly defines EXEC maps to a CanonicalOp marked
  // routesExecThroughStoreExec) is wired once the EXEC-writing handlers land;
  // it cannot hold while the handler set is still being filled in.

  // ==== Phase 1: Disassemble + identify block boundaries ====
  //
  // The decode loop (and its two LLVM-drift guards) lives in decode.cpp so
  // this function stays focused on IR emission. decodeKernel returns a
  // linearised instruction stream + the set of CFG block-start offsets.
  if (KernelSize != 0 && KernelSize > UINT64_MAX - KernelOffset)
    return RaiseFailure::internalFailure(
        "transpiler: kernel decode extent overflows");

  const uint64_t KernelEndOffset =
      KernelSize == 0 ? 0 : KernelOffset + KernelSize;
  Expected<DecodeResult> DecodedOrErr = decodeKernel(
      Mc, OpcMap, ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
      KernelOffset, KernelEndOffset);
  if (!DecodedOrErr)
    return DecodedOrErr.takeError();
  DecodeResult Decoded = std::move(*DecodedOrErr);
  auto &Insts = Decoded.Insts;
  auto &BlockStarts = Decoded.BlockStarts;

  // ==== Phase 1.1: s_set_pc_i64 analysis ====
  // s_set_pc classification and the extra block leaders it discovers only
  // matter once control flow lands; a branchless kernel has no set-PC sites,
  // so an empty analysis suffices here.
  SetPcAnalysis SetpcAnalysis;
  bool FollowedOutOfExtentCallee = false;

  if (Stats)
    Stats->TotalCount = static_cast<int>(Insts.size());

  // The SPE (SIMT Predicated Execution) EXEC-writer safety gate lands with the
  // first EXEC-writing handler; the scalar-move path here writes no EXEC.

  // ==== Phase 2: Build LLVM IR module + function ====
  // LLVMContext + i32/i64 were created earlier for the WaveProjection.
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions Opts;
  std::unique_ptr<TargetMachine> Tm(Mc.Target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"),
      CompilationTargetIsa.empty() ? SourceIsa : CompilationTargetIsa, "", Opts,
      Reloc::PIC_));
  if (!Tm) {
    errs() << "transpiler: Failed to create TargetMachine\n";
    return RaiseFailure::targetMachineCreationFailed();
  }
  M.setDataLayout(Tm->createDataLayout());

  auto *VoidTy = Type::getVoidTy(C);
  auto *I1Ty = Type::getInt1Ty(C);
  auto *I8Ty = Type::getInt8Ty(C);

  // Build function signature: a single opaque
  // `ptr byref([N x i8]) align 16` placeholder whose only job is to
  // make the AMDGPU backend emit `kernarg_segment_size = N` and
  // `kernarg_segment_align = 16` in the lifted kernel's KD/metadata,
  // so the runtime's kernarg buffer reaches the kernel intact and
  // the metadata reports the AMDGPU ABI's 16-byte minimum.
  //
  // The handlers do NOT read this argument -- kernarg loads lift to
  // GEP+load against `amdgcn_kernarg_segment_ptr` and let the AMDGPU
  // backend re-select `s_load_*` against the kernarg segment. The
  // typed source-ABI signature (ptr addrspace(1) / i32 / i64 / per-
  // dword aggregate split) is therefore unnecessary on the lifted
  // side.
  //
  // Why `byref` + `align`: AMDGPULowerKernelArguments consults the
  // `align` parameter attribute only for byref kernel args (see
  // `MaybeAlign ParamAlign = IsByRef ? Arg.getParamAlign() :
  // std::nullopt;` in LLVM's `AMDGPULowerKernelArguments.cpp`). For
  // a non-byref `[N x i8]` arg, the IR-level alignment is the
  // type's natural alignment (1 byte), and the YAML metadata's
  // `.kernarg_segment_align` field reports a smaller value than the
  // ABI's 16-byte minimum. Using `byref` with an explicit
  // `align(16)` lets the backend honour the alignment without
  // forcing a vector or padding type, and the byref semantics --
  // "pointer to an aggregate that's actually placed in the kernarg
  // segment" -- match the placeholder's intent: a stable region of
  // `kernarg_segment_size` bytes that handlers don't need a typed
  // view of.
  //
  // AMDGPULowerKernelArguments skips load emission for arguments
  // that are `use_empty()` but still bumps the cumulative arg
  // offset, so the unused placeholder still contributes to
  // `kernarg_segment_size`.
  //
  // Test back-reference: every lit fixture under `lit_tests/` pins
  // either a `ptr addrspace(4)` GEP shape or an addrspace(1) global
  // GEP shape against the segment_ptr intrinsic -- none of them rely
  // on the kernarg buffer being a typed Function argument list.
  SmallVector<Type *, 1> ParamTypes;
  KernargLayout Kernargs;
  int ParamIdx = 0;
  Type *KernargByrefTy = nullptr;
  if (Meta.KernargSegmentSize > 0) {
    KernargByrefTy =
        ArrayType::get(I8Ty, static_cast<uint64_t>(Meta.KernargSegmentSize));
    ParamTypes.push_back(PointerType::get(C, /*addrspace=*/4));
    ParamIdx = 1;
  }
  Kernargs.ImplicitArgsBase = Meta.implicitArgsBase();
  Kernargs.Args = Meta.Args;
  Kernargs.KernargSegmentSize = Meta.KernargSegmentSize;

  auto *FuncTy = FunctionType::get(VoidTy, ParamTypes, false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // Attach `byref([N x i8])` + `align(16)` to the placeholder kernarg
  // pointer. AMDGPULowerKernelArguments only honours param-align on
  // byref kernel args, so this combo is what gets the lifted KD's
  // kernarg-segment alignment to the AMDGPU ABI's 16-byte minimum
  // without forcing an aggregate / vector type for the parameter.
  if (KernargByrefTy != nullptr) {
    F->addParamAttr(0, Attribute::getWithByRefType(C, KernargByrefTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, Align(16)));
  }
  // The kernel-entry v0 holds the packed workitem id, x[0:9] | y[10:19] |
  // z[20:29]. ENABLE_VGPR_WORKITEM_ID (COMPUTE_PGM_RSRC2 bits [12:11]) records
  // how many of x/y/z the source enabled: 0 -> X, 1 -> X+Y, 2 -> X+Y+Z. The
  // packed v0 seed below reconstructs exactly those fields; seeding only X left
  // every threadIdx.y / threadIdx.z read folding to 0.
  unsigned WorkitemIdCnt =
      (Meta.ComputePgmRsrc2 >>
       llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) &
      ((1u << llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_WIDTH) -
       1u);
  unsigned NumWorkitemDims = WorkitemIdCnt >= 2 ? 3u : WorkitemIdCnt + 1u;
  {
    // Pin the workgroup size to exactly what the source kernel declared, so
    // the backend lays out LDS / workitem IDs the same way the original
    // gfx1250 binary did.
    int MaxWg =
        Meta.MaxFlatWorkgroupSize > 0 ? Meta.MaxFlatWorkgroupSize : 1024;
    if (Projection.usesDoubledDispatch()) {
      // The runtime launches this block scaled by the doubled-dispatch factor
      // along x. `amdgpu-flat-work-group-size` must advertise the scaled size
      // or ROCR/HIP would reject the larger launch as exceeding the declared
      // bound; the in-kernel workgroup-size query is virtualized back to the
      // source size via source-hidden-args, so kernel logic still sees MaxWg.
      MaxWg *= static_cast<int>(Projection.doubledDispatchFactor());
      // IR-level breadcrumb recording the doubled dimension and factor (e.g.
      // "x2") for offline inspection and the raise_cli lit tests. This is not
      // the runtime signal: the launch runtime learns the doubled dim/factor
      // from the transpile result (RaiseResult -> comgr result info fields ->
      // loader), because this function attribute does not survive to the kernel
      // descriptor metadata. See hotswap/docs/modrep-predicate-chain.md
      // sec. 10.
      assert(Projection.doubledDispatchDim() < 3 &&
             "doubled dispatch dim must be x/y/z");
      const char DimChar = "xyz"[Projection.doubledDispatchDim()];
      F->addFnAttr("hotswap-modrep-doubled-dispatch",
                   std::string(1, DimChar) +
                       std::to_string(Projection.doubledDispatchFactor()));
    }
    F->addFnAttr("amdgpu-flat-work-group-size",
                 std::to_string(MaxWg) + "," + std::to_string(MaxWg));

    // Deliberately do NOT set "amdgpu-waves-per-eu".  Pinning occupancy
    // constrains register allocation and caused spurious VGPR spills for
    // wide kernels (e.g. the Triton 128x128 matmul on gfx942), which then
    // triggered memory faults because our raised IR is register-pressure
    // heavy compared to a from-source compile.  Letting the backend choose
    // occupancy freely keeps register pressure safe.
    // TODO(gfx1250->gfx942): revisit once the raiser emits tighter IR; we may
    // want to propagate the source kernel's waves-per-eu for parity.

    // The hotswap caller still launches with the source kernel's host-side
    // kernarg buffer.  Hotswap materialises every source-visible value either
    // as a normal formal parameter, as source-ABI preloaded SGPR state seeded
    // explicitly in IR below, or as an intrinsic for architected dispatch
    // state.  Suppress backend-invented implicit kernarg slots so the emitted
    // descriptor keeps the source kernarg size instead of appending a
    // target-default hidden-arg block that the host never populated.
    F->addFnAttr("amdgpu-no-cluster-id-x");
    F->addFnAttr("amdgpu-no-cluster-id-y");
    F->addFnAttr("amdgpu-no-cluster-id-z");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-dispatch-id");
    // Do not suppress dispatch-ptr: source hidden-arg synthesis materialises
    // values such as hidden_group_size_* and hidden_block_count_* from the
    // target dispatch packet, because the lifted HSACO intentionally does not
    // ask HIP to append source-ABI hidden args after the opaque kargs blob.
    F->addFnAttr("amdgpu-no-heap-ptr");
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-implicitarg-ptr");
    F->addFnAttr("amdgpu-no-lds-kernel-id");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-queue-ptr");
    F->addFnAttr("amdgpu-no-workitem-id-x");
    // Only suppress the Y/Z workitem-id fields the source did not enable. The
    // packed v0 seed uses workitem.id.{y,z} for 2-D/3-D blocks; a stale "no"
    // attribute would pin ENABLE_VGPR_WORKITEM_ID at 0 so the backend never
    // loads those fields and threadIdx.y/z would read garbage.
    if (NumWorkitemDims < 2)
      F->addFnAttr("amdgpu-no-workitem-id-y");
    if (NumWorkitemDims < 3)
      F->addFnAttr("amdgpu-no-workitem-id-z");
    F->addFnAttr("uniform-work-group-size", "true");
  }

  // Propagate static LDS allocation from the source kernel descriptor.
  //
  // The raiser's `ds_write_b128` / `ds_load_b128` / `ds_bpermute` emit
  // pointer-arithmetic into `addrspace(3)` DIRECTLY (via `inttoptr i64
  // to ptr addrspace(3)`), without declaring an LDS `GlobalVariable`.
  // LLVM's AMDGPU backend derives `group_segment_fixed_size` from
  // addrspace(3) GlobalVariables plus the `amdgpu-lds-size` function
  // attribute (see `AMDGPUMachineFunctionInfo` -- `LDSSizeRange.first`
  // is read from the attr), so a raised kernel that only manipulates
  // addrspace(3) via int-to-ptr conversion and never sets the attr
  // gets `group_segment_fixed_size: 0` in the emitted HSACO.  The
  // hardware then treats every LDS op as out-of-segment and returns
  // zero / drops writes.  This silently miscompiled every lifted
  // kernel with a non-trivial LDS round-trip, most visibly Triton's
  // `matmul_fp16` (mode-5 B-only-varying input returned all zeros
  // because the cross-thread LDS fragment shuffle read from an
  // uninitialised segment; see matrix-translation.md sec. 12.4 for the
  // bisection).
  //
  // We mirror the source's `.group_segment_fixed_size` by setting the
  // per-function `amdgpu-lds-size` attribute in the source-declared
  // range.  The attribute takes "min,max" -- we pass the same value
  // for both since the source's static size is known exactly.
  if (Meta.GroupSegmentFixedSize > 0) {
    std::string SizeStr = std::to_string(Meta.GroupSegmentFixedSize);
    F->addFnAttr("amdgpu-lds-size", SizeStr + "," + SizeStr);
  }

  if (ParamIdx > 0)
    F->getArg(0)->setName("kargs");

  Function *FnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *FnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  Function *FnDispatchPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_dispatch_ptr);
  Function *FnKargPtr = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_kernarg_segment_ptr);
  // Build the source-ISA user-SGPR ABI from the kernel descriptor.
  // Phase 4 seeding and handler-side ABI-sensitive decoding (e.g.
  // handle_smem's kernarg-pointer detection) both key off this layout.
  UserSgprLayout UserSgprLayout;
  if (llvm::Error LayoutErr = UserSgprLayout::tryFromKernelMeta(
          Meta, Isa, SourceIsa, UserSgprLayout)) {
    std::string UserSgprFailureDetail =
        llvm::toStringWithoutConsuming(LayoutErr);
    if (!UserSgprFailureDetail.empty())
      llvm::errs() << UserSgprFailureDetail << "\n";
    return std::move(LayoutErr);
  }
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo) &&
      Meta.hasNonDisabledClusterDims()) {
    const std::array<uint32_t, 3> &Dims = *Meta.ClusterDims;
    return RaiseFailure::unsupportedSourceClusterDims(
        KernelName,
        ".cluster_dims=[" + Twine(Dims[0]) + "," + Twine(Dims[1]) + "," +
            Twine(Dims[2]) +
            "] requires real TTMP6 cluster workgroup state; the current "
            "HotSwap ABI model only supports disabled source clusters");
  }
  // ==== Phase 3: Create basic blocks ====
  // `blockStarts` is a std::set (see decode.h) so it iterates in
  // ascending source-address order, giving deterministic BB labels.
  // `offsetToBB` is a DenseMap and intentionally unordered; for the
  // thread-loop entry BB we need the lowest-address BB as InsertBefore
  // (so the entry sorts above the kernel body in IR), which we capture
  // explicitly during the create loop.
  llvm::DenseMap<uint64_t, BasicBlock *> OffsetToBb;
  BasicBlock *FirstBodyBb = nullptr;
  for (uint64_t Addr : BlockStarts) {
    BasicBlock *Bb =
        BasicBlock::Create(C, "bb_0x" + utohexstr(Addr - KernelOffset), F);
    OffsetToBb[Addr] = Bb;
    if (!FirstBodyBb)
      FirstBodyBb = Bb;
  }
  // The register-seeding block must be a predecessor-free entry block that
  // control-flows into the kernel's real start (KernelOffset). Normally the
  // KernelOffset block is itself the lowest-addressed block, so it can serve as
  // the entry directly. But when an out-of-extent callee was merged, a helper
  // block at a lower offset would otherwise become the LLVM entry (BlockStarts
  // iterates ascending) yet has predecessors (the caller's branch into it),
  // violating the verifier. In that case (as in the thread-loop case) use a
  // dedicated "entry" block inserted before all body blocks and branch it to
  // KernelOffset, so the seeding is separate from -- and never mis-merged into
  // -- the body blocks.
  bool UseDedicatedEntry = UseThreadLoop || FollowedOutOfExtentCallee;
  BasicBlock *EntryBb = UseDedicatedEntry
                            ? BasicBlock::Create(C, "entry", F, FirstBodyBb)
                            : OffsetToBb[KernelOffset];

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(EntryBb);

  AllocaRegFile Regs;
  Regs.init(B, I32Ty, I1Ty, Isa, *Mc.RegInfo, Projection);

  // Seed kernel-entry SGPR state from the descriptor-derived user-SGPR ABI.
  //
  // Crucial invariant: never hardcode SGPR indices. Kernarg preload and
  // enable_sgpr_* toggles legally move the kernarg pointer and workgroup-id
  // SGPRs away from s[0:1]/s2/s3. Hardcoding those indices mis-seeds entry
  // state and turns real source values into undef reads on the JIT path.
  //
  // Seed ABI-provided entry pointers with the matching AMDGPU intrinsics. The
  // source descriptor's dispatch_ptr bit means the corresponding SGPR pair
  // holds the AQL dispatch packet base, and source SMEM may legally load
  // through it just like it loads through kernarg_segment_ptr.
  if (UserSgprLayout.DispatchPtrSgpr >= 0) {
    Regs.storeSGPR64(B, UserSgprLayout.DispatchPtrSgpr,
                     B.CreateCall(FnDispatchPtr, {}, "dispatch_ptr"));
  }
  if (UserSgprLayout.KernargSegmentPtrSgpr >= 0) {
    Regs.storeSGPR64(B, UserSgprLayout.KernargSegmentPtrSgpr,
                     B.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
  }
  if (UserSgprLayout.WorkgroupIdXSgpr >= 0) {
    Regs.storeSGPR32(B, UserSgprLayout.WorkgroupIdXSgpr,
                     B.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
  }
  if (UserSgprLayout.WorkgroupIdYSgpr >= 0) {
    Regs.storeSGPR32(B, UserSgprLayout.WorkgroupIdYSgpr,
                     B.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
  }
  // Hidden-arg remaps use the ABI version the backend will emit for this
  // module. If target emission starts pinning a module flag, thread that value
  // here instead of relying on LLVM's default.
  unsigned TargetCodeObjectVersion =
      AMDGPU::getDefaultAMDHSACodeObjectVersion();
  auto EmitPreloadedKernargDword = [&](IRBuilder<> &SeedB,
                                       int ByteOffset) -> Expected<Value *> {
    SourceHiddenArgContext HiddenCtx{C,
                                     M,
                                     SeedB,
                                     I8Ty,
                                     I32Ty,
                                     I64Ty,
                                     Meta.Args,
                                     AssumeHipGlobalOffsetZero,
                                     TargetCodeObjectVersion};
    if (Projection.usesDoubledDispatch()) {
      HiddenCtx.DoubledDispatchDim =
          static_cast<int>(Projection.doubledDispatchDim());
      HiddenCtx.DoubledDispatchFactor = Projection.doubledDispatchFactor();
    }
    SourceHiddenArgValue Hidden = emitSourceHiddenDword(HiddenCtx, ByteOffset);
    if (Hidden.Matched && Hidden.Value)
      return Hidden.Value;

    if (Hidden.Matched) {
      return RaiseFailure::preloadedHiddenArgFailure(KernelName, ByteOffset,
                                                     Hidden.FailureDetail);
    }

    if (Kernargs.ImplicitArgsBase > 0 &&
        ByteOffset >= Kernargs.ImplicitArgsBase) {
      // Strict mode (a pipeline option that refuses implicit-arg preload
      // instead of falling back to an implicitarg_ptr load) is not plumbed
      // through this entry point yet; take the non-strict fallback.
      if (/*isStrictMode()=*/false) {
        return RaiseFailure::preloadedImplicitArgFailure(KernelName,
                                                         ByteOffset);
      }

      Function *FnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_implicitarg_ptr);
      Value *ImplPtr =
          SeedB.CreateCall(FnImplicitArgPtr, {}, "preload_implicitarg_ptr");
      int64_t ImplOffset = ByteOffset - Kernargs.ImplicitArgsBase;
      Value *Gep = ImplOffset == 0
                       ? ImplPtr
                       : SeedB.CreateInBoundsGEP(I8Ty, ImplPtr,
                                                 SeedB.getInt64(ImplOffset),
                                                 "preload_impl_gep");
      return SeedB.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_impl_dw");
    }

    Value *SegPtr = SeedB.CreateCall(FnKargPtr, {}, "preload_kernarg_ptr");
    Value *Gep = SeedB.CreateInBoundsGEP(
        I8Ty, SegPtr, SeedB.getInt64(ByteOffset), "preload_gep");
    return SeedB.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_dw");
  };
  // Kernarg preload SGPRs carry dwords copied by hardware from the kernarg
  // segment before kernel entry. Materialize the same dwords by loading
  // through `amdgcn_kernarg_segment_ptr` so the AMDGPU backend handles the
  // ABI lowering uniformly: the GEP+load lowers back to `s_load_b32` (or a
  // hardware-preload SGPR read on gfx12+) against the kernarg segment, with
  // identical bytes to what the source kernel saw at entry.
  //
  // Hidden block counts (Triton's hidden_block_count_* ABI) still need
  // dispatch-packet synthesis since their values aren't stored in the
  // kernarg segment at all. Unmatched implicit-range preload offsets are
  // handled by the same strict/permissive boundary as SMEM hidden-arg loads.
  for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size(); ++SgprIdx) {
    const auto &Entry = UserSgprLayout.Entries[SgprIdx];
    if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
      continue;

    Expected<Value *> DwOrErr =
        EmitPreloadedKernargDword(B, Entry.KernargByteOffset);
    if (!DwOrErr)
      return DwOrErr.takeError();

    Value *Dw = *DwOrErr;
    Regs.storeSGPR32(B, static_cast<int>(SgprIdx), Dw);
  }
  // NumWorkitemDims (computed above) selects how many of x/y/z to fold into the
  // packed v0 seed.
  auto SeedWorkitemId = [&](IRBuilder<> &SeedB) {
    Regs.storeVGPR32(SeedB, 0,
                     Projection.emitPackedWorkitemId(SeedB, NumWorkitemDims));
  };

  if (!UseThreadLoop)
    SeedWorkitemId(B);

  // On gfx12+ the hardware command processor uses TTMP registers for
  // workgroup scheduling (RDNA4+ / CDNA-next layout):
  //   ttmp7[15:0]  = workgroup_id_y  (low 16 bits)
  //   ttmp7[31:16] = workgroup_id_z  (high 16 bits; 0 when grid has no Z)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  // The packed-Y-and-Z layout in ttmp7 is from the AMDGPU backend's
  // `loadInputValue` path (see LLVM's `AMDGPULegalizerInfo.cpp` --
  // `WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu)`,
  // `WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u)`).
  // Triton-generated gfx1250 kernels read the Y component via
  // `s_and_b32 sN, ttmp7, 0xffff` (e.g. matmul_fp16_16x16's `pid_n =
  // tl.program_id(1)` lowering), so a kernel raised without ttmp7
  // initialised always sees `workgroup_id_y == 0` -- only the
  // leftmost column of workgroups in a 2D-grid kernel writes its
  // tile, and the right-side tiles stay at whatever the destination
  // memory held at dispatch (verified empirically: matmul_fp16_16x16
  // M=32 with an all-1s input shows cols 0..15 = correct 32.0,
  // cols 16..31 = poison-fill from the host's pre-launch memset).
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  std::function<void(IRBuilder<> &)> SeedTtmp8 = [](IRBuilder<> &) {};
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
    // TTMP6 carries the source workgroup-cluster fields on gfx12+. This
    // HotSwap path models non-cluster source execution, so use the singleton
    // cluster encoding: per-cluster workgroup IDs and max IDs are all zero.
    B.CreateStore(B.getInt32(0), Regs.Ttmp[6]);
    B.CreateStore(B.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"),
                  Regs.Ttmp[9]);

    // ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF).
    // We mask Y to 16 bits before shifting Z so a stray-high-bit Y
    // doesn't bleed into the Z field.  CAVEAT: upstream's mask is
    // conditional -- `AMDGPULegalizerInfo::loadInputValue` uses `~0u`
    // on no-Z-grid entry-function kernels (letting a consumer that
    // reads ttmp7 unmasked see the FULL 32-bit workgroup_id_y, for
    // Y up to UINT_MAX).  Our unconditional 16-bit mask clips Y on
    // no-Z grids with Y >= 65536, which is a hypothetical silent
    // miscompile.  We have not observed a lifted kernel that does
    // this in practice -- every Triton-emitted consumer I surveyed
    // reads via `s_and ttmp7, 0xffff` -- but if a Y >= 65536 no-Z
    // kernel shows up we'll need to either thread `hasWorkGroupIDZ`
    // through `meta` and emit the conditional mask here, or switch
    // to the `~0u` mask and let `s_and ttmp7, 0xffff` consumers
    // tolerate the Z bits bleeding into their read (they already do
    // per the consumer pattern definition).
    Value *WgIdY = B.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
    Function *FnWorkgroupIdZ =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_z);
    Value *WgIdZ = B.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
    Value *WgIdYLo = B.CreateAnd(WgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
    Value *WgIdZHi = B.CreateShl(WgIdZ, B.getInt32(16), "wg_id_z_hi16");
    Value *Ttmp7Val = B.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
    B.CreateStore(Ttmp7Val, Regs.Ttmp[7]);

    SeedTtmp8 = [&](IRBuilder<> &SeedB) {
      // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
      Value *TidForTtmp = Projection.emitWorkitemIdX(SeedB);
      TidForTtmp->setName("ttmp8_tid");
      Value *WaveId =
          SeedB.CreateLShr(TidForTtmp, SeedB.getInt32(5), "wave_id_in_wg");
      Value *Ttmp8Val =
          SeedB.CreateShl(WaveId, SeedB.getInt32(25), "ttmp8_val");
      SeedB.CreateStore(Ttmp8Val, Regs.Ttmp[8]);
    };
    if (!UseThreadLoop)
      SeedTtmp8(B);
  }

  auto SeedThreadLoopIterationState = [&](IRBuilder<> &SeedB) -> Error {
    for (auto *Slot : Regs.Sgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Vgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Agpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Ttmp)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.M0);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[0]);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[1]);

    // Mirror the entry-BB user-SGPR seeding above so the thread-loop body sees
    // the same source ABI state as a normal source wave.
    if (UserSgprLayout.DispatchPtrSgpr >= 0) {
      Regs.storeSGPR64(SeedB, UserSgprLayout.DispatchPtrSgpr,
                       SeedB.CreateCall(FnDispatchPtr, {}, "dispatch_ptr"));
    }
    if (UserSgprLayout.KernargSegmentPtrSgpr >= 0) {
      Regs.storeSGPR64(SeedB, UserSgprLayout.KernargSegmentPtrSgpr,
                       SeedB.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
    }
    if (UserSgprLayout.WorkgroupIdXSgpr >= 0) {
      Regs.storeSGPR32(SeedB, UserSgprLayout.WorkgroupIdXSgpr,
                       SeedB.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
    }
    if (UserSgprLayout.WorkgroupIdYSgpr >= 0) {
      Regs.storeSGPR32(SeedB, UserSgprLayout.WorkgroupIdYSgpr,
                       SeedB.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
    }
    for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size();
         ++SgprIdx) {
      const auto &Entry = UserSgprLayout.Entries[SgprIdx];
      if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
        continue;
      Expected<Value *> DwOrErr =
          EmitPreloadedKernargDword(SeedB, Entry.KernargByteOffset);
      if (!DwOrErr)
        return DwOrErr.takeError();

      Regs.storeSGPR32(SeedB, static_cast<int>(SgprIdx), *DwOrErr);
    }

    if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
      SeedB.CreateStore(SeedB.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"),
                        Regs.Ttmp[9]);
      Value *WgIdY = SeedB.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
      Function *FnWorkgroupIdZ = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_workgroup_id_z);
      Value *WgIdZ = SeedB.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
      Value *WgIdYLo =
          SeedB.CreateAnd(WgIdY, SeedB.getInt32(0xFFFF), "wg_id_y_lo16");
      Value *WgIdZHi =
          SeedB.CreateShl(WgIdZ, SeedB.getInt32(16), "wg_id_z_hi16");
      Value *Ttmp7Val = SeedB.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
      SeedB.CreateStore(Ttmp7Val, Regs.Ttmp[7]);
      SeedTtmp8(SeedB);
    }

    SeedWorkitemId(SeedB);
    Regs.storeVCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeSCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeExec(SeedB, Projection.emitInitialExec(SeedB));
    return Error::success();
  };

  // ==== Phase 5: Raise each instruction; collect all failures in allFailures.
  // ====

  // `userSgprLayout` was built above before Phase 4 so entry SGPR seeding
  // and handler-side ABI decisions use the same descriptor-derived mapping.
  RaiseContext Ctx{C,
                   M,
                   B,
                   Regs,
                   Projection,
                   Mc,
                   Isa,
                   TargetIsa,
                   TargetCodeObjectVersion,
                   Kernargs,
                   &UserSgprLayout,
                   F,
                   nullptr,
                   OffsetToBb,
                   ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
                   TextBaseAddress,
                   SourceImageSections,
                   KernelOffset,
                   KernelEndOffset};
  Ctx.SetpcAnalysis = &SetpcAnalysis;
  Ctx.SourcePrivateSegmentFixedSize = Meta.PrivateSegmentFixedSize;
  Ctx.SourceComputePgmRsrc2 = Meta.ComputePgmRsrc2;
  Ctx.SourceKernelCodeProperties = Meta.KernelCodeProperties;
  Ctx.AssumeHipGlobalOffsetZero = AssumeHipGlobalOffsetZero;

  // Dominance-safe SGPR wave-mask shadow storage.
  // One EXEC-width mask + one scalar-valid bit per SGPR base index.
  // Consumers can combine `(valid ? shadow : fallback)` across BBs without
  // carrying non-dominating SSA values in `lastSgprWaveMaskI1`.
  Ctx.SgprWaveMaskExecShadow.reserve(Regs.Sgpr.size());
  Ctx.SgprWaveMaskValidShadow.reserve(Regs.Sgpr.size());
  Ctx.SourceWaveSgprPairShadow.reserve(Regs.Sgpr.size());
  Ctx.SourceWaveSgprPairValidShadow.reserve(Regs.Sgpr.size());
  for (unsigned I = 0; I < Regs.Sgpr.size(); ++I) {
    auto *MaskA =
        B.CreateAlloca(Regs.ExecTy, nullptr, "sgpr_mask_shadow_" + Twine(I));
    auto *ValidA = B.CreateAlloca(I1Ty, nullptr, "sgpr_mask_valid_" + Twine(I));
    auto *PairA =
        B.CreateAlloca(I64Ty, nullptr, "source_wave_sgpr_pair_" + Twine(I));
    auto *PairValidA = B.CreateAlloca(
        I1Ty, nullptr, "source_wave_sgpr_pair_valid_" + Twine(I));
    B.CreateStore(ConstantInt::get(Regs.ExecTy, 0), MaskA);
    B.CreateStore(B.getFalse(), ValidA);
    B.CreateStore(ConstantInt::get(I64Ty, 0), PairA);
    B.CreateStore(B.getFalse(), PairValidA);
    Ctx.SgprWaveMaskExecShadow.push_back(MaskA);
    Ctx.SgprWaveMaskValidShadow.push_back(ValidA);
    Ctx.SourceWaveSgprPairShadow.push_back(PairA);
    Ctx.SourceWaveSgprPairValidShadow.push_back(PairValidA);
  }

  llvm::Error RaiseReadFailure = llvm::Error::success();
  auto ReadFailureHandler = [&](llvm::Error Err) {
    if (RaiseReadFailure) {
      RaiseReadFailure =
          llvm::joinErrors(std::move(RaiseReadFailure), std::move(Err));
    } else {
      RaiseReadFailure = std::move(Err);
    }
  };
  Ctx.recordReadFailure = ReadFailureHandler;

  // Wire the reg-file's EXEC-write invalidation hook to ctx's lane_active
  // memo. This catches every EXEC mutation -- ctx.storeExec, the various
  // ctx.writeReg*(EXEC, ...) wrappers, *and* the handful of handlers that
  // still call ctx.Regs.storeExec / ctx.Regs.writeRegExecWidth directly
  // (SAVEEXEC family in handle_sop1, V_CMPX in handle_valu). Without
  // this hook those direct paths would leave the memo pointing at a
  // pre-write `lane_active`, silently mispredicating subsequent
  // emitUnderExec diamonds.
  Regs.OnExecWritten = [&Ctx] { Ctx.resetLaneActiveCache(); };

  // Wire the reg-file's per-SGPR write invalidation hook to ctx's
  // V_CMP -> V_CNDMASK per-lane-i1 shadow map
  // (`lastSgprWaveMaskI1`). Fires on every `storeSGPR32 / storeSGPR64`
  // and therefore on every path that mutates an SGPR -- including
  // handlers that bypass `writeReg32 / writeReg64` to call the
  // low-level stores directly (handle_smem's multi-dword load
  // splitting, handle_valu's SCC-flag SGPR writes, etc.). The V_CMP
  // wave-mask write path also fires this hook; the V_CMP handler
  // immediately re-populates the shadow with the per-lane `i1`
  // afterwards via `ctx.recordSgprWaveMaskI1`. See hotswap/docs/sgpr-
  // wave-mask-translation.md section 3.1 for the full contract.
  Regs.OnSgprWritten = [&Ctx](int Idx) { Ctx.invalidateSgprWaveMaskI1(Idx); };

  // Wire the reg-file's M0-write hook to ctx's raise-time M0 constant
  // shadow. Fires on every M0 store; a constant store records the value,
  // any other store clears it. The v_movrel* handlers consult
  // `Ctx.getM0Const()` to resolve the M0-relative VGPR index statically.
  Regs.OnM0Written = [&Ctx](llvm::Value *V) { Ctx.updateM0Const(V); };

  if (UseThreadLoop) {
    auto *IterA = B.CreateAlloca(I32Ty, nullptr, "tl_iter_alloca");
    B.CreateStore(B.getInt32(0), IterA);
    static_cast<ThreadLoopProjection *>(ProjectionPtr.get())
        ->setIterationAlloca(IterA);

    BasicBlock *CondBb = BasicBlock::Create(C, "tl_cond", F);
    BasicBlock *LatchBb = BasicBlock::Create(C, "tl_latch", F);
    BasicBlock *DoneBb = BasicBlock::Create(C, "tl_done", F);
    Ctx.ThreadLoopLatch = LatchBb;

    B.CreateBr(CondBb);
    B.SetInsertPoint(CondBb);

    Value *Iter = B.CreateLoad(I32Ty, IterA, "tl_iter_val");
    Value *IterOk = B.CreateICmpULT(
        Iter, B.getInt32(TargetIsa.waveSize() / Isa.waveSize()), "tl_iter_ok");
    Value *Lane = Projection.emitLaneIdx(B);
    Value *LaneOk =
        B.CreateICmpULT(Lane, B.getInt32(Isa.waveSize()), "tl_lane_ok");
    Value *EnterBody = B.CreateAnd(IterOk, LaneOk, "tl_enter_body");

    if (Error Err = SeedThreadLoopIterationState(B))
      return Err;

    for (auto *ValidA : Ctx.SgprWaveMaskValidShadow)
      B.CreateStore(B.getFalse(), ValidA);
    for (auto *ValidA : Ctx.SourceWaveSgprPairValidShadow)
      B.CreateStore(B.getFalse(), ValidA);

    B.CreateCondBr(EnterBody, OffsetToBb[KernelOffset], LatchBb);

    B.SetInsertPoint(LatchBb);
    Value *OldIter = B.CreateLoad(I32Ty, IterA, "tl_iter_old");
    Value *NextIter = B.CreateAdd(OldIter, B.getInt32(1), "tl_iter_next");
    B.CreateStore(NextIter, IterA);
    Value *More = B.CreateICmpULT(
        NextIter, B.getInt32(TargetIsa.waveSize() / Isa.waveSize()), "tl_more");
    B.CreateCondBr(More, CondBb, DoneBb);

    B.SetInsertPoint(DoneBb);
    B.CreateRetVoid();
  }

  // Non-thread-loop dedicated entry (out-of-extent callee merged): the seeding
  // lives in a standalone "entry" block; terminate it with a branch to the
  // kernel's real start so the body blocks are reached only via real edges.
  // (The thread-loop path wired its own entry->body edge above.)
  if (UseDedicatedEntry && !UseThreadLoop)
    B.CreateBr(OffsetToBb[KernelOffset]);

  if (RaiseReadFailure) {
    assert(false && "Unexpected read failures before raise loop");
  }

  llvm::Error RaiseFailures = llvm::Error::success();
  int RaisedCount = 0;
  for (size_t InstIdx = 0; InstIdx < Insts.size(); ++InstIdx) {
    const DecodedInst &Di = Insts[InstIdx];

    // If a terminator ended the recovered CFG path and the next decoded
    // instruction is not a known block leader, that instruction is unreachable
    // fallthrough bytes (often code after an unconditional branch). Do not emit
    // it into the already-terminated LLVM block.
    auto BbIt = OffsetToBb.find(Di.Offset);
    if (B.GetInsertBlock()->hasTerminator() && BbIt == OffsetToBb.end())
      continue;

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at -- which is the
    // `spe_skip` tail when the last emission was wrapped -- not from the
    // block that started the source instruction.
    if (BbIt != OffsetToBb.end() && BbIt->second != B.GetInsertBlock()) {
      BasicBlock *InsertBb = B.GetInsertBlock();
      if (!InsertBb->hasTerminator())
        B.CreateBr(BbIt->second);
      B.SetInsertPoint(BbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      Ctx.VgprMsBs = 0;
      // Drop the V_CMP -> V_CNDMASK per-lane-i1 shadow at every BB
      // transition. The cached `i1` SSA values dominate only the BB
      // they were emitted in; carrying them into a successor would
      // read an SSA value out of its dominance scope. A future
      // reaching-definitions pass on the raised IR could upgrade this
      // to a proper per-BB merge (see sgpr-wave-mask-translation.md
      // section 7 evolution path).
      Ctx.clearSgprWaveMaskShadow();
      // M0's raise-time constant shadow only dominates within its BB.
      Ctx.clearM0Const();
    }

    if (Error E = Ctx.computeVGPRAdjust(Di))
      return E;
    // Invalidate the SPE lane_active memoisation at every instruction
    // boundary. Any instruction is a potential EXEC writer (either through
    // our modeled CanonicalOp allow-list, or through a path we haven't yet
    // covered), and emitLaneActiveBit is load-bearing for per-lane
    // predication correctness: reusing a stale lane_active from before an
    // EXEC write would silently mispredicate side effects. See
    // RaiseContext::resetLaneActiveCache in raise-context.h for the full
    // invalidation contract.
    Ctx.resetLaneActiveCache();
    OpResolver Op{Ctx, Di};

    // Dispatch to the format-specific handler by querying TSFlags (and
    // `AMDGPU::isVOPD` for the one encoding without a dedicated flag bit)
    // directly, rather than going through a hand-rolled FormatKind enum.
    // Check precedence mirrors LLVM's decoder:
    //   * VOPD first -- it has no TSFlags bit; detect by named-operand id.
    //   * IsMAI before VOP3 -- MFMA is a VOP3 subclass with its own handler.
    //   * DPP / SDWA / VOPC / VOP3P / VOP3 / VOP2 / VOP1 all route to
    //     handleVALU, so they're collapsed into one mask test; ordering
    //     within the VOP family is therefore irrelevant here.
    //   * Scalar / memory family bits are mutually exclusive.
    // `default: break;` semantics are preserved: anything without a matching
    // bit falls through with `hr.Handled == false` and hits the unsupported-
    // instruction error path below.

    llvm::Expected<HandlerResult> HrOrErr =
        [&]() -> llvm::Expected<HandlerResult> {
      // The dispatch grows one instruction-family edge per patch: an opcode
      // whose handler has not landed yet falls through to the unhandled path
      // below and refuses cleanly. This milestone lifts the scalar-move /
      // program-end pair, so only the SOPP and SOP1 arms are wired.
      const uint64_t Flags = Di.TargetSpecificFlags;
      const unsigned Opc = Di.Inst.getOpcode();

      if (Flags & SIInstrFlags::SOPP)
        return handleSOPP(Ctx, Di, Op);
      else if (Flags & SIInstrFlags::SOP1)
        return handleSOP1(Ctx, Di, Op);

      StringRef Format = formatName(Di.TargetSpecificFlags, Opc);
      return RaiseFailure::unsupportedInstructionForm(
          strippedMnemonic(Mc, Di.Inst), Di.Offset, Format);
    }();

    if (RaiseReadFailure || !HrOrErr) {
      if (RaiseFailures && RaiseReadFailure) {
        RaiseFailures = llvm::joinErrors(std::move(RaiseFailures),
                                         std::move(RaiseReadFailure));
        RaiseReadFailure = llvm::Error::success();
      } else if (RaiseReadFailure) {
        RaiseFailures = std::move(RaiseReadFailure);
        RaiseReadFailure = llvm::Error::success();
      }

      if (RaiseFailures && !HrOrErr) {
        RaiseFailures =
            llvm::joinErrors(std::move(RaiseFailures), HrOrErr.takeError());
      } else if (!HrOrErr) {
        RaiseFailures = HrOrErr.takeError();
      }
      continue;
    }

    HandlerResult Hr = *HrOrErr;

    // A handler recognised the instruction but refused it
    if (!Hr.Handled) {
      StringRef Format =
          formatName(Di.TargetSpecificFlags, Di.Inst.getOpcode());
      errs() << "transpiler: Unsupported instruction: "
             << printInst(Mc, Di.Inst) << " [format=" << Format << "]"
             << " at offset 0x" << format_hex(Di.Offset, 1) << "\n";
      RaiseFailures = llvm::joinErrors(
          std::move(RaiseFailures),
          RaiseFailure::unsupportedOpcode(strippedMnemonic(Mc, Di.Inst),
                                          Di.Offset, Format));
      continue;
    }

    if (Di.defsScc() && !Hr.SccHandled && Hr.SccResult) {
      Value *Zero = Constant::getNullValue(Hr.SccResult->getType());
      Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateICmpNE(Hr.SccResult, Zero));
    }
    if (Di.defsExec())
      Result.HasDivergentExec = true;

    RaisedCount++;
    continue;
  }

  if (RaiseReadFailure) {
    assert(false && "unhandled read failure after raise loop");
  }

  // If the function's entry block has predecessors (e.g. a backward
  // branch targeting the kernel's first instruction), LLVM's verifier
  // rejects the IR.  Insert an empty prolog block that falls through to
  // the original entry so the entry becomes predecessor-free.
  if (!pred_empty(&F->getEntryBlock())) {
    BasicBlock *OldEntry = &F->getEntryBlock();
    BasicBlock *Prolog = BasicBlock::Create(C, "prolog", F, OldEntry);
    B.SetInsertPoint(Prolog);
    B.CreateBr(OldEntry);
  }

  // Ensure all BBs have terminators.  Reachable unterminated blocks arise
  // when a kernel falls off its symbol boundary without an explicit
  // s_endpgm -- emit `ret void` (or branch to the thread-loop latch)
  // so the lifted kernel terminates cleanly.  Blocks with no predecessors
  // that are not the entry block are dead fallthrough bytes after a
  // recovered branch; keep their defensive `unreachable`.
  for (auto &BB : *F) {
    if (!BB.hasTerminator()) {
      B.SetInsertPoint(&BB);
      if (!pred_empty(&BB) || &BB == &F->getEntryBlock()) {
        if (Ctx.ThreadLoopLatch)
          B.CreateBr(Ctx.ThreadLoopLatch);
        else
          B.CreateRetVoid();
      } else {
        B.CreateUnreachable();
      }
    }
  }

  if (Stats)
    Stats->LiftedCount = RaisedCount;

  // If any instructions failed to raise, skip Phases 6-7.
  if (RaiseFailures) {
    return RaiseFailures;
  }

  // ==== Phase 6: Promote allocas to SSA ====
  {
    DominatorTree DT(*F);
    AssumptionCache AC(*F);
    SmallVector<AllocaInst *, 512> Allocas;
    Regs.collectAllocas(Allocas);
    Ctx.collectSgprWaveMaskShadowAllocas(Allocas);
    PromoteMemToReg(Allocas, DT, &AC);
  }

  // ==== Phase 7: Verify IR ====
  std::string VerifyErr;
  raw_string_ostream VerifyOs(VerifyErr);
  if (verifyModule(M, &VerifyOs)) {
    errs() << "transpiler: IR verification failed:\n" << VerifyErr << "\n";
    return RaiseFailure::irVerificationFailed(VerifyErr);
  }

  Result.UsesScratchPrivateSegment = Ctx.UsesScratchPrivateSegment;
  Result.SourcePrivateSegmentFixedSize = Ctx.SourcePrivateSegmentFixedSize;
  return Result;
}

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
          bool EnableWaveNative, uint64_t TextBaseAddress,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
          RaiseStats *Stats) {
  return raiseToIR(TextBytes, SourceIsa, KernelName, Meta,
                   /*KernelOffset=*/0,
                   /*KernelSize=*/0, CompilationTargetIsa,
                   EnableWritelaneRewrite, EnableWaveNative,
                   /*AssumeHipGlobalOffsetZero=*/false,
                   /*ForceModrepDoubled=*/false, TextBaseAddress,
                   SourceImageSections, /*FunctionExtents=*/{}, Stats);
}

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          uint64_t KernelOffset, uint64_t KernelSize,
          llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
          bool EnableWaveNative, bool AssumeHipGlobalOffsetZero,
          bool ForceModrepDoubled, uint64_t TextBaseAddress,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
          llvm::ArrayRef<KernelSymbolExtent> FunctionExtents,
          RaiseStats *Stats) {
  return raiseToIRImpl(
      TextBytes, SourceIsa, KernelName, Meta, KernelOffset, KernelSize,
      TextBaseAddress, SourceImageSections, CompilationTargetIsa,
      EnableWritelaneRewrite, EnableWaveNative,
      /*forceThreadLoopProjection=*/false,
      /*suppressC5ForThreadLoopRoute=*/false, ForceModrepDoubled,
      AssumeHipGlobalOffsetZero, FunctionExtents, Stats);
}

} // namespace COMGR::hotswap
