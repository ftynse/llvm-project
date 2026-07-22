//===- code-object-utils.h - AMDGPU code-object metadata ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API for extracting the AMDGPU-specific metadata the hotswap raiser
// needs from an ELF code object: the per-kernel MsgPack-derived ABI surface
// and the kernel descriptor register fields read directly from .rodata. Each
// entry point returns `llvm::Expected<...>` on failure -- forwarded LLVM
// errors keep their original ErrorInfo type, hotswap-detected mismatches use
// `HotswapError` from `hotswap-error.h`.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

/// Raw bytes of the AMDGPU `.text` section, as returned by
/// `extractTextSection`. The byte buffer is owned by the `TextSection` --
/// the underlying ELF MemoryBuffer is not borrowed across the call.
struct TextSection {
  /// Raw `.text` bytes, indexed by text-relative decoded instruction offsets.
  llvm::SmallVector<uint8_t> Bytes;
  /// Runtime address of `.text`; PC-relative instructions use this source
  /// code-object address domain.
  uint64_t Address = 0;

  /// Allocated source sections whose bytes may be read through PC-relative
  /// SMEM.
  struct ImageSection {
    /// Section bytes, indexed by source code-object address minus `Address`.
    llvm::SmallVector<uint8_t> Bytes;
    /// Runtime address of this source code-object section.
    uint64_t Address = 0;
  };
  /// Minimal source image used for literal-table materialisation.
  llvm::SmallVector<ImageSection> ImageSections;
};

/// Resolved text-section extent for a kernel symbol. `Offset` is relative to
/// `.text`; `Size` bounds decoding to the selected symbol's byte range.
struct KernelSymbolExtent {
  uint64_t Offset = 0;
  uint64_t Size = 0;
};

/// One entry of the kernel argument table extracted from the AMDGPU MsgPack
/// notes. Mirrors the AMDHSA `.args` schema; absent fields stay at the
/// constructor defaults below.
struct KernelArgMeta {
  std::string Name;
  uint32_t Offset = 0;
  uint32_t Size = 0;
  /// AMDHSA `.value_kind` enum spelling (e.g. `by_value`, `global_buffer`,
  /// `hidden_global_offset_x`). Kept as a string because the AMDHSA spec
  /// adds new kinds without bumping the metadata version, so a hand-rolled
  /// enum here would silently lose round-trip fidelity for unrecognised
  /// kinds.
  std::string ValueKind;
  /// LLVM AMDGPU address space id for pointer-typed arguments. Defaults to
  /// 0 -- the LLVM-default flat address space -- which matches non-pointer
  /// arguments where AMDHSA omits the field entirely.
  unsigned AddressSpace = 0;
};

/// Per-kernel metadata extracted from the AMDGPU code object's MsgPack notes
/// + kernel descriptor (`<name>.kd`).
struct KernelMeta {
  std::string Name;
  uint32_t KernargSegmentSize = 0;
  uint32_t GroupSegmentFixedSize = 0;
  uint32_t PrivateSegmentFixedSize = 0;
  uint32_t MaxFlatWorkgroupSize = 256;
  // Code object v6 `.cluster_dims`: [0,0,0] means clusters are disabled.
  // Any non-zero value carries source cluster state that HotSwap does not
  // reconstruct yet, so the raiser refuses it before seeding TTMP6.
  bool HasClusterDims = false;
  llvm::SmallVector<uint32_t, 3> ClusterDims;
  llvm::SmallVector<KernelArgMeta> Args;

  bool hasNonDisabledClusterDims() const {
    return HasClusterDims &&
           llvm::any_of(ClusterDims, [](uint32_t Dim) { return Dim != 0; });
  }

  // ---------------------------------------------------------------------
  // Kernel descriptor (KD) raw fields.
  //
  // Populated by extractKernelMeta from the 64-byte kernel_descriptor_t
  // block that lives at the symbol named `<kernelName>.kd` (always in the
  // .rodata section for amdhsa code objects). These fields are the surface
  // needed to derive the source-ISA SGPR ABI:
  //
  //   * PrivateSegmentFixedSize: source private/scratch bytes per work-item.
  //
  //   * KernelCodeProperties (KD bytes 56-57): bit field selecting which
  //     `enable_sgpr_*` user SGPRs the loader pre-populates before entry.
  //     See LLVM's AMDHSAKernelDescriptor.h
  //     KERNEL_CODE_PROPERTY_ENABLE_SGPR_* for the bit positions.
  //
  //   * KernargPreload (KD bytes 58-59): packed {LENGTH[6:0], OFFSET[15:7]}
  //     per LLVM's KERNARG_PRELOAD_SPEC enum -- the gfx1250 kernarg preload
  //     mechanism the user-SGPR layout consumer needs to know about.
  //
  //   * ComputePgmRsrc2 (KD bytes 52-55): ENABLE_SGPR_WORKGROUP_ID_{X,Y,Z} /
  //     WORKGROUP_INFO bits and the USER_SGPR_COUNT field.
  //
  //   * ComputePgmRsrc1 (KD bytes 48-51): captured for diagnostics and
  //     future wave-size-aware decisions.
  //
  // `HasKernelDescriptor` is true iff parsing succeeded. We do not silently
  // fall back to a hardcoded layout when it is false -- the caller is
  // expected to refuse the lift instead.
  bool HasKernelDescriptor = false;
  uint32_t ComputePgmRsrc1 = 0;
  uint32_t ComputePgmRsrc2 = 0;
  uint16_t KernelCodeProperties = 0;
  uint16_t KernargPreload = 0;

  /// Byte offset (8-byte aligned) of the first hidden argument in the
  /// kernarg segment. Hidden arguments (`hidden_*` value kinds) are
  /// appended after every explicit argument.
  uint64_t implicitArgsBase() const {
    uint64_t MaxEnd = 0;
    for (const KernelArgMeta &Arg : Args) {
      if (llvm::StringRef(Arg.ValueKind).starts_with("hidden_")) {
        continue;
      }
      uint64_t End = static_cast<uint64_t>(Arg.Offset) + Arg.Size;
      if (End > MaxEnd) {
        MaxEnd = End;
      }
    }
    return llvm::alignTo(MaxEnd, 8);
  }
};

/// Extract the `.text` section bytes from `ElfData`. Returns a
/// `HotswapError` when the ELF parses but has no `.text` section;
/// forwards `llvm::object` parse errors unchanged.
llvm::Expected<TextSection> extractTextSection(llvm::MemoryBufferRef ElfData);

/// List the kernel names declared in the AMDGPU MsgPack notes embedded in
/// `ElfData`. Returns a `HotswapError` when no AMDGPU metadata note is
/// present.
llvm::Expected<llvm::SmallVector<std::string>>
listKernelNames(llvm::MemoryBufferRef ElfData);

/// Extract the per-kernel metadata for `KernelName` from the MsgPack notes
/// in `ElfData`, including the kernel-descriptor register fields read out
/// of `.rodata`. KD-bytes lookup is best-effort: a usable KernelMeta is
/// still returned for the MsgPack-derived fields when the .rodata KD blob
/// is unreachable, with `HasKernelDescriptor == false`.
llvm::Expected<KernelMeta> extractKernelMeta(llvm::MemoryBufferRef ElfData,
                                             llvm::StringRef KernelName);

/// Resolve the byte offset and byte extent for `KernelName` within `.text`.
/// When the ELF symbol size is missing or zero, the extent is bounded by the
/// next metadata kernel symbol where possible, so helper/device functions
/// between kernels stay inside the selected kernel's extent.
llvm::Expected<KernelSymbolExtent>
findKernelSymbolExtent(llvm::MemoryBufferRef ElfData,
                       llvm::StringRef KernelName);

/// List the byte extent of every function symbol in `.text`, sorted by
/// ascending offset. Offsets are `.text`-relative (symbol address minus the
/// section base), matching `findKernelSymbolExtent`. Zero-sized symbols are
/// bounded by the next function symbol (or the end of `.text`). This lets the
/// raiser resolve a call/branch target that lands in a *different* function
/// (an outlined device helper) to that callee's extent so it can be decoded
/// and lifted alongside the caller.
llvm::Expected<llvm::SmallVector<KernelSymbolExtent>>
listTextFunctionExtents(llvm::MemoryBufferRef ElfData);

} // namespace COMGR::hotswap

#endif
