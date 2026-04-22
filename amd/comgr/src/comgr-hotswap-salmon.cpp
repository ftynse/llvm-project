//===- comgr-hotswap-salmon.cpp - ISA transpilation via LLVM IR (salmon) --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// COMgr entry point for the salmon transpiler. Where the byte-level
/// `amd_comgr_hotswap_rewrite` path patches a small set of stepping-specific
/// instruction encodings in place, this entry point hands the whole code
/// object to the salmon pipeline - every kernel is disassembled, raised to
/// LLVM IR, re-lowered through the stock AMDGPU backend for the target ISA,
/// and re-linked into a single merged HSACO via
/// `transpiler::runPipelineAllKernels` (see amd/salmon/pipeline.hpp and
/// amd/salmon/raise_cli.cpp for the standalone driver this entry point
/// mirrors).
///
/// Failure is loud: any per-kernel raise failure surfaced by the salmon
/// pipeline turns into `AMD_COMGR_STATUS_ERROR`. The salmon library logs
/// the offending kernel and mnemonic on stderr (use salmon's CLI with the
/// `--write-hsaco` mode for the same output).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"

#include "salmon/pipeline.hpp"

#include <cstdint>
#include <vector>

using namespace COMGR;

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE || !source_isa_name ||
      !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate both ISA names through the same parser the byte-level
  // `amd_comgr_hotswap_rewrite` uses, so the public contract is identical:
  // malformed identifiers are rejected up-front and never reach the salmon
  // pipeline. We do not gate on the processor name here — salmon decides
  // per-kernel whether the source/target pair is supported, and surfaces
  // unsupported instructions as a pipeline failure (see
  // RaiseFailure::reason in amd/salmon/raise_failure.hpp).
  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Salmon takes the code object by `std::vector<uint8_t>` (see
  // pipeline.hpp). DataObject stores its bytes in a `char *` buffer; copy
  // into the salmon-shaped container rather than reinterpret-casting the
  // pointer, since the pipeline reads through this buffer many times across
  // kernels and the temporary lifetime needs to be unambiguous.
  const auto *InputBegin = reinterpret_cast<const uint8_t *>(InputP->Data);
  std::vector<uint8_t> InputBytes(InputBegin, InputBegin + InputP->Size);

  // Drive the same all-kernels merge path that raise_cli.cpp's --write-hsaco
  // mode falls back on for whole-file flows. We pass salmon's defaults for
  // the writelane / wave-native toggles (both on, post-graduation) — the
  // public comgr surface intentionally hides those knobs since they are
  // either correctness-preserving rewrites (writelane) or projection
  // strategies (wave-native) that callers should not have to reason about.
  // If an opt-out is ever needed at the comgr boundary it should land as a
  // separate options struct rather than overloading this entry point.
  transpiler::PipelineResult Pipeline = transpiler::runPipelineAllKernels(
      InputBytes, SourceIdent.Processor.str(), TargetIdent.Processor.str());
  if (!Pipeline.success || Pipeline.hsaco.empty())
    return AMD_COMGR_STATUS_ERROR;

  DataObject *OutputP = DataObject::allocate(AMD_COMGR_DATA_KIND_EXECUTABLE);
  if (!OutputP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (auto Status = OutputP->setData(
          llvm::StringRef(reinterpret_cast<const char *>(Pipeline.hsaco.data()),
                          Pipeline.hsaco.size()))) {
    OutputP->release();
    return Status;
  }

  *output = DataObject::convert(OutputP);
  return AMD_COMGR_STATUS_SUCCESS;
}
