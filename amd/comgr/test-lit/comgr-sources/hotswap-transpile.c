//===- hotswap-transpile.c - Test salmon-backed transpile API ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Driver for amd_comgr_hotswap_transpile (the salmon-backed entry point).
//
// Mirrors the call/return shape of hotswap-rewrite.c for the validation
// paths so the two entry points stay in lockstep at the comgr boundary.
// The "real HSACO" path (positional invocation with a code-object file) is
// gated on the caller supplying a known-good HSACO; in that case it just
// asserts that the call returns SUCCESS and emits a non-empty output.
//
// The salmon pipeline shells out to llc and ld.lld at runtime, so end-to-end
// success on a real HSACO requires an LLVM build tree on PATH. The lit test
// only exercises the validation paths by default for that reason.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  // No args -> NULL-pointer validation. Mirrors hotswap-rewrite.c's first
  // mode so the two entry points have the same negative-test surface.
  if (argc < 2) {
    amd_comgr_data_t dummy_output;
    amd_comgr_data_t dummy_input = {0};
    amd_comgr_status_t Status =
        amd_comgr_hotswap_transpile(dummy_input, NULL, NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("transpile with NULL args: expected INVALID_ARGUMENT, got %d",
           (int)Status);
    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail("usage: hotswap-transpile <elf_file> <source_isa> <target_isa> "
         "[--zero-size|--wrong-kind] [--output=<path>]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  int ZeroSize = 0;
  int WrongKind = 0;
  // Optional path to dump the transpiled bytes to. lit tests use this to
  // hand the output to llvm-readelf / llvm-objdump for ISA-level smoke
  // checks; the validation paths leave it NULL and only inspect stdout.
  const char *OutputPath = NULL;
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--zero-size") == 0)
      ZeroSize = 1;
    else if (strcmp(argv[i], "--wrong-kind") == 0)
      WrongKind = 1;
    else if (strncmp(argv[i], "--output=", 9) == 0)
      OutputPath = argv[i] + 9;
    else
      fail("unknown option: %s", argv[i]);
  }

  char *ElfBuf;
  size_t ElfSize = (size_t)setBuf(ElfFile, &ElfBuf);

  amd_comgr_data_t InputData;
  // --wrong-kind: feed BC instead of EXECUTABLE. Exercises the data-kind
  // gate in amd_comgr_hotswap_transpile, which mirrors the gate in the
  // byte-level rewriter.
  amd_comgr_data_kind_t Kind =
      WrongKind ? AMD_COMGR_DATA_KIND_BC : AMD_COMGR_DATA_KIND_EXECUTABLE;
  amd_comgr_(create_data(Kind, &InputData));
  if (!ZeroSize)
    amd_comgr_(set_data(InputData, ElfSize, ElfBuf));

  amd_comgr_data_t OutputData;
  amd_comgr_status_t Status =
      amd_comgr_hotswap_transpile(InputData, SourceISA, TargetISA, &OutputData);

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  // The salmon pipeline reports per-kernel failures as
  // AMD_COMGR_STATUS_ERROR (e.g. unsupported instruction, missing kernels,
  // backend compile failure). Surface this as a distinct line so lit can
  // assert it independently from INVALID_ARGUMENT.
  if (Status == AMD_COMGR_STATUS_ERROR) {
    printf("RESULT: ERROR\n");
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %d", (int)Status);

  size_t OutSize = 0;
  amd_comgr_(get_data(OutputData, &OutSize, NULL));

  if (OutSize == 0)
    fail("transpile produced empty output on supposedly successful call");

  if (OutputPath != NULL) {
    char *OutBuf = (char *)malloc(OutSize);
    if (OutBuf == NULL)
      fail("malloc(%zu) for output buffer failed", OutSize);
    amd_comgr_(get_data(OutputData, &OutSize, OutBuf));
    FILE *OutFile = fopen(OutputPath, "wb");
    if (OutFile == NULL)
      fail("cannot open output path '%s' for writing", OutputPath);
    if (fwrite(OutBuf, 1, OutSize, OutFile) != OutSize)
      fail("short write to '%s'", OutputPath);
    fclose(OutFile);
    free(OutBuf);
  }

  amd_comgr_(release_data(OutputData));
  amd_comgr_(release_data(InputData));
  free(ElfBuf);

  printf("RESULT: SUCCESS bytes=%zu\n", OutSize);
  return 0;
}
