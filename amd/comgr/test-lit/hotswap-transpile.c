// COM: Test the salmon-backed amd_comgr_hotswap_transpile API.
// COM: Only registered when comgr was configured with -DCOMGR_ENABLE_SALMON=ON;
// COM: the REQUIRES guard makes the test xfail gracefully on builds that
// COM: did not opt in to salmon (mirrors the comgr-has-spirv pattern).

// REQUIRES: comgr-salmon

// COM: Create a minimal not-an-ELF blob for the negative-path tests. The
// COM: positive end-to-end path below uses a real HSACO checked into the
// COM: salmon tree; deeper coverage (full corpus, GPU execution) lives in
// COM: the salmon gtest harness.
// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > \
// RUN:        %t.elf

// COM: NULL-argument validation (no positional args).
// RUN: hotswap-transpile | %FileCheck --check-prefix=NULL %s
// NULL: NULL_ARGS: INVALID_ARGUMENT

// COM: Malformed ISA strings -> parseTargetIdentifier rejects.
// RUN: hotswap-transpile %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// BADISA: RESULT: INVALID_ARGUMENT

// COM: Zero-size input data with a well-formed ISA pair -> rejected
// COM: at the data-pointer gate (mirrors hotswap-rewrite.c's ZEROSIZE).
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// ZEROSIZE: RESULT: INVALID_ARGUMENT

// COM: Wrong data kind (BC instead of EXECUTABLE) -> rejected at the
// COM: kind gate. Both hotswap entry points share this contract.
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --wrong-kind \
// RUN:   | %FileCheck --check-prefix=WRONGKIND %s
// WRONGKIND: RESULT: INVALID_ARGUMENT

// COM: Well-formed ISA pair plus a non-ELF / no-kernels payload -> the
// COM: salmon pipeline runs (input passed validation) but
// COM: runPipelineAllKernels surfaces "no kernels" as a pipeline failure,
// COM: which the comgr wrapper maps to AMD_COMGR_STATUS_ERROR. This is the
// COM: cheapest way to assert that the salmon call site is actually wired
// COM: up without staging a real HSACO into the comgr tree.
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=NOKERNELS %s
// NOKERNELS: RESULT: ERROR

// COM: End-to-end transpile of a real HSACO. The salmon tree ships a tiny
// COM: gfx950 vecadd code object (single kernel, no cross-lane / MFMA / wave
// COM: ops) under amd/salmon/tests/. Re-lower it to gfx942 and verify both
// COM: the API contract (SUCCESS + non-empty bytes) and the binary contents
// COM: (ELF e_flags retargeted to gfx942, kernel symbol preserved). The
// COM: salmon backend shells out to llc and ld.lld; the lit site config
// COM: prepends llvm_tools_dir to PATH so both are reachable from the test
// COM: environment.
// RUN: hotswap-transpile %S/../../salmon/tests/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --output=%t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=VECADD %s
// VECADD: RESULT: SUCCESS bytes={{[1-9][0-9]*}}

// COM: Sanity-check the source binary's e_flags so the negative check below
// COM: is meaningful: confirm it really is a gfx950 ELF before asserting
// COM: that the *output* is not.
// RUN: %llvm-readelf -h %S/../../salmon/tests/vecadd_gfx950.co \
// RUN:   | %FileCheck --check-prefix=SRCISA %s
// SRCISA: Flags: {{.*}}gfx950

// COM: The transpiled binary's AMDGPU machine ID must reflect the target
// COM: ISA, not the source. llvm-readelf renders EF_AMDGPU_MACH as a
// COM: human-readable "gfxNNN" tag in the Flags line, so we can FileCheck
// COM: it directly without parsing hex. CHECK-NOT pins the negative side:
// COM: a no-op pipeline that just copied the input through would fail this.
// RUN: %llvm-readelf -h %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTISA %s
// TGTISA:     Flags: {{.*}}gfx942
// TGTISA-NOT: gfx950

// COM: Symbol-level smoke check: the kernel name must round-trip through
// COM: the salmon raise -> llc -> ld.lld pipeline. If the kernel were
// COM: dropped (e.g. raise failure swallowed silently) the executable
// COM: would link empty and this would fail.
// RUN: %llvm-objdump --syms %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTSYM %s
// TGTSYM: vecadd
