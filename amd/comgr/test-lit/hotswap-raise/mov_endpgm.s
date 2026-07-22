; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The raiser loads the code object, reads the kernel metadata and descriptor,
; and emits the kernel function shell. Instruction lifting is not wired up yet,
; so the body is a placeholder ret.
; RUN: %raise_cli %t.hsaco --emit-ir | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @mov_endpgm_kernel(
; CHECK: ret void

; --dump-decoded runs the MC stack, opcode map, and decoder over the kernel's
; .text and lists each instruction's canonical op and disassembly. The DECODE
; lines below sit next to the instructions they match.
; RUN: %raise_cli %t.hsaco --dump-decoded | %FileCheck %s --check-prefix=DECODE

; An unrecognised source ISA is refused with a diagnostic, not a crash.
; RUN: not %raise_cli %t.hsaco --isa=gfxbogus --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; REFUSE: does not name an AMDGPU GPU

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	mov_endpgm_kernel
	.p2align	8
	.type	mov_endpgm_kernel,@function
mov_endpgm_kernel:
; DECODE: S_MOV_B32{{.+}}s_mov_b32 s0, 0
	s_mov_b32 s0, 0
; DECODE: S_ENDPGM{{.+}}s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mov_endpgm_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           mov_endpgm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         mov_endpgm_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
