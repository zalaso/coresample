// Start-up code for C programs on Coresample (and unchanged on Spike).
//
// Clears every register (Spike's boot ROM leaves values in a0, a1 and t0;
// the core starts from zero, and the commit logs must match), sets up the
// stack, clears .bss, calls main and reports main's return value through
// `tohost`: 0 becomes 1 (pass), n becomes (n << 1) | 1 (fail with code n).
  .section .text.init, "ax"
  .global _start
_start:
  .option push
  .option norelax
  li x1, 0;  li x2, 0;  li x3, 0;  li x4, 0;  li x5, 0;  li x6, 0;  li x7, 0
  li x8, 0;  li x9, 0;  li x10, 0; li x11, 0; li x12, 0; li x13, 0; li x14, 0
  li x15, 0; li x16, 0; li x17, 0; li x18, 0; li x19, 0; li x20, 0; li x21, 0
  li x22, 0; li x23, 0; li x24, 0; li x25, 0; li x26, 0; li x27, 0; li x28, 0
  li x29, 0; li x30, 0; li x31, 0
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top

  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sw zero, 0(t0)
  addi t0, t0, 4
  j 1b
2:
  call main

  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
3:
  sw a0, 0(t0)
  j 3b

  .section .tohost, "aw", @progbits
  .balign 64
  .global tohost
tohost: .dword 0
  .global fromhost
fromhost: .dword 0
