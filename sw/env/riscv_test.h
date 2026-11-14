// Minimal test environment for running riscv-tests on Coresample.
//
// Replaces the official "p" environment, which needs CSRs and trap handling
// that the first core does not have yet. Everything runs in machine mode from
// _start. The test reports its result by writing to the `tohost` symbol:
// 1 means pass, (test_number << 1) | 1 means fail. Spike uses the same
// protocol, so one ELF runs unchanged on both.
#ifndef CORESAMPLE_RISCV_TEST_H
#define CORESAMPLE_RISCV_TEST_H

#define RVTEST_RV32U
#define RVTEST_RV64U RVTEST_RV32U
#define RVTEST_RV32M
#define RVTEST_RV64M RVTEST_RV32M

#define TESTNUM gp

// Spike's boot ROM leaves values in a0, a1 and t0; our core starts from zero.
// Clearing every register first keeps the two commit logs comparable.
#define INIT_XREG \
  li x1, 0;  li x2, 0;  li x3, 0;  li x4, 0;  li x5, 0;  li x6, 0;  \
  li x7, 0;  li x8, 0;  li x9, 0;  li x10, 0; li x11, 0; li x12, 0; \
  li x13, 0; li x14, 0; li x15, 0; li x16, 0; li x17, 0; li x18, 0; \
  li x19, 0; li x20, 0; li x21, 0; li x22, 0; li x23, 0; li x24, 0; \
  li x25, 0; li x26, 0; li x27, 0; li x28, 0; li x29, 0; li x30, 0; \
  li x31, 0;

#define RVTEST_CODE_BEGIN \
  .section .text.init;    \
  .align 6;               \
  .globl _start;          \
_start:                   \
  INIT_XREG

#define RVTEST_CODE_END unimp

#define RVTEST_PASS        \
  fence;                   \
  li TESTNUM, 1;           \
  la t5, tohost;           \
  sw TESTNUM, 0(t5);       \
1: j 1b;

#define RVTEST_FAIL              \
  fence;                         \
1: beqz TESTNUM, 1b;             \
  sll TESTNUM, TESTNUM, 1;       \
  or TESTNUM, TESTNUM, 1;        \
  la t5, tohost;                 \
  sw TESTNUM, 0(t5);             \
2: j 2b;

#define EXTRA_DATA

#define RVTEST_DATA_BEGIN                              \
  EXTRA_DATA                                           \
  .pushsection .tohost, "aw", @progbits;               \
  .align 6; .global tohost; tohost: .dword 0;          \
  .size tohost, 8;                                     \
  .align 6; .global fromhost; fromhost: .dword 0;      \
  .size fromhost, 8;                                   \
  .popsection;                                         \
  .align 4; .global begin_signature; begin_signature:

#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:

#endif
