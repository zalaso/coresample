// rvmodel_macros.h
// DUT-specific macros for running riscv-arch-test on the Coresample core.
// The testbench ends the simulation when the program writes a nonzero value
// to `tohost`: 1 means pass, any other value means fail.
// SPDX-License-Identifier: Apache-2.0

#ifndef _RVMODEL_MACROS_H
#define _RVMODEL_MACROS_H

#define RVMODEL_DATA_SECTION \
        .pushsection .tohost,"aw",@progbits;                \
        .balign 8; .global tohost; tohost: .dword 0;         \
        .balign 8; .global fromhost; fromhost: .dword 0;     \
        .popsection

#define RVMODEL_HALT_PASS  \
  li x1, 1                ;\
  la t0, tohost           ;\
  write_tohost_pass:      ;\
    sw x1, 0(t0)          ;\
    sw x0, 4(t0)          ;\
    j write_tohost_pass   ;\

#define RVMODEL_HALT_FAIL \
  li x1, 3                ;\
  la t0, tohost           ;\
  write_tohost_fail:      ;\
    sw x1, 0(t0)          ;\
    sw x0, 4(t0)          ;\
    j write_tohost_fail   ;\

// No console: failures are reported through tohost and the commit log.
#define RVMODEL_IO_WRITE_STR(_R1, _R2, _R3, _STR_PTR)

#endif // _RVMODEL_MACROS_H
