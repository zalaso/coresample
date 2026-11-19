// rvtest_config.h
// Coresample core configuration for riscv-arch-test (ACT4).
// ACT4 normally generates this file from a UDB configuration; the core is
// small enough to describe by hand.
//
// RV32IM + Zifencei, machine mode only, no CSRs. STANDARD_SM_SUPPORTED is
// deliberately not defined: without CSRs the framework cannot install trap
// handlers, so only tests that need no traps can run.
// SPDX-License-Identifier: Apache-2.0

#ifndef RVTEST_CONFIG_H
#define RVTEST_CONFIG_H

#define I_SUPPORTED
#define M_SUPPORTED
#define ZMMUL_SUPPORTED
#define ZIFENCEI_SUPPORTED

#define UDB_MXLEN 32
#define UDB_NUM_PMP_ENTRIES 0

#endif // RVTEST_CONFIG_H
