# FENCE.I must make a store to the very next instruction visible.
#
# On the 5-stage pipeline, when the store writes memory (MEM), the FENCE.I is
# in EX and the instruction after it has already been fetched (ID) with its
# old contents. Only FENCE.I's refetch can pick up the new one. The riscv-tests
# and riscv-arch-test FENCE.I tests always jump to the new code, and the jump
# refetches anyway, so they pass even when FENCE.I does nothing.
#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  li TESTNUM, 2
  la t0, 1f
  li t1, 0x02a00513        # addi a0, zero, 42
  li a0, 0
  sw t1, 0(t0)
  fence.i
1:
  addi a0, zero, 1         # overwritten by the store: must execute as addi a0, zero, 42
  li t2, 42
  bne a0, t2, fail

  # Test 3: the same with the patched instruction one slot further (IF).
  li TESTNUM, 3
  la t0, 2f
  li t1, 0x01b00593        # addi a1, zero, 27
  li a1, 0
  sw t1, 0(t0)
  fence.i
  nop
2:
  addi a1, zero, 1         # overwritten: must execute as addi a1, zero, 27
  li t2, 27
  bne a1, t2, fail

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
