/* Memory layout for C programs: one 1 MiB RAM at 0x80000000, the address
   where Spike starts after its boot ROM and where the core resets. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  ram (rwx) : ORIGIN = 0x80000000, LENGTH = 1M
}

SECTIONS
{
  .text.init : { *(.text.init) } > ram
  .text      : { *(.text) *(.text.*) } > ram
  . = ALIGN(64);
  .tohost    : { *(.tohost) } > ram
  .rodata    : { *(.rodata) *(.rodata.*) *(.srodata) *(.srodata.*) } > ram
  .data      : { *(.data) *(.data.*) } > ram
  __global_pointer$ = . + 0x800;
  .sdata     : { *(.sdata) *(.sdata.*) } > ram
  . = ALIGN(4);
  __bss_start = .;
  .sbss      : { *(.sbss) *(.sbss.*) } > ram
  .bss       : { *(.bss) *(.bss.*) *(COMMON) } > ram
  . = ALIGN(4);
  __bss_end = .;
  . = ALIGN(16);
  . += 0x4000;
  __stack_top = .;
  _end = .;
}
