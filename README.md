# Coresample

A core sample of a neural network: follow one token from the model, through the
compiled instructions, down to the pipeline stall that slowed it down, on a
RISC-V CPU designed from scratch and verified at every level.

> Status: phase 1, step 1. Single-cycle RV32IM core passes riscv-tests rv32ui and rv32um, with every instruction identical to Spike.

## Levels and what checks them

| Level | What is built | Checked against |
| --- | --- | --- |
| Model | Neural network simulator with autograd, then int8 quantization | PyTorch, within a tolerance |
| Compiler | Network to RV32IM code, with an instruction → operation → layer map | The int8 reference model |
| CPU | RV32IM core in SystemVerilog, simulated with Verilator | Spike, riscv-tests, riscv-arch-test |
| Inspector | Multi-resolution trace index and a cross-level view | Cycle totals agree at every level |

## Layout

```
hw/rtl/       SystemVerilog sources (cs_core.sv: single-cycle RV32IM core)
hw/sim/       Verilator testbench: ELF loader, RAM, Spike-format commit log
sw/env/       Minimal environment to run riscv-tests on the core and on Spike
scripts/      Dependency installer, test runner and Spike comparison
docs/         Roadmap and decisions
```

## Getting started (Ubuntu / Linux Mint)

```bash
scripts/install-deps.sh   # Verilator, RISC-V GCC, Spike, riscv-tests (needs sudo)
make test                 # run riscv-tests rv32ui + rv32um on the core
make compare              # same, plus an instruction-by-instruction check against Spike
```

Traces end up in `build/traces/`, one commit log per test for the core and for Spike.

## How verification works

Every test ELF runs twice: on the core (through Verilator) and on Spike, the
RISC-V reference simulator. Both write a commit log, one line per retired
instruction with its register write and memory access. A test passes only if
the program reports success and the two logs are identical.
