# Coresample

A core sample of a neural network: follow one token from the model, through the
compiled instructions, down to the pipeline stall that slowed it down, on a
RISC-V CPU designed from scratch and verified at every level.

> Status: phase 1. Single-cycle RV32IM core passes riscv-tests (rv32ui, rv32um) and riscv-arch-test (I, M, Zmmul, Zifencei), with every instruction identical to Spike. A C program (one int8 dense layer) runs on it and can be inspected as a Perfetto flame graph and as VCD waveforms. The network simulator (Rust) trains an MLP, a small CNN and a one-block GPT that match PyTorch step by step.

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
sw/arch-test/ riscv-arch-test (ACT4) configuration for the core
sw/runtime/   Start-up code and linker script for C programs
sw/programs/  C programs for the core (dense_int8: one int8 dense layer with ReLU)
scripts/      Dependency installer, test runners, Spike comparison, trace converter
sim/nn/       Network simulator in Rust: tensors, autograd graph, training
docs/         Roadmap and decisions
```

## Getting started (Ubuntu / Linux Mint)

```bash
scripts/install-deps.sh   # Verilator, RISC-V GCC, Spike, riscv-tests, riscv-arch-test (needs sudo)
make test                 # run riscv-tests rv32ui + rv32um on the core
make compare              # same, plus an instruction-by-instruction check against Spike
make arch-test            # riscv-arch-test, self-checking, plus the same check against Spike
make trace                # run dense_int8, check it against Spike, write its trace and waveforms
make unit                 # unit tests of the Python tools
make nn-test              # network simulator tests against PyTorch (needs cargo)
```

Spike must be on `PATH` (the installer puts it in `~/.local/bin`), or pass `SPIKE=/path/to/spike`.
Traces end up in `build/traces/`, one commit log per test for the core and for Spike.

## How verification works

Every test ELF runs twice: on the core (through Verilator) and on Spike, the
RISC-V reference simulator. Both write a commit log, one line per retired
instruction with its register write and memory access. A test passes only if
the program reports success and the two logs are identical.

### riscv-arch-test

The official ACT4 flow computes expected results with the Sail model and needs
mise, uv, Ruby and Sail. `scripts/build_arch_tests.py` runs the same pipeline
with Spike as the reference instead: it compiles each test in signature mode,
runs it on Spike, post-processes the signature with ACT's own code and
recompiles the test as a self-checking ELF. The core then runs that ELF, and
its commit log is compared with Spike's as above.

Tests are selected from their header: only those whose required extensions the
core implements (I, M, Zmmul, Zifencei). The core has no CSRs, so tests that
need trap handlers are out of reach for now.

## Looking at a run

`make trace` writes three views of the same run to `build/programs/dense_int8/`:

| File | Open with | Shows |
| --- | --- | --- |
| `dense_int8.perfetto.json` | [ui.perfetto.dev](https://ui.perfetto.dev): drag the file onto the page | Flame graph of function calls; one slice per instruction with its register write and memory access |
| `dense_int8.vcd` | [Surfer](https://surfer-project.org) (web app or desktop) | Every signal of the core, clock by clock, including the 32 registers |
| `dense_int8.core.log` | any text editor | Spike-format commit log, identical to Spike's |

Both viewers use the same time axis: cycle n is the span [20 + 10n, 30 + 10n) ns
(a nominal 100 MHz clock after two reset cycles), so a slice in Perfetto and the
same moment in the waveforms carry the same timestamp. The tests in
`scripts/tests/` check that the levels agree: every retired instruction is in
the trace, frames nest, every instruction runs inside the function the symbol
table assigns it, call counts match the program, and the VCD's program counter
matches the commit log cycle by cycle.

## License

Apache License 2.0, see [LICENSE](LICENSE). Two files in `sw/arch-test/`
(`rvmodel_macros.h`, `link.ld`) are adapted from riscv-arch-test and stay under
its BSD 3-Clause license, see [sw/arch-test/LICENSE.BSD](sw/arch-test/LICENSE.BSD).
riscv-tests, riscv-arch-test and Spike are downloaded by `scripts/install-deps.sh`
into `third_party/` and `~/.local`; they are not part of this repository.

## Network simulator

`sim/nn` is a Rust crate with no dependencies: f32 tensors, a computation graph
whose nodes are explicit operations (so later stages can point back at them),
reverse-mode autograd and SGD. Every operation, forward and backward, is
compared with PyTorch, element by element, within |a - b| <= 1e-5 + 1e-4 |b|.

The reference values come from `sim/nn/reference/gen_reference.py` (PyTorch
2.5.1, fixed seeds, one thread) and are committed in `sim/nn/tests/data`, so
`cargo test` does not need PyTorch. Besides PyTorch, the gradients are checked
by finite differences. A 2-16-3 MLP trained for 100 SGD steps on a spiral
dataset stays within 4e-7 of PyTorch in every loss and every final weight. A
small CNN (conv 3x3, ReLU, max pool, linear) trained for 60 steps on synthetic
8x8 images of bars stays within 3e-7. Convolution and pooling use direct loops
on NCHW tensors, and max pooling breaks ties like PyTorch (first maximum).

The transformer is a one-block GPT (pre-norm, causal multi-head attention,
GELU MLP) in `sim/nn/src/models.rs`, built from elementary graph operations:
embedding, layer norm, matmul, permute, batched matmul, scale, causal mask,
softmax, GELU. The PyTorch reference is written with the same operations, so
the two graphs correspond node by node. Trained for 80 SGD steps on a copy
task, it stays within 8e-7 of PyTorch. The reference run must be stable: the
generator rejects runs whose loss ever rises, because where training spikes,
rounding differences grow exponentially and no two implementations agree.

### int8 quantization

`sim/nn/src/quant.rs` runs a network with integers only, the arithmetic the
core will execute: uint8 activations with a zero point, int8 symmetric weights,
int32 bias and accumulator, and requantization by a fixed-point multiplier
M = m0 / 2^shift with a single rounding (on RV32IM: `mul`, `mulh`, add, shift).

It is checked against PyTorch's own quantized kernels (qnnpack, static
post-training quantization). On the spiral MLP every integer matches PyTorch:
input and weight quantization, all 1536 hidden values and all 288 outputs. The
tests allow 1 in the last place, since PyTorch requantizes in floating point,
but require 99% of values to be identical: a truncating requantization stays
within 1 and only the 99% rule catches it. This integer pipeline is the
reference the compiled program on the core must match bit for bit.
