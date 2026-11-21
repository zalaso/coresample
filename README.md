# Coresample

A core sample of a neural network: follow one token from the model, through the
compiled instructions, down to the pipeline stall that slowed it down, on a
RISC-V CPU designed from scratch and verified at every level.

> Status: phase 2 complete. G2 reached: a quantized CNN, compiled by the project's own compiler, runs on the core with output identical to the int8 reference, on the single-cycle core and on the 5-stage pipeline, whose every lost cycle is traced to its cause. Phase 1: Single-cycle RV32IM core passes riscv-tests (rv32ui, rv32um) and riscv-arch-test (I, M, Zmmul, Zifencei), with every instruction identical to Spike. A C program (one int8 dense layer) runs on it and can be inspected as a Perfetto flame graph and as VCD waveforms. The network simulator (Rust) trains an MLP, a small CNN and a one-block GPT that match PyTorch step by step.

## Levels and what checks them

| Level | What is built | Checked against |
| --- | --- | --- |
| Model | Neural network simulator with autograd, then int8 quantization | PyTorch, within a tolerance |
| Compiler | Network to RV32IM code, with an instruction → operation → layer map | The int8 reference model |
| CPU | RV32IM core in SystemVerilog, simulated with Verilator | Spike, riscv-tests, riscv-arch-test |
| Inspector | Multi-resolution trace index and a cross-level view | Cycle totals agree at every level |

## Layout

```
hw/rtl/       SystemVerilog sources: cs_core.sv (single-cycle), cs_pipe.sv (5-stage pipeline)
hw/sim/       Verilator testbench: ELF loader, RAM, Spike-format commit log
sw/env/       Minimal environment to run riscv-tests on the core and on Spike
sw/arch-test/ riscv-arch-test (ACT4) configuration for the core
sw/tests/     Directed tests for cases the suites miss (e.g. FENCE.I on the next instruction)
sw/runtime/   Start-up code and linker script for C programs
sw/programs/  C programs for the core (dense_int8: one int8 dense layer with ReLU)
scripts/      Dependency installer, test runners, Spike comparison, trace converter
sim/nn/       Network simulator in Rust: tensors, autograd graph, training, int8 quantization
compiler/     Compiler from a quantized network to an RV32IM program, with a region map
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
make compiler-test        # compiled networks on the core, checked against the int8 reference
make g2                   # the int8 CNN on the core: all 48 images, plus Spike and a trace
make g2 CORE=pipe         # every target also runs on the pipeline; this one adds a Kanata trace
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

The bars CNN (conv 3x3 + ReLU, max pool, linear) is quantized the same way and
described as a `QNetwork`, a list of quantized layers: the input of the
compiler. Every integer of every layer matches PyTorch's qnnpack (12288 conv
values, 3072 pooled, 144 outputs), and both classify 46 of 48 images. In the
int8 convolution, padding is a real zero, stored as the input zero point.

## Compiler (G2)

`compiler/` turns a `QNetwork` into RV32IM assembly. Each layer is a function
(`layer0_conv2d`, `layer1_max_pool2d`, `layer3_linear`), so the Perfetto flame
graph shows layers directly. Every instruction belongs to a region naming its
operation (`bias`, `mac`, `requant`, `store`, `max`, `loop`, ...), and the ELF
carries the region addresses in a `.cs_map` section: the instruction →
operation → layer map is exact by construction, not recovered from debug info.

The program runs the network on every sample and compares its output with the
expected bytes from the Rust int8 reference. `make g2` runs the bars CNN on all
48 images in 3.6 million cycles (about 75,000 per image, 89% of them in the
convolution) and passes on the core and on Spike; on one image every retired
instruction matches Spike's. The three checks split the work: Spike checks the
CPU, the Rust int8 reference checks the compiler, PyTorch checks the reference.
The generated code is deliberately plain (loops and bounds checks); making it
fast is a later step that the inspector will measure.

## 5-stage pipeline

`hw/rtl/cs_pipe.sv` is a classic IF ID EX MEM WB pipeline: forwarding from MEM
and WB into EX, a register-file bypass from WB, a one-cycle stall on load-use,
branches and jumps resolved in EX (a taken one discards the two younger
instructions), FENCE.I as a refetch. It must retire exactly what the
single-cycle core retires: every suite runs on it (`CORE=pipe`) with the same
instruction-by-instruction comparison against Spike.

With `--pipeline` the simulator logs which instruction is in each stage every
cycle; `scripts/pipeline_to_kanata.py` turns that into a Kanata trace for
[Konata](https://github.com/shioyadan/Konata), with every stall explained (and
an arrow to the load it waits for) and every discarded instruction attributed
to the branch or jump that discarded it. It also accounts for every cycle: an
empty write-back slot must have exactly one cause. On one CNN image:

| | cycles | share |
| --- | ---: | ---: |
| instructions retired | 75,805 | 86.1% |
| taken branches (2 discarded instructions each) | 7,426 | 8.4% |
| load-use stalls | 4,515 | 5.1% |
| jumps | 280 | 0.3% |
| pipeline fill | 4 | 0.0% |
| **total** | **88,030** | CPI 1.161 |

Taken branches cost more than load-use stalls: the compiler's loops branch
back on almost every iteration. That is where optimization should start.

The existing FENCE.I tests did not notice a pipeline whose FENCE.I does
nothing, because they always jump to the new code and the jump refetches.
`sw/tests/fence_i_next.S` patches the very next instruction, which only
FENCE.I can make visible, and catches it.
