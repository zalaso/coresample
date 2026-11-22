# Coresample

**What actually happens inside a computer when a neural network runs?**

Coresample answers that question for one real network, all the way down. It
follows an input from the model, through the integers it becomes, through the
RISC-V instructions a compiler turns it into, down to the single clock cycle a
processor lost waiting for a value. Every piece of that chain is built from
scratch in this repository, and every piece is checked against an independent
reference.

**Try it in your browser: [zalaso.github.io/coresample](https://zalaso.github.io/coresample/)**
(a small image-classifying network, 48 images, about 4 million cycles, every
one accounted for).

The name comes from geology. A core sample is a cylinder drilled straight down
through the ground, so you can see every layer at once, in order, with nothing
smoothed over. This project does the same with the software and hardware stack
under a neural network.

## Why I built it

Most explanations of how AI runs on hardware stay at one level, or simplify
until the real mechanism disappears. You read about matrix multiplications, or
about pipelines and caches, but rarely about how one becomes the other. I
wanted to see the whole stack at once, for real and at full depth: the actual
network, the actual instructions, the actual cycles.

A project like this used to take a team: a model, a quantizer, a compiler, a
processor and the tools to look inside them. With an AI agent doing much of
the heavy lifting, one person can attempt it, so I chose to aim high instead
of building a simplified toy. The agent (Claude, by Anthropic) wrote most of
the code under my direction; the tests decide what is correct, not the agent.

Depth alone is not enough, though. A system this deep is only useful if you can
look into it, so everything is made visible: you can start from the whole run
and click down to a single instruction, its encoding and the stall that
delayed it.

## What you can see

The [inspector](https://zalaso.github.io/coresample/) shows where the cycles of
a run go, level by level:

- **run → image → layer → operation**, drawn as bands proportional to the
  cycles they take;
- a **timeline** of every cycle, retired or lost, with the reason it was lost;
- the **instructions** of any operation, with how many cycles each one cost and
  its binary encoding split into fields;
- two processors side by side: a simple **single-cycle** core and a
  **5-stage pipeline**, and exactly what the pipeline adds to each operation.

A few things it shows about this network:

- **80%** of all cycles are spent in one loop: the multiply-accumulate of the
  convolution.
- The pipeline needs **574,308** cycles more than the single-cycle core for the
  same 3.6 million instructions. 356,388 of them are lost to taken branches and
  216,720 to instructions waiting for a value just loaded from memory.
- Not every image costs the same: on the single-cycle core they range from
  75,160 to 75,192 cycles, depending on the data.

## How it works

Five layers, each built in this repository and each checked against something
independent of it:

```
 model ──────▶ int8 integers ──────▶ compiler ──────▶ RISC-V core ──────▶ inspector
 Rust, with      the arithmetic       network to        SystemVerilog,      trace index
 autograd        a small chip runs    RV32IM assembly   two designs         and web page
```

| Layer | What it is | Checked against |
| --- | --- | --- |
| **Model** (`sim/nn`) | A neural network simulator in Rust with automatic differentiation. It trains an MLP, a small CNN and a one-block GPT. | PyTorch, step by step; gradients also by finite differences |
| **Quantization** (`sim/nn`) | The trained network rewritten to use small integers only (int8), the way it would run on a microcontroller | PyTorch's own quantized kernels, every integer |
| **Compiler** (`compiler/`) | Turns the quantized network into RISC-V assembly, and records which layer and operation every instruction belongs to | The integer reference, byte for byte, with the program running on the core |
| **Processor** (`hw/`) | An RV32IM core in SystemVerilog, simulated with Verilator, in two versions: single-cycle and a 5-stage pipeline | Spike (the reference RISC-V simulator) instruction by instruction, plus the official riscv-tests and riscv-arch-test suites |
| **Inspector** (`scripts/`, `web/`) | Traces of every cycle, indexed so millions of cycles can be explored, and the web page above | Cycle totals that must agree at every level |

The checks split the work. Spike checks the processor, the integer reference
checks the compiler, and PyTorch checks the reference. A bug in one layer is
caught by the check that owns it.

## How it was built

In three milestones, each one released:

1. **G1, a verified processor** ([v0.1.0](https://github.com/zalaso/coresample/releases/tag/v0.1.0)).
   A single-cycle RV32IM core that passes riscv-tests and matches Spike on every
   instruction, with traces you can open in Perfetto and waveforms in Surfer.
2. **G2, a network on the processor** ([v0.2.0](https://github.com/zalaso/coresample/releases/tag/v0.2.0)).
   The network simulator, int8 quantization and the compiler. A quantized CNN
   runs on the core, and its output matches the reference byte for byte.
3. **G3, everything visible** ([v0.3.0](https://github.com/zalaso/coresample/releases/tag/v0.3.0)).
   The 5-stage pipeline, cycle accounting in which every lost cycle has exactly
   one cause, a trace index and the web inspector.

The rules were the same throughout:

- Tests are written before the code.
- No milestone starts until the previous one is verified.
- Every test suite is itself tested: bugs are inserted on purpose (mutation
  tests) to prove the suite catches them.

That last rule found real gaps. For example, the official FENCE.I tests passed
on a pipeline whose FENCE.I did nothing, so the project has its own test for
that case.

The plan and its decisions are in [docs/roadmap.md](docs/roadmap.md). Each work
session is logged, with where it got stuck, in
[docs/diario-costi.md](docs/diario-costi.md). Both are in Italian.

**What's next:** run the transformer on the core, and follow one of its tokens
from the model down to the pipeline stall.

## Run it yourself

On Ubuntu or Linux Mint:

```bash
scripts/install-deps.sh   # Verilator, RISC-V GCC, Spike, the test suites, Rust, PyTorch (asks for sudo)
make compare              # the processor against the official tests and Spike
make g2 CORE=pipe         # the CNN on the pipeline: all 48 images, checked, with traces
make index && make index CORE=pipe && make web
python3 -m http.server 8765 --directory web   # the inspector at http://localhost:8765
```

<details>
<summary>All make targets</summary>

Every target runs on the single-cycle core; add `CORE=pipe` for the pipeline.

```bash
make test                 # riscv-tests rv32ui + rv32um on the core
make compare              # same, plus an instruction-by-instruction check against Spike
make arch-test            # riscv-arch-test, self-checking, plus the same check against Spike
make trace                # a small C program, with its Perfetto trace and VCD waveforms
make unit                 # tests of the Python tools
make nn-test              # network simulator against PyTorch references (needs cargo)
make nn-ref               # regenerate those references (needs PyTorch in .venv)
make compiler-test        # compiled networks on the core, against the int8 reference
make g2                   # the int8 CNN on the core: all 48 images, plus Spike and a trace
make index                # trace index of all 48 images (build/index, or build/index-pipe)
make web                  # put both indexes in web/data for the inspector
make pages                # publish web/ with its data to GitHub Pages
```

Spike must be on `PATH` (the installer puts it in `~/.local/bin`), or pass
`SPIKE=/path/to/spike`.

</details>

## Repository layout

```
hw/rtl/       The processor: cs_core.sv (single-cycle), cs_pipe.sv (5-stage pipeline)
hw/sim/       Verilator testbench: program loader, memory, commit log, pipeline log
sim/nn/       Network simulator in Rust: tensors, autograd, training, int8 quantization
compiler/     Compiler from a quantized network to RISC-V assembly, with the region map
scripts/      Test runners, trace converters, cycle accounting, trace index
web/          The inspector: a static page that reads the trace index
sw/           Test environments, C runtime, directed tests, the C example program
docs/         Roadmap and work log (Italian)
```

## Technical details

<details>
<summary>How the processor is verified</summary>

Every test program runs twice: on the core (through Verilator) and on Spike,
the RISC-V reference simulator. Both write a commit log, one line per retired
instruction with its register write and memory access. A test passes only if
the program reports success and the two logs are identical. Results: riscv-tests
50/50 and riscv-arch-test 52/52 (I, M, Zmmul, Zifencei), on both cores.

The official riscv-arch-test flow computes expected results with the Sail model
and needs mise, uv, Ruby and Sail. `scripts/build_arch_tests.py` runs the same
pipeline with Spike as the reference instead: it compiles each test in
signature mode, runs it on Spike, post-processes the signature with the suite's
own code and recompiles the test as a self-checking program. Only tests whose
required extensions the core implements are selected. The core has no CSRs, so
tests that need trap handlers are out of reach for now.

</details>

<details>
<summary>Network simulator and int8 quantization</summary>

`sim/nn` is a Rust crate with no dependencies: f32 tensors, a computation graph
whose nodes are explicit operations (so later stages can point back at them),
reverse-mode autograd and SGD. Every operation, forward and backward, is
compared with PyTorch element by element, within |a − b| ≤ 1e-5 + 1e-4·|b|.
The reference values come from `sim/nn/reference/gen_reference.py` (PyTorch
2.5.1, fixed seeds, one thread) and are committed, so `cargo test` does not
need PyTorch.

Trained step by step against PyTorch, the MLP stays within 4e-7 after 100 SGD
steps, the CNN within 3e-7 after 60, and the one-block GPT (pre-norm, causal
multi-head attention, GELU) within 8e-7 after 80. Reference runs must be
stable: where training spikes, rounding differences grow exponentially and no
two implementations agree, so the generator rejects runs whose loss ever rises.

`sim/nn/src/quant.rs` runs a network with integers only, the arithmetic the
core executes: uint8 activations with a zero point, int8 symmetric weights,
int32 bias and accumulator, and requantization by a fixed-point multiplier with
a single rounding (on RV32IM: `mul`, `mulh`, add, shift). Against PyTorch's
quantized kernels (qnnpack), every integer of every layer of the CNN matches
(12,288 convolution values, 3,072 pooled, 144 outputs), and both classify 46 of
the 48 images correctly.

</details>

<details>
<summary>Compiler</summary>

`compiler/` turns a quantized network into RV32IM assembly. Each layer is a
function (`layer0_conv2d`, `layer1_max_pool2d`, `layer3_linear`). Every
instruction belongs to a region that names its operation (`bias`, `mac`,
`requant`, `store`, `max`, `loop`, ...), and the ELF carries the region
addresses in a `.cs_map` section. So the instruction → operation → layer map is
exact by construction, not recovered from debug information.

The generated program runs the network on every sample and checks its output
against the expected bytes. On all 48 images it takes 3.6 million cycles on the
single-cycle core, passes on the core and on Spike, and on one image every
retired instruction matches Spike's. The generated code is deliberately plain
(loops and bounds checks): making it fast is a next step the inspector can
measure.

</details>

<details>
<summary>The 5-stage pipeline and cycle accounting</summary>

`hw/rtl/cs_pipe.sv` is a classic IF ID EX MEM WB pipeline: forwarding from MEM
and WB into EX, a register-file bypass from WB, a one-cycle stall when an
instruction needs a value loaded just before it (load-use), branches and jumps
resolved in EX (a taken one discards the two younger instructions), and
FENCE.I as a refetch. It must retire exactly what the single-cycle core
retires, and every suite runs on it with the same comparison against Spike.

With `--pipeline` the simulator logs which instruction is in each stage every
cycle. `scripts/pipeline_to_kanata.py` turns that into a trace for
[Konata](https://github.com/shioyadan/Konata), with every stall explained. It
also accounts for every cycle: an empty write-back slot must have exactly one
cause. One CNN image on the pipeline:

| | cycles | share |
| --- | ---: | ---: |
| instructions retired | 75,805 | 86.1% |
| taken branches | 7,426 | 8.4% |
| load-use stalls | 4,515 | 5.1% |
| jumps | 280 | 0.3% |
| pipeline fill | 4 | 0.0% |
| **total** | **88,030** | CPI 1.161 |

`scripts/op_report.py` charges every cycle to exactly one layer and operation: a
stall to the instruction that waited, a flush to the branch that caused it.
The convolution's multiply-accumulate loop takes 79.7% of the cycles.

</details>

<details>
<summary>Trace index and inspector</summary>

All 48 images on the pipeline are 4.2 million cycles, and the raw logs weigh
440 MB. `scripts/trace_index.py` streams them, with constant memory, into an
index that answers exact questions about any range of cycles:

* raw chunks of 4,096 cycles, 7 bytes per cycle (address, region, cause),
  loaded only when zooming in;
* a pyramid of bins counting cycles per region and lost cycles per cause;
* the call tree run → image → layer, and a table with every instruction's
  cycles and stall causes.

Tests compare every range of small runs with counting cycle by cycle, and the
index's totals with `op_report`'s, layer by layer.

`web/` is a static page with no build step that reads the two indexes. Its
visual style follows the RISC-V reference card printed with architecture
textbooks: card stock, ruled tables, green section bands (see
[DESIGN.md](DESIGN.md) and [PRODUCT.md](PRODUCT.md)). The public copy lives on
the `gh-pages` branch, published by `make pages` as a single commit that each
publish replaces, so the trace data never piles up in the history of `main`.

</details>

<details>
<summary>Looking at a single program</summary>

`make trace` runs a small C program (one int8 dense layer) and writes three
views of the same run to `build/programs/dense_int8/`:

| File | Open with | Shows |
| --- | --- | --- |
| `dense_int8.perfetto.json` | [ui.perfetto.dev](https://ui.perfetto.dev) | Flame graph of function calls, one slice per instruction |
| `dense_int8.vcd` | [Surfer](https://surfer-project.org) | Every signal of the core, clock by clock |
| `dense_int8.core.log` | any text editor | Commit log, identical to Spike's |

Both viewers use the same time axis, so a slice in Perfetto and the same moment
in the waveforms carry the same timestamp.

</details>

## License

Apache License 2.0, see [LICENSE](LICENSE). Two files in `sw/arch-test/`
(`rvmodel_macros.h`, `link.ld`) are adapted from riscv-arch-test and stay under
its BSD 3-Clause license, see [sw/arch-test/LICENSE.BSD](sw/arch-test/LICENSE.BSD).
riscv-tests, riscv-arch-test and Spike are downloaded by `scripts/install-deps.sh`
into `third_party/` and `~/.local`; they are not part of this repository.
