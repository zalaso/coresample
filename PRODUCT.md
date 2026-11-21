# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML, CSS and JavaScript in `web/`, no build step and no npm
dependencies, publishable as is on GitHub Pages. Chosen by the user.

## Users

Engineers and researchers in computer architecture and ML systems who reach
the project from its GitHub repository and want to understand, in depth,
where a neural network's cycles go on a real core. They use it as an analysis
tool, not as a teaching simplification. Interface language: English.

## Product Purpose

Coresample follows one input of a neural network from the model, through the
compiled RISC-V instructions, down to the pipeline stall that slowed it down,
on a core designed from scratch. The inspector is where that becomes visible:
a run explored across levels (run, image, layer, operation, instruction,
cycle), with every number exact. Success: a technical visitor can answer
"which operation costs the most, and why" for a real run, and trust the answer.

## Positioning

Every level is built and verified in the same project: the network against
PyTorch, the int8 pipeline against PyTorch's quantized kernels, the compiler
against the int8 reference bit for bit, the CPU against Spike and the official
test suites. The compiler records which operation each instruction belongs to
(`.cs_map`), so the cross-level map is exact by construction, not recovered
from debug info. Cycle accounting is exact: every lost cycle has exactly one
cause, and totals agree at every level.

## Operating Context

The inspector reads a precomputed trace index (`scripts/trace_index.py`):
`index.json` with the call tree (run, image, layer), a pyramid of cycle bins
counting cycles per region and bubbles per cause, the region table, and raw
4096-cycle chunks loaded on demand. Today's run: a quantized CNN (conv 3x3 +
ReLU, max pool, linear) classifying 48 8x8 images, on a single-cycle RV32IM
core (3.6 M cycles) and on a 5-stage pipeline (4.2 M cycles). Detailed views
already exist elsewhere and remain reachable: Perfetto (flame graph JSON),
Konata (Kanata pipeline traces), Surfer (VCD waveforms).

## Capabilities and Constraints

- Levels: run → image (sample) → layer (function) → operation (region: bias,
  mac, requant, store, max, loop, prologue, epilogue, check) → instruction → cycle.
- Bubble causes on the pipeline: load-use, branch, jump, fence.i, trap, fill.
  A load-use stall is charged to the instruction that waited, a flush to the
  branch or jump that caused it.
- Numbers shown must be exact counts from the index; no estimates, no
  smoothing presented as data.
- Undecided: the first view to land on, and whether the run comparison
  (single-cycle versus pipeline, before versus after an optimization) lives in
  the same screen. The transformer is not on the core yet: no "token" level.

## Evidence on Hand

Real data only: `build/index` and `build/index-pipe` (48-image runs), the
per-operation report (`scripts/op_report.py`), the G2 release artifacts. No
users, testimonials or benchmarks against other tools exist; do not invent them.

## Product Principles

- Exact before pretty: every figure on screen is a count the tests verify.
- Every level links to the levels around it; nothing is a dead end.
- Show the cause, not only the cost: lost cycles always carry their reason.
- Depth without simplification: the visitor can always reach the instruction
  and the cycle.
