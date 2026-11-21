#!/usr/bin/env python3
"""Convert the pipelined core's per-cycle log into a Kanata trace for Konata.

`cs_sim --pipeline FILE` (CORE=pipe) writes one line per cycle:
    <cycle> F <id>:<pc> D <id> X <id> M <id> W <id> [stall] [flush=<cause>]
where <id> is the instruction's fetch sequence number and "-" is a bubble.

The Kanata output shows every instruction's trip through F D X M W, explains
each stall (load-use, with an arrow to the load it waits for) and each
discarded instruction (which branch or jump flushed it).

It also accounts for every cycle: an empty write-back slot must have exactly
one cause (pipeline fill, a load-use stall, or an instruction discarded by a
branch, jump, fence.i or trap), so cycles = retired instructions + bubbles.
`--summary` prints that breakdown.

Konata: https://github.com/shioyadan/Konata (open the .kanata file).
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_to_perfetto import load_disassembly  # noqa: E402

STAGES = ("F", "D", "X", "M", "W")


@dataclass
class Cycle:
    n: int
    stages: dict[str, int | None]
    fetch: int | None = None
    fetch_pc: int | None = None
    stall: bool = False
    flush: str | None = None


def parse_pipeline_log(text: str) -> list[Cycle]:
    cycles = []
    for line in text.splitlines():
        if not line.strip():
            continue
        tok = line.split()
        if len(tok) < 11 or tok[1::2][:5] != list(STAGES):
            raise ValueError(f"malformed pipeline line: {line!r}")
        c = Cycle(n=int(tok[0]), stages={})
        for name, value in zip(STAGES, tok[2:11:2]):
            if value == "-":
                c.stages[name] = None
            elif name == "F":
                fid, pc = value.split(":")
                c.stages["F"] = c.fetch = int(fid)
                c.fetch_pc = int(pc, 16)
            else:
                c.stages[name] = int(value)
        for extra in tok[11:]:
            if extra == "stall":
                c.stall = True
            elif extra.startswith("flush="):
                c.flush = extra.split("=", 1)[1]
            else:
                raise ValueError(f"unknown field {extra!r} in {line!r}")
        cycles.append(c)
    return cycles


@dataclass
class Accounting:
    retired: int = 0
    discarded: int = 0
    bubbles: dict[str, int] = field(default_factory=dict)


def account(cycles: list[Cycle]) -> Accounting:
    """Explains every empty write-back slot by tracing the bubble back to the
    cycle that made it. Raises ValueError if a bubble has no cause."""

    def why(stage: str, t: int) -> str:
        c = cycles[t]
        if c.stages[stage] is not None:
            raise ValueError(f"cycle {t}: {stage} is not empty")
        if t == 0:
            return "fill"
        prev = cycles[t - 1]
        if stage == "D":
            if prev.flush:  # the instruction in IF was discarded
                return prev.flush
            if prev.stages["F"] is None:
                return "fill" if prev.n < 1 else "fetch stopped"
        elif stage == "X":
            if prev.stages["D"] is None:
                return why("D", t - 1)
            if prev.stall:
                return "load-use"
            if prev.flush:  # the instruction in ID was discarded
                return prev.flush
        else:  # M and W just pass on what the previous stage had
            before = STAGES[STAGES.index(stage) - 1]
            if prev.stages[before] is None:
                return why(before, t - 1)
        raise ValueError(f"cycle {t}: bubble in {stage} has no cause")

    acc = Accounting()
    counts: collections.Counter[str] = collections.Counter()
    for t, c in enumerate(cycles):
        if c.stages["W"] is None:
            counts[why("W", t)] += 1
        else:
            acc.retired += 1
    acc.bubbles = dict(counts)
    seen = {c.fetch for c in cycles if c.fetch is not None}
    retired = {c.stages["W"] for c in cycles if c.stages["W"] is not None}
    in_flight = {v for v in cycles[-1].stages.values() if v is not None} if cycles else set()
    acc.discarded = len(seen - retired - in_flight)
    return acc


def retired_pcs(cycles: list[Cycle]) -> list[int]:
    pc_of = {c.fetch: c.fetch_pc for c in cycles if c.fetch is not None}
    return [pc_of[c.stages["W"]] for c in cycles if c.stages["W"] is not None]


def to_kanata(cycles: list[Cycle], disasm: dict[int, str]) -> str:
    out = ["Kanata\t0004"]
    pc_of: dict[int, int] = {}
    where: dict[int, str] = {}  # instructions in flight -> stage
    rid = 0

    def text(i: int) -> str:
        pc = pc_of[i]
        return f"{pc:08x} {disasm.get(pc, '')}".rstrip()

    for t, c in enumerate(cycles):
        out.append(f"C=\t{c.n}" if t == 0 else "C\t1")
        now = {i: s for s, i in c.stages.items() if i is not None}
        for i, stage in where.items():
            if i in now:
                continue
            if stage == "W":
                out.append(f"R\t{i}\t{rid}\t0")
                rid += 1
            else:
                out.append(f"R\t{i}\t0\t1")
        for s in STAGES:
            i = c.stages[s]
            if i is None:
                continue
            if i not in where:
                pc_of[i] = c.fetch_pc if s == "F" and c.fetch_pc is not None else pc_of.get(i, 0)
                out.append(f"I\t{i}\t{i}\t0")
                out.append(f"L\t{i}\t0\t{text(i)}")
                out.append(f"S\t{i}\t0\t{s}")
            elif where[i] != s:
                out.append(f"E\t{i}\t0\t{where[i]}")
                out.append(f"S\t{i}\t0\t{s}")
        if c.stall and c.stages["D"] is not None and c.stages["X"] is not None:
            consumer, producer = c.stages["D"], c.stages["X"]
            out.append(f"L\t{consumer}\t1\tcycle {c.n}: stall, load-use: waits for {text(producer)}")
            out.append(f"W\t{consumer}\t{producer}\t0")
        if c.flush and c.stages["X"] is not None:
            cause = f"{c.flush} at {text(c.stages['X'])}"
            for s in ("F", "D"):
                if c.stages[s] is not None:
                    out.append(f"L\t{c.stages[s]}\t1\tcycle {c.n}: discarded by {cause}")
        where = now
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, type=Path, help="written by cs_sim --pipeline")
    ap.add_argument("--elf", type=Path, help="for disassembly in the labels")
    ap.add_argument("-o", "--output", type=Path, help="Kanata file to write")
    ap.add_argument("--summary", action="store_true", help="print the cycle accounting")
    args = ap.parse_args()

    cycles = parse_pipeline_log(args.log.read_text())
    acc = account(cycles)
    if args.output:
        disasm = {}
        if args.elf:
            disasm = load_disassembly(args.elf, os.environ.get("RISCV_PREFIX", "riscv64-unknown-elf-") + "objdump")
        args.output.write_text(to_kanata(cycles, disasm))
        print(f"{args.output}: {len(cycles)} cycles, {acc.retired} instructions retired")
    if args.summary:
        lost = sum(acc.bubbles.values())
        print(f"cycles {len(cycles)} = {acc.retired} retired + {lost} bubbles "
              f"(CPI {len(cycles) / max(acc.retired, 1):.3f})")
        for cause, n in sorted(acc.bubbles.items(), key=lambda kv: -kv[1]):
            print(f"  {cause:<14} {n:>9}  {100 * n / len(cycles):5.1f}% of cycles")
        print(f"  discarded instructions: {acc.discarded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
