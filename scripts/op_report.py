#!/usr/bin/env python3
"""Where the cycles of a compiled network go, per layer and operation.

Single-cycle core: one cycle per retired instruction, so the report counts
instructions per region of the `.cs_map` map.

Pipeline (`--pipeline LOG` from cs_sim --pipeline): every cycle is charged to
exactly one row. A retired instruction is one cycle of its region; a lost
cycle is charged where pipeline_to_kanata.account blames it: a load-use stall
to the instruction that waited, a flush to the branch or jump that caused it,
the initial fill to the pipeline itself. The rows add up to the total.

    op_report.py --elf cnn1.elf --log cnn1.core.log [--pipeline cnn1.pipeline.log]
"""
from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_to_kanata as p2k  # noqa: E402
from csmap import RegionMap  # noqa: E402
from run_tests import parse_commit_log  # noqa: E402

OUTSIDE = "(outside the network)"
FILL_ROW = ("(pipeline)", "fill")


@dataclass
class Row:
    function: str
    layer: int | None
    op: str
    instructions: int = 0
    bubbles: dict[str, int] = field(default_factory=dict)

    @property
    def cycles(self) -> int:
        return self.instructions + sum(self.bubbles.values())


class _Table:
    def __init__(self, regions: RegionMap):
        self.regions = regions
        self.rows: dict[tuple[str, str], Row] = {}

    def row(self, pc: int | None) -> Row:
        if pc is None:
            key, layer = FILL_ROW, None
        else:
            r = self.regions.at(pc)
            key, layer = ((r.function, r.op), r.layer) if r else ((OUTSIDE, "-"), None)
        if key not in self.rows:
            self.rows[key] = Row(key[0], layer, key[1])
        return self.rows[key]

    def sorted(self) -> list[Row]:
        return sorted(self.rows.values(), key=lambda r: -r.cycles)


def report(retired_pcs: list[int], regions: RegionMap) -> list[Row]:
    """Single-cycle core: instructions (= cycles) per region."""
    t = _Table(regions)
    for pc in retired_pcs:
        t.row(pc).instructions += 1
    return t.sorted()


def report_pipeline(cycles: list[p2k.Cycle], regions: RegionMap) -> list[Row]:
    """Pipeline: every cycle charged to one row."""
    acc = p2k.account(cycles)
    pc_of = p2k.pcs_by_id(cycles)
    t = _Table(regions)
    for pc in p2k.retired_pcs(cycles):
        t.row(pc).instructions += 1
    for cause, culprit in acc.blame:
        row = t.row(pc_of[culprit] if culprit is not None else None)
        row.bubbles[cause] = row.bubbles.get(cause, 0) + 1
    return t.sorted()


def format_table(rows: list[Row]) -> str:
    total = sum(r.cycles for r in rows)
    causes = sorted({c for r in rows for c in r.bubbles}, key=lambda c: -sum(r.bubbles.get(c, 0) for r in rows))
    head = f"{'function.operation':<28}{'cycles':>10}{'share':>8}{'instr':>10}{'CPI':>7}" + "".join(
        f"{c:>11}" for c in causes)
    lines = [head, "-" * len(head)]
    for r in rows:
        cpi = f"{r.cycles / r.instructions:.2f}" if r.instructions else "-"
        lines.append(f"{r.function + '.' + r.op:<28}{r.cycles:>10}{100 * r.cycles / total:>7.1f}%"
                     f"{r.instructions:>10}{cpi:>7}" + "".join(f"{r.bubbles.get(c, 0):>11}" for c in causes))
    by_layer: dict[str, int] = collections.Counter()
    for r in rows:
        by_layer[r.function] += r.cycles
    lines.append("-" * len(head))
    lines.append(f"{'total':<28}{total:>10}")
    lines.append("per function: " + ", ".join(f"{f} {100 * n / total:.1f}%" for f, n in by_layer.most_common()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elf", required=True, type=Path, help="built by cs-compile (has a .cs_map section)")
    ap.add_argument("--log", type=Path, help="commit log (single-cycle core)")
    ap.add_argument("--pipeline", type=Path, help="cs_sim --pipeline log (pipelined core)")
    args = ap.parse_args()
    regions = RegionMap.load(args.elf)
    if args.pipeline:
        rows = report_pipeline(p2k.parse_pipeline_log(args.pipeline.read_text()), regions)
    elif args.log:
        rows = report([c.pc for c in parse_commit_log(args.log.read_text())], regions)
    else:
        ap.error("give --log or --pipeline")
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
