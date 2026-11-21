#!/usr/bin/env python3
"""Multi-resolution index of a run, for exploring millions of cycles.

The unit is the write-back slot of every cycle: either a retired instruction,
or a bubble charged where op_report charges it (load-use on the instruction
that waited, a flush on the branch or jump that caused it, the fill on
nobody). The index stores the run three ways:

* chunks/<k>.bin: the raw slots, `chunk_cycles` per file, 7 bytes each:
  pc (u32), region id (i16), cause (u8: 0 retired, else 1 + index in CAUSES).
* a pyramid of bins in index.json: level 0 has one bin per chunk, each level
  above merges `fanout` bins. Each bin counts cycles per region and bubbles
  per cause.
* the call tree in index.json: run -> sample -> layer, with each node's own
  and total counts.
* the code table in index.json: every instruction address that was charged a
  cycle, with its encoding, disassembly, region, cycles (its own plus the bubbles blamed
  on it), retirements and bubbles per cause.

Any range of cycles is answered exactly: whole bins from the highest level
that fits, raw slots at the edges. The builder streams the logs with constant
memory, so the 48-image run (4 million cycles) fits on a small machine.

    trace_index.py --elf cnn48.elf --log cnn48.core.log [--pipeline cnn48.pipeline.log] -o build/index
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import struct
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csmap import RegionMap  # noqa: E402
from pipeline_to_kanata import BubbleTracer, Cycle, parse_line  # noqa: E402
from trace_to_perfetto import OPC_JAL, OPC_JALR, LINK_REGS, Symbol, load_disassembly, load_symbols  # noqa: E402

CAUSES = ["fill", "load-use", "branch", "jump", "fence.i", "trap", "fetch stopped"]
OUTSIDE = -1  # region id of an instruction outside the compiled network
NOBODY = -2   # region id of a bubble blamed on no instruction (the fill)
SLOT = struct.Struct("<IhB")


@dataclass
class Stats:
    cycles: int = 0
    regions: dict[int, int] = field(default_factory=dict)
    bubbles: dict[str, int] = field(default_factory=dict)

    def add(self, region: int, cause: str | None, n: int = 1) -> None:
        self.cycles += n
        self.regions[region] = self.regions.get(region, 0) + n
        if cause is not None:
            self.bubbles[cause] = self.bubbles.get(cause, 0) + n

    def merge(self, other: Stats) -> None:
        self.cycles += other.cycles
        for k, v in other.regions.items():
            self.regions[k] = self.regions.get(k, 0) + v
        for k, v in other.bubbles.items():
            self.bubbles[k] = self.bubbles.get(k, 0) + v

    def to_json(self) -> dict:
        return {"cycles": self.cycles, "regions": {str(k): v for k, v in sorted(self.regions.items())},
                "bubbles": dict(sorted(self.bubbles.items()))}

    @classmethod
    def from_json(cls, d: dict) -> Stats:
        return cls(d["cycles"], {int(k): v for k, v in d["regions"].items()}, dict(d["bubbles"]))


@dataclass(frozen=True)
class Slot:
    pc: int
    region: int
    cause: str | None  # None: an instruction retired in this cycle


# ------------------------------------------------------------------ building

class _Node:
    def __init__(self, name: str, start: int):
        self.name, self.start, self.end = name, start, start
        self.own = Stats()
        self.children: list[_Node] = []

    def to_json(self) -> dict:
        kids = [c.to_json() for c in self.children]
        total = Stats()
        total.merge(self.own)
        for c in kids:
            total.merge(Stats.from_json(c["total"]))
        if kids:
            self.start = min(self.start, *(c["start"] for c in kids))
            self.end = max(self.end, *(c["end"] for c in kids))
        return {"name": self.name, "start": self.start, "end": self.end, "self": self.own.to_json(),
                "total": total.to_json(), "children": kids}


class _Calls:
    """Function frames from the retired instruction stream (call, return and
    tail call, as in trace_to_perfetto), with layer calls grouped into
    samples: each call of the first layer starts a new sample."""

    def __init__(self, symbols: list[Symbol], regions: RegionMap):
        self.starts = [s.addr for s in symbols]
        self.symbols = symbols
        self.layer_of: dict[str, int] = {}
        for r in regions.regions:
            if r.layer is not None:
                self.layer_of[r.function] = r.layer
        self.first_layer = min(self.layer_of, key=self.layer_of.get) if self.layer_of else None
        self.root: _Node | None = None
        self.stack: list[_Node] = []
        self.samples = 0
        self.sample_of: dict[int, _Node] = {}  # id(frame) -> its current sample node
        self.after: str | None = None  # "call", "ret" or "jump" from the previous instruction
        self._name_cache: dict[int, str] = {}

    def name(self, pc: int) -> str:
        if pc not in self._name_cache:
            i = bisect.bisect_right(self.starts, pc) - 1
            s = self.symbols[i] if i >= 0 else None
            self._name_cache[pc] = s.name if s and s.contains(pc) else f"0x{pc:08x}"
        return self._name_cache[pc]

    def _enter(self, name: str, t: int) -> None:
        node = _Node(name, t)
        parent = self.stack[-1]
        if name in self.layer_of:
            if name == self.first_layer or id(parent) not in self.sample_of:
                sample = _Node(f"sample {self.samples}", t)
                self.samples += 1
                parent.children.append(sample)
                self.sample_of[id(parent)] = sample
            self.sample_of[id(parent)].children.append(node)
        else:
            parent.children.append(node)
        self.stack.append(node)

    def retire(self, pc: int, insn: int, t: int) -> _Node:
        name = self.name(pc)
        if self.root is None:
            self.root = _Node(name, t)
            self.stack = [self.root]
        elif self.after == "call":
            self._enter(name, t)
        elif self.after == "ret" and len(self.stack) > 1:
            self.stack.pop()
        elif self.after == "jump" and name != self.stack[-1].name and self._is_entry(pc) and len(self.stack) > 1:
            self.stack.pop()  # tail call: the callee replaces the caller's frame
            self._enter(name, t)
        node = self.stack[-1]
        node.end = t + 1
        op, rd, rs1 = insn & 0x7F, (insn >> 7) & 0x1F, (insn >> 15) & 0x1F
        self.after = None
        if op == OPC_JAL:
            self.after = "call" if rd in LINK_REGS else "jump" if rd == 0 else None
        elif op == OPC_JALR:
            self.after = "call" if rd in LINK_REGS else ("ret" if rs1 in LINK_REGS else "jump") if rd == 0 else None
        return node

    def _is_entry(self, pc: int) -> bool:
        i = bisect.bisect_left(self.starts, pc)
        return i < len(self.starts) and self.starts[i] == pc


def _commits(lines: Iterable[str]) -> Iterator[tuple[int, int]]:
    """(pc, encoding) of each commit-log line."""
    for line in lines:
        tok = line.split()
        if len(tok) >= 5 and tok[0] == "core" and tok[3].startswith("0x"):
            yield int(tok[3], 16) & 0xFFFF_FFFF, int(tok[4].strip("()"), 16) & 0xFFFF_FFFF


def build(commit_lines: Iterable[str], cycle_lines: Iterable[str | Cycle] | None, symbols: list[Symbol],
          regions: RegionMap, out: Path, chunk_cycles: int = 4096, fanout: int = 16,
          disasm: dict[int, str] | None = None) -> dict:
    """Builds the index in `out` (replaced). Without `cycle_lines` every
    retired instruction is one cycle (the single-cycle core)."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "chunks").mkdir(parents=True)
    region_cache: dict[int, int] = {}

    def region_of(pc: int) -> int:
        if pc not in region_cache:
            r = regions.at(pc)
            region_cache[pc] = r.id if r else OUTSIDE
        return region_cache[pc]

    calls = _Calls(symbols, regions)
    bins: list[Stats] = []
    chunk = bytearray()
    t = 0
    instructions = 0
    nobody = Stats()  # bubbles blamed on no instruction: charged to the root at the end
    per_pc: dict[int, list] = {}  # pc -> [cycles, retirements, {cause: bubbles}]
    encoding: dict[int, int] = {}  # pc -> instruction word, from the commit log
    unattributed = 0

    def emit(pc: int, region: int, cause: str | None) -> None:
        nonlocal chunk, t, unattributed
        if region == NOBODY:
            unattributed += 1
        else:
            row = per_pc.setdefault(pc, [0, 0, {}])
            row[0] += 1
            if cause is None:
                row[1] += 1
            else:
                row[2][cause] = row[2].get(cause, 0) + 1
        if t % chunk_cycles == 0:
            bins.append(Stats())
        bins[-1].add(region, cause)
        chunk += SLOT.pack(pc, region, 0 if cause is None else CAUSES.index(cause) + 1)
        t += 1
        if t % chunk_cycles == 0:
            (out / "chunks" / f"{t // chunk_cycles - 1:06d}.bin").write_bytes(chunk)
            chunk = bytearray()

    commits = _commits(commit_lines)
    if cycle_lines is None:
        for pc, insn in commits:
            encoding[pc] = insn
            node = calls.retire(pc, insn, t)
            r = region_of(pc)
            node.own.add(r, None)
            emit(pc, r, None)
            instructions += 1
    else:
        tracer = BubbleTracer()
        pc_of: dict[int, int] = {}          # fetch id -> pc, recent instructions only
        node_of: dict[int, _Node] = {}      # fetch id -> frame it retired in, recent only
        waiting: dict[int, list[tuple[int, str]]] = {}  # bubbles blamed on not-yet-retired ids
        for item in cycle_lines:
            c = parse_line(item) if isinstance(item, str) else item
            if c.fetch is not None:
                pc_of[c.fetch] = c.fetch_pc
            bubble = tracer.push(c)
            if bubble is None:
                wid = c.stages["W"]
                try:
                    pc, insn = next(commits)
                except StopIteration:
                    raise ValueError(f"cycle {c.n}: an instruction retires but the commit log has ended") from None
                if pc != pc_of.get(wid):
                    raise ValueError(f"cycle {c.n}: commit log pc 0x{pc:08x} is not the retiring instruction")
                encoding[pc] = insn
                node = calls.retire(pc, insn, t)
                r = region_of(pc)
                node.own.add(r, None)
                node_of[wid] = node
                for region, cause in waiting.pop(wid, []):
                    node.own.add(region, cause)
                emit(pc, r, None)
                instructions += 1
            else:
                cause, culprit = bubble
                if culprit is None:
                    nobody.add(NOBODY, cause)
                    emit(0, NOBODY, cause)
                else:
                    pc = pc_of[culprit]
                    r = region_of(pc)
                    if culprit in node_of:
                        node_of[culprit].own.add(r, cause)
                    else:
                        waiting.setdefault(culprit, []).append((r, cause))
                    emit(pc, r, cause)
            if c.n % 1024 == 0:  # forget instructions long gone
                horizon = (c.stages["W"] or 0) - 64
                for d in (pc_of, node_of):
                    for k in [k for k in d if k < horizon]:
                        del d[k]
        if next(commits, None) is not None:
            raise ValueError("the commit log has instructions the pipeline log never retires")
        for items in waiting.values():  # waited for an instruction still in flight at the end
            for region, cause in items:
                nobody.add(region, cause)
    if chunk:
        (out / "chunks" / f"{(t - 1) // chunk_cycles:06d}.bin").write_bytes(chunk)
    if calls.root is None:
        raise ValueError("no instruction retired")
    calls.root.own.merge(nobody)

    levels = [{"width": chunk_cycles, "bins": bins}]
    while len(levels[-1]["bins"]) > 1:
        below = levels[-1]["bins"]
        merged = []
        for i in range(0, len(below), fanout):
            s = Stats()
            for b in below[i:i + fanout]:
                s.merge(b)
            merged.append(s)
        levels.append({"width": levels[-1]["width"] * fanout, "bins": merged})

    index = {
        "version": 1, "cycles": t, "instructions": instructions,
        "chunk_cycles": chunk_cycles, "fanout": fanout, "causes": CAUSES,
        "regions": [{"id": r.id, "function": r.function, "layer": r.layer, "op": r.op} for r in regions.regions],
        "levels": [{"width": lv["width"], "bins": [b.to_json() for b in lv["bins"]]} for lv in levels],
        "tree": calls.root.to_json(),
        "unattributed": unattributed,
        "code": [{"pc": pc, "insn": encoding.get(pc, 0), "text": (disasm or {}).get(pc, ""),
                  "region": region_of(pc), "cycles": row[0],
                  "instructions": row[1], "bubbles": dict(sorted(row[2].items()))}
                 for pc, row in sorted(per_pc.items())],
    }
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return index


def build_from_files(elf: Path, commit_log: Path, pipeline_log: Path | None, out: Path,
                     chunk_cycles: int = 4096, fanout: int = 16) -> dict:
    symbols = load_symbols(elf)
    regions = RegionMap.load(elf)
    disasm = load_disassembly(elf)
    with open(commit_log) as commits:
        if pipeline_log is None:
            return build(commits, None, symbols, regions, out, chunk_cycles, fanout, disasm)
        with open(pipeline_log) as cycles:
            return build(commits, cycles, symbols, regions, out, chunk_cycles, fanout, disasm)


# ------------------------------------------------------------------- reading

class Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        d = json.loads((self.path / "index.json").read_text())
        self.cycles: int = d["cycles"]
        self.instructions: int = d["instructions"]
        self.chunk_cycles: int = d["chunk_cycles"]
        self.causes: list[str] = d["causes"]
        self.regions = {r["id"]: r for r in d["regions"]}
        self.levels = [(lv["width"], [Stats.from_json(b) for b in lv["bins"]]) for lv in d["levels"]]
        self.tree: dict = d["tree"]
        self.code: list[dict] = d["code"]
        self.unattributed: int = d["unattributed"]
        self._chunks: dict[int, bytes] = {}

    def region_label(self, rid: int) -> str:
        """The same labels as op_report: `function.op`."""
        if rid == OUTSIDE:
            return "(outside the network).-"
        if rid == NOBODY:
            return "(pipeline).fill"
        r = self.regions[rid]
        return f"{r['function']}.{r['op']}"

    def samples(self) -> list[dict]:
        found = []

        def walk(node: dict) -> None:
            for c in node["children"]:
                if c["name"].startswith("sample "):
                    found.append(c)
                else:
                    walk(c)
        walk(self.tree)
        return found

    def _chunk(self, k: int) -> bytes:
        if k not in self._chunks:
            if len(self._chunks) > 64:
                self._chunks.clear()
            self._chunks[k] = (self.path / "chunks" / f"{k:06d}.bin").read_bytes()
        return self._chunks[k]

    def slot(self, t: int) -> Slot:
        if not 0 <= t < self.cycles:
            raise ValueError(f"cycle {t} out of range")
        pc, region, code = SLOT.unpack_from(self._chunk(t // self.chunk_cycles), (t % self.chunk_cycles) * SLOT.size)
        return Slot(pc, region, None if code == 0 else self.causes[code - 1])

    def stats(self, a: int, b: int) -> Stats:
        """Exact counts for cycles [a, b)."""
        if not 0 <= a <= b <= self.cycles:
            raise ValueError(f"range [{a}, {b}) outside [0, {self.cycles})")
        res = Stats()
        t = a
        while t < b:
            for width, bins in reversed(self.levels):
                if t % width == 0 and t + width <= b:
                    res.merge(bins[t // width])
                    t += width
                    break
            else:  # no whole bin fits: raw slots up to the next chunk boundary
                end = min(b, (t // self.chunk_cycles + 1) * self.chunk_cycles)
                while t < end:
                    s = self.slot(t)
                    res.add(s.region, s.cause)
                    t += 1
        return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elf", required=True, type=Path, help="built by cs-compile (has a .cs_map section)")
    ap.add_argument("--log", required=True, type=Path, help="commit log written by cs_sim --trace")
    ap.add_argument("--pipeline", type=Path, help="cs_sim --pipeline log (pipelined core)")
    ap.add_argument("-o", "--output", required=True, type=Path, help="index directory (replaced)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--fanout", type=int, default=16)
    args = ap.parse_args()
    index = build_from_files(args.elf, args.log, args.pipeline, args.output, args.chunk, args.fanout)
    idx = Index(args.output)
    size = (args.output / "index.json").stat().st_size
    chunks = sum(f.stat().st_size for f in (args.output / "chunks").iterdir())
    print(f"{args.output}: {index['cycles']} cycles, {index['instructions']} instructions, "
          f"{len(idx.samples())} samples, {len(index['levels'])} levels; "
          f"index.json {size / 1e3:.0f} kB, chunks {chunks / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
