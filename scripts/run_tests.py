#!/usr/bin/env python3
"""Run test ELFs on the Coresample simulator and, optionally, check them
instruction by instruction against Spike.

A test passes when:
  * the simulator reports PASS (the program wrote 1 to `tohost`), and
  * with --spike: every instruction the core retired matches Spike's commit
    log in pc, encoding, register write and memory access.

Both logs use Spike's commit-log line format, for example
    core   0: 3 0x80000004 (0x00000093) x1  0x00000000
    core   0: 3 0x80000100 (0x00e12023) mem 0x80002000 0x0000000a
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MASK32 = 0xFFFF_FFFF
RAM_BASE = 0x8000_0000
RAM_SIZE = 0x10_0000
LINE_RE = re.compile(r"core\s+\d+:\s+\d+\s+0x([0-9a-fA-F]+)\s+\(0x([0-9a-fA-F]+)\)(.*)")
REG_RE = re.compile(r"x(\d+)$")


@dataclass(frozen=True)
class Commit:
    pc: int
    insn: int
    rd: int | None = None
    rd_val: int | None = None
    mem_addr: int | None = None
    mem_val: int | None = None  # None for loads, the stored value for stores

    def describe(self) -> str:
        s = f"pc=0x{self.pc:08x} insn=0x{self.insn:08x}"
        if self.rd is not None:
            s += f" x{self.rd}=0x{self.rd_val:08x}"
        if self.mem_addr is not None:
            if self.mem_val is None:
                s += f" load@0x{self.mem_addr:08x}"
            else:
                s += f" store@0x{self.mem_addr:08x}=0x{self.mem_val:x}"
        return s


def parse_commit_log(text: str) -> list[Commit]:
    """Parse commit-log lines; ignores CSR writes and anything else Spike prints."""
    commits = []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        pc = int(m.group(1), 16) & MASK32
        insn = int(m.group(2), 16) & MASK32
        tokens = m.group(3).split()
        rd = rd_val = mem_addr = mem_val = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            reg = REG_RE.match(tok)
            if reg and i + 1 < len(tokens):
                if int(reg.group(1)) != 0:
                    rd, rd_val = int(reg.group(1)), int(tokens[i + 1], 16) & MASK32
                i += 2
            elif tok == "mem" and i + 1 < len(tokens):
                mem_addr = int(tokens[i + 1], 16) & MASK32
                i += 2
                if i < len(tokens) and tokens[i].startswith("0x"):
                    mem_val = int(tokens[i], 16) & MASK32
                    i += 1
            else:
                i += 2 if i + 1 < len(tokens) else 1  # e.g. "c768_mstatus 0x..."
        commits.append(Commit(pc, insn, rd, rd_val, mem_addr, mem_val))
    return commits


def first_divergence(core: list[Commit], ref: list[Commit]) -> str | None:
    """Every instruction the core retired must equal Spike's at the same index."""
    for i, c in enumerate(core):
        if i >= len(ref):
            return f"Spike stopped after {len(ref)} instructions, the core retired {len(core)}"
        if c != ref[i]:
            return (f"instruction #{i} differs\n"
                    f"      core : {c.describe()}\n"
                    f"      spike: {ref[i].describe()}")
    return None


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_one(elf: Path, sim: str, spike: str | None, trace_dir: Path) -> tuple[bool, str]:
    core_log = trace_dir / f"{elf.stem}.core.log"
    try:
        res = run([sim, "--trace", str(core_log), str(elf)], timeout=60)
    except subprocess.TimeoutExpired:
        return False, "simulator timed out"
    verdict = next((l for l in res.stdout.splitlines() if l.startswith("RESULT")), None)
    if verdict != "RESULT PASS":
        detail = verdict or (res.stderr.strip().splitlines() or ["no RESULT line"])[-1]
        return False, detail

    if spike is None:
        return True, "pass"

    spike_log = trace_dir / f"{elf.stem}.spike.log"
    try:
        sres = run([spike, "--isa=rv32im", f"-m0x{RAM_BASE:x}:0x{RAM_SIZE:x}",
                    "--log-commits", str(elf)], timeout=60)
    except subprocess.TimeoutExpired:
        return False, "spike timed out"
    spike_log.write_text(sres.stderr)
    ref = [c for c in parse_commit_log(sres.stderr) if c.pc >= RAM_BASE]  # skip boot ROM
    core = parse_commit_log(core_log.read_text())
    diff = first_divergence(core, ref)
    if diff:
        return False, "diverges from Spike: " + diff
    return True, f"pass, {len(core)} instructions identical to Spike"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", required=True, help="path to the cs_sim binary")
    ap.add_argument("--spike", help="path to spike; enables the step-by-step comparison")
    ap.add_argument("--trace-dir", default="build/traces")
    ap.add_argument("elfs", nargs="+", type=Path)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for elf in sorted(args.elfs):
        ok, detail = run_one(elf, args.sim, args.spike, trace_dir)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {elf.stem:<16} {detail}")
    total = len(args.elfs)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
