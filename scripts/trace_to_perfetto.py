#!/usr/bin/env python3
"""Convert a Coresample commit log into a Perfetto trace (Chrome JSON format).

The trace has one track for the hart with two levels of slices:
  * function frames, nested by call depth: a flame graph of the run;
  * one slice per retired instruction, named by its disassembly, with the
    register write and memory access in its arguments.

Open the result at https://ui.perfetto.dev (Open trace file).

Time axis: the single-cycle core retires one instruction per cycle, and
instruction n covers [20 + 10n, 30 + 10n) ns, exactly the span of cycle n in
the simulator's VCD (100 MHz, after two reset cycles). A slice in Perfetto
and the same moment in the waveforms therefore share one timestamp.

Calls and returns follow the RISC-V calling convention hints: jal/jalr that
write ra or t0 are calls, `jalr x0, 0(ra|t0)` is a return. Any other jump
that lands exactly on the first instruction of a different function is a tail
call (`tail f` at -O2): the callee's frame replaces the caller's, as on the
real stack. The function of a frame is the ELF symbol that contains the
callee's first instruction.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tests import Commit, parse_commit_log  # noqa: E402

RESET_CYCLES = 2
CYCLE_NS = 10
OPC_JAL, OPC_JALR = 0x6F, 0x67
LINK_REGS = {1, 5}  # ra, t0


@dataclass(frozen=True)
class Symbol:
    addr: int
    size: int
    name: str

    def contains(self, pc: int) -> bool:
        return self.addr <= pc < self.addr + self.size


def cycle_start_us(n: int) -> float:
    """Start of cycle n in microseconds, the unit of Chrome JSON timestamps."""
    return round((RESET_CYCLES * CYCLE_NS + n * CYCLE_NS) / 1000, 6)


def symbol_at(symbols: list[Symbol], pc: int) -> Symbol | None:
    i = bisect.bisect_right([s.addr for s in symbols], pc) - 1
    if i >= 0 and symbols[i].contains(pc):
        return symbols[i]
    return None


def load_symbols(elf: Path, nm: str = "riscv64-unknown-elf-nm") -> list[Symbol]:
    """Code symbols, sorted by address. Labels without a size (like _start)
    extend to the next symbol."""
    out = subprocess.run([nm, "-n", "-S", "--defined-only", str(elf)],
                         capture_output=True, text=True, check=True).stdout
    raw = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "tT":
            raw.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
        elif len(parts) == 3 and parts[1] in "tT":
            raw.append((int(parts[0], 16), None, parts[2]))
    symbols = []
    for i, (addr, size, name) in enumerate(raw):
        if size is None:
            size = (raw[i + 1][0] - addr) if i + 1 < len(raw) else 4
        symbols.append(Symbol(addr, size, name))
    return symbols


def load_disassembly(elf: Path, objdump: str = "riscv64-unknown-elf-objdump") -> dict[int, str]:
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(elf)],
                         capture_output=True, text=True, check=True).stdout
    disasm = {}
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(.+)$", line)
        if m:
            text = m.group(2).split("#")[0].split("<")[0].strip()
            disasm[int(m.group(1), 16)] = re.sub(r"\s+", " ", text)
    return disasm


def classify(c: Commit) -> str:
    """'call', 'ret', 'jump' (no link) or '' for one retired instruction."""
    opcode = c.insn & 0x7F
    rd = (c.insn >> 7) & 0x1F
    rs1 = (c.insn >> 15) & 0x1F
    if opcode == OPC_JAL:
        return "call" if rd in LINK_REGS else "jump" if rd == 0 else ""
    if opcode == OPC_JALR:
        if rd in LINK_REGS:
            return "call"
        if rd == 0:
            return "ret" if rs1 in LINK_REGS else "jump"
    return ""


def describe_effects(c: Commit) -> dict[str, str]:
    args = {}
    if c.rd is not None:
        args["rd"] = f"x{c.rd} = 0x{c.rd_val:08x}"
    if c.mem_addr is not None:
        if c.mem_val is None:
            args["mem"] = f"load 0x{c.mem_addr:08x}"
        else:
            args["mem"] = f"store 0x{c.mem_addr:08x} = 0x{c.mem_val:x}"
    return args


def build_events(commits: list[Commit], symbols: list[Symbol], disasm: dict[int, str]) -> list[dict]:
    def name_of(pc: int) -> str:
        sym = symbol_at(symbols, pc)
        return sym.name if sym else f"0x{pc:08x}"

    events: list[dict] = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Coresample core"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "hart 0"}},
    ]
    frames: list[dict] = []
    stack: list[tuple[str, int]] = []  # (function, first cycle)

    def close(name: str, first: int, end: int) -> None:
        frames.append({
            "ph": "X", "cat": "function", "name": name, "pid": 1, "tid": 1,
            "ts": cycle_start_us(first), "dur": round((end - first) * CYCLE_NS / 1000, 6),
            "args": {"first_cycle": first, "cycles": end - first},
        })

    for n, c in enumerate(commits):
        if not stack:
            stack.append((name_of(c.pc), n))
        events.append({
            "ph": "X", "cat": "instruction", "name": disasm.get(c.pc, f"0x{c.insn:08x}"),
            "pid": 1, "tid": 1, "ts": cycle_start_us(n), "dur": CYCLE_NS / 1000,
            "args": {"cycle": n, "pc": f"0x{c.pc:08x}", "insn": f"0x{c.insn:08x}", **describe_effects(c)},
        })
        kind = classify(c)
        if kind == "call" and n + 1 < len(commits):
            stack.append((name_of(commits[n + 1].pc), n + 1))
        elif kind == "ret" and len(stack) > 1:
            name, first = stack.pop()
            close(name, first, n + 1)
        elif kind == "jump" and n + 1 < len(commits):
            target = commits[n + 1].pc
            sym = symbol_at(symbols, target)
            if sym and sym.addr == target and sym.name != stack[-1][0]:  # tail call
                name, first = stack.pop()
                close(name, first, n + 1)
                stack.append((sym.name, n + 1))
    while stack:  # the program halted with frames still open
        name, first = stack.pop()
        close(name, first, len(commits))
    return events + frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elf", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path, help="commit log written by cs_sim --trace")
    ap.add_argument("-o", "--output", required=True, type=Path)
    prefix = os.environ.get("RISCV_PREFIX", "riscv64-unknown-elf-")
    args = ap.parse_args()

    commits = parse_commit_log(args.log.read_text())
    symbols = load_symbols(args.elf, prefix + "nm")
    disasm = load_disassembly(args.elf, prefix + "objdump")
    events = build_events(commits, symbols, disasm)
    args.output.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}))
    n_frames = sum(e.get("cat") == "function" for e in events)
    print(f"{args.output}: {len(commits)} instructions, {n_frames} function frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
