"""Tests for scripts/trace_to_perfetto.py.

Unit tests use hand-written commit logs. The integration tests run on the
dense_int8 program built by `make programs` and are skipped if it is missing.

Run with:  python3 -m unittest discover -s scripts/tests
"""
from __future__ import annotations

import bisect
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trace_to_perfetto as t2p  # noqa: E402
from run_tests import parse_commit_log  # noqa: E402

# Encodings used below (RV32I):
JAL_RA_8 = 0x008000EF   # jal ra, +8
RET = 0x00008067        # jalr x0, 0(ra)
ADDI = 0x00000013       # addi x0, x0, 0 (nop)
J_8 = 0x0080006F        # jal x0, +8 (plain jump, not a call)
JALR_RA_A5 = 0x000780E7  # jalr ra, 0(a5) (indirect call)

SYMBOLS = [
    t2p.Symbol(0x8000_0000, 0x10, "main"),
    t2p.Symbol(0x8000_0100, 0x10, "leaf"),
]


def encode_jal_offset(off: int) -> int:
    """The immediate bits of a JAL instruction for a byte offset."""
    off &= 0x1F_FFFF
    return (((off >> 20) & 1) << 31) | (((off >> 1) & 0x3FF) << 21) | \
           (((off >> 11) & 1) << 20) | (((off >> 12) & 0xFF) << 12)


def log(*lines: tuple[int, int, str]) -> list:
    text = "\n".join(f"core   0: 3 0x{pc:08x} (0x{insn:08x}){extra}" for pc, insn, extra in lines)
    return parse_commit_log(text)


def slices(events: list[dict], cat: str) -> list[dict]:
    return [e for e in events if e.get("ph") == "X" and e.get("cat") == cat]


class UnitTests(unittest.TestCase):
    def test_cycle_n_maps_to_the_vcd_time_axis(self):
        # Instruction n covers [20 + 10n, 30 + 10n) ns, the same span as cycle n in the VCD.
        self.assertEqual(t2p.cycle_start_us(0), 0.020)
        self.assertEqual(t2p.cycle_start_us(3), 0.050)

    def test_one_slice_per_instruction(self):
        commits = log((0x8000_0000, ADDI, ""), (0x8000_0004, ADDI, ""), (0x8000_0008, ADDI, ""))
        ins = slices(t2p.build_events(commits, SYMBOLS, {}), "instruction")
        self.assertEqual(len(ins), 3)
        self.assertEqual([e["args"]["cycle"] for e in ins], [0, 1, 2])
        self.assertEqual([e["args"]["pc"] for e in ins], ["0x80000000", "0x80000004", "0x80000008"])

    def test_disassembly_names_the_instruction_slices(self):
        commits = log((0x8000_0000, ADDI, ""))
        ins = slices(t2p.build_events(commits, SYMBOLS, {0x8000_0000: "nop"}), "instruction")
        self.assertEqual(ins[0]["name"], "nop")

    def test_call_and_return_make_a_nested_frame(self):
        commits = log(
            (0x8000_0000, ADDI, ""),
            (0x8000_0004, JAL_RA_8, " x1  0x80000008"),   # call leaf (target taken from the next pc)
            (0x8000_0100, ADDI, ""),
            (0x8000_0104, RET, ""),                       # return to main
            (0x8000_0008, ADDI, ""),
        )
        frames = slices(t2p.build_events(commits, SYMBOLS, {}), "function")
        by_name = {f["name"]: f for f in frames}
        self.assertEqual(set(by_name), {"main", "leaf"})
        leaf, main = by_name["leaf"], by_name["main"]
        # leaf runs cycles 2 and 3; main covers all 5 cycles.
        self.assertEqual(leaf["args"]["first_cycle"], 2)
        self.assertEqual(leaf["args"]["cycles"], 2)
        self.assertEqual(main["args"]["first_cycle"], 0)
        self.assertEqual(main["args"]["cycles"], 5)

    def test_indirect_call_through_jalr(self):
        commits = log(
            (0x8000_0000, JALR_RA_A5, " x1  0x80000004"),
            (0x8000_0100, RET, ""),
            (0x8000_0004, ADDI, ""),
        )
        frames = slices(t2p.build_events(commits, SYMBOLS, {}), "function")
        self.assertIn("leaf", [f["name"] for f in frames])

    def test_plain_jump_is_not_a_call(self):
        commits = log((0x8000_0000, J_8, ""), (0x8000_0008, ADDI, ""))
        frames = slices(t2p.build_events(commits, SYMBOLS, {}), "function")
        self.assertEqual([f["name"] for f in frames], ["main"])

    def test_tail_call_replaces_the_callers_frame(self):
        # main: call middle; middle ends with `tail leaf` (j leaf); leaf returns to main.
        symbols = SYMBOLS + [t2p.Symbol(0x8000_0200, 0x10, "middle")]
        j_to_leaf = 0x0000006F | encode_jal_offset(0x8000_0100 - 0x8000_0204)
        commits = log(
            (0x8000_0000, JAL_RA_8, " x1  0x80000004"),
            (0x8000_0200, ADDI, ""),
            (0x8000_0204, j_to_leaf, ""),
            (0x8000_0100, RET, ""),
            (0x8000_0004, ADDI, ""),
        )
        frames = slices(t2p.build_events(commits, symbols, {}), "function")
        spans = {f["name"]: (f["args"]["first_cycle"], f["args"]["cycles"]) for f in frames}
        self.assertEqual(spans, {"main": (0, 5), "middle": (1, 2), "leaf": (3, 1)})

    def test_jump_to_a_label_inside_the_same_function_is_not_a_tail_call(self):
        symbols = [t2p.Symbol(0x8000_0000, 0x10, "main"), t2p.Symbol(0x8000_0008, 0, "loop")]
        commits = log((0x8000_0000, J_8, ""), (0x8000_0008, ADDI, ""))
        frames = slices(t2p.build_events(commits, symbols, {}), "function")
        self.assertEqual([f["name"] for f in frames], ["main"])

    def test_frames_left_open_are_closed_at_the_end(self):
        # The program halts inside leaf (like _start spinning on tohost).
        commits = log((0x8000_0000, JAL_RA_8, " x1  0x80000004"), (0x8000_0100, ADDI, ""))
        frames = slices(t2p.build_events(commits, SYMBOLS, {}), "function")
        self.assertEqual(sorted(f["args"]["cycles"] for f in frames), [1, 2])

    def test_unknown_pc_gets_a_placeholder_name(self):
        commits = log((0x9000_0000, ADDI, ""))
        frames = slices(t2p.build_events(commits, SYMBOLS, {}), "function")
        self.assertEqual(frames[0]["name"], "0x90000000")

    def test_register_and_memory_effects_go_into_args(self):
        commits = log((0x8000_0000, ADDI, " x5  0x0000002a mem 0x80001000 0x07"))
        ins = slices(t2p.build_events(commits, SYMBOLS, {}), "instruction")[0]
        self.assertEqual(ins["args"]["rd"], "x5 = 0x0000002a")
        self.assertEqual(ins["args"]["mem"], "store 0x80001000 = 0x7")


PROGRAM = ROOT / "build" / "programs" / "dense_int8"


@unittest.skipUnless((PROGRAM / "dense_int8.core.log").exists(), "run `make programs` first")
class DenseInt8Trace(unittest.TestCase):
    """Cross-checks on a real program: the levels must agree with each other."""

    @classmethod
    def setUpClass(cls):
        cls.commits = parse_commit_log((PROGRAM / "dense_int8.core.log").read_text())
        cls.symbols = t2p.load_symbols(PROGRAM / "dense_int8.elf")
        cls.trace = json.loads((PROGRAM / "dense_int8.perfetto.json").read_text())
        events = cls.trace["traceEvents"]
        cls.frames = slices(events, "function")
        cls.instructions = slices(events, "instruction")

    def test_every_retired_instruction_is_in_the_trace(self):
        self.assertEqual(len(self.instructions), len(self.commits))

    def test_root_frame_spans_the_whole_run(self):
        root = min(self.frames, key=lambda f: f["ts"])
        self.assertEqual(root["name"], "_start")
        self.assertEqual(root["args"]["cycles"], len(self.commits))

    def test_frames_nest(self):
        # Sorted by start, each frame must end before (or with) every open ancestor.
        stack: list[tuple[int, int]] = []
        for f in sorted(self.frames, key=lambda f: (f["args"]["first_cycle"], -f["args"]["cycles"])):
            start = f["args"]["first_cycle"]
            end = start + f["args"]["cycles"]
            while stack and stack[-1][1] <= start:
                stack.pop()
            if stack:
                self.assertLessEqual(end, stack[-1][1], f["name"])
            stack.append((start, end))

    def test_each_instruction_runs_inside_its_innermost_frame(self):
        # This is what proves call/return detection right: the function the
        # symbol table says owns each pc is the frame on top of the stack.
        starts = sorted((f["args"]["first_cycle"], -f["args"]["cycles"], f["name"]) for f in self.frames)
        owner = [None] * len(self.commits)
        for first, neg_len, name in starts:  # outer frames first, inner ones overwrite
            for c in range(first, first - neg_len):
                owner[c] = name
        for i, c in enumerate(self.commits):
            sym = t2p.symbol_at(self.symbols, c.pc)
            self.assertEqual(owner[i], sym.name if sym else f"0x{c.pc:08x}", f"cycle {i} pc 0x{c.pc:08x}")

    def test_call_counts_match_the_program(self):
        names = [f["name"] for f in self.frames]
        self.assertEqual(names.count("main"), 1)
        self.assertEqual(names.count("dense_int8"), 4)    # BATCH
        self.assertEqual(sum(n.startswith("dot_int8") for n in names), 32)  # BATCH * OUT
        self.assertEqual(names.count("requantize_relu"), 32)
        self.assertEqual(names.count("check"), 1)

    def test_vcd_and_commit_log_agree_cycle_by_cycle(self):
        # In the VCD, cycle n is settled at 20 + 10n ns: rvfi_pc_rdata must be
        # the pc of the n-th retired instruction.
        pcs = read_vcd_signal(PROGRAM / "dense_int8.vcd", "rvfi_pc_rdata")
        for n, c in enumerate(self.commits):
            self.assertEqual(pcs(t2p.cycle_start_us(n) * 1_000_000), c.pc, f"cycle {n}")


def read_vcd_signal(path: Path, name: str):
    """Minimal VCD reader for one top-level signal: returns value_at(time_ps)."""
    ident = None
    changes: list[tuple[int, int]] = []
    time = 0
    with path.open() as f:
        for line in f:
            if ident is None:
                m = re.match(rf"\s*\$var wire \d+ (\S+) {name} ", line)
                if m:
                    ident = m.group(1)
                continue
            if line.startswith("#"):
                time = int(line[1:])
            elif line.startswith("b"):
                value, sig = line[1:].split()
                if sig == ident:
                    changes.append((time, int(value, 2)))
    if ident is None:
        raise ValueError(f"{name} not in {path}")

    times = [ct for ct, _ in changes]

    def value_at(t: float) -> int | None:
        i = bisect.bisect_right(times, round(t)) - 1
        return changes[i][1] if i >= 0 else None

    return value_at


if __name__ == "__main__":
    unittest.main()
