"""Tests for scripts/pipeline_to_kanata.py.

Unit tests use hand-written pipeline logs (the format cs_sim --pipeline writes).
The integration tests run on the G2 CNN built by `make g2 CORE=pipe` and are
skipped if it is missing.

Run with:  python3 -m unittest discover -s scripts/tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline_to_kanata as p2k  # noqa: E402
from run_tests import parse_commit_log  # noqa: E402


def log(*lines: str) -> list[p2k.Cycle]:
    return p2k.parse_pipeline_log("\n".join(lines))


# Three independent instructions flowing through an empty pipeline.
STRAIGHT = [
    "0 F 0:80000000 D - X - M - W -",
    "1 F 1:80000004 D 0 X - M - W -",
    "2 F 2:80000008 D 1 X 0 M - W -",
    "3 F 3:8000000c D 2 X 1 M 0 W -",
    "4 F 4:80000010 D 3 X 2 M 1 W 0",
    "5 F 5:80000014 D 4 X 3 M 2 W 1",
    "6 F 6:80000018 D 5 X 4 M 3 W 2",
]

# Instruction 1 is a load, 2 uses its result: one stall cycle in ID.
LOAD_USE = [
    "0 F 0:80000000 D - X - M - W -",
    "1 F 1:80000004 D 0 X - M - W -",
    "2 F 2:80000008 D 1 X 0 M - W -",
    "3 F 3:8000000c D 2 X 1 M 0 W - stall",
    "4 F 3:8000000c D 2 X - M 1 W 0",
    "5 F 4:80000010 D 3 X 2 M - W 1",
    "6 F 5:80000014 D 4 X 3 M 2 W -",
    "7 F 6:80000018 D 5 X 4 M 3 W 2",
]

# Instruction 1 is a taken branch: 2 and 3 (in ID and IF) are discarded.
BRANCH = [
    "0 F 0:80000000 D - X - M - W -",
    "1 F 1:80000004 D 0 X - M - W -",
    "2 F 2:80000008 D 1 X 0 M - W -",
    "3 F 3:8000000c D 2 X 1 M 0 W - flush=branch",
    "4 F 4:80000040 D - X - M 1 W 0",
    "5 F 5:80000044 D 4 X - M - W 1",
    "6 F 6:80000048 D 5 X 4 M - W -",
    "7 F 7:8000004c D 6 X 5 M 4 W -",
    "8 F 8:80000050 D 7 X 6 M 5 W 4",
]


class Parsing(unittest.TestCase):
    def test_fields(self):
        c = log("3 F 3:8000000c D 2 X 1 M 0 W - stall flush=jump")[0]
        self.assertEqual((c.n, c.fetch, c.fetch_pc), (3, 3, 0x8000000C))
        self.assertEqual(c.stages, {"F": 3, "D": 2, "X": 1, "M": 0, "W": None})
        self.assertTrue(c.stall)
        self.assertEqual(c.flush, "jump")

    def test_rejects_a_malformed_line(self):
        with self.assertRaises(ValueError):
            log("3 F 3:8000000c D 2 X")


class Accounting(unittest.TestCase):
    def test_straight_line_has_only_fill_bubbles(self):
        a = p2k.account(log(*STRAIGHT))
        self.assertEqual(a.retired, 3)
        self.assertEqual(a.bubbles, {"fill": 4})

    def test_a_stall_costs_one_bubble(self):
        a = p2k.account(log(*LOAD_USE))
        self.assertEqual(a.retired, 3)
        self.assertEqual(a.bubbles, {"fill": 4, "load-use": 1})

    def test_a_taken_branch_costs_two_bubbles(self):
        a = p2k.account(log(*BRANCH))
        self.assertEqual(a.retired, 3)
        self.assertEqual(a.bubbles, {"fill": 4, "branch": 2})
        self.assertEqual(a.discarded, 2)

    def test_every_empty_writeback_has_exactly_one_cause(self):
        # Drop the stall flag: the bubble it made now has no cause.
        broken = [line.replace(" stall", "") for line in LOAD_USE]
        with self.assertRaisesRegex(ValueError, "no cause"):
            p2k.account(log(*broken))


class Kanata(unittest.TestCase):
    def convert(self, lines, disasm=None):
        return p2k.to_kanata(log(*lines), disasm or {}).splitlines()

    def test_header_and_stage_sequence(self):
        out = self.convert(STRAIGHT)
        self.assertEqual(out[0], "Kanata\t0004")
        self.assertEqual(out[1], "C=\t0")
        starts = [l.split("\t")[3] for l in out if l.startswith("S\t0\t")]
        self.assertEqual(starts, ["F", "D", "X", "M", "W"])
        self.assertIn("R\t0\t0\t0", out)  # first instruction retires as number 0

    def test_labels_use_the_disassembly(self):
        out = self.convert(STRAIGHT, {0x80000000: "addi a0,a0,1"})
        self.assertIn("L\t0\t0\t80000000 addi a0,a0,1", out)

    def test_stall_is_explained_and_linked_to_its_producer(self):
        out = self.convert(LOAD_USE, {0x80000004: "lbu t0,0(t0)"})
        self.assertIn("W\t2\t1\t0", out)  # consumer 2 depends on producer 1
        details = [l for l in out if l.startswith("L\t2\t1\t")]
        self.assertTrue(any("load-use" in d and "lbu t0,0(t0)" in d for d in details), details)

    def test_discarded_instructions_are_flushed_not_retired(self):
        out = self.convert(BRANCH)
        self.assertIn("R\t2\t0\t1", out)
        self.assertIn("R\t3\t0\t1", out)
        self.assertTrue(any(l.startswith("L\t2\t1\t") and "branch" in l for l in out))


G2 = ROOT / "build" / "g2-pipe"


@unittest.skipUnless((G2 / "cnn1.pipeline.log").exists(), "run `make g2 CORE=pipe` first")
class G2Pipeline(unittest.TestCase):
    """The pipeline trace of the CNN must agree with the commit log."""

    @classmethod
    def setUpClass(cls):
        cls.cycles = p2k.parse_pipeline_log((G2 / "cnn1.pipeline.log").read_text())
        cls.commits = parse_commit_log((G2 / "cnn1.core.log").read_text())
        cls.acc = p2k.account(cls.cycles)

    def test_retired_count_matches_the_commit_log(self):
        self.assertEqual(self.acc.retired, len(self.commits))

    def test_instructions_retire_in_fetch_order_with_the_logged_pcs(self):
        pcs = p2k.retired_pcs(self.cycles)
        self.assertEqual(pcs, [c.pc for c in self.commits])

    def test_cycles_are_instructions_plus_explained_bubbles(self):
        total = self.acc.retired + sum(self.acc.bubbles.values())
        self.assertEqual(total, len(self.cycles))


if __name__ == "__main__":
    unittest.main()
