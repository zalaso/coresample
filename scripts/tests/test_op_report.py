"""Tests for scripts/op_report.py: cycles and lost cycles per layer and operation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tests"))

import op_report  # noqa: E402
import pipeline_to_kanata as p2k  # noqa: E402
from csmap import Region, RegionMap  # noqa: E402
from test_pipeline_to_kanata import BRANCH, LOAD_USE  # noqa: E402

# The synthetic pipeline logs fetch from 0x80000000 upward; 0x80000040.. is a branch target.
REGIONS = RegionMap([
    Region(0x8000_0000, 0x8000_0008, 0, "layer0_conv2d", 0, "mac"),
    Region(0x8000_0008, 0x8000_0020, 1, "layer0_conv2d", 0, "requant"),
    Region(0x8000_0040, 0x8000_0060, 2, "main", None, "check"),
])


def rows(table):
    return {(r.function, r.op): r for r in table}


class Report(unittest.TestCase):
    def test_single_cycle_counts_instructions(self):
        table = rows(op_report.report([0x8000_0000, 0x8000_0004, 0x8000_0008, 0x9000_0000], REGIONS))
        self.assertEqual(table[("layer0_conv2d", "mac")].instructions, 2)
        self.assertEqual(table[("layer0_conv2d", "mac")].cycles, 2)
        self.assertEqual(table[("layer0_conv2d", "requant")].instructions, 1)
        self.assertEqual(table[(op_report.OUTSIDE, "-")].instructions, 1)

    def test_load_use_stall_is_charged_to_the_waiting_instruction(self):
        cycles = p2k.parse_pipeline_log("\n".join(LOAD_USE))
        table = rows(op_report.report_pipeline(cycles, REGIONS))
        # Instruction 2 (pc 0x80000008, requant) waited for the load (pc 0x80000004, mac).
        self.assertEqual(table[("layer0_conv2d", "requant")].bubbles, {"load-use": 1})
        self.assertEqual(table[("layer0_conv2d", "mac")].bubbles, {})
        self.assertEqual(table[op_report.FILL_ROW].bubbles, {"fill": 4})
        self.assertEqual(sum(r.cycles for r in table.values()), len(cycles))

    def test_flush_is_charged_to_the_branch(self):
        cycles = p2k.parse_pipeline_log("\n".join(BRANCH))
        table = rows(op_report.report_pipeline(cycles, REGIONS))
        # Instruction 1 (pc 0x80000004, mac) is the taken branch.
        self.assertEqual(table[("layer0_conv2d", "mac")].bubbles, {"branch": 2})
        self.assertEqual(sum(r.cycles for r in table.values()), len(cycles))


G2 = ROOT / "build" / "g2-pipe"


@unittest.skipUnless((G2 / "cnn1.pipeline.log").exists(), "run `make g2 CORE=pipe` first")
class G2Report(unittest.TestCase):
    def test_every_cycle_is_charged_somewhere(self):
        cycles = p2k.parse_pipeline_log((G2 / "cnn1.pipeline.log").read_text())
        table = op_report.report_pipeline(cycles, RegionMap.load(G2 / "cnn1.elf"))
        self.assertEqual(sum(r.cycles for r in table), len(cycles))
        by_op = rows(table)
        self.assertGreater(by_op[("layer0_conv2d", "mac")].bubbles.get("load-use", 0), 0,
                           "the mac loop loads a byte and uses it at once: it must stall")


if __name__ == "__main__":
    unittest.main()
