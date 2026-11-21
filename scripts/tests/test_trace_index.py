"""Tests for scripts/trace_index.py, the multi-resolution trace index.

The index must be exact: any range of cycles, answered from the pyramid of
bins plus the raw chunks at the edges, equals counting the cycles one by one;
and its totals equal op_report's, which is computed independently.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tests"))

import op_report  # noqa: E402
import pipeline_to_kanata as p2k  # noqa: E402
import trace_index as ti  # noqa: E402
from csmap import Region, RegionMap  # noqa: E402
from run_tests import parse_commit_log  # noqa: E402
from test_pipeline_to_kanata import BRANCH, LOAD_USE  # noqa: E402
from trace_to_perfetto import Symbol  # noqa: E402

ADDI, JAL_RA, RET = 0x00000013, 0x000000EF, 0x00008067

SYMBOLS = [Symbol(0x8000_0000, 0x100, "main"), Symbol(0x8000_0100, 0x100, "layer0_conv2d"),
           Symbol(0x8000_0200, 0x100, "layer1_linear")]
REGIONS = RegionMap([
    Region(0x8000_0000, 0x8000_0100, 0, "main", None, "loop"),
    Region(0x8000_0100, 0x8000_0180, 1, "layer0_conv2d", 0, "mac"),
    Region(0x8000_0180, 0x8000_0200, 2, "layer0_conv2d", 0, "store"),
    Region(0x8000_0200, 0x8000_0300, 3, "layer1_linear", 1, "mac"),
])


def commit(pc: int, insn: int = ADDI) -> str:
    return f"core   0: 3 0x{pc:08x} (0x{insn:08x})"


def two_samples() -> list[str]:
    """main calls layer0 then layer1, twice: two samples of a two-layer network."""
    lines = []
    for _ in range(2):
        lines += [commit(0x8000_0000), commit(0x8000_0004, JAL_RA)]
        lines += [commit(0x8000_0100), commit(0x8000_0104), commit(0x8000_0180), commit(0x8000_0184, RET)]
        lines += [commit(0x8000_0008, JAL_RA), commit(0x8000_0200), commit(0x8000_0204, RET)]
    lines.append(commit(0x8000_000C))
    return lines


def assert_totals_add_up(test: unittest.TestCase, node: dict) -> None:
    kids = sum(c["total"]["cycles"] for c in node["children"])
    test.assertEqual(node["total"]["cycles"], node["self"]["cycles"] + kids, node["name"])
    for c in node["children"]:
        assert_totals_add_up(test, c)


def build(commits: list[str], cycles=None, chunk=4, fanout=2) -> ti.Index:
    d = Path(tempfile.mkdtemp())
    ti.build(iter(commits), iter(cycles) if cycles is not None else None, SYMBOLS, REGIONS, d,
             chunk_cycles=chunk, fanout=fanout)
    return ti.Index(d)


class Tree(unittest.TestCase):
    def test_samples_group_the_layer_calls(self):
        idx = build(two_samples())
        root = idx.tree
        self.assertEqual(root["name"], "main")
        samples = idx.samples()
        self.assertEqual([s["name"] for s in samples], ["sample 0", "sample 1"])
        self.assertEqual([c["name"] for c in samples[0]["children"]], ["layer0_conv2d", "layer1_linear"])
        conv = samples[0]["children"][0]
        self.assertEqual(conv["total"]["cycles"], 4)
        self.assertEqual(conv["total"]["regions"], {"1": 2, "2": 2})
        self.assertEqual(root["total"]["cycles"], 19)
        self.assertEqual(root["self"]["cycles"], 19 - 2 * 6)  # main's own loop instructions

    def test_totals_add_up_at_every_level(self):
        assert_totals_add_up(self, build(two_samples()).tree)


class Ranges(unittest.TestCase):
    def test_every_range_matches_counting_cycle_by_cycle(self):
        idx = build(two_samples(), chunk=2, fanout=2)  # 19 cycles: 10 chunks, 4 pyramid levels
        slots = [idx.slot(t) for t in range(idx.cycles)]
        self.assertGreaterEqual(len(idx.levels), 3)
        for a in range(idx.cycles + 1):
            for b in range(a, idx.cycles + 1):
                expected = ti.Stats()
                for s in slots[a:b]:
                    expected.add(s.region, s.cause)
                self.assertEqual(idx.stats(a, b), expected, f"[{a}, {b})")

    def test_every_range_with_bubbles_matches_the_pipeline_log(self):
        # Expected counts straight from the pipeline log and the bubble blame,
        # not through the index reader: this exercises the raw edges with bubbles.
        for log_lines in (LOAD_USE, BRANCH):
            cycles = p2k.parse_pipeline_log("\n".join(log_lines))
            pc_of = p2k.pcs_by_id(cycles)
            blame = iter(p2k.account(cycles).blame)
            expected_slots = []
            for c in cycles:
                if c.stages["W"] is not None:
                    pc, cause = pc_of[c.stages["W"]], None
                else:
                    cause, culprit = next(blame)
                    pc = pc_of[culprit] if culprit is not None else None
                region = ti.NOBODY if pc is None else (REGIONS.at(pc).id if REGIONS.at(pc) else ti.OUTSIDE)
                expected_slots.append((region, cause))
            idx = build([commit(pc) for pc in p2k.retired_pcs(cycles)], cycles, chunk=2, fanout=2)
            for a in range(idx.cycles + 1):
                for b in range(a, idx.cycles + 1):
                    expected = ti.Stats()
                    for region, cause in expected_slots[a:b]:
                        expected.add(region, cause)
                    self.assertEqual(idx.stats(a, b), expected, f"[{a}, {b})")

    def test_out_of_range_is_rejected(self):
        idx = build(two_samples())
        with self.assertRaises(ValueError):
            idx.stats(0, idx.cycles + 1)


class Pipeline(unittest.TestCase):
    def test_bubbles_are_charged_like_op_report(self):
        cycles = p2k.parse_pipeline_log("\n".join(LOAD_USE))
        pcs = p2k.retired_pcs(cycles)
        idx = build([commit(pc) for pc in pcs], cycles)
        total = idx.stats(0, idx.cycles)
        self.assertEqual(idx.cycles, len(cycles))
        self.assertEqual(total.bubbles, {"fill": 4, "load-use": 1})
        # Instruction 2 (pc 0x80000008, outside these regions) waited for the load.
        self.assertEqual(idx.slot(6).cause, "load-use")
        self.assertEqual(idx.slot(6).pc, 0x8000_0008)

    def test_commit_log_and_pipeline_must_agree(self):
        cycles = p2k.parse_pipeline_log("\n".join(LOAD_USE))
        with self.assertRaises(ValueError):
            build([commit(0x8000_0000)], cycles)  # three instructions retire, one is logged


G2 = ROOT / "build" / "g2-pipe"


@unittest.skipUnless((G2 / "cnn1.pipeline.log").exists(), "run `make g2 CORE=pipe` first")
class G2OneImage(unittest.TestCase):
    """The index of one CNN image agrees with op_report, computed independently."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        ti.build_from_files(G2 / "cnn1.elf", G2 / "cnn1.core.log", G2 / "cnn1.pipeline.log", cls.dir)
        cls.idx = ti.Index(cls.dir)
        regions = RegionMap.load(G2 / "cnn1.elf")
        cycles = p2k.parse_pipeline_log((G2 / "cnn1.pipeline.log").read_text())
        cls.rows = op_report.report_pipeline(cycles, regions)
        cls.regions = regions
        cls.commits = parse_commit_log((G2 / "cnn1.core.log").read_text())

    def test_same_cycles_and_instructions(self):
        self.assertEqual(self.idx.cycles, sum(r.cycles for r in self.rows))
        self.assertEqual(self.idx.instructions, len(self.commits))

    def test_every_operation_row_matches(self):
        total = self.idx.stats(0, self.idx.cycles)
        by_label = {}
        for rid, n in total.regions.items():  # several regions can share a label (two `loop`s)
            label = self.idx.region_label(rid)
            by_label[label] = by_label.get(label, 0) + n
        for row in self.rows:
            label = f"{row.function}.{row.op}"
            self.assertEqual(by_label.get(label, 0), row.cycles, label)

    def test_each_layer_in_the_tree_matches_op_report(self):
        # The tree charges a bubble to the frame of the instruction it blames;
        # summed over a function's nodes, that is op_report's rows for the function.
        per_function: dict[str, int] = {}

        def walk(node):
            per_function[node["name"]] = per_function.get(node["name"], 0) + node["self"]["cycles"]
            for c in node["children"]:
                walk(c)
        walk(self.idx.tree)
        for f in ["layer0_conv2d", "layer1_max_pool2d", "layer3_linear", "main"]:
            expected = sum(r.cycles for r in self.rows if r.function == f)
            self.assertEqual(per_function.get(f, 0), expected, f)

    def test_per_instruction_counts_add_up(self):
        # Every cycle is charged to one instruction (or to nobody: the fill),
        # and per region the instructions add up to the region's cycles.
        code = self.idx.code
        self.assertEqual(sum(c["cycles"] for c in code) + self.idx.unattributed, self.idx.cycles)
        self.assertEqual(sum(c["instructions"] for c in code), self.idx.instructions)
        total = self.idx.stats(0, self.idx.cycles)
        per_region: dict[int, int] = {}
        for c in code:
            per_region[c["region"]] = per_region.get(c["region"], 0) + c["cycles"]
        for rid, n in total.regions.items():
            if rid != ti.NOBODY:
                self.assertEqual(per_region.get(rid, 0), n, self.idx.region_label(rid))
        lbu = [c for c in code if c["text"].startswith("lbu")]
        self.assertTrue(lbu, "the disassembly is in the index")
        self.assertTrue(all(c["insn"] & 0x7F == 0x03 for c in lbu), "lbu encodings carry the LOAD opcode")

    def test_one_sample_with_three_layers(self):
        samples = self.idx.samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual([c["name"] for c in samples[0]["children"]],
                         ["layer0_conv2d", "layer1_max_pool2d", "layer3_linear"])
        self.assertEqual(self.idx.tree["total"]["cycles"], self.idx.cycles, "the root covers every cycle")
        assert_totals_add_up(self, self.idx.tree)


INDEX = ROOT / "build" / "index-pipe"


@unittest.skipUnless((INDEX / "index.json").exists(), "run `make index CORE=pipe` first")
class G2AllImages(unittest.TestCase):
    def test_48_samples_and_a_small_index(self):
        idx = ti.Index(INDEX)
        self.assertEqual(len(idx.samples()), 48)
        self.assertLess((INDEX / "index.json").stat().st_size, 2_000_000)
        self.assertEqual(idx.stats(0, idx.cycles).cycles, idx.cycles)
        self.assertEqual(idx.tree["total"]["cycles"], idx.cycles)
        assert_totals_add_up(self, idx.tree)


if __name__ == "__main__":
    unittest.main()
