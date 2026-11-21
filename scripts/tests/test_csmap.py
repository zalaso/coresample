"""Tests for scripts/csmap.py (the .cs_map region map of compiled networks)."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from csmap import Region, RegionMap, read_section  # noqa: E402


def tiny_elf(sections: dict[str, bytes]) -> bytes:
    """A minimal ELF32 little-endian file with the given sections (no program)."""
    names = b"\0" + b"".join(n.encode() + b"\0" for n in [*sections, ".shstrtab"])
    body = b"".join(sections.values())
    data_off = 0x34
    strtab_off = data_off + len(body)
    shoff = strtab_off + len(names)
    header = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header += struct.pack("<HHIIIIIHHHHHH", 2, 0xF3, 1, 0, 0, shoff, 0, 0x34, 0, 0, 40, len(sections) + 2,
                          len(sections) + 1)
    shdrs = struct.pack("<10I", *([0] * 10))  # null section
    off, name_off = data_off, 1
    for name, content in sections.items():
        shdrs += struct.pack("<10I", name_off, 1, 0, 0, off, len(content), 0, 0, 1, 0)
        off += len(content)
        name_off += len(name) + 1
    shdrs += struct.pack("<10I", name_off, 3, 0, 0, strtab_off, len(names), 0, 0, 1, 0)
    return header + body + names + shdrs


class ReadSection(unittest.TestCase):
    def test_finds_sections_by_name(self):
        elf = tiny_elf({".text": b"\x13\x00\x00\x00", ".cs_map": b"abcdef"})
        self.assertEqual(read_section(elf, ".cs_map"), b"abcdef")
        self.assertEqual(read_section(elf, ".text"), b"\x13\x00\x00\x00")
        self.assertIsNone(read_section(elf, ".data"))

    def test_rejects_other_files(self):
        self.assertIsNone(read_section(b"not an elf", ".text"))


class Map(unittest.TestCase):
    def test_load_and_lookup(self):
        entries = struct.pack("<9I", 0x100, 0x108, 0, 0x108, 0x120, 1, 0x200, 0x204, 2)
        names = [{"id": 0, "function": "main", "layer": None, "op": "loop"},
                 {"id": 1, "function": "layer0_conv2d", "layer": 0, "op": "mac"},
                 {"id": 2, "function": "layer0_conv2d", "layer": 0, "op": "store"}]
        with tempfile.TemporaryDirectory() as d:
            elf = Path(d) / "p.elf"
            elf.write_bytes(tiny_elf({".cs_map": entries}))
            elf.with_suffix(".regions.json").write_text(json.dumps(names))
            m = RegionMap.load(elf)
        self.assertEqual(m.at(0x104).label, "main.loop")
        self.assertEqual(m.at(0x108).label, "layer0_conv2d.mac")  # end is exclusive
        self.assertEqual(m.at(0x11C).layer, 0)
        self.assertIsNone(m.at(0x120))  # gap between regions
        self.assertIsNone(m.at(0x0FC))

    def test_overlapping_regions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            RegionMap([Region(0, 8, 0, "f", None, "a"), Region(4, 12, 1, "f", None, "b")])


if __name__ == "__main__":
    unittest.main()
