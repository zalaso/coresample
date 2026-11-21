#!/usr/bin/env python3
"""Read the instruction -> operation -> layer map of a compiled network.

The compiler (`cs-compile`) puts a `.cs_map` section in the ELF: one entry per
region, (begin address, end address, region id) as three little-endian
32-bit words. The region names (function, layer, operation) are in the
`<program>.regions.json` file written next to the assembly.
"""
from __future__ import annotations

import bisect
import json
import struct
from dataclasses import dataclass
from pathlib import Path


def read_section(elf: bytes, name: str) -> bytes | None:
    """Contents of section `name` of a 32-bit little-endian ELF, or None."""
    if elf[:4] != b"\x7fELF" or len(elf) < 0x34 or elf[4] != 1 or elf[5] != 1:
        return None
    shoff, = struct.unpack_from("<I", elf, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x2E)

    def header(i: int) -> tuple[int, int, int]:
        name_off, _, _, _, off, size = struct.unpack_from("<IIIIII", elf, shoff + i * shentsize)
        return name_off, off, size

    _, strtab, _ = header(shstrndx)
    for i in range(shnum):
        name_off, off, size = header(i)
        start = strtab + name_off
        if elf[start:elf.index(b"\0", start)] == name.encode():
            return elf[off:off + size]
    return None


@dataclass(frozen=True)
class Region:
    begin: int
    end: int
    id: int
    function: str
    layer: int | None
    op: str

    @property
    def label(self) -> str:
        """`layer0_conv2d.mac`, `main.loop`, ..."""
        return f"{self.function}.{self.op}"


class RegionMap:
    def __init__(self, regions: list[Region]):
        self.regions = sorted(regions, key=lambda r: r.begin)
        self._starts = [r.begin for r in self.regions]
        for a, b in zip(self.regions, self.regions[1:]):
            if a.end > b.begin:
                raise ValueError(f"regions {a.label} and {b.label} overlap")

    def at(self, pc: int) -> Region | None:
        i = bisect.bisect_right(self._starts, pc) - 1
        if i >= 0 and pc < self.regions[i].end:
            return self.regions[i]
        return None

    @classmethod
    def load(cls, elf_path: Path, regions_json: Path | None = None) -> RegionMap:
        """Regions of a compiled program; names from `<elf stem>.regions.json`
        unless another file is given."""
        raw = read_section(Path(elf_path).read_bytes(), ".cs_map")
        if raw is None:
            raise ValueError(f"{elf_path}: no .cs_map section (not built by cs-compile?)")
        names = json.loads(Path(regions_json or Path(elf_path).with_suffix(".regions.json")).read_text())
        by_id = {n["id"]: n for n in names}
        regions = []
        for begin, end, rid in struct.iter_unpack("<III", raw):
            n = by_id[rid]
            regions.append(Region(begin, end, rid, n["function"], n["layer"], n["op"]))
        return cls(regions)
