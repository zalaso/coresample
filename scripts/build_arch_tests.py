#!/usr/bin/env python3
"""Build self-checking riscv-arch-test (ACT4) ELFs for the Coresample core.

ACT4 computes expected results with the Sail model. This script runs the same
pipeline with Spike as the reference instead, so no extra tools are needed:

    test.S --(-DSIGNATURE)--> test.sig.elf --(Spike)--> test.sig
           --(ACT sig_modify)--> test.results --(-DRVTEST_SELFCHECK)--> test.elf

The final ELF carries the expected values and checks every result itself, so
running it on the core needs only the usual tohost pass/fail protocol.

Tests are selected from their START_TEST_CONFIG header: every required
extension must be one the core implements, and every parameter must match.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "sw" / "arch-test"

# What the core implements, in the ACT test-header vocabulary.
SUPPORTED_EXTENSIONS = {"I", "M", "Zmmul", "Zifencei"}
SUPPORTED_PARAMS = {"MXLEN": "32"}

RAM_BASE = 0x8000_0000
RAM_SIZE = 0x10_0000  # must match sw/arch-test/link.ld and the testbench

HEADER_RE = re.compile(r"##### START_TEST_CONFIG #####\n(.*?)##### END_TEST_CONFIG #####", re.S)


def parse_header(path: Path) -> tuple[set[str], dict[str, str], str]:
    """Return (required extensions, params, march) from a test's config header."""
    m = HEADER_RE.search(path.read_text())
    if not m:
        raise ValueError(f"{path}: no START_TEST_CONFIG header")
    exts: set[str] = set()
    params: dict[str, str] = {}
    march = ""
    in_params = False
    for raw in m.group(1).splitlines():
        line = raw.lstrip("#").rstrip()
        if not line.strip():
            continue
        if line.startswith(" REQUIRED_EXTENSIONS:"):
            exts = set(re.findall(r"'([^']+)'", line))
            in_params = False
        elif line.startswith(" MARCH:"):
            march = line.split(":", 1)[1].strip()
            in_params = False
        elif line.startswith(" params:"):
            in_params = True
        elif in_params and line.startswith("   "):
            key, _, value = line.strip().partition(":")
            params[key.strip()] = value.strip()
        else:
            in_params = False
    return exts, params, march


def select_tests(act: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    selected, skipped = [], []
    for test in sorted((act / "tests" / "rv32i").glob("*/*.S")):
        exts, params, _ = parse_header(test)
        missing = exts - SUPPORTED_EXTENSIONS
        bad = {k: v for k, v in params.items() if SUPPORTED_PARAMS.get(k) != v}
        if not missing and not bad:
            selected.append(test)
        elif test.parent.name in SUPPORTED_EXTENSIONS:
            # Only worth reporting for suites the core claims to implement.
            skipped.append((test, f"needs {sorted(missing)} {bad}".strip()))
    return selected, skipped


def load_sig_modify(act: Path):
    """ACT's own signature post-processing, so the .results format stays in sync."""
    path = act / "framework" / "src" / "act" / "sig_modify.py"
    spec = importlib.util.spec_from_file_location("act_sig_modify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_signature_file


def run(cmd: list[str], timeout: int = 30) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stdout}{res.stderr}")


def build_one(test: Path, act: Path, out: Path, gcc: str, spike: str, process_sig) -> Path:
    _, _, march = parse_header(test)
    name = test.stem
    work = out / "build"
    sig_elf, sig, results = work / f"{name}.sig.elf", work / f"{name}.sig", work / f"{name}.results"
    final = out / "elfs" / f"{name}.elf"

    common = [
        # -mno-relax: ACT turns relaxation off inside an .option push/pop block.
        # binutils 2.42 decides alignment padding at the end of the file, when
        # relaxation is back on, and leaves zeros instead of NOPs in the code.
        gcc, "-Wa,-mno-relax", f"-march={march}", "-mabi=ilp32", "-mcmodel=medany", "-nostdlib", "-O0", "-g",
        f"-I{CONFIG_DIR}", f"-I{act / 'tests' / 'env'}", f"-T{CONFIG_DIR / 'link.ld'}",
        "-Wl,--defsym=__stack_size=128", "-DTEST_FLEN=0", f'-DTEST_FILE="{test.name}"',
    ]
    # 1. Signature ELF. sail_macros.h needs these addresses; the tests that use
    #    them (interrupts) are never selected for this core.
    run([*common, "-DSIGNATURE", "-DSAIL_CLINT_BASE_ADDRESS=0x2000000",
         "-DSAIL_SIMPLE_INTERRUPT_GENERATOR_BASE_ADDRESS=0x3000000",
         "-o", str(sig_elf), str(test)])
    # 2. Expected results from Spike.
    run([spike, "--isa=rv32im", f"-m0x{RAM_BASE:x}:0x{RAM_SIZE:x}",
         f"+signature={sig}", "+signature-granularity=4", str(sig_elf)])
    # 3. Same post-processing as ACT.
    process_sig(sig, 32)
    # 4. Self-checking ELF.
    run([*common, "-DRVTEST_SELFCHECK", f'-DSIGNATURE_FILE="{results}"', "-DXLEN=32",
         "-o", str(final), str(test)])
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--act", type=Path, default=ROOT / "third_party" / "riscv-arch-test")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "arch")
    ap.add_argument("--gcc", default=os.environ.get("RISCV_PREFIX", "riscv64-unknown-elf-") + "gcc")
    ap.add_argument("--spike", default="spike")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    if not (args.act / "tests" / "env").is_dir():
        print(f"riscv-arch-test not found in {args.act}: run scripts/install-deps.sh", file=sys.stderr)
        return 1
    (args.out / "build").mkdir(parents=True, exist_ok=True)
    (args.out / "elfs").mkdir(parents=True, exist_ok=True)

    tests, skipped = select_tests(args.act)
    for test, why in skipped:
        print(f"skip  {test.stem}: {why}")
    process_sig = load_sig_modify(args.act)

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_one, t, args.act, args.out, args.gcc, args.spike, process_sig): t
                   for t in tests}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # report every broken test, not just the first
                failed += 1
                print(f"ERROR {futures[fut].stem}: {e}", file=sys.stderr)
    print(f"built {len(tests) - failed}/{len(tests)} self-checking ELFs in {args.out / 'elfs'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
