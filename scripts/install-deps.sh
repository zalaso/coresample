#!/usr/bin/env bash
# Installs the tools Coresample needs on Ubuntu / Linux Mint:
#   1. system packages (needs sudo): Verilator, RISC-V GCC, Spike's build deps
#   2. Spike, the RISC-V reference simulator, built from source into $PREFIX
#   3. riscv-tests, the official ISA test suite, cloned into third_party/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"

echo "==> 1/3 System packages (sudo)"
sudo apt-get update
sudo apt-get install -y \
  verilator \
  gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
  device-tree-compiler libboost-regex-dev libboost-system-dev \
  build-essential git python3

echo "==> 2/3 Spike"
if command -v spike >/dev/null 2>&1; then
  echo "spike already installed: $(command -v spike)"
else
  tmp="$(mktemp -d)"
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/riscv-software-src/riscv-isa-sim "$tmp/riscv-isa-sim"
  mkdir -p "$tmp/riscv-isa-sim/build"
  (
    cd "$tmp/riscv-isa-sim/build"
    ../configure --prefix="$PREFIX"
    # Two jobs: with more, the build can run out of memory on 6-8 GB machines.
    make -j2
    make install
  )
  git -C "$tmp/riscv-isa-sim" rev-parse HEAD > "$ROOT/docs/.spike-commit"
  rm -rf "$tmp"
fi

echo "==> 3/3 riscv-tests"
if [ -d "$ROOT/third_party/riscv-tests" ]; then
  echo "riscv-tests already present"
else
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/riscv-software-src/riscv-tests "$ROOT/third_party/riscv-tests"
fi
git -C "$ROOT/third_party/riscv-tests" rev-parse HEAD > "$ROOT/docs/.riscv-tests-commit"

echo
echo "Done. If 'spike' is not found, add $PREFIX/bin to PATH:"
echo "  export PATH=\"$PREFIX/bin:\$PATH\""
