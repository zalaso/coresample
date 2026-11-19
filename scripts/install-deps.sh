#!/usr/bin/env bash
# Installs the tools Coresample needs on Ubuntu / Linux Mint:
#   1. system packages (needs sudo): Verilator, RISC-V GCC, Spike's build deps
#   2. Spike, the RISC-V reference simulator, built from source into $PREFIX
#   3. riscv-tests, the official ISA test suite, cloned into third_party/
#   4. riscv-arch-test (ACT4), the architectural tests, cloned into third_party/
#   5. Rust, through rustup, for the neural network simulator (sim/nn)
#   6. PyTorch (CPU only) in .venv/, to generate the simulator's reference values
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"

echo "==> 1/6 System packages (sudo)"
sudo apt-get update
sudo apt-get install -y \
  verilator \
  gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
  device-tree-compiler libboost-regex-dev libboost-system-dev \
  build-essential git python3

echo "==> 2/6 Spike"
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

echo "==> 3/6 riscv-tests"
if [ -d "$ROOT/third_party/riscv-tests" ]; then
  echo "riscv-tests already present"
else
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/riscv-software-src/riscv-tests "$ROOT/third_party/riscv-tests"
fi
git -C "$ROOT/third_party/riscv-tests" rev-parse HEAD > "$ROOT/docs/.riscv-tests-commit"

echo "==> 4/6 riscv-arch-test"
if [ -d "$ROOT/third_party/riscv-arch-test" ]; then
  echo "riscv-arch-test already present"
else
  mkdir -p "$ROOT/third_party"
  # Only the generated tests and headers are used: no mise, uv, Ruby or Sail needed.
  git clone --depth 1 https://github.com/riscv-non-isa/riscv-arch-test "$ROOT/third_party/riscv-arch-test"
fi
git -C "$ROOT/third_party/riscv-arch-test" rev-parse HEAD > "$ROOT/docs/.riscv-arch-test-commit"

echo "==> 5/6 Rust"
if command -v cargo >/dev/null 2>&1 || [ -x "$HOME/.cargo/bin/cargo" ]; then
  echo "cargo already installed"
else
  # Official installer; --no-modify-path leaves shell startup files alone.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi
# The exact toolchain is pinned in sim/nn/rust-toolchain.toml and installed on first use.

echo "==> 6/6 PyTorch (CPU) in .venv"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
"$ROOT/.venv/bin/python" -c "import torch; print('torch', torch.__version__)"

echo
echo "Done. If 'cargo' is not found, add $HOME/.cargo/bin to PATH:"
echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
echo "If 'spike' is not found, add $PREFIX/bin to PATH:"
echo "  export PATH=\"$PREFIX/bin:\$PATH\""
