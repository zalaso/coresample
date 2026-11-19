// Verilator testbench for the Coresample core.
//
// Loads a RISC-V ELF into a flat RAM, runs the core until the program writes
// to `tohost` (the riscv-tests / Spike pass-fail protocol) and optionally
// writes a commit log in Spike's format, one line per retired instruction.
//
//   cs_sim [--trace FILE] [--max-cycles N] program.elf
//
// Prints a final line "RESULT PASS", "RESULT FAIL <test>" or
// "RESULT ERROR <reason>"; the exit code is 0, 1 or 2 respectively.

#include <elf.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "Vcs_core.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

namespace {

constexpr uint32_t kRamBase = 0x80000000u;
constexpr uint32_t kRamSize = 0x00100000u;  // 1 MiB, same as the Spike run

class Ram {
 public:
  Ram() : bytes_(kRamSize, 0) {}

  bool contains(uint32_t addr, uint32_t len) const {
    return addr >= kRamBase && uint64_t(addr - kRamBase) + len <= kRamSize;
  }
  uint8_t* at(uint32_t addr) { return &bytes_[addr - kRamBase]; }

  uint32_t read32(uint32_t addr) {
    uint32_t v;
    std::memcpy(&v, at(addr), 4);  // host is little-endian, like RISC-V
    return v;
  }
  void write32(uint32_t addr, uint32_t data, uint32_t mask) {
    for (int i = 0; i < 4; ++i)
      if (mask & (1u << i)) at(addr)[i] = uint8_t(data >> (8 * i));
  }

 private:
  std::vector<uint8_t> bytes_;
};

struct Program {
  uint32_t entry = 0;
  uint32_t tohost = 0;
};

bool load_elf(const char* path, Ram& ram, Program& prog, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = std::string("cannot open ") + path; return false; }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});

  Elf32_Ehdr eh;
  if (buf.size() < sizeof eh) { err = "file too small for an ELF header"; return false; }
  std::memcpy(&eh, buf.data(), sizeof eh);
  if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 || eh.e_ident[EI_CLASS] != ELFCLASS32 ||
      eh.e_machine != EM_RISCV) {
    err = "not a 32-bit RISC-V ELF";
    return false;
  }
  prog.entry = eh.e_entry;

  for (int i = 0; i < eh.e_phnum; ++i) {
    Elf32_Phdr ph;
    std::memcpy(&ph, buf.data() + eh.e_phoff + i * eh.e_phentsize, sizeof ph);
    if (ph.p_type != PT_LOAD || ph.p_memsz == 0) continue;
    if (!ram.contains(ph.p_paddr, ph.p_memsz) || ph.p_offset + ph.p_filesz > buf.size()) {
      err = "segment outside RAM or file";
      return false;
    }
    std::memcpy(ram.at(ph.p_paddr), buf.data() + ph.p_offset, ph.p_filesz);
    std::memset(ram.at(ph.p_paddr) + ph.p_filesz, 0, ph.p_memsz - ph.p_filesz);
  }

  for (int i = 0; i < eh.e_shnum; ++i) {
    Elf32_Shdr sh;
    std::memcpy(&sh, buf.data() + eh.e_shoff + i * eh.e_shentsize, sizeof sh);
    if (sh.sh_type != SHT_SYMTAB) continue;
    Elf32_Shdr strtab;
    std::memcpy(&strtab, buf.data() + eh.e_shoff + sh.sh_link * eh.e_shentsize, sizeof strtab);
    for (uint32_t off = 0; off + sizeof(Elf32_Sym) <= sh.sh_size; off += sizeof(Elf32_Sym)) {
      Elf32_Sym sym;
      std::memcpy(&sym, buf.data() + sh.sh_offset + off, sizeof sym);
      const char* name = reinterpret_cast<const char*>(buf.data() + strtab.sh_offset + sym.st_name);
      if (std::strcmp(name, "tohost") == 0) prog.tohost = sym.st_value;
    }
  }
  if (prog.tohost == 0) { err = "no 'tohost' symbol"; return false; }
  return true;
}

int lowest_set_bit(uint32_t mask) { return __builtin_ctz(mask); }

// One commit-log line in Spike's format.
void log_commit(FILE* out, const Vcs_core& c) {
  std::fprintf(out, "core   0: 3 0x%08" PRIx32 " (0x%08" PRIx32 ")", uint32_t(c.rvfi_pc_rdata),
               uint32_t(c.rvfi_insn));
  if (c.rvfi_rd_addr != 0)
    std::fprintf(out, " x%-2d 0x%08" PRIx32, int(c.rvfi_rd_addr), uint32_t(c.rvfi_rd_wdata));
  if (c.rvfi_mem_rmask != 0) {
    std::fprintf(out, " mem 0x%08" PRIx32,
                 uint32_t(c.rvfi_mem_addr + lowest_set_bit(c.rvfi_mem_rmask)));
  }
  if (c.rvfi_mem_wmask != 0) {
    const int off = lowest_set_bit(c.rvfi_mem_wmask);
    const int size = __builtin_popcount(c.rvfi_mem_wmask);
    const uint64_t value = (uint64_t(c.rvfi_mem_wdata) >> (8 * off)) & ((1ull << (8 * size)) - 1);
    std::fprintf(out, " mem 0x%08" PRIx32 " 0x%0*" PRIx64, uint32_t(c.rvfi_mem_addr + off),
                 2 * size, value);
  }
  std::fputc('\n', out);
}

int finish(int code, const std::string& line, uint64_t cycles) {
  std::printf("cycles %" PRIu64 "\n%s\n", cycles, line.c_str());
  return code;
}

}  // namespace

int main(int argc, char** argv) {
  const char* elf_path = nullptr;
  const char* trace_path = nullptr;
  const char* vcd_path = nullptr;
  uint64_t max_cycles = 1'000'000;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!std::strcmp(argv[i], "--vcd") && i + 1 < argc) vcd_path = argv[++i];
    else if (!std::strcmp(argv[i], "--max-cycles") && i + 1 < argc) max_cycles = std::stoull(argv[++i]);
    else elf_path = argv[i];
  }
  if (!elf_path) {
    std::fprintf(stderr, "usage: %s [--trace FILE] [--vcd FILE] [--max-cycles N] program.elf\n", argv[0]);
    return 2;
  }

  Ram ram;
  Program prog;
  std::string err;
  if (!load_elf(elf_path, ram, prog, err)) return finish(2, "RESULT ERROR " + err, 0);
  if (prog.entry != kRamBase) return finish(2, "RESULT ERROR entry point is not 0x80000000", 0);

  FILE* trace = nullptr;
  if (trace_path && !(trace = std::fopen(trace_path, "w")))
    return finish(2, std::string("RESULT ERROR cannot write ") + trace_path, 0);
  std::unique_ptr<FILE, int (*)(FILE*)> trace_guard(trace, [](FILE* f) { return f ? std::fclose(f) : 0; });

  auto ctx = std::make_unique<VerilatedContext>();
  ctx->commandArgs(argc, argv);
  auto core = std::make_unique<Vcs_core>(ctx.get());

  // Waveforms at a nominal 100 MHz: one dump per clock half-period, 5 ns
  // apart (the VCD unit is 1 ps). After the two reset cycles, cycle n of the
  // run (the number the commit log and "cycles" count from 0) has its
  // rising edge at 25 + 10n ns.
  std::unique_ptr<VerilatedVcdC> vcd;
  uint64_t vcd_time = 0;
  if (vcd_path) {
    ctx->traceEverOn(true);
    vcd = std::make_unique<VerilatedVcdC>();
    core->trace(vcd.get(), 99);
    vcd->open(vcd_path);
    if (!vcd->isOpen()) return finish(2, std::string("RESULT ERROR cannot write ") + vcd_path, 0);
  }
  auto dump = [&]() { if (vcd) { vcd->dump(vcd_time); vcd_time += 5000; } };
  struct VcdCloser { std::unique_ptr<VerilatedVcdC>& v; ~VcdCloser() { if (v) v->close(); } } vcd_closer{vcd};

  // Reset for two cycles.
  core->rst = 1;
  for (int i = 0; i < 2; ++i) {
    core->clk = 0; core->eval(); dump();
    core->clk = 1; core->eval(); dump();
  }
  core->rst = 0;

  for (uint64_t cycle = 0; cycle < max_cycles; ++cycle) {
    // Settle the combinational paths: pc -> instruction -> data address -> load data.
    core->clk = 0;
    core->eval();
    if (!ram.contains(core->imem_addr, 4))
      return finish(2, "RESULT ERROR fetch outside RAM", cycle);
    core->imem_rdata = ram.read32(core->imem_addr);
    core->eval();
    if ((core->dmem_re || core->dmem_we) && !ram.contains(core->dmem_addr, 4)) {
      char msg[96];
      std::snprintf(msg, sizeof msg, "RESULT ERROR data access outside RAM at pc 0x%08x",
                    uint32_t(core->rvfi_pc_rdata));
      return finish(2, msg, cycle);
    }
    core->dmem_rdata = core->dmem_re ? ram.read32(core->dmem_addr) : 0;
    core->eval();
    dump();

    if (!core->rvfi_valid) return finish(2, "RESULT ERROR core is not retiring", cycle);
    if (trace) log_commit(trace, *core);
    if (core->rvfi_trap) {
      char msg[96];
      std::snprintf(msg, sizeof msg, "RESULT ERROR trap at pc 0x%08x insn 0x%08x",
                    uint32_t(core->rvfi_pc_rdata), uint32_t(core->rvfi_insn));
      return finish(2, msg, cycle);
    }

    // Memory write, then the clock edge.
    const bool to_host = core->dmem_we && core->dmem_addr == (prog.tohost & ~3u) &&
                         (core->dmem_wmask & 1u) && core->dmem_wdata != 0;
    const uint32_t host_value = core->dmem_wdata;
    if (core->dmem_we) ram.write32(core->dmem_addr, core->dmem_wdata, core->dmem_wmask);
    core->clk = 1;
    core->eval();
    dump();

    if (to_host) {
      if (host_value == 1) return finish(0, "RESULT PASS", cycle + 1);
      return finish(1, "RESULT FAIL " + std::to_string(host_value >> 1), cycle + 1);
    }
  }
  return finish(2, "RESULT ERROR timeout", max_cycles);
}
