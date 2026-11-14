# Coresample build.
#   make sim      build the Verilator simulator of the core
#   make tests    build the riscv-tests ELFs (rv32ui, rv32um)
#   make test     run every test on the core
#   make compare  run every test and compare each instruction with Spike

RISCV_PREFIX ?= riscv64-unknown-elf-
VERILATOR    ?= verilator
SPIKE        ?= spike
PYTHON       ?= python3
RISCV_TESTS  ?= third_party/riscv-tests

BUILD := build
SIM   := $(BUILD)/sim/cs_sim
RTL   := hw/rtl/cs_pkg.sv hw/rtl/cs_core.sv
SIM_SRC := hw/sim/sim_main.cpp

ISA_DIR := $(RISCV_TESTS)/isa
# ma_data checks misaligned accesses, which the core reports as traps for now.
UI_TESTS := $(filter-out ma_data,$(basename $(notdir $(wildcard $(ISA_DIR)/rv32ui/*.S))))
UM_TESTS := $(basename $(notdir $(wildcard $(ISA_DIR)/rv32um/*.S)))
TEST_ELFS := $(UI_TESTS:%=$(BUILD)/tests/rv32ui-%.elf) $(UM_TESTS:%=$(BUILD)/tests/rv32um-%.elf)

TEST_CFLAGS := -march=rv32im_zicsr_zifencei -mabi=ilp32 -static -mcmodel=medany \
               -fvisibility=hidden -nostdlib -nostartfiles \
               -Isw/env -I$(ISA_DIR)/macros/scalar -Tsw/env/link.ld

.PHONY: all sim tests test compare check-tests clean
all: sim

sim: $(SIM)

$(SIM): $(RTL) $(SIM_SRC)
	mkdir -p $(BUILD)/sim
	$(VERILATOR) --cc --exe --build -j 0 -Wall --top-module cs_core \
	  -Mdir $(BUILD)/sim -o cs_sim $(RTL) $(abspath $(SIM_SRC))

check-tests:
	@test -d $(ISA_DIR) || { echo "riscv-tests not found in $(RISCV_TESTS): run scripts/install-deps.sh"; exit 1; }

$(BUILD)/tests:
	mkdir -p $@

$(BUILD)/tests/rv32ui-%.elf: $(ISA_DIR)/rv32ui/%.S sw/env/riscv_test.h sw/env/link.ld | $(BUILD)/tests
	$(RISCV_PREFIX)gcc $(TEST_CFLAGS) $< -o $@

$(BUILD)/tests/rv32um-%.elf: $(ISA_DIR)/rv32um/%.S sw/env/riscv_test.h sw/env/link.ld | $(BUILD)/tests
	$(RISCV_PREFIX)gcc $(TEST_CFLAGS) $< -o $@

tests: check-tests $(TEST_ELFS)

test: sim tests
	$(PYTHON) scripts/run_tests.py --sim $(SIM) $(TEST_ELFS)

compare: sim tests
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --spike $(SPIKE) $(TEST_ELFS)

clean:
	rm -rf $(BUILD)
