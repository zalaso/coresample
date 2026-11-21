# Coresample build. Every target runs on the single-cycle core; add CORE=pipe
# for the 5-stage pipeline (build directories get a -pipe suffix).
#   make sim      build the Verilator simulator of the core
#   make tests    build the riscv-tests ELFs (rv32ui, rv32um)
#   make test     run every test on the core
#   make compare  run every test and compare each instruction with Spike
#   make arch-test  build riscv-arch-test (ACT4) self-checking ELFs with Spike
#                   as the reference, run them and compare with Spike
#   make trace    run the dense_int8 C program, compare it with Spike and write
#                 its Perfetto trace and VCD waveforms to build/programs/
#   make unit     unit tests of the Python tools
#   make nn-test  tests of the network simulator (Rust) against PyTorch references
#   make nn-ref   regenerate those references (needs PyTorch in .venv)
#   make g2       compile the int8 CNN, run it on the core and on Spike (all 48
#                 images), and write the Perfetto trace of one image
#   make compiler-test  end-to-end tests of the compiler on the core
#   make index    trace index of the CNN on all 48 images (build/index[-pipe])
#   make web      put both indexes in web/data for the inspector (serve web/ over HTTP)

RISCV_PREFIX ?= riscv64-unknown-elf-
VERILATOR    ?= verilator
SPIKE        ?= spike
PYTHON       ?= python3
RISCV_TESTS  ?= third_party/riscv-tests
ARCH_TEST    ?= third_party/riscv-arch-test

BUILD := build
CORE  ?= single
ifeq ($(CORE),pipe)
  TOP      := cs_pipe
  SUFFIX   := -pipe
  SIM_DEFS := -CFLAGS -DCS_PIPELINE
else ifeq ($(CORE),single)
  TOP      := cs_core
  SUFFIX   :=
  SIM_DEFS :=
else
  $(error CORE must be single or pipe)
endif
SIM   := $(BUILD)/sim$(SUFFIX)/cs_sim
RTL   := hw/rtl/cs_pkg.sv hw/rtl/$(TOP).sv
SIM_SRC := hw/sim/sim_main.cpp
TRACES := $(BUILD)/traces$(SUFFIX)

ISA_DIR := $(RISCV_TESTS)/isa
# ma_data checks misaligned accesses, which the core reports as traps for now.
UI_TESTS := $(filter-out ma_data,$(basename $(notdir $(wildcard $(ISA_DIR)/rv32ui/*.S))))
UM_TESTS := $(basename $(notdir $(wildcard $(ISA_DIR)/rv32um/*.S)))
# Coresample's own directed tests (sw/tests), for cases the suites miss.
CS_TESTS := $(basename $(notdir $(wildcard sw/tests/*.S)))
TEST_ELFS := $(UI_TESTS:%=$(BUILD)/tests/rv32ui-%.elf) $(UM_TESTS:%=$(BUILD)/tests/rv32um-%.elf) \
             $(CS_TESTS:%=$(BUILD)/tests/cs-%.elf)

TEST_CFLAGS := -march=rv32im_zicsr_zifencei -mabi=ilp32 -static -mcmodel=medany \
               -fvisibility=hidden -nostdlib -nostartfiles \
               -Isw/env -I$(ISA_DIR)/macros/scalar -Tsw/env/link.ld

.PHONY: all sim tests test compare arch-test trace unit nn-test nn-ref g2 compiler-test index web check-tests clean
all: sim

sim: $(SIM)

$(SIM): $(RTL) $(SIM_SRC)
	mkdir -p $(dir $(SIM))
	$(VERILATOR) --cc --exe --build -j 0 -Wall --trace --top-module $(TOP) --prefix Vcs_core \
	  $(SIM_DEFS) -Mdir $(dir $(SIM)) -o cs_sim $(RTL) $(abspath $(SIM_SRC))

check-tests:
	@test -d $(ISA_DIR) || { echo "riscv-tests not found in $(RISCV_TESTS): run scripts/install-deps.sh"; exit 1; }

$(BUILD)/tests:
	mkdir -p $@

$(BUILD)/tests/rv32ui-%.elf: $(ISA_DIR)/rv32ui/%.S sw/env/riscv_test.h sw/env/link.ld | $(BUILD)/tests
	$(RISCV_PREFIX)gcc $(TEST_CFLAGS) $< -o $@

$(BUILD)/tests/cs-%.elf: sw/tests/%.S sw/env/riscv_test.h sw/env/link.ld | $(BUILD)/tests
	$(RISCV_PREFIX)gcc $(TEST_CFLAGS) $< -o $@

$(BUILD)/tests/rv32um-%.elf: $(ISA_DIR)/rv32um/%.S sw/env/riscv_test.h sw/env/link.ld | $(BUILD)/tests
	$(RISCV_PREFIX)gcc $(TEST_CFLAGS) $< -o $@

tests: check-tests $(TEST_ELFS)

test: sim tests
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --trace-dir $(TRACES) $(TEST_ELFS)

compare: sim tests
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --spike $(SPIKE) --trace-dir $(TRACES) $(TEST_ELFS)

# The ELF list depends on the test headers, so the Python script builds them all
# and the runner picks up whatever it produced.
arch-test: sim
	$(PYTHON) scripts/build_arch_tests.py --act $(ARCH_TEST) --spike $(SPIKE) --gcc $(RISCV_PREFIX)gcc
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --spike $(SPIKE) --trace-dir $(TRACES)/arch \
	  $(BUILD)/arch/elfs/*.elf

# C programs: crt0 + one directory per program under sw/programs/.
PROG_CFLAGS := -march=rv32im_zicsr -mabi=ilp32 -O2 -g -static -mcmodel=medany \
               -nostdlib -nostartfiles -ffreestanding -fno-builtin -Wall -Wextra \
               -Wl,--no-warn-rwx-segments -Tsw/runtime/link.ld
DENSE := $(BUILD)/programs$(SUFFIX)/dense_int8

$(DENSE)/dense_int8_data.h: scripts/gen_dense_int8.py
	mkdir -p $(DENSE)
	$(PYTHON) scripts/gen_dense_int8.py $@

$(DENSE)/dense_int8.elf: sw/programs/dense_int8/main.c sw/runtime/crt0.S sw/runtime/link.ld $(DENSE)/dense_int8_data.h
	$(RISCV_PREFIX)gcc $(PROG_CFLAGS) -I$(DENSE) sw/runtime/crt0.S $< -lgcc -o $@

# The commit log comes from the Spike comparison, so a trace is only written
# for a run that matches Spike instruction by instruction.
trace: sim $(DENSE)/dense_int8.elf
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --spike $(SPIKE) --trace-dir $(DENSE) $(DENSE)/dense_int8.elf
	$(SIM) --vcd $(DENSE)/dense_int8.vcd $(DENSE)/dense_int8.elf
	$(PYTHON) scripts/trace_to_perfetto.py --elf $(DENSE)/dense_int8.elf --log $(DENSE)/dense_int8.core.log \
	  -o $(DENSE)/dense_int8.perfetto.json

unit:
	$(PYTHON) -m unittest discover -s scripts/tests

CARGO ?= cargo
VENV_PYTHON ?= .venv/bin/python

nn-test:
	$(CARGO) test -p coresample-nn

nn-ref:
	$(VENV_PYTHON) sim/nn/reference/gen_reference.py

G2 := $(BUILD)/g2$(SUFFIX)
# On the pipeline, also the per-cycle stage log, its Kanata trace for Konata,
# and where the cycles went.
# Perfetto uses the pipeline's real cycles; the report charges every cycle to a
# layer and operation.
G2_TIMING := $(if $(filter pipe,$(CORE)),--pipeline $(G2)/cnn1.pipeline.log)
G2_REPORT := $(if $(filter pipe,$(CORE)),--pipeline $(G2)/cnn1.pipeline.log,--log $(G2)/cnn1.core.log)
ifeq ($(CORE),pipe)
define G2_PIPELINE
$(SIM) --pipeline $(G2)/cnn1.pipeline.log $(G2)/cnn1.elf
	$(PYTHON) scripts/pipeline_to_kanata.py --log $(G2)/cnn1.pipeline.log --elf $(G2)/cnn1.elf \
	  -o $(G2)/cnn1.kanata --summary
endef
endif
CNN_REF := sim/nn/tests/data/quant_cnn.ref
CS_COMPILE := $(CARGO) run -q --release -p coresample-compiler --bin cs-compile --

# G2: the CNN on the core with output identical to the int8 reference. The
# program checks its own output; Spike must agree, and on one image every
# retired instruction must match Spike's.
g2: sim
	mkdir -p $(G2)
	$(CS_COMPILE) --ref $(CNN_REF) --samples 48 -o $(G2)/cnn48.S
	$(RISCV_PREFIX)gcc $(PROG_CFLAGS) sw/runtime/crt0.S $(G2)/cnn48.S -o $(G2)/cnn48.elf
	$(SIM) --max-cycles 50000000 $(G2)/cnn48.elf
	$(SPIKE) --isa=rv32im -m0x80000000:0x100000 $(G2)/cnn48.elf && echo "Spike: PASS"
	$(CS_COMPILE) --ref $(CNN_REF) --samples 1 -o $(G2)/cnn1.S
	$(RISCV_PREFIX)gcc $(PROG_CFLAGS) sw/runtime/crt0.S $(G2)/cnn1.S -o $(G2)/cnn1.elf
	$(PYTHON) scripts/run_tests.py --sim $(SIM) --spike $(SPIKE) --trace-dir $(G2) $(G2)/cnn1.elf
	$(G2_PIPELINE)
	$(PYTHON) scripts/trace_to_perfetto.py --elf $(G2)/cnn1.elf --log $(G2)/cnn1.core.log $(G2_TIMING) \
	  -o $(G2)/cnn1.perfetto.json
	$(PYTHON) scripts/op_report.py --elf $(G2)/cnn1.elf $(G2_REPORT)

# The index of the whole 48-image run. The raw logs (hundreds of MB) are
# streamed by the builder and can be deleted afterwards.
INDEX := $(BUILD)/index$(SUFFIX)
INDEX_PIPE := $(if $(filter pipe,$(CORE)),--pipeline $(G2)/cnn48.pipeline.log)
index: g2
	$(SIM) --max-cycles 50000000 --trace $(G2)/cnn48.core.log $(INDEX_PIPE) $(G2)/cnn48.elf
	$(PYTHON) scripts/trace_index.py --elf $(G2)/cnn48.elf --log $(G2)/cnn48.core.log $(INDEX_PIPE) -o $(INDEX)

# The inspector reads both cores' indexes; build them first with
# `make index` and `make index CORE=pipe`.
web:
	test -f $(BUILD)/index/index.json && test -f $(BUILD)/index-pipe/index.json
	rm -rf web/data && mkdir -p web/data
	cp -r $(BUILD)/index web/data/single
	cp -r $(BUILD)/index-pipe web/data/pipe
	printf '%s\n' '{"runs": [{"id": "single", "path": "data/single"}, {"id": "pipe", "path": "data/pipe"}]}' > web/data/runs.json

compiler-test: sim
	CS_SIM=$(abspath $(SIM)) $(CARGO) test -p coresample-compiler

clean:
	rm -rf $(BUILD)
