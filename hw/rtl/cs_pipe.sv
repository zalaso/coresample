// Coresample 5-stage pipelined RV32IM core: IF ID EX MEM WB.
//
// Same ports and memory model as cs_core (combinational instruction and data
// reads, data writes on the rising edge) and the same architectural behavior:
// it must retire exactly the instructions the single-cycle core retires.
//
// Hazards:
// * data: results are forwarded to EX from the EX/MEM and MEM/WB registers,
//   and ID reads the register file with a bypass from WB. A load followed by
//   an instruction that uses its result costs one stall cycle (load-use).
// * control: branches, JAL and JALR resolve in EX. A taken branch or a jump
//   flushes the two younger instructions (IF and ID). FENCE.I does the same,
//   so code written by stores is fetched again.
// * traps (illegal instruction, misaligned access or target) are detected in
//   EX: younger instructions are flushed, fetching stops, and the trapping
//   instruction retires with rvfi_trap set; then the core halts.
//
// The dbg_* ports expose which instruction (by fetch sequence number) is in
// each stage and why cycles were lost; the testbench turns them into a
// pipeline trace.
module cs_pipe
  import cs_pkg::*;
#(
  parameter logic [31:0] RESET_PC = 32'h8000_0000
) (
  input  logic        clk,
  input  logic        rst,

  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,

  output logic [31:0] dmem_addr,
  output logic        dmem_re,
  output logic        dmem_we,
  output logic [3:0]  dmem_wmask,
  output logic [31:0] dmem_wdata,
  input  logic [31:0] dmem_rdata,

  output logic        rvfi_valid,
  output logic [63:0] rvfi_order,
  output logic [31:0] rvfi_insn,
  output logic        rvfi_trap,
  output logic        rvfi_halt,
  output logic        rvfi_intr,
  output logic [1:0]  rvfi_mode,
  output logic [1:0]  rvfi_ixl,
  output logic [4:0]  rvfi_rs1_addr,
  output logic [4:0]  rvfi_rs2_addr,
  output logic [31:0] rvfi_rs1_rdata,
  output logic [31:0] rvfi_rs2_rdata,
  output logic [4:0]  rvfi_rd_addr,
  output logic [31:0] rvfi_rd_wdata,
  output logic [31:0] rvfi_pc_rdata,
  output logic [31:0] rvfi_pc_wdata,
  output logic [31:0] rvfi_mem_addr,
  output logic [3:0]  rvfi_mem_rmask,
  output logic [3:0]  rvfi_mem_wmask,
  output logic [31:0] rvfi_mem_rdata,
  output logic [31:0] rvfi_mem_wdata,

  // Pipeline visibility: valid flag and fetch sequence number per stage.
  output logic        dbg_if_valid,
  output logic [31:0] dbg_if_id,
  output logic [31:0] dbg_if_pc,
  output logic        dbg_id_valid,
  output logic [31:0] dbg_id_id,
  output logic        dbg_ex_valid,
  output logic [31:0] dbg_ex_id,
  output logic        dbg_mem_valid,
  output logic [31:0] dbg_mem_id,
  output logic        dbg_wb_valid,
  output logic [31:0] dbg_wb_id,
  output logic        dbg_stall,       // load-use: IF and ID hold, EX gets a bubble
  output logic        dbg_flush,       // IF and ID are discarded
  output logic [2:0]  dbg_flush_cause  // 1 branch, 2 jump, 3 fence.i, 4 trap
);

  // ------------------------------------------------------------- decoding
  // Instructions travel down the pipeline as raw words; each stage decodes
  // the fields it needs.
  // Field extractors read only some bits of the word, by design.
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic [6:0] op_of(input logic [31:0] i);  return i[6:0];   endfunction
  function automatic logic [4:0] rd_of(input logic [31:0] i);  return i[11:7];  endfunction
  function automatic logic [4:0] rs1_of(input logic [31:0] i); return i[19:15]; endfunction
  function automatic logic [4:0] rs2_of(input logic [31:0] i); return i[24:20]; endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  function automatic logic uses_rs1(input logic [31:0] i);
    return !(op_of(i) inside {OPC_LUI, OPC_AUIPC, OPC_JAL});
  endfunction
  function automatic logic uses_rs2(input logic [31:0] i);
    return op_of(i) inside {OPC_BRANCH, OPC_STORE, OPC_OP};
  endfunction
  function automatic logic writes_rd(input logic [31:0] i);
    return op_of(i) inside {OPC_LUI, OPC_AUIPC, OPC_JAL, OPC_JALR, OPC_LOAD, OPC_OP_IMM, OPC_OP};
  endfunction

  function automatic logic is_legal(input logic [31:0] i);
    logic [2:0] f3;
    logic [6:0] f7;
    f3 = i[14:12];
    f7 = i[31:25];
    if (i[1:0] != 2'b11) return 1'b0;  // no compressed instructions
    unique case (op_of(i))
      OPC_LUI, OPC_AUIPC, OPC_JAL: return 1'b1;
      OPC_JALR:     return f3 == 3'b000;
      OPC_BRANCH:   return (f3 != 3'b010) && (f3 != 3'b011);
      OPC_LOAD:     return f3 inside {3'b000, 3'b001, 3'b010, 3'b100, 3'b101};
      OPC_STORE:    return f3 inside {3'b000, 3'b001, 3'b010};
      OPC_OP_IMM:   return (f3 == 3'b001) ? (f7 == F7_BASE)
                         : (f3 == 3'b101) ? (f7 == F7_BASE || f7 == F7_ALT) : 1'b1;
      OPC_OP:       return (f7 == F7_BASE) || (f7 == F7_MULDIV)
                         || (f7 == F7_ALT && (f3 == 3'b000 || f3 == 3'b101));
      OPC_MISC_MEM: return (f3 == 3'b000) || (f3 == 3'b001);  // FENCE, FENCE.I
      OPC_SYSTEM:   return 1'b0;  // ECALL, EBREAK and CSRs are not implemented yet
      default:      return 1'b0;
    endcase
  endfunction

  // -------------------------------------------------------- pipeline state
  logic [31:0] regs [32];
  logic        halted, stop_fetch;
  logic [63:0] order;
  logic [31:0] fetch_id;

  logic [31:0] pc_f;                                   // IF
  logic        d_valid;  logic [31:0] d_pc, d_insn, d_id;  // IF/ID
  logic        e_valid;  logic [31:0] e_pc, e_insn, e_id, e_rs1v, e_rs2v;  // ID/EX
  logic        m_valid, m_trap, m_rd_we, m_re, m_we;   // EX/MEM
  logic [31:0] m_pc, m_insn, m_id, m_result, m_addr, m_store, m_next_pc, m_rs1v, m_rs2v;
  logic [3:0]  m_mask;
  logic        w_valid, w_trap, w_rd_we, w_re, w_we;   // MEM/WB
  logic [31:0] w_pc, w_insn, w_id, w_wdata, w_next_pc, w_rs1v, w_rs2v, w_addr, w_rdata, w_store;
  logic [3:0]  w_mask;

  logic active;
  assign active = !rst && !halted;

  // ------------------------------------------------------------------- IF
  assign imem_addr = pc_f;

  // ------------------------------------------------------------------- ID
  // Register read with a bypass from the instruction writing back this cycle.
  function automatic logic [31:0] read_reg(input logic [4:0] r);
    if (r == 5'd0) return 32'd0;
    if (w_valid && w_rd_we && rd_of(w_insn) == r) return w_wdata;
    return regs[r];
  endfunction

  logic load_use;
  assign load_use = d_valid && e_valid && op_of(e_insn) == OPC_LOAD && rd_of(e_insn) != 5'd0
                 && ((uses_rs1(d_insn) && rs1_of(d_insn) == rd_of(e_insn))
                  || (uses_rs2(d_insn) && rs2_of(d_insn) == rd_of(e_insn)));

  // ------------------------------------------------------------------- EX
  // Operands with forwarding: the youngest older result wins.
  function automatic logic [31:0] forward(input logic [4:0] r, input logic [31:0] from_id);
    if (r == 5'd0) return 32'd0;
    if (m_valid && m_rd_we && !m_re && rd_of(m_insn) == r) return m_result;  // loads: see load_use
    if (w_valid && w_rd_we && rd_of(w_insn) == r) return w_wdata;
    return from_id;
  endfunction

  logic [31:0] a, b;  // rs1 and rs2 values as the instruction in EX sees them
  assign a = forward(rs1_of(e_insn), e_rs1v);
  assign b = forward(rs2_of(e_insn), e_rs2v);

  logic [6:0] e_op, e_f7;
  logic [2:0] e_f3;
  assign e_op = op_of(e_insn);
  assign e_f3 = e_insn[14:12];
  assign e_f7 = e_insn[31:25];

  logic [31:0] imm_i, imm_s, imm_b, imm_u, imm_j;
  assign imm_i = {{20{e_insn[31]}}, e_insn[31:20]};
  assign imm_s = {{20{e_insn[31]}}, e_insn[31:25], e_insn[11:7]};
  assign imm_b = {{19{e_insn[31]}}, e_insn[31], e_insn[7], e_insn[30:25], e_insn[11:8], 1'b0};
  assign imm_u = {e_insn[31:12], 12'b0};
  assign imm_j = {{11{e_insn[31]}}, e_insn[31], e_insn[19:12], e_insn[20], e_insn[30:21], 1'b0};

  // ALU, shared by OP and OP-IMM; insn[30] selects SUB and SRA/SRAI.
  logic [31:0] alu_b, alu_res;
  logic [4:0]  shamt;
  assign alu_b = (e_op == OPC_OP) ? b : imm_i;
  assign shamt = alu_b[4:0];
  always_comb begin
    unique case (e_f3)
      3'b000: alu_res = (e_op == OPC_OP && e_insn[30]) ? a - alu_b : a + alu_b;
      3'b001: alu_res = a << shamt;
      3'b010: alu_res = {31'd0, $signed(a) < $signed(alu_b)};
      3'b011: alu_res = {31'd0, a < alu_b};
      3'b100: alu_res = a ^ alu_b;
      3'b101: alu_res = e_insn[30] ? 32'($signed(a) >>> shamt) : a >> shamt;
      3'b110: alu_res = a | alu_b;
      3'b111: alu_res = a & alu_b;
    endcase
  end

  // M extension, one cycle (see cs_core for the 33-bit signedness trick).
  logic signed [32:0] mul_a, mul_b;
  logic signed [63:0] mul_p;
  logic [31:0] m_res;
  logic div_by_zero, div_overflow;
  assign mul_a = {(e_f3 != 3'b011) && a[31], a};
  assign mul_b = {(e_f3 == 3'b001) && b[31], b};
  assign mul_p = mul_a * mul_b;
  assign div_by_zero  = (b == 32'd0);
  assign div_overflow = (a == 32'h8000_0000) && (b == 32'hFFFF_FFFF);
  always_comb begin
    unique case (e_f3)
      3'b000:  m_res = mul_p[31:0];
      3'b001,
      3'b010,
      3'b011:  m_res = mul_p[63:32];
      3'b100:  m_res = div_by_zero ? 32'hFFFF_FFFF : div_overflow ? 32'h8000_0000
                                                  : 32'($signed(a) / $signed(b));
      3'b101:  m_res = div_by_zero ? 32'hFFFF_FFFF : a / b;
      3'b110:  m_res = div_by_zero ? a : div_overflow ? 32'd0 : 32'($signed(a) % $signed(b));
      3'b111:  m_res = div_by_zero ? a : a % b;
    endcase
  end

  logic br_taken;
  always_comb begin
    unique case (e_f3)
      3'b000:  br_taken = (a == b);
      3'b001:  br_taken = (a != b);
      3'b100:  br_taken = ($signed(a) <  $signed(b));
      3'b101:  br_taken = ($signed(a) >= $signed(b));
      3'b110:  br_taken = (a <  b);
      3'b111:  br_taken = (a >= b);
      default: br_taken = 1'b0;
    endcase
  end

  logic [31:0] ls_addr;
  logic [1:0]  ls_off;
  logic        e_load, e_store, misaligned;
  assign e_load  = (e_op == OPC_LOAD);
  assign e_store = (e_op == OPC_STORE);
  assign ls_addr = a + (e_store ? imm_s : imm_i);
  assign ls_off  = ls_addr[1:0];
  assign misaligned = (e_f3[1:0] == 2'b01 && ls_off[0]) || (e_f3[1:0] == 2'b10 && ls_off != 2'b00);

  logic [3:0]  access_mask;
  logic [31:0] store_data;
  always_comb begin
    unique case (e_f3[1:0])
      2'b00:   begin access_mask = 4'b0001 << ls_off; store_data = {4{b[7:0]}};  end
      2'b01:   begin access_mask = 4'b0011 << ls_off; store_data = {2{b[15:0]}}; end
      default: begin access_mask = 4'b1111;           store_data = b;            end
    endcase
  end

  // Result, next pc and redirect for the instruction in EX.
  logic [31:0] ex_result, ex_next_pc;
  logic        ex_rd_we, ex_jump, ex_trap, ex_redirect;
  logic [2:0]  ex_cause;
  always_comb begin
    ex_result  = alu_res;
    ex_next_pc = e_pc + 32'd4;
    ex_rd_we   = writes_rd(e_insn);
    ex_jump    = 1'b0;
    ex_cause   = 3'd0;
    unique case (e_op)
      OPC_LUI:      ex_result = imm_u;
      OPC_AUIPC:    ex_result = e_pc + imm_u;
      OPC_JAL:      begin ex_result = e_pc + 32'd4; ex_next_pc = e_pc + imm_j; ex_jump = 1'b1; ex_cause = 3'd2; end
      OPC_JALR:     begin ex_result = e_pc + 32'd4; ex_next_pc = (a + imm_i) & ~32'd1;
                          ex_jump = 1'b1; ex_cause = 3'd2; end
      OPC_BRANCH:   if (br_taken) begin ex_next_pc = e_pc + imm_b; ex_jump = 1'b1; ex_cause = 3'd1; end
      OPC_OP:       if (e_f7 == F7_MULDIV) ex_result = m_res;
      OPC_MISC_MEM: if (e_f3 == 3'b001) begin ex_jump = 1'b1; ex_cause = 3'd3; end  // FENCE.I: refetch
      default:      ;
    endcase
    ex_trap = e_valid && (!is_legal(e_insn) || ((e_load || e_store) && misaligned)
                          || ex_next_pc[1:0] != 2'b00);
    if (ex_trap) begin
      ex_rd_we   = 1'b0;
      ex_next_pc = e_pc;
      ex_cause   = 3'd4;
    end
    ex_redirect = e_valid && ex_jump && !ex_trap;
  end

  // ------------------------------------------------------------------ MEM
  assign dmem_addr  = {m_addr[31:2], 2'b00};
  assign dmem_re    = active && m_valid && m_re;
  assign dmem_we    = active && m_valid && m_we;
  assign dmem_wmask = m_mask;
  assign dmem_wdata = m_store;

  logic [15:0] load_shifted;
  logic [31:0] load_val;
  assign load_shifted = 16'(dmem_rdata >> {m_addr[1:0], 3'b000});
  always_comb begin
    unique case (m_insn[14:12])
      3'b000:  load_val = {{24{load_shifted[7]}},  load_shifted[7:0]};
      3'b001:  load_val = {{16{load_shifted[15]}}, load_shifted[15:0]};
      3'b100:  load_val = {24'd0, load_shifted[7:0]};
      3'b101:  load_val = {16'd0, load_shifted[15:0]};
      default: load_val = dmem_rdata;
    endcase
  end

  // -------------------------------------------------------------- control
  logic flush, stall;
  assign flush = ex_redirect || ex_trap;
  assign stall = load_use && !flush;

  // ----------------------------------------------------------- sequential
  always_ff @(posedge clk) begin
    if (rst) begin
      pc_f       <= RESET_PC;
      fetch_id   <= 32'd0;
      halted     <= 1'b0;
      stop_fetch <= 1'b0;
      order      <= 64'd0;
      d_valid    <= 1'b0;
      e_valid    <= 1'b0;
      m_valid    <= 1'b0;
      w_valid    <= 1'b0;
    end else if (!halted) begin
      // IF and IF/ID
      if (ex_trap) begin
        stop_fetch <= 1'b1;
        d_valid    <= 1'b0;
      end else if (ex_redirect) begin
        pc_f    <= ex_next_pc;
        d_valid <= 1'b0;
      end else if (!stall) begin
        d_valid <= !stop_fetch;
        d_pc    <= pc_f;
        d_insn  <= imem_rdata;
        d_id    <= fetch_id;
        if (!stop_fetch) begin
          pc_f     <= pc_f + 32'd4;
          fetch_id <= fetch_id + 32'd1;
        end
      end
      if (ex_redirect) fetch_id <= fetch_id + 32'd1;  // the discarded fetch keeps its number

      // ID/EX: a bubble on flush or stall
      e_valid <= d_valid && !flush && !stall;
      e_pc    <= d_pc;
      e_insn  <= d_insn;
      e_id    <= d_id;
      e_rs1v  <= read_reg(rs1_of(d_insn));
      e_rs2v  <= read_reg(rs2_of(d_insn));

      // EX/MEM
      m_valid   <= e_valid;
      m_pc      <= e_pc;
      m_insn    <= e_insn;
      m_id      <= e_id;
      m_trap    <= ex_trap;
      m_rd_we   <= ex_rd_we;
      m_re      <= e_load && !ex_trap;
      m_we      <= e_store && !ex_trap;
      m_result  <= ex_result;
      m_addr    <= ls_addr;
      m_store   <= store_data;
      m_mask    <= access_mask;
      m_next_pc <= ex_next_pc;
      m_rs1v    <= uses_rs1(e_insn) ? a : 32'd0;
      m_rs2v    <= uses_rs2(e_insn) ? b : 32'd0;

      // MEM/WB
      w_valid   <= m_valid;
      w_pc      <= m_pc;
      w_insn    <= m_insn;
      w_id      <= m_id;
      w_trap    <= m_trap;
      w_rd_we   <= m_rd_we;
      w_re      <= m_re;
      w_we      <= m_we;
      w_wdata   <= m_re ? load_val : m_result;
      w_next_pc <= m_next_pc;
      w_rs1v    <= m_rs1v;
      w_rs2v    <= m_rs2v;
      w_addr    <= dmem_addr;
      w_mask    <= m_mask;
      w_rdata   <= dmem_rdata;
      w_store   <= m_store;

      // WB
      if (w_valid) begin
        order <= order + 64'd1;
        if (w_trap) halted <= 1'b1;
        if (w_rd_we && rd_of(w_insn) != 5'd0) regs[rd_of(w_insn)] <= w_wdata;
      end
    end
  end

  // ----------------------------------------------------------------- RVFI
  assign rvfi_valid     = active && w_valid;
  assign rvfi_order     = order;
  assign rvfi_insn      = w_insn;
  assign rvfi_trap      = w_trap;
  assign rvfi_halt      = w_trap;
  assign rvfi_intr      = 1'b0;
  assign rvfi_mode      = 2'd3;
  assign rvfi_ixl       = 2'd1;
  assign rvfi_rs1_addr  = uses_rs1(w_insn) ? rs1_of(w_insn) : 5'd0;
  assign rvfi_rs2_addr  = uses_rs2(w_insn) ? rs2_of(w_insn) : 5'd0;
  assign rvfi_rs1_rdata = w_rs1v;
  assign rvfi_rs2_rdata = w_rs2v;
  assign rvfi_rd_addr   = w_rd_we ? rd_of(w_insn) : 5'd0;
  assign rvfi_rd_wdata  = (w_rd_we && rd_of(w_insn) != 5'd0) ? w_wdata : 32'd0;
  assign rvfi_pc_rdata  = w_pc;
  assign rvfi_pc_wdata  = w_next_pc;
  assign rvfi_mem_addr  = (w_re || w_we) ? w_addr : 32'd0;
  assign rvfi_mem_rmask = w_re ? w_mask : 4'd0;
  assign rvfi_mem_wmask = w_we ? w_mask : 4'd0;
  assign rvfi_mem_rdata = w_re ? w_rdata : 32'd0;
  assign rvfi_mem_wdata = w_we ? w_store : 32'd0;

  // ---------------------------------------------------------------- debug
  assign dbg_if_valid    = active && !stop_fetch;
  assign dbg_if_id       = fetch_id;
  assign dbg_if_pc       = pc_f;
  assign dbg_id_valid    = active && d_valid;
  assign dbg_id_id       = d_id;
  assign dbg_ex_valid    = active && e_valid;
  assign dbg_ex_id       = e_id;
  assign dbg_mem_valid   = active && m_valid;
  assign dbg_mem_id      = m_id;
  assign dbg_wb_valid    = active && w_valid;
  assign dbg_wb_id       = w_id;
  assign dbg_stall       = active && stall;
  assign dbg_flush       = active && flush;
  assign dbg_flush_cause = ex_cause;

endmodule
