// Coresample single-cycle RV32IM core.
//
// One instruction retires every clock cycle. Memory lives outside the core:
// instruction and data reads are combinational, data writes take effect on
// the rising clock edge. There are no CSRs and no trap handler yet: an
// illegal instruction, ECALL/EBREAK or a misaligned access or jump target is
// reported as a trap on the RVFI port and the core halts.
//
// The rvfi_* ports follow the riscv-formal interface (RVFI): every retired
// instruction is reported with the registers and memory it read and wrote.
module cs_core
  import cs_pkg::*;
#(
  parameter logic [31:0] RESET_PC = 32'h8000_0000
) (
  input  logic        clk,
  input  logic        rst,

  // Instruction memory: combinational read
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,

  // Data memory: word-aligned address, combinational read, write on posedge
  output logic [31:0] dmem_addr,
  output logic        dmem_re,
  output logic        dmem_we,
  output logic [3:0]  dmem_wmask,
  output logic [31:0] dmem_wdata,   // already placed in its byte lanes
  input  logic [31:0] dmem_rdata,

  // RISC-V Formal Interface
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
  output logic [31:0] rvfi_mem_wdata
);

  // ------------------------------------------------------------------ state
  logic [31:0] pc;
  logic        halted;
  logic [63:0] order;
  logic [31:0] regs [32];  // regs[0] is never written and never read

  logic active;
  assign active = !rst && !halted;

  // ------------------------------------------------------------------ fetch
  logic [31:0] insn;
  assign imem_addr = pc;
  assign insn      = imem_rdata;

  // ----------------------------------------------------------------- decode
  logic [6:0] opcode, funct7;
  logic [4:0] rd, rs1, rs2;
  logic [2:0] funct3;
  assign opcode = insn[6:0];
  assign rd     = insn[11:7];
  assign funct3 = insn[14:12];
  assign rs1    = insn[19:15];
  assign rs2    = insn[24:20];
  assign funct7 = insn[31:25];

  logic [31:0] imm_i, imm_s, imm_b, imm_u, imm_j;
  assign imm_i = {{20{insn[31]}}, insn[31:20]};
  assign imm_s = {{20{insn[31]}}, insn[31:25], insn[11:7]};
  assign imm_b = {{19{insn[31]}}, insn[31], insn[7], insn[30:25], insn[11:8], 1'b0};
  assign imm_u = {insn[31:12], 12'b0};
  assign imm_j = {{11{insn[31]}}, insn[31], insn[19:12], insn[20], insn[30:21], 1'b0};

  logic legal;
  always_comb begin
    legal = 1'b0;
    if (insn[1:0] == 2'b11) begin  // no compressed instructions
      unique case (opcode)
        OPC_LUI, OPC_AUIPC, OPC_JAL: legal = 1'b1;
        OPC_JALR:     legal = (funct3 == 3'b000);
        OPC_BRANCH:   legal = (funct3 != 3'b010) && (funct3 != 3'b011);
        OPC_LOAD:     legal = (funct3 inside {3'b000, 3'b001, 3'b010, 3'b100, 3'b101});
        OPC_STORE:    legal = (funct3 inside {3'b000, 3'b001, 3'b010});
        OPC_OP_IMM: begin
          if (funct3 == 3'b001)      legal = (funct7 == F7_BASE);
          else if (funct3 == 3'b101) legal = (funct7 == F7_BASE) || (funct7 == F7_ALT);
          else                       legal = 1'b1;
        end
        OPC_OP:       legal = (funct7 == F7_BASE) || (funct7 == F7_MULDIV)
                           || (funct7 == F7_ALT && (funct3 == 3'b000 || funct3 == 3'b101));
        OPC_MISC_MEM: legal = (funct3 == 3'b000) || (funct3 == 3'b001);  // FENCE, FENCE.I
        OPC_SYSTEM:   legal = 1'b0;  // ECALL, EBREAK and CSRs are not implemented yet
        default:      legal = 1'b0;
      endcase
    end
  end

  logic uses_rs1, uses_rs2;
  assign uses_rs1 = !(opcode inside {OPC_LUI, OPC_AUIPC, OPC_JAL});
  assign uses_rs2 = (opcode inside {OPC_BRANCH, OPC_STORE, OPC_OP});

  // ------------------------------------------------------- register read
  logic [31:0] rs1_val, rs2_val;
  assign rs1_val = (rs1 == 5'd0) ? 32'd0 : regs[rs1];
  assign rs2_val = (rs2 == 5'd0) ? 32'd0 : regs[rs2];

  // -------------------------------------------------------------- ALU
  // Shared by OP (register operand) and OP-IMM (immediate operand).
  logic [31:0] alu_b, alu_res;
  logic [4:0]  shamt;
  logic        is_sub;
  assign alu_b  = (opcode == OPC_OP) ? rs2_val : imm_i;
  assign shamt  = alu_b[4:0];
  // insn[30] selects SUB and SRA/SRAI; for ADDI it is an immediate bit.
  assign is_sub = (opcode == OPC_OP) && insn[30];

  always_comb begin
    unique case (funct3)
      3'b000: alu_res = is_sub ? rs1_val - alu_b : rs1_val + alu_b;
      3'b001: alu_res = rs1_val << shamt;
      3'b010: alu_res = {31'd0, $signed(rs1_val) < $signed(alu_b)};
      3'b011: alu_res = {31'd0, rs1_val < alu_b};
      3'b100: alu_res = rs1_val ^ alu_b;
      3'b101: alu_res = insn[30] ? 32'($signed(rs1_val) >>> shamt) : rs1_val >> shamt;
      3'b110: alu_res = rs1_val | alu_b;
      3'b111: alu_res = rs1_val & alu_b;
    endcase
  end

  // --------------------------------------------------- M extension
  // Multiplication: extend both operands to 33 bits with the right
  // signedness, so bits [63:32] of one signed product give MULH, MULHSU and
  // MULHU alike. Division is combinational here; an FPGA build will need an
  // iterative divider.
  logic signed [32:0] mul_a, mul_b;
  logic signed [63:0] mul_p;  // low 64 bits of the 66-bit product are all we need
  logic [31:0] m_res;
  logic div_by_zero, div_overflow;

  assign mul_a = {(funct3 != 3'b011) && rs1_val[31], rs1_val};  // MULHU: unsigned
  assign mul_b = {(funct3 == 3'b001) && rs2_val[31], rs2_val};  // only MULH: signed
  assign mul_p = mul_a * mul_b;
  assign div_by_zero  = (rs2_val == 32'd0);
  assign div_overflow = (rs1_val == 32'h8000_0000) && (rs2_val == 32'hFFFF_FFFF);

  always_comb begin
    unique case (funct3)
      3'b000:  m_res = mul_p[31:0];                                       // MUL
      3'b001,
      3'b010,
      3'b011:  m_res = mul_p[63:32];                                      // MULH, MULHSU, MULHU
      3'b100:  m_res = div_by_zero  ? 32'hFFFF_FFFF :                     // DIV
                       div_overflow ? 32'h8000_0000 :
                       32'($signed(rs1_val) / $signed(rs2_val));
      3'b101:  m_res = div_by_zero  ? 32'hFFFF_FFFF : rs1_val / rs2_val;  // DIVU
      3'b110:  m_res = div_by_zero  ? rs1_val :                           // REM
                       div_overflow ? 32'd0 :
                       32'($signed(rs1_val) % $signed(rs2_val));
      3'b111:  m_res = div_by_zero  ? rs1_val : rs1_val % rs2_val;        // REMU
    endcase
  end

  // ----------------------------------------------------------- branches
  logic br_taken;
  always_comb begin
    unique case (funct3)
      3'b000:  br_taken = (rs1_val == rs2_val);                    // BEQ
      3'b001:  br_taken = (rs1_val != rs2_val);                    // BNE
      3'b100:  br_taken = ($signed(rs1_val) <  $signed(rs2_val));  // BLT
      3'b101:  br_taken = ($signed(rs1_val) >= $signed(rs2_val));  // BGE
      3'b110:  br_taken = (rs1_val <  rs2_val);                    // BLTU
      3'b111:  br_taken = (rs1_val >= rs2_val);                    // BGEU
      default: br_taken = 1'b0;
    endcase
  end

  // ------------------------------------------------------ loads and stores
  logic [31:0] ls_addr;
  logic [1:0]  byte_off;
  logic        is_load, is_store, misaligned;
  assign is_load  = (opcode == OPC_LOAD);
  assign is_store = (opcode == OPC_STORE);
  assign ls_addr  = rs1_val + (is_store ? imm_s : imm_i);
  assign byte_off = ls_addr[1:0];
  // funct3[1:0]: 00 byte, 01 halfword, 10 word
  assign misaligned = (funct3[1:0] == 2'b01 && byte_off[0])
                   || (funct3[1:0] == 2'b10 && byte_off != 2'b00);

  logic [3:0]  access_mask;  // bytes touched, in their lanes
  always_comb begin
    unique case (funct3[1:0])
      2'b00:   access_mask = 4'b0001 << byte_off;
      2'b01:   access_mask = 4'b0011 << byte_off;
      default: access_mask = 4'b1111;
    endcase
  end

  logic [31:0] store_data;
  always_comb begin
    unique case (funct3[1:0])
      2'b00:   store_data = {4{rs2_val[7:0]}};
      2'b01:   store_data = {2{rs2_val[15:0]}};
      default: store_data = rs2_val;
    endcase
  end

  logic [15:0] load_shifted;  // the addressed byte or halfword, moved to bit 0
  logic [31:0] load_val;
  assign load_shifted = 16'(dmem_rdata >> {byte_off, 3'b000});
  always_comb begin
    unique case (funct3)
      3'b000:  load_val = {{24{load_shifted[7]}},  load_shifted[7:0]};   // LB
      3'b001:  load_val = {{16{load_shifted[15]}}, load_shifted[15:0]};  // LH
      3'b100:  load_val = {24'd0, load_shifted[7:0]};                    // LBU
      3'b101:  load_val = {16'd0, load_shifted[15:0]};                   // LHU
      default: load_val = dmem_rdata;                                    // LW
    endcase
  end

  // ------------------------------------------------- execute and write-back
  logic [31:0] next_pc, rd_wdata;
  logic        rd_we, mem_re, mem_we, trap;

  always_comb begin
    next_pc  = pc + 32'd4;
    rd_we    = 1'b0;
    rd_wdata = 32'd0;
    mem_re   = 1'b0;
    mem_we   = 1'b0;

    unique case (opcode)
      OPC_LUI:    begin rd_we = 1'b1; rd_wdata = imm_u; end
      OPC_AUIPC:  begin rd_we = 1'b1; rd_wdata = pc + imm_u; end
      OPC_JAL:    begin rd_we = 1'b1; rd_wdata = pc + 32'd4; next_pc = pc + imm_j; end
      OPC_JALR:   begin rd_we = 1'b1; rd_wdata = pc + 32'd4;
                        next_pc = (rs1_val + imm_i) & ~32'd1; end
      OPC_BRANCH: if (br_taken) next_pc = pc + imm_b;
      OPC_LOAD:   begin rd_we = 1'b1; rd_wdata = load_val; mem_re = 1'b1; end
      OPC_STORE:  mem_we = 1'b1;
      OPC_OP_IMM: begin rd_we = 1'b1; rd_wdata = alu_res; end
      OPC_OP:     begin rd_we = 1'b1; rd_wdata = (funct7 == F7_MULDIV) ? m_res : alu_res; end
      default:    ;  // FENCE and FENCE.I are no-ops: memory is not cached
    endcase

    trap = !legal
        || ((is_load || is_store) && misaligned)
        || (next_pc[1:0] != 2'b00);
    if (trap) begin
      next_pc = pc;
      rd_we   = 1'b0;
      mem_re  = 1'b0;
      mem_we  = 1'b0;
    end
  end

  assign dmem_addr  = {ls_addr[31:2], 2'b00};
  assign dmem_re    = active && mem_re;
  assign dmem_we    = active && mem_we;
  assign dmem_wmask = access_mask;
  assign dmem_wdata = store_data;

  // ------------------------------------------------------------ sequential
  always_ff @(posedge clk) begin
    if (rst) begin
      pc     <= RESET_PC;
      halted <= 1'b0;
      order  <= 64'd0;
    end else if (!halted) begin
      order <= order + 64'd1;
      if (trap) halted <= 1'b1;
      else      pc     <= next_pc;
      if (rd_we && rd != 5'd0) regs[rd] <= rd_wdata;
    end
  end

  // ------------------------------------------------------------------ RVFI
  assign rvfi_valid     = active;
  assign rvfi_order     = order;
  assign rvfi_insn      = insn;
  assign rvfi_trap      = trap;
  assign rvfi_halt      = trap;
  assign rvfi_intr      = 1'b0;
  assign rvfi_mode      = 2'd3;  // machine mode
  assign rvfi_ixl       = 2'd1;  // XLEN = 32
  assign rvfi_rs1_addr  = uses_rs1 ? rs1 : 5'd0;
  assign rvfi_rs2_addr  = uses_rs2 ? rs2 : 5'd0;
  assign rvfi_rs1_rdata = uses_rs1 ? rs1_val : 32'd0;
  assign rvfi_rs2_rdata = uses_rs2 ? rs2_val : 32'd0;
  assign rvfi_rd_addr   = rd_we ? rd : 5'd0;
  assign rvfi_rd_wdata  = (rd_we && rd != 5'd0) ? rd_wdata : 32'd0;
  assign rvfi_pc_rdata  = pc;
  assign rvfi_pc_wdata  = next_pc;
  assign rvfi_mem_addr  = (mem_re || mem_we) ? dmem_addr : 32'd0;
  assign rvfi_mem_rmask = mem_re ? access_mask : 4'd0;
  assign rvfi_mem_wmask = mem_we ? access_mask : 4'd0;
  assign rvfi_mem_rdata = mem_re ? dmem_rdata : 32'd0;
  assign rvfi_mem_wdata = mem_we ? store_data : 32'd0;

endmodule
