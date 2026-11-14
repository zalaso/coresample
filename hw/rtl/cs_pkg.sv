// Shared constants for the Coresample cores.
package cs_pkg;

  // RV32 base opcodes (insn[6:0])
  localparam logic [6:0] OPC_LUI      = 7'b0110111;
  localparam logic [6:0] OPC_AUIPC    = 7'b0010111;
  localparam logic [6:0] OPC_JAL      = 7'b1101111;
  localparam logic [6:0] OPC_JALR     = 7'b1100111;
  localparam logic [6:0] OPC_BRANCH   = 7'b1100011;
  localparam logic [6:0] OPC_LOAD     = 7'b0000011;
  localparam logic [6:0] OPC_STORE    = 7'b0100011;
  localparam logic [6:0] OPC_OP_IMM   = 7'b0010011;
  localparam logic [6:0] OPC_OP       = 7'b0110011;
  localparam logic [6:0] OPC_MISC_MEM = 7'b0001111;
  localparam logic [6:0] OPC_SYSTEM   = 7'b1110011;

  // funct7 values used by OP / OP-IMM
  localparam logic [6:0] F7_BASE = 7'b0000000;
  localparam logic [6:0] F7_ALT  = 7'b0100000;  // SUB, SRA, SRAI
  localparam logic [6:0] F7_MULDIV = 7'b0000001;  // M extension

endpackage
