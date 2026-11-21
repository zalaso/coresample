// Reading the trace index written by scripts/trace_index.py.
//
// index.json holds the call tree, the pyramid of cycle bins and the code
// table; chunks/<k>.bin hold the raw cycles, 7 bytes each:
// pc (u32), region (i16), cause (u8: 0 retired, else 1 + index in causes).
// Every figure the inspector shows comes from these, exactly.

export const OUTSIDE = -1; // an instruction outside the compiled network
export const NOBODY = -2;  // a bubble blamed on no instruction (pipeline fill)
const SLOT = 7;

export class RunIndex {
  static async load(base) {
    const res = await fetch(`${base}/index.json`);
    if (!res.ok) throw new Error(`${base}/index.json: ${res.status} ${res.statusText}`);
    return new RunIndex(base, await res.json());
  }

  constructor(base, d) {
    this.base = base;
    this.cycles = d.cycles;
    this.instructions = d.instructions;
    this.chunkCycles = d.chunk_cycles;
    this.causes = d.causes;
    this.regions = new Map(d.regions.map((r) => [r.id, r]));
    this.levels = d.levels;
    this.tree = d.tree;
    this.code = d.code;
    this.unattributed = d.unattributed;
    this.chunks = new Map();
    this.samples = [];
    const walk = (n) => n.children.forEach((c) => (c.name.startsWith("sample ") ? this.samples.push(c) : walk(c)));
    walk(this.tree);
  }

  /** "function.op" of a region id, matching op_report and the index. */
  label(rid) {
    if (rid === OUTSIDE) return "(outside the network)";
    if (rid === NOBODY) return "(pipeline fill)";
    const r = this.regions.get(rid);
    return `${r.function}.${r.op}`;
  }

  async chunk(k) {
    if (!this.chunks.has(k)) {
      const p = fetch(`${this.base}/chunks/${String(k).padStart(6, "0")}.bin`).then(async (r) => {
        if (!r.ok) throw new Error(`chunk ${k}: ${r.status}`);
        return new DataView(await r.arrayBuffer());
      });
      this.chunks.set(k, p);
    }
    return this.chunks.get(k);
  }

  /** Whole-run summary from the code table: exact, no chunk needed. */
  wholeRun() {
    const s = emptySummary(0, this.cycles);
    for (const c of this.code) {
      addTo(s.pcs, c.pc, c.cycles, c.instructions, c.bubbles);
      addTo(s.regions, c.region, c.cycles, c.instructions, c.bubbles);
      s.instructions += c.instructions;
      for (const [k, v] of Object.entries(c.bubbles)) s.bubbles[k] = (s.bubbles[k] || 0) + v;
    }
    if (this.unattributed) {
      const fill = { fill: this.unattributed };
      addTo(s.regions, NOBODY, this.unattributed, 0, fill);
      s.bubbles.fill = (s.bubbles.fill || 0) + this.unattributed;
    }
    s.cycles = this.cycles;
    return s;
  }

  /** Exact summary of cycles [a, b) by scanning the raw slots, plus
   *  timeline bins of `binWidth` cycles. */
  async scan(a, b, binWidth) {
    const s = emptySummary(a, b);
    s.bins = [];
    const C = this.chunkCycles;
    for (let k = Math.floor(a / C); k * C < b; k++) {
      const view = await this.chunk(k);
      const from = Math.max(a, k * C), to = Math.min(b, (k + 1) * C);
      for (let t = from; t < to; t++) {
        const o = (t - k * C) * SLOT;
        const pc = view.getUint32(o, true), region = view.getInt16(o + 4, true), code = view.getUint8(o + 6);
        const cause = code === 0 ? null : this.causes[code - 1];
        const bub = cause ? { [cause]: 1 } : null;
        addTo(s.regions, region, 1, cause ? 0 : 1, bub);
        if (region !== NOBODY) addTo(s.pcs, pc, 1, cause ? 0 : 1, bub);
        if (cause) s.bubbles[cause] = (s.bubbles[cause] || 0) + 1;
        else s.instructions += 1;
        const bi = Math.floor((t - a) / binWidth);
        const bin = (s.bins[bi] ||= { start: a + bi * binWidth, cycles: 0, bubbles: {}, regions: {} });
        bin.cycles += 1;
        bin.regions[region] = (bin.regions[region] || 0) + 1;
        if (cause) bin.bubbles[cause] = (bin.bubbles[cause] || 0) + 1;
      }
    }
    s.cycles = b - a;
    return s;
  }

  /** Timeline bins of the whole run, straight from the pyramid's level 0. */
  runBins() {
    const width = this.levels[0].width;
    return this.levels[0].bins.map((b, i) => ({ start: i * width, cycles: b.cycles, bubbles: b.bubbles, regions: b.regions }));
  }
}

function emptySummary(a, b) {
  return { start: a, end: b, cycles: 0, instructions: 0, bubbles: {}, regions: new Map(), pcs: new Map() };
}

function addTo(map, key, cycles, instructions, bubbles) {
  let e = map.get(key);
  if (!e) map.set(key, (e = { cycles: 0, instructions: 0, bubbles: {} }));
  e.cycles += cycles;
  e.instructions += instructions;
  if (bubbles) for (const [k, v] of Object.entries(bubbles)) e.bubbles[k] = (e.bubbles[k] || 0) + v;
}

/** Sum of a bubbles object. */
export function lost(bubbles) {
  return Object.values(bubbles).reduce((x, y) => x + y, 0);
}

/** RV32 encoding split into its format's fields, as on the reference card. */
export function fields(insn) {
  const bits = (hi, lo) => (insn >>> lo) & ((1 << (hi - lo + 1)) - 1);
  const bin = (v, w) => v.toString(2).padStart(w, "0");
  const f = (name, hi, lo) => ({ name, bits: bin(bits(hi, lo), hi - lo + 1), width: hi - lo + 1 });
  const op = insn & 0x7f;
  const tail = [f("rs1", 19, 15), f("funct3", 14, 12)];
  switch (op) {
    case 0x33: return { format: "R", fields: [f("funct7", 31, 25), f("rs2", 24, 20), ...tail, f("rd", 11, 7), f("opcode", 6, 0)] };
    case 0x13: case 0x03: case 0x67: case 0x0f:
      return { format: "I", fields: [f("imm[11:0]", 31, 20), ...tail, f("rd", 11, 7), f("opcode", 6, 0)] };
    case 0x23: return { format: "S", fields: [f("imm[11:5]", 31, 25), f("rs2", 24, 20), ...tail, f("imm[4:0]", 11, 7), f("opcode", 6, 0)] };
    case 0x63: return { format: "B", fields: [f("imm[12|10:5]", 31, 25), f("rs2", 24, 20), ...tail, f("imm[4:1|11]", 11, 7), f("opcode", 6, 0)] };
    case 0x37: case 0x17: return { format: "U", fields: [f("imm[31:12]", 31, 12), f("rd", 11, 7), f("opcode", 6, 0)] };
    case 0x6f: return { format: "J", fields: [f("imm[20|10:1|11|19:12]", 31, 12), f("rd", 11, 7), f("opcode", 6, 0)] };
    default: return { format: "?", fields: [f("word", 31, 0)] };
  }
}
