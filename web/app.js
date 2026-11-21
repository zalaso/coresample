// Coresample inspector: where the cycles go, level by level.
import { RunIndex, lost, fields } from "./data.js";

const CORES = { single: "Single-cycle", pipe: "Pipeline" };
const CAUSES = [
  ["branch", "Taken branches", "--branch"],
  ["load-use", "Load-use stalls", "--loaduse"],
  ["jump", "Jumps", "--jump"],
  ["fill", "Pipeline fill", "--fill"],
  ["fence.i", "fence.i", "--jump"],
  ["trap", "Trap", "--branch"],
  ["fetch stopped", "Fetch stopped", "--fill"],
];
const OTHER = "other";
const ABBR = { branch: "br", "load-use": "lu", jump: "jp", fill: "fill", "fence.i": "fi", trap: "trap", "fetch stopped": "stop" };

const fmt = (n) => n.toLocaleString("en-US");
const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : "–");
const cpi = (c, i) => (i ? (c / i).toFixed(3) : "–");
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const state = { core: "pipe", mode: "run", image: null, layer: null, op: null, open: new Set() };
const runs = {};
const scopes = new Map();

// ------------------------------------------------------------------ data

async function scopeOf(core, image) {
  const key = `${core}:${image}`;
  if (!scopes.has(key)) {
    const run = runs[core];
    if (image === null) scopes.set(key, Promise.resolve(run.wholeRun()));
    else {
      const s = run.samples[image];
      const bin = Math.max(64, Math.ceil((s.end - s.start) / 160 / 64) * 64);
      scopes.set(key, run.scan(s.start, s.end, bin));
    }
  }
  return scopes.get(key);
}

/** Regions of a scope grouped into layers and operations, in program order. */
function groups(run, s) {
  const layers = new Map();
  for (const [rid, e] of s.regions) {
    const r = run.regions.get(rid);
    const isLayer = r && r.layer !== null;
    const key = isLayer ? r.function : OTHER;
    if (!layers.has(key)) {
      layers.set(key, { key, name: isLayer ? r.function : "main and start-up", order: isLayer ? r.layer : 1e9,
        cycles: 0, instructions: 0, bubbles: {}, rids: [], ops: new Map() });
    }
    const g = layers.get(key);
    const opName = r ? (isLayer ? r.op : run.label(rid)) : run.label(rid);
    const opKey = r ? `${r.function}.${r.op}` : run.label(rid);
    add(g, e, rid);
    if (!g.ops.has(opKey)) g.ops.set(opKey, { key: opKey, name: opName, cycles: 0, instructions: 0, bubbles: {}, rids: [], first: rid });
    add(g.ops.get(opKey), e, rid);
  }
  const out = [...layers.values()].sort((a, b) => a.order - b.order);
  for (const g of out) g.ops = [...g.ops.values()].sort((a, b) => b.cycles - a.cycles);
  return out;
}

function add(g, e, rid) {
  g.cycles += e.cycles;
  g.instructions += e.instructions;
  for (const [k, v] of Object.entries(e.bubbles)) g.bubbles[k] = (g.bubbles[k] || 0) + v;
  g.rids.push(rid);
}

// --------------------------------------------------------------- rendering

async function render() {
  const run = runs[state.core];
  const compare = state.mode === "compare";
  $("icicle").hidden = compare;
  $("timeline").hidden = compare;
  $("compare").hidden = !compare;
  const s = await scopeOf(state.core, state.image);
  const gs = groups(run, s);
  await renderBand(s);
  renderFacts(s);
  if (compare) await renderCompare(gs);
  else {
    renderIcicle(run, s, gs);
    lastScope = s;
    lastGroups = gs;
    drawTimeline(run, s, gs);
  }
  await renderCard(run, s, gs);
  fitCounts();
  writeHash();
}

function scopeName() {
  return state.image === null ? "all 48 images" : `image ${state.image}`;
}

async function renderBand(s) {
  if (state.mode === "compare") {
    const [a, b] = await Promise.all([scopeOf("single", state.image), scopeOf("pipe", state.image)]);
    $("band").innerHTML = `<span class="scope">Single-cycle → pipeline · ${scopeName()}</span>
      <span class="line"><b>${fmt(a.cycles)} → ${fmt(b.cycles)}</b><span class="unit">cycles, +${fmt(b.cycles - a.cycles)} on the pipeline</span></span>`;
  } else {
    $("band").innerHTML = `<span class="scope">${CORES[state.core]} core · ${scopeName()}</span>
      <span class="line"><b>${fmt(s.cycles)}</b><span class="unit">cycles, every one accounted for</span></span>`;
  }
}

function renderFacts(s) {
  const l = lost(s.bubbles);
  const who = state.mode === "compare" ? `${CORES[state.core]}: ` : "";
  const facts = [
    [fmt(s.instructions), "Instructions retired"],
    [cpi(s.cycles, s.instructions), "Cycles per instruction"],
    [fmt(s.bubbles.branch || 0), "Lost to taken branches"],
    [fmt(s.bubbles["load-use"] || 0), "Lost to load-use stalls"],
  ];
  if (!l) facts[2] = ["0", "Lost cycles: one per instruction"];
  if (!l) facts[3] = [fmt(s.cycles), "Cycles in scope"];
  $("facts").innerHTML = facts.map(([v, k]) => `<div><dt>${who}${k}</dt><dd>${v}</dd></div>`).join("");
}

function stallStripes(g) {
  if (!g.cycles) return "";
  let left = 0;
  return CAUSES.filter(([c]) => c !== "fill" && g.bubbles[c]).map(([c, , color]) => {
    const w = (100 * g.bubbles[c]) / g.cycles;
    const html = `<span class="stall" style="left:${left}%;width:${w}%;background:var(${color})"></span>`;
    left += w;
    return html;
  }).join("");
}

function cell({ on, flex, label, title, sub, data, stripes = "", thin = false }) {
  return `<button class="cell${on ? " on" : ""}${thin ? " thin" : ""}" style="flex:${flex} 1 0" data-sel='${esc(JSON.stringify(data))}'
    aria-pressed="${on}" aria-label="${esc(title)}" title="${esc(title)}">${thin ? "" : `${esc(label)}<em>${esc(sub)}</em>`}${stripes}</button>`;
}

function renderIcicle(run, s, gs) {
  const total = s.cycles;
  const rows = [];
  rows.push(level("Run", cell({ on: state.image === null && !state.layer && !state.op, flex: 1, label: "All 48 images",
    title: `Whole run: ${fmt(run.cycles)} cycles`, sub: `${fmt(run.cycles)} cycles`, data: { image: null } })));
  const imgs = run.samples.map((smp, i) => cell({ on: state.image === i, flex: smp.end - smp.start, thin: true,
    title: `Image ${i}: ${fmt(smp.end - smp.start)} cycles`, data: { image: i } })).join("");
  const n = run.samples.length;
  const options = [`<option value="">All ${n} images</option>`, ...run.samples.map((smp, i) =>
    `<option value="${i}"${state.image === i ? " selected" : ""}>Image ${i} · ${fmt(smp.end - smp.start)} cycles</option>`)].join("");
  const prev = state.image === null ? n - 1 : (state.image + n - 1) % n;
  const next = state.image === null ? 0 : (state.image + 1) % n;
  const stepper = `<div class="stepper"><button data-sel='{"image":${prev}}' aria-label="Previous image">Prev</button>
    <select id="img-select" aria-label="Image">${options}</select><button data-sel='{"image":${next}}' aria-label="Next image">Next</button></div>`;
  rows.push(level("Image", imgs, "imgs", stepper));
  // Cells under 4% of the row are too narrow for text: their figures live in the tooltip and the card.
  const narrow = (x) => x / total < 0.04;
  rows.push(level("Layer", gs.map((g) => cell({ on: state.layer === g.key, flex: g.cycles, label: g.name, thin: narrow(g.cycles),
    title: `${g.name}: ${fmt(g.cycles)} cycles, ${pct(g.cycles, total)} of ${scopeName()}`,
    sub: `${fmt(g.cycles)} · ${pct(g.cycles, total)}`, data: { layer: g.key }, stripes: stallStripes(g) })).join("")));
  rows.push(level("Operation", gs.flatMap((g) => g.ops.map((o) => cell({ on: state.op === o.key, flex: o.cycles, label: o.name, thin: narrow(o.cycles),
    title: `${o.key}: ${fmt(o.cycles)} cycles, ${pct(o.cycles, total)} of ${scopeName()}`,
    sub: `${fmt(o.cycles)} · ${pct(o.cycles, total)}`, data: { layer: g.key, op: o.key }, stripes: stallStripes(o) }))).join("")));
  $("icicle").innerHTML = rows.join("");
  $("img-select").addEventListener("change", (ev) => select({ image: ev.target.value === "" ? null : Number(ev.target.value) }));
}

function level(name, cells, extra = "", after = "") {
  return `<div class="lvl" role="group" aria-label="${name}"><span>${name}</span><div><div class="cells ${extra}">${cells}</div>${after}</div></div>`;
}

/** A count is never shown clipped: a sub-line that does not fit is hidden;
 *  the full figure stays in the cell's tooltip and in the card. */
function fitCounts() {
  document.querySelectorAll(".cell em").forEach((em) => {
    em.hidden = false;
    if (em.scrollWidth > em.clientWidth + 1) em.hidden = true;
  });
}

// --------------------------------------------------------------- timeline

const tl = { bins: [], cursor: null, run: null, rids: null };

function selectedRids(gs) {
  const g = gs.find((x) => x.key === state.layer);
  const o = g && g.ops.find((x) => x.key === state.op);
  return o ? new Set(o.rids) : g ? new Set(g.rids) : null;
}

function drawTimeline(run, s, gs) {
  const canvas = $("tl");
  const bins = state.image === null ? run.runBins() : s.bins.filter(Boolean);
  Object.assign(tl, { bins, run, rids: selectedRids(gs) });
  if (tl.cursor !== null && tl.cursor >= bins.length) tl.cursor = null;
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const full = Math.max(...bins.map((b) => b.cycles));
  const bw = w / bins.length;
  const colors = { retired: css("--retired") };
  for (const [c, , v] of CAUSES) colors[c] = css(v);
  const sel = tl.rids;
  bins.forEach((b, i) => {
    let y = h;
    g.globalAlpha = sel ? 0.28 : 1;
    const seg = (n, color) => {
      const hh = (h * n) / full;
      g.fillStyle = color;
      g.fillRect(i * bw, y - hh, Math.max(bw, 0.6), hh);
      y -= hh;
    };
    seg(b.cycles - lost(b.bubbles), colors.retired);
    for (const [c] of CAUSES) if (b.bubbles[c]) seg(b.bubbles[c], colors[c]);
    if (sel && b.regions) {
      // The selected layer or operation, at full strength, from the cell's floor.
      let mine = 0;
      for (const [rid, n] of Object.entries(b.regions)) if (sel.has(Number(rid))) mine += n;
      g.globalAlpha = 1;
      g.fillStyle = css("--ink");
      g.fillRect(i * bw, h - (h * mine) / full, Math.max(bw, 0.6), (h * mine) / full);
    }
  });
  g.globalAlpha = 1;
  // The cycle grid: a rule between cells where they are wide enough, a
  // heavier one every 8 cells.
  const rule = css("--rule-soft"), heavy = css("--green");
  for (let i = 1; i < bins.length; i++) {
    const every8 = i % 8 === 0;
    if (!every8 && bw < 4) continue;
    g.fillStyle = every8 ? heavy : rule;
    g.fillRect(Math.round(i * bw), 0, 1, h);
  }
  if (tl.cursor !== null) {
    g.strokeStyle = css("--focus");
    g.lineWidth = 2;
    g.strokeRect(tl.cursor * bw + 1, 1, Math.max(bw - 2, 2), h - 2);
  }
  const start = bins[0].start, end = bins.at(-1).start + bins.at(-1).cycles;
  $("tl-axis").innerHTML = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span>${fmt(Math.round(start + f * (end - start)))}</span>`).join("");
  const keys = [["retired", "Retired", "--retired"], ...CAUSES.filter(([c]) => bins.some((b) => b.bubbles[c]))];
  if (sel) keys.push(["sel", state.op || state.layer, "--ink"]);
  $("tl-legend").innerHTML = keys.map(([, label, v]) => `<span><span class="key" style="background:var(${v})"></span>${esc(label)}</span>`).join("");
  $("tl").setAttribute("aria-label", `Timeline of ${scopeName()}: ${bins.length} cells of ${fmt(bins[0].cycles)} cycles, cycles ${fmt(start)} to ${fmt(end)}`);
  readCell();
}

/** The operation (region label) with the most cycles in a cell. */
function dominant(b) {
  if (!b.regions) return null;
  let best = null, n = 0;
  for (const [rid, c] of Object.entries(b.regions)) {
    if (Number(rid) < 0 || c <= n) continue;
    best = Number(rid);
    n = c;
  }
  if (best === null) return null;
  const r = tl.run.regions.get(best);
  return { key: `${r.function}.${r.op}`, layer: r.layer !== null ? r.function : OTHER, cycles: n };
}

function readCell() {
  const out = $("tl-read");
  if (tl.cursor === null) {
    out.textContent = "Move over the timeline, or focus it and use the arrow keys, to read a cell. Click it, or press Enter, to select the operation that fills it.";
    return;
  }
  const b = tl.bins[tl.cursor];
  const parts = [`<b>${fmt(b.cycles - lost(b.bubbles))}</b> retired`];
  for (const [c, label] of CAUSES) if (b.bubbles[c]) parts.push(`<b>${fmt(b.bubbles[c])}</b> ${label.toLowerCase()}`);
  const d = dominant(b);
  const most = d ? ` Mostly <b>${esc(d.key)}</b> (${pct(d.cycles, b.cycles)}).` : "";
  out.innerHTML = `Cycles ${fmt(b.start)}–${fmt(b.start + b.cycles - 1)}: ${parts.join(" · ")}.${most}`;
}

function cellAt(ev) {
  const r = $("tl").getBoundingClientRect();
  return Math.min(tl.bins.length - 1, Math.max(0, Math.floor(((ev.clientX - r.left) / r.width) * tl.bins.length)));
}

let lastScope = null, lastGroups = null;

/** Moves the timeline cursor; only the timeline is redrawn. */
function moveCursor(i) {
  tl.cursor = i;
  if (lastGroups) drawTimeline(tl.run, lastScope, lastGroups);
}

function pickCell() {
  if (tl.cursor === null) return;
  const d = dominant(tl.bins[tl.cursor]);
  if (d) select({ layer: d.layer, op: d.key });
}

// ------------------------------------------------------------------ card

function selected(gs) {
  const g = gs.find((x) => x.key === state.layer);
  const o = g && g.ops.find((x) => x.key === state.op);
  return { g, o };
}

async function renderCard(run, s, gs) {
  const { g, o } = selected(gs);
  const node = o || g || s;
  const title = o ? o.key : g ? g.name : state.image === null ? "Whole run" : `Image ${state.image}`;
  const route = ["run", state.image === null ? null : `image ${state.image}`, g ? g.name : null, o ? o.name : null]
    .filter(Boolean).map((x, i, a) => (i === a.length - 1 ? `<b>${esc(x)}</b>` : esc(x))).join(" → ");
  let html = `<div><h2 class="title">${esc(title)}</h2><p class="route">${route}</p></div>`;
  if (state.mode === "compare") html += await compareMeasures();
  else html += measures(node, s.cycles);
  html += await codeTable(run, s, o, g);
  $("card").innerHTML = html;
}

function measures(n, total) {
  const rows = [
    ["Cycles", fmt(n.cycles)],
    [`Share of ${scopeName()}`, pct(n.cycles, total)],
    ["Instructions retired", fmt(n.instructions)],
    ["Cycles per instruction", cpi(n.cycles, n.instructions)],
  ];
  for (const [c, label, color] of CAUSES) {
    if (n.bubbles[c]) rows.push([`<span class="key" style="background:var(${color})"></span>${label}`, fmt(n.bubbles[c])]);
  }
  if (!lost(n.bubbles)) rows.push(["Lost cycles", "0"]);
  return `<div><h3 class="sect">Measures</h3><table><tbody>${rows
    .map(([k, v]) => `<tr><td>${k}</td><td class="n">${v}</td></tr>`).join("")}</tbody></table></div>`;
}

async function compareMeasures() {
  const [a, b] = await Promise.all([scopeOf("single", state.image), scopeOf("pipe", state.image)]);
  const pick = (run, s) => {
    const gg = groups(run, s);
    const sel = selectedIn(gg);
    return sel || s;
  };
  const na = pick(runs.single, a), nb = pick(runs.pipe, b);
  const rows = [
    ["Cycles", fmt(na.cycles), fmt(nb.cycles)],
    ["Instructions retired", fmt(na.instructions), fmt(nb.instructions)],
    ["Cycles per instruction", cpi(na.cycles, na.instructions), cpi(nb.cycles, nb.instructions)],
  ];
  for (const [c, label, color] of CAUSES) {
    if (nb.bubbles[c] || na.bubbles[c]) rows.push([`<span class="key" style="background:var(${color})"></span>${label}`, fmt(na.bubbles[c] || 0), fmt(nb.bubbles[c] || 0)]);
  }
  return `<div><h3 class="sect">Measures, both cores</h3><table><thead><tr><th><span class="sr">Measure</span></th><th class="n">Single</th><th class="n">Pipeline</th></tr></thead><tbody>${rows
    .map(([k, x, y]) => `<tr><td>${k}</td><td class="n">${x}</td><td class="n">${y}</td></tr>`).join("")}</tbody></table></div>`;
}

function selectedIn(gs) {
  const g = gs.find((x) => x.key === state.layer);
  const o = g && g.ops.find((x) => x.key === state.op);
  return o || g || null;
}

async function codeTable(run, s, o, g) {
  const info = new Map(run.code.map((c) => [c.pc, c]));
  const rids = new Set((o || g || { rids: null }).rids || []);
  let pcs = [...s.pcs.entries()].filter(([pc]) => !rids.size || rids.has(info.get(pc)?.region));
  let heading = "Instructions, in program order";
  if (rids.size) pcs.sort((x, y) => x[0] - y[0]);
  else {
    pcs.sort((x, y) => y[1].cycles - x[1].cycles);
    pcs = pcs.slice(0, 16);
    heading = "The 16 instructions that cost the most";
  }
  let other = null;
  if (state.mode === "compare") {
    const sc = await scopeOf(state.core === "pipe" ? "single" : "pipe", state.image);
    other = sc.pcs;
  }
  const cols = other ? 4 : 3;
  const rows = pcs.map(([pc, e]) => {
    const c = info.get(pc) || { text: "", insn: 0 };
    const marks = CAUSES.filter(([k]) => e.bubbles[k]).map(([k, label, color]) =>
      `<span class="mk" style="color:var(${color}-ink)" title="${fmt(e.bubbles[k])} cycles lost to ${label.toLowerCase()}"><span class="key" style="background:var(${color})"></span>${ABBR[k]} ${fmt(e.bubbles[k])}</span>`).join("");
    const open = state.open.has(pc);
    let enc = "";
    if (open) {
      const f = fields(c.insn);
      enc = `<tr class="encrow"><td colspan="${cols}"><div class="enc" style="grid-template-columns:${f.fields.map((x) => `${x.width}fr`).join(" ")}">${f.fields
        .map((x) => `<div>${x.bits}<small>${esc(x.name)}</small></div>`).join("")}</div>
        <span class="note">${f.format}-type · 0x${(c.insn >>> 0).toString(16).padStart(8, "0")} · retired ${fmt(e.instructions)} times</span></td></tr>`;
    }
    const delta = other ? `<td class="n">${fmt(e.cycles - (other.get(pc)?.cycles || 0))}</td>` : "";
    return `<tr><td>${pc.toString(16)}</td><td><button data-pc="${pc}" aria-expanded="${open}"><span class="insn">${esc(c.text)}</span><span class="marks">${marks}</span></button></td>
      <td class="n">${fmt(e.cycles)}</td>${delta}</tr>${enc}`;
  }).join("");
  const dh = other ? `<th class="n">${state.core === "pipe" ? "Added" : "Saved"}</th>` : "";
  return `<div><h3 class="sect">${heading}</h3><table class="code"><thead><tr><th>pc</th><th>Instruction</th><th class="n">Cycles</th>${dh}</tr></thead><tbody>${rows ||
    `<tr><td colspan="${cols}" class="note">No instruction in this selection.</td></tr>`}</tbody></table>
    <p class="note">Open an instruction to see its encoding, split into its format's fields. Marks: br taken branch, lu load-use stall, jp jump.</p></div>`;
}

// --------------------------------------------------------------- compare

async function renderCompare() {
  const [a, b] = await Promise.all([scopeOf("single", state.image), scopeOf("pipe", state.image)]);
  const ga = groups(runs.single, a), gb = groups(runs.pipe, b);
  const ops = new Map();
  for (const [side, gs] of [["a", ga], ["b", gb]]) {
    for (const g of gs) for (const o of g.ops) {
      if (!ops.has(o.key)) ops.set(o.key, { key: o.key, layer: g.key, a: 0, b: 0, bubbles: {} });
      const row = ops.get(o.key);
      row[side] += o.cycles;
      if (side === "b") row.bubbles = o.bubbles;
    }
  }
  const rows = [...ops.values()].sort((x, y) => y.b - y.a - (x.b - x.a));
  const cell = (r, c) => fmt(r.bubbles[c] || 0);
  const body = rows.map((r) => `<tr class="pick${state.op === r.key ? " sel" : ""}" data-sel='${esc(JSON.stringify({ layer: r.layer, op: r.key }))}' tabindex="0">
    <td>${esc(r.key)}</td><td class="n">${fmt(r.a)}</td><td class="n">${fmt(r.b)}</td><td class="n">${fmt(r.b - r.a)}</td>
    <td class="n">${cell(r, "branch")}</td><td class="n">${cell(r, "load-use")}</td><td class="n">${cell(r, "jump")}</td><td class="n">${cell(r, "fill")}</td></tr>`).join("");
  const tot = `<tr class="total"><td>Total, ${scopeName()}</td><td class="n">${fmt(a.cycles)}</td><td class="n">${fmt(b.cycles)}</td><td class="n">${fmt(b.cycles - a.cycles)}</td>
    <td class="n">${fmt(b.bubbles.branch || 0)}</td><td class="n">${fmt(b.bubbles["load-use"] || 0)}</td><td class="n">${fmt(b.bubbles.jump || 0)}</td><td class="n">${fmt(b.bubbles.fill || 0)}</td></tr>`;
  $("cmp").innerHTML = `<thead><tr><th>Operation</th><th class="n">Single-cycle</th><th class="n">Pipeline</th><th class="n">Added</th>
    <th class="n"><span class="key" style="background:var(--branch)"></span>Branch</th><th class="n"><span class="key" style="background:var(--loaduse)"></span>Load-use</th>
    <th class="n"><span class="key" style="background:var(--jump)"></span>Jump</th><th class="n"><span class="key" style="background:var(--fill)"></span>Fill</th></tr></thead><tbody>${body}${tot}</tbody>`;
}

// ------------------------------------------------------------- behaviour

function select(data) {
  if ("image" in data) {
    state.image = data.image;
    if (!("layer" in data)) { state.layer = null; state.op = null; }
  }
  if ("layer" in data) state.layer = data.layer;
  if ("op" in data) state.op = data.op;
  else if ("layer" in data) state.op = null;
  render().catch(fail);
}

let lastHash = "";
function writeHash() {
  const p = new URLSearchParams({ core: state.core, view: state.mode });
  if (state.image !== null) p.set("image", state.image);
  if (state.layer) p.set("layer", state.layer);
  if (state.op) p.set("op", state.op);
  lastHash = `#${p}`;
  history.replaceState(null, "", lastHash);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  state.core = p.get("core") in CORES ? p.get("core") : "pipe";
  state.mode = p.get("view") === "compare" ? "compare" : "run";
  const img = Number(p.get("image"));
  const images = runs[state.core].samples.length;
  state.image = p.has("image") && Number.isInteger(img) && img >= 0 && img < images ? img : null;
  state.layer = p.get("layer");
  state.op = p.get("op");
  document.querySelector(`input[name=core][value=${state.core}]`).checked = true;
  document.querySelector(`input[name=mode][value=${state.mode}]`).checked = true;
}

function fail(err) {
  const e = $("error");
  e.hidden = false;
  e.textContent = `Could not read the trace index (${err.message}). Build it with \`make web\`, then serve web/ over HTTP.`;
  console.error(err);
}

document.addEventListener("click", (ev) => {
  const sel = ev.target.closest("[data-sel]");
  if (sel) return select(JSON.parse(sel.dataset.sel));
  const code = ev.target.closest("button[data-pc]");
  if (code) {
    const pc = Number(code.dataset.pc);
    state.open.has(pc) ? state.open.delete(pc) : state.open.add(pc);
    render().catch(fail);
  }
});
document.addEventListener("keydown", (ev) => {
  const row = ev.target.closest("tr[data-sel]");
  if (row && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); select(JSON.parse(row.dataset.sel)); }
});
document.querySelectorAll("input[name=core]").forEach((i) => i.addEventListener("change", () => { state.core = i.value; render().catch(fail); }));
document.querySelectorAll("input[name=mode]").forEach((i) => i.addEventListener("change", () => { state.mode = i.value; render().catch(fail); }));
const canvas = $("tl");
canvas.addEventListener("pointermove", (ev) => moveCursor(cellAt(ev)));
canvas.addEventListener("pointerdown", (ev) => moveCursor(cellAt(ev)));
canvas.addEventListener("click", pickCell);
canvas.addEventListener("keydown", (ev) => {
  const n = tl.bins.length;
  const at = tl.cursor ?? -1;
  const step = ev.shiftKey ? 8 : 1;
  const to = { ArrowRight: Math.min(n - 1, at + step), ArrowLeft: Math.max(0, at - step), Home: 0, End: n - 1 }[ev.key];
  if (to !== undefined) { ev.preventDefault(); moveCursor(to); }
  else if (ev.key === "Enter") { ev.preventDefault(); pickCell(); }
});
window.addEventListener("hashchange", () => {
  if (location.hash === lastHash) return;
  readHash();
  render().catch(fail);
});
let resize;
window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => render().catch(fail), 120); });

(async () => {
  try {
    const res = await fetch("data/runs.json");
    if (!res.ok) throw new Error(`data/runs.json: ${res.status}`);
    const manifest = await res.json();
    await Promise.all(manifest.runs.map(async (r) => { runs[r.id] = await RunIndex.load(r.path); }));
    readHash();
    await render();
    // Web fonts change text widths: check again for counts that no longer fit.
    document.fonts.ready.then(fitCounts);
  } catch (err) {
    fail(err);
  }
})();
