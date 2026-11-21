---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: []
---

# Surface: Coresample inspector (web/index.html)

Mode: Operate. Audience: engineers and researchers in computer architecture and ML systems arriving from GitHub. Job: find where a network's cycles go on a real core, and why, down to the instruction and the stall cause. First view: where the cycles go, with the core switch (single-cycle / pipeline) and a cores comparison. Desktop first, legible on a phone. Every figure is an exact count from the trace index.

## Direction contract

THESIS: The run's cycle accounting set like the RISC-V reference card: dense ruled tables and banded hierarchy, not a profiler's dark flame graph and not a dashboard of hero numbers.

OWN-WORLD: Pale green card stock ground, dark green shell rail and reversed section bands, black ink, rules in a muted green; red marks branch flushes, amber marks load-use stalls. Archivo and Archivo Narrow for words, JetBrains Mono only for addresses, encodings and counts. Two type sizes; rank by weight, case, reversal, rule.

STORY: The visitor sees the whole run's cycles and CPI, picks an image, a layer, an operation, and lands on instructions with their encodings and the cycles each one cost or waited. Switching core or comparing cores shows what the pipeline added.

FIRST VIEWPORT: Left green rail with wordmark, core switch, compare switch, views. Centre: reversed band with total cycles; four fact cells; the ruled icicle run → image → layer → operation, selection reversed; timeline of 4096-cycle cells below. Right: selection card with measures table and the instruction table with bit-field encodings.

FORM: position 5 on my ordered list (the RISC-V reference card); seed:2e2b4488642b4fa9. Raises: stall restyles a row inside a fixed grid; visible cycle grid; one plot owns the viewport; selected route highlighted across panels; two type sizes; focus reveals detail in place.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
