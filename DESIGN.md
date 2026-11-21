---
name: Coresample inspector
description: Where a neural network's cycles go on a RISC-V core, set like the RISC-V green reference card.
colors:
  green: "#0f3b2a"
  on-green: "#f1f6ee"
  stock: "#dfe9d8"
  stock-2: "#d1dfc8"
  cell: "#ecf2e7"
  ink: "#141a16"
  ink-2: "#33473b"
  rule: "#7f9a86"
  rule-soft: "#b9cbb4"
  branch: "#b4351f"
  loaduse: "#b7861c"
  jump: "#6a4c93"
  fill: "#7f9a86"
  branch-ink: "#8f2512"
  loaduse-ink: "#6b4f0b"
  jump-ink: "#553a7a"
  retired: "#0f3b2a"
  sel: "#0f3b2a"
  on-sel: "#f1f6ee"
  focus: "#141a16"
  focus-rail: "#d7a93b"
  green-dark: "#1f5a40"
  on-green-dark: "#eef5ea"
  stock-dark: "#0f1813"
  stock-2-dark: "#142019"
  cell-dark: "#18261e"
  ink-dark: "#dde8d7"
  ink-2-dark: "#a9bdab"
  rule-dark: "#4d6a58"
  rule-soft-dark: "#2c4236"
  branch-dark: "#e0674e"
  loaduse-dark: "#e0b44a"
  jump-dark: "#a98bd6"
  fill-dark: "#6b8a76"
  branch-ink-dark: "#f08a74"
  loaduse-ink-dark: "#e9c46a"
  jump-ink-dark: "#c4aef0"
  retired-dark: "#5aa57d"
  sel-dark: "#cfe3c9"
  on-sel-dark: "#0f1813"
  focus-dark: "#e0b44a"
typography:
  display:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "0em"
  wordmark:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.04em"
  title:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.03em"
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "tnum"
  label:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    letterSpacing: "0.14em"
  label-dense:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    letterSpacing: "0.1em"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "13px"
    fontWeight: 400
    fontFeature: "tnum"
rounded:
  none: "0px"
spacing:
  2xs: "4px"
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  card: "20px"
  rail: "22px"
  gutter: "24px"
components:
  rail:
    backgroundColor: "{colors.green}"
    textColor: "{colors.on-green}"
    padding: "22px 18px"
    width: "236px"
  seg-option:
    backgroundColor: "transparent"
    textColor: "{colors.on-green}"
    typography: "{typography.title}"
    rounded: "{rounded.none}"
    padding: "7px 4px"
  seg-option-selected:
    backgroundColor: "{colors.on-green}"
    textColor: "{colors.green}"
  band:
    backgroundColor: "{colors.green}"
    textColor: "{colors.on-green}"
    typography: "{typography.display}"
    padding: "12px 24px"
  fact:
    backgroundColor: "{colors.stock}"
    textColor: "{colors.ink}"
    typography: "{typography.display}"
    padding: "12px 24px"
  section-band:
    backgroundColor: "{colors.green}"
    textColor: "{colors.on-green}"
    typography: "{typography.label}"
    padding: "5px 10px"
  icicle-cell:
    backgroundColor: "{colors.cell}"
    textColor: "{colors.ink}"
    typography: "{typography.title}"
    rounded: "{rounded.none}"
    padding: "5px 8px 9px"
    height: "46px"
  icicle-cell-hover:
    backgroundColor: "color-mix(in srgb, #0f3b2a 12%, #ecf2e7)"
  icicle-cell-selected:
    backgroundColor: "{colors.sel}"
    textColor: "{colors.on-sel}"
  table-head:
    textColor: "{colors.ink-2}"
    typography: "{typography.label-dense}"
    padding: "5px 6px"
  table-row-selected:
    backgroundColor: "{colors.sel}"
    textColor: "{colors.on-sel}"
  timeline:
    backgroundColor: "{colors.cell}"
    height: "300px"
  encoding-grid:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.mono}"
    padding: "3px 4px"
  stepper-control:
    backgroundColor: "{colors.cell}"
    textColor: "{colors.ink}"
    typography: "{typography.title}"
    rounded: "{rounded.none}"
    padding: "8px 12px"
    height: "40px"
  card:
    backgroundColor: "{colors.stock-2}"
    textColor: "{colors.ink}"
    padding: "20px"
    width: "420px"
---

# Design System: Coresample inspector

## Overview

**Creative North Star: "The Reference Card"**

The inspector is the RISC-V green reference card made into a working screen. A run's cycle accounting is set the way the card sets an instruction set: pale green card stock, a dark green shell, reversed green bands for headings, black ink, and ruled tables in a muted green. Hierarchy comes from what a printed card has to hand: weight, case, reversal and rule. There are two type sizes and no third.

The density is a reference sheet's. Every surface is a table, a ruled row of proportional cells, or a figure with its label, and every figure is an exact count in tabular numerals. Colour beyond the greens is reserved for one meaning: why a cycle was lost. Red marks a taken-branch flush, amber a load-use stall, violet a jump. Nothing else on the page is red, amber or violet.

The build is flat, square-cornered and still. It has no shadows, no radius and no motion. Selection is shown by reversal into the shell green, and keyboard focus by a separate ring that is never the selection colour. The direction contract rejects two looks outright: a profiler's dark flame graph and a dashboard of hero numbers.

**Key Characteristics:**
- Pale green stock ground, dark green rail and reversed section bands, black ink, muted green rules.
- Two type sizes (13px and 24px); rank set by weight, case, reversal and rule.
- Archivo and Archivo Narrow for words, JetBrains Mono for addresses, encodings and counts.
- Cause colours (red, amber, violet) only for lost cycles, each with a darker text partner.
- Selection reverses into green; focus is a separate ring drawn outside the element.
- Flat and square: no shadows, no radius, no transitions.
- A dark scheme that follows `prefers-color-scheme`, keeping every role.

## Colors

One structural green carries the shell, the bands, the heavy rules and the selection; everything else is card stock, ink and rule, plus three cause colours that mean "cycles lost here".

### Primary
- **Reference Card Green** (green): the rail background, the run band, every section heading band, the 2px structural rules (under the facts, above table heads, the main column's right edge, icicle cell borders, the timeline frame and its every-8th-cell grid line), and the text-selection highlight. In light mode it is also the selection fill (sel) and the retired-cycle colour in the timeline (retired). In dark mode it lifts to green-dark, and selection flips to a pale reversal (sel-dark with on-sel-dark).
- **Reversed Type** (on-green): text and outlines on any green ground; the fill of the checked segment, with green text, so the rail's own selection is a reversal of the reversal.

### Secondary
- **Flush Red** (branch): taken-branch flushes in stall stripes, timeline segments, legend keys and cause swatches. Also covers trap bubbles.
- **Stall Amber** (loaduse): load-use stalls, in the same places.
- **Jump Violet** (jump): jump flushes, and fence.i bubbles.
- **Cause inks** (branch-ink, loaduse-ink, jump-ink): the same three causes as text, darkened to body contrast on the side card (about 5.5:1 to 6.6:1). Cause marks in the instruction table use these for their abbreviation and count, next to a swatch in the full cause colour.

### Neutral
- **Card Stock** (stock): the page ground, behind the main column and the facts.
- **Side Card** (stock-2): the selection card on the right, one step darker than the stock, so the card reads as a second card laid beside the first.
- **Table Cell** (cell): the fill inside ruled containers: icicle rows, the timeline canvas, stepper controls.
- **Press Black** (ink): primary text, and the timeline overlay that shows the selected layer or operation at full strength.
- **Green-Grey Ink** (ink-2): secondary text: labels, table heads, the route line, addresses, the axis, notes and the readout. About 8:1 on stock.
- **Muted Rule** (rule): 1px rules between table rows, fact cells and encoding fields; the scrollbar thumb. The fill bubble (pipeline fill, fetch stopped) uses the same value (fill), deliberately quiet.
- **Faint Rule** (rule-soft): the per-cell grid lines inside the timeline.
- **Focus Ink** (focus): the keyboard focus ring on the stock. Ink in light mode, amber (focus-dark) in dark mode; inside the rail it is a fixed amber (focus-rail) in both schemes.

### Named Rules
**The One Green Rule.** Structure has one colour. Shell, bands, heavy rules, cell borders and selection all use the same green; never introduce a second structural hue.

**The Cause Colour Rule.** Red, amber and violet mean lost cycles and nothing else. They appear as stripes, segments, swatches and cause marks; as text they always switch to their -ink partner. Never use them for emphasis, status or decoration.

**The Focus Is Not Selection Rule.** The focus ring is never the selection colour: ink in light, amber in dark and in the rail. It is drawn outside the element (2px, offset 2px) so it stays visible against a reversed selected cell; a focused selected cell also gets an inset 2px ring in on-sel.

## Typography

**Display Font:** Archivo (with system-ui)
**Body Font:** Archivo (with system-ui)
**Label/Mono Font:** Archivo Narrow (with Arial Narrow) for labels and controls; JetBrains Mono (with ui-monospace) for figures

**Character:** A grotesque and its narrow cut, as on a printed reference card: wide Archivo for figures and prose, condensed Archivo Narrow for every label, control and cell name, so dense rows fit without shrinking the type. JetBrains Mono appears only where alignment matters: addresses, encodings and counts. The body sets `font-variant-numeric: tabular-nums` throughout.

### Hierarchy
- **Display** (800, 24px, 1.2): the band figure, the fact values, the selection card title. The only large size.
- **Wordmark** (800, 24px, 1, uppercase, 0.04em): "Coresample" in the rail; the same size as display, ranked by case and tracking.
- **Title** (Narrow 600, 13px, 1.3, 0.03em): icicle cell names, segment options, rail links, stepper controls, the cause-mark run (at 700).
- **Body** (400, 13px, 1.45): prose, the readout (max 90ch), notes (max 62ch), the rail note (max 32ch), table cells.
- **Label** (Narrow 700, 13px, 0.14em, uppercase): section band headings, segment legends, rail group labels, icicle level names.
- **Label, dense** (Narrow 700, 13px, 0.1em, uppercase): table heads and fact labels. The band's scope line and unit sit between the two at 0.12em.
- **Mono** (400, 13px): addresses, instruction text, encoding bits, the timeline axis, the route line, right-aligned numeric columns. Emphasis inside mono runs is 600.

### Named Rules
**The Two Sizes Rule.** Only 13px and 24px exist. Rank comes from weight (400, 600, 700, 800), case, reversal into green and rule weight, never from a third size.

**The Whole Count Rule.** A count is never truncated. When a cell's count line does not fit, the line is hidden whole; the full figure stays in the cell's tooltip and in the selection card.

**The Mono For Figures Rule.** JetBrains Mono is for addresses, encodings and counts only, right-aligned and unwrapped in numeric columns. Words are never set in mono.

## Layout

A three-column desktop grid: the rail (236px, sticky, full height), the main column (`minmax(0, 1fr)`, closed by a 2px green rule on its right), and the selection card (420px, sticky, full height, scrolling on its own). The main column stacks the run band, a four-cell facts row, the icicle, and the timeline; in compare mode the icicle and timeline give way to a full-width comparison table.

Spacing follows a small set of steps. The main column's gutter is 24px; sections take 16 to 20px at the top. The card pads 20px with an 18px gap between blocks; the rail pads 22px vertically and 18px horizontally with a 22px gap. Inside, steps of 4, 6, 8 and 12px separate rows, legend items and cells. The band and legends wrap with a 4px row gap and 14px column gap.

The icicle is a proportional ruled grid: four levels (Run, Image, Layer, Operation), each a 92px level name and a row of cells whose widths are flex-proportional to cycles. A cell under 4% of its row becomes a thin cell (no text, 2px minimum), its figures moving to the tooltip and the card. The timeline is a fixed-height canvas (300px) of equal cycle cells.

Two breakpoints:
- **1180px and below:** the rail narrows to 220px and the card drops under the main column in column 2, unstuck, with a 2px green rule on top.
- **760px and below:** one column. The rail stops being sticky, pads 16px, and drops its note and section navigation. The gutter falls to 16px, the facts become a 2x2 grid ruled in both directions, level names stack above their rows, the timeline shortens to 180px, and the image stepper appears (image cells are too thin to tap).

### Named Rules
**The Proportional Cell Rule.** A cell's width is its share of cycles, always. A cell too narrow for text keeps its width and loses its text, never the reverse.

## Elevation & Depth

The system is flat. There are no drop shadows, no overlays that float, no blur. Depth is the printed card's: a reversed green band sits above plain stock, a 2px green rule outranks a 1px muted rule, and the side card is one tonal step darker than the main stock. The only box-shadow in the build is functional, the inset 2px on-sel ring that marks focus on an already selected icicle cell.

### Named Rules
**The Printed Card Rule.** If a printed reference card could not do it, the screen does not: hierarchy is reversal, rule weight and tone, never shadow or lift.

## Shapes

Every corner is square (0px). Containers are drawn with rules, not filled panels: icicle rows and the timeline are boxed in 1px green, cells are divided by 1px green, table rows by 1px muted rule, table heads and totals by 2px green. Segmented options share borders (each option after the first drops its left border), so a switch reads as one ruled strip. Stall stripes are 5px bars laid flat along the bottom of a cell; legend and cause swatches are 10px squares.

## Components

### Segmented switches (rail)
Two-option switches set as one ruled strip, the card's own toggle.
- **Shape:** square, 1px border in on-green at 45%, shared inner borders, 7px 4px padding.
- **Default:** title type in on-green on the rail green, under a tracked uppercase legend at 80% opacity.
- **Selected:** reversed to an on-green fill with green text.
- **Focus:** the native radio is invisible over the label; the visible span takes a 2px focus-rail ring, offset 2px.

### Navigation (rail)
- **Style:** groups under a tracked uppercase rail label; each link is title type with a 1px on-green (25%) rule on top and 6px vertical padding, no underline.
- **Hover:** underline.
- **Mobile:** both navigation groups and the rail note are hidden at 760px and below.

### Run band
The page heading, set as a reversed green band. Two lines: the scope in tracked uppercase narrow caps (core and scope, at 90% opacity), then the figure in display weight with its unit in the same narrow caps. The scope line always states what the figure counts; it is never a slogan. In compare mode the figure reads "single → pipeline" with the added cycles in the unit.

### Facts
Four ruled cells under the band, divided by 1px muted rules and closed by a 2px green rule. Each shows the figure in display weight above its dense uppercase label in ink-2. At 760px they reflow to a 2x2 grid.

### Section bands
Every section heading (Timeline, Measures, the instruction table, the comparison) is a reversed green band in label type, 5px 10px padding. A band may carry a right-hand payload, as the timeline's legend does; legend swatches inside the band get a 1px on-green outline so a green swatch stays visible on green. The first table row under a band drops its top rule.

### Icicle cells
The signature component: ruled rows of proportional cells, one row per level.
- **Shape:** square cells in a 1px green box, 46px minimum height, divided by 1px green rules, 5px 8px 9px padding.
- **Content:** the name in title type, then the count line in mono ink-2 ("cycles · share"). A count line that does not fit is hidden, never clipped.
- **Stall stripes:** a 5px bar along the cell's bottom edge, one segment per lost-cycle cause in the cause colours, each as wide as that cause's share of the cell's cycles. Pipeline fill is left out of the stripe.
- **Hover:** cell fill tinted with 12% green.
- **Selected:** reversed to sel with on-sel text (count line included).
- **Focus:** 2px focus ring outside the cell, raised above its neighbours; a selected cell adds an inset 2px on-sel ring.
- **Thin:** cells under 4% of the row, and every image cell, carry no text.

### Timeline
A canvas of equal cycle cells framed in 1px green on the cell fill, 300px tall (180px on phones).
- **Bars:** each cell stacks retired cycles from the floor, then each cause's lost cycles in its cause colour.
- **Selection:** when a layer or operation is selected, the stacks drop to 28% opacity and the selection's own cycles are drawn over them in ink from the floor.
- **Grid:** a 1px faint rule between cells wide enough (4px and up), and a 1px green rule every 8 cells regardless.
- **Cursor:** a 2px focus-colour outline on the cell under the pointer or arrow keys (Shift steps by 8, Home and End jump).
- **Axis and readout:** five mono figures under the canvas at 0, 25, 50, 75 and 100%; a live readout below in body ink-2 with figures bolded in ink.
- **Legend:** in the section band; retired, then only the causes present, then the selection in ink.

### Tables
Ruled tables for measures, instructions and the comparison.
- **Rules:** 1px muted rule between rows; table head and total row ruled with 2px green above.
- **Head:** dense label type in ink-2, right-aligned over numeric columns.
- **Numeric cells:** mono, right-aligned, unwrapped.
- **Pickable rows (compare):** hover tints 10% green; selected rows reverse to sel; focus is a 2px ring inset by 2px, switching to on-sel on a selected row.
- **Cause marks:** after an instruction, a 10px swatch in the cause colour plus the abbreviation (br, lu, jp) and count in the matching -ink colour, in narrow 700.

### Encoding grid
An opened instruction shows its bit fields as a ruled strip: a 1px green box, columns proportional to field width, fields divided by 1px muted rules, the bits in mono with the field name under them in narrow 600 ink-2. A note under it gives format, hex word and retire count.

### Image stepper
Shown only at 760px and below. Prev, a full-width image select, Next: narrow 600 type on the cell fill, 1px green border, 8px 12px padding, 40px minimum height.

### Selection card
The side card on stock-2: a display-weight title (sentence case, wraps anywhere), the route line in mono ink-2 with the last step bolded in ink, then the Measures and instruction tables under their section bands.

## Do's and Don'ts

### Do:
- **Do** rank with weight, case, reversal and rule at 13px and 24px only.
- **Do** put every section heading in a reversed green band (5px 10px, label type).
- **Do** reverse selection into green (sel and on-sel) and keep focus a separate ring: ink in light, amber in dark and in the rail, drawn outside the element.
- **Do** mark lost cycles with a swatch plus an abbreviation (br, lu, jp) in the cause's -ink text colour.
- **Do** set counts, addresses and encodings in JetBrains Mono, right-aligned, with tabular numerals.
- **Do** hide a count line that does not fit and keep the full figure in the tooltip and the card.
- **Do** size proportional cells by cycles, and drop text, not width, when a cell gets thin.
- **Do** keep the band to two lines: scope in narrow caps, then figure plus unit.

### Don't:
- **Don't** look like a profiler's dark flame graph or a dashboard of hero numbers.
- **Don't** add a third type size.
- **Don't** use red, amber or violet for anything but lost cycles, or set cause text in the swatch colour.
- **Don't** make the focus ring the selection colour.
- **Don't** truncate or ellipsize a count.
- **Don't** round corners, add shadows or animate state changes.
- **Don't** set words in mono.
