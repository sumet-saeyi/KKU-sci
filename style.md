# Air Canvas — Network Visualization Style Guide

This documents the visual language used by `_draw_dynamic_network` in
`air_canvas.py`, so future stages/panels stay consistent instead of drifting
back toward a "neon hacker terminal" look.

## Design principles

1. **One accent color means something is happening now.** Teal = active
   flow. Amber = the final, singular result (the predicted class). Nothing
   else competes for attention — most of the frame is quiet slate.
2. **State is always legible at a glance.** Every stage panel and every
   stepper node has exactly three states — `pending`, `active`, `done` —
   each with one consistent visual treatment. No stage should invent its own
   one-off styling.
3. **Motion should mean something.** Animation timing follows the actual
   computation: kernels scan in raster order and reveal all channels at
   once, pooling resolves as one synchronized pulse, dense layers flash
   instantaneously. See the code comments in `_draw_dynamic_network` for the
   accuracy rationale — this file only covers the *visual* system.
4. **Quiet by default.** Panels, grids, and borders are low-contrast until a
   stage becomes active or complete. Nothing pulses, glows, or animates
   unless it is currently relevant.

## Color palette (BGR tuples, since OpenCV expects BGR)

| Role                     | Hex       | BGR             | Usage                                  |
|--------------------------|-----------|-----------------|-----------------------------------------|
| Background (`BG`)        | `#0f172a` | `(36, 26, 18)`  | Full-frame fill, base for the grid      |
| Panel fill (`PANEL_BG`)  | `#1a2332` | `(46, 35, 26)`  | Card backgrounds                        |
| Border (`BORDER`)        | `#334155` | `(85, 65, 51)`  | Panel outlines, dividers                |
| Border, dim (`BORDER_DIM`)| `#26313f`| `(58, 46, 38)`  | Pending-state outlines, stepper track   |
| Grid (`GRID`)            | `#1e2530` | `(48, 38, 30)`  | Faint background grid lines             |
| Text (`TEXT`)            | `#e2e8e8` | `(240, 232, 226)`| Primary labels, active-state text      |
| Text, muted (`TEXT_DIM`) | `#94a3b8` | `(184, 163, 148)`| Secondary labels, pending-state text   |
| Accent — teal (`TEAL`)   | `#2dd42d`*| `(191, 212, 45)`| Active flow, kernels, connections       |
| Accent — teal dim (`TEAL_DIM`)| —    | `(120, 135, 40)`| Completed-state borders/dots           |
| Accent — amber (`AMBER`) | `#fbbf24` | `(36, 191, 251)`| The winning class, the result card only|

\* approximate; tuned by eye against the dark background rather than a strict
hex round-trip. When adding a color, pick the BGR tuple that reads correctly
on-screen and add its nearest hex here for reference.

Never introduce a new accent color for a one-off effect. If something needs
emphasis, it's either "active" (teal) or "the result" (amber) — nothing
else earns its own hue.

## Typography

- Font family: `cv2.FONT_HERSHEY_SIMPLEX` for all body/label text;
  `cv2.FONT_HERSHEY_DUPLEX` reserved for the single large result digit in the
  prediction card (the one place a heavier weight is warranted).
- Always pass `cv2.LINE_AA` for text and thin accent lines — this is what
  keeps the dashboard from looking like a raw OpenCV demo.
- Scale conventions:
  - `0.55` — top-level page title only ("GE-2019 / FORWARD PASS").
  - `0.42–0.45` — panel titles (e.g. "CONV1  32 channels, shared kernel").
  - `0.36–0.38` — stepper labels, sub-captions under a panel.
  - `0.34–0.36` — inline annotations (e.g. "kernel", "dense: all neurons…").
- Panel titles are two-part: a short stage name in caps, then a plain-English
  description, separated by extra spacing rather than punctuation —
  `"CONV1   32 channels, shared kernel"`.

## Layout patterns

### Panel (`panel(x, y, w, h, border=BORDER, fill=PANEL_BG, accent=None)`)
Every stage lives in a panel: filled rect + 1px border + an optional 2px
accent line along the top edge. The accent line is the *only* signal for
active/done — don't also change the fill color or add a glow.

- `accent=TEAL` → stage is currently active.
- `accent=TEAL_DIM` → stage has completed.
- `accent=None` → stage hasn't started (pending); border stays `BORDER` or
  `BORDER_DIM`.

### Pipeline stepper
A single horizontal stepper at the top (y≈70) is the source of truth for
"where are we in the forward pass." Each stage gets one node:

- Pending: hollow circle (`BORDER_DIM`), radius 5, dim label.
- Active: filled circle that pulses in brightness (`sin`-based), radius 6,
  plus a radius-8 ring around it, bright label.
- Done: filled circle, steady `TEAL_DIM`, dim label.

Connector lines between nodes are `TEAL_DIM` once the earlier node is done,
otherwise `BORDER_DIM`. Add new stages to the single `STAGES` list
(`name, start_frame, end_frame`) — the stepper and all panel accents read
from that one list, so timing changes never need to be updated in two
places.

### Data flow between stages (`burst(...)`)
A handful of parallel lines fade in together from a source panel's edge to
the next panel's inlet, over a short fixed life (~10 frames). This is
intentionally not a single traveling particle — a moving dot implies serial
transfer, which is wrong for a full tensor moving to the next layer.

### Result card
Bottom-center, appears once softmax has resolved: a bordered card with an
amber top accent, a small muted "PREDICTION" label, the class digit large in
amber (Duplex font), and a confidence percentage in primary text. This is
the only place amber and the large type scale are used — it should always
read as "the answer," never as decoration elsewhere in the frame.

## Adding a new stage/panel checklist

1. Add `("NAME", start_frame, end_frame)` to `STAGES`.
2. Draw it with `panel(...)`, computing `accent` from `stage_state(i)` the
   same way every other stage does.
3. Title it with `title("NAME   plain-english description", x, y-14)`.
4. If it needs a highlight color, reuse `TEAL` for "computing now" — don't
   add a new hue.
5. If it involves incoming data from the previous stage, use `burst(...)`,
   not a custom particle effect.
