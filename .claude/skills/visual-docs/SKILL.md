---
name: visual-docs
description: Add or update animated, interactive architecture diagrams in the ECGEN MkDocs site (docs/visual/ "Visual Guide") — input, architecture, output and loss for a model. Use when documenting a new model family (e.g. diffusion, SSM), changing an existing model's architecture/loss so its diagrams must follow, or adding any visual/interactive explanation to the docs.
---

# ECGEN visual docs

The Visual Guide (`docs/visual/`) explains each model with inline SVG diagrams styled by
`docs/stylesheets/architecture.css` and driven by `docs/javascripts/architecture.js` (both
registered in `mkdocs.yml` under `extra_css` / `extra_javascript`). Reuse that system; do not
add new libraries (no D3/Mermaid for these figures).

## Workflow

1. **Read the code first, not the papers.** Take every shape, layer, default and loss term from
   `src/ecgen/families/<family>/`, `src/ecgen/nn/` and the family's `losses.py`. Trace the real tensor shapes
   (strides, padding, which outputs are kept/discarded, what is `.detach()`ed, which optimizer
   steps). If the code deviates from the paper, draw what the code does and add a `!!! note`.
2. **Read an existing page** as the template — `docs/visual/vqvae.md` covers almost every
   pattern (static SVG, data-driven diagram, widgets, loss + gradient figure).
3. **Plan the page** with these sections, in order:
   - one-line summary + code paths + link to the written reference in `docs/models/`
   - *forward pass / setup* — input trace → modules → output trace, with shapes
   - *architecture* — layer-level diagram with hover info on every block
   - *mechanism widget(s)* — the one idea that makes the model work (quantizer snap,
     reparameterisation, causal receptive field, phase shuffle, …)
   - *how the loss is computed* — code snippet, MathJax formula, then a diagram with loss
     boxes (orange), gradient paths (purple) and `sg[·]` for stop-gradients
   - *sampling* — how a new ECG is generated at inference time
4. **Wire it up:** add the page to the `Visual Guide` nav in `mkdocs.yml`, a row to the table
   and the "At a glance" table in `docs/visual/index.md`, and a `!!! tip "Visual guide"` link
   at the top of the matching `docs/models/*.md` page.
5. **Build and look at it** (see *Verify*). Fix every overlap you see; iterate until clean.

## Building blocks

### Static figure (preferred for simple flows)

```html
<figure class="arch">
<svg viewBox="0 0 760 200" role="img" aria-label="...">
<g data-info="<b>Encoder1D</b> — what it does, exact layer args."><rect x="10" y="20" width="90" height="40" rx="5" class="module"/><text x="55" y="45" class="sm c">Encoder</text></g>
<path d="M102,40 L140,40" class="edge flow"/>
<path data-ecg="150,10,200,60" data-lead="II" data-seed="2" data-beats="2" class="trace draw"/>
</svg>
<div class="arch-info">Hover over a block for details.</div>
</figure>
```

Rules that bite:

- **No blank lines inside `<figure>`/`<svg>`** — Python-Markdown would end the raw HTML block.
  Keep each `<g>…</g>` on one line.
- Use a `viewBox` ~760 wide; the CSS scales it to the content width (min 560px, scrolls on phones).
- Any `[data-info]` element in a `figure.arch` that also contains `.arch-info` gets hover/tap
  text automatically. `data-info` may contain `<b>`, `<code>`, `<sub>`.
- Arrow markers are injected by the JS (`#arch-arrow`, `-data`, `-grad`, `-ok`, `-loss`); just
  use the edge classes.

### CSS classes (all theme-aware; never hard-code colours except code hues)

| Purpose | Class |
|---|---|
| tensor / data box | `tensor` |
| trainable module | `module` |
| frozen module (dashed) | `frozen` |
| loss term | `loss` |
| skip / sum / "good" | `ok` |
| neutral box | `plain` |
| faded / discarded | `ghost` |
| arrows | `edge` + `flow` (animated forward), `grad` (purple backward), `skip` (green), `loss` (orange), `noarrow` |
| traces | `trace` (real, blue), `trace fake` (red), `trace mix` (green), `trace noise` (grey); add `draw` to animate drawing |
| text | `sm` `xs` `lg`, `c` (centre) `r` (right), `muted`, `mono` |
| pulsing node | `pulse` |

Colour tokens are `--arch-*` variables defined on `:root, [data-md-color-scheme]` and
overridden for `[data-md-color-scheme="slate"]`. Use `style="fill:var(--arch-grad)"` for
coloured text. A moving packet: `<circle r="4" cy="Y" class="packet"><animateMotion dur="2.4s" repeatCount="indefinite" path="M x0,0 L x1,0"/></circle>`.

### ECG / noise traces

`<path data-ecg="x,y,w,h" ...>` is filled by the JS with a synthetic (schematic) trace:
`data-lead` (I, II, III, aVR, aVL, aVF, V1–V6), `data-seed`, `data-beats`, and `data-kind`
(`ecg` default, `fake` = jittered/noisier, `noise` = U(−1, 1)). Say "schematic" in a caption
wherever a reader might mistake traces for real data or model output.

### Data-driven diagrams and widgets (in `architecture.js`)

- `<svg data-diagram="NAME">` → `DIAGRAMS[NAME](svg)` renders it (existing: `unet`, `critic`).
- `<figure class="arch" data-widget="NAME">` → `WIDGETS[NAME](fig)` (existing: `leads`,
  `backbone` (+ `data-mode="vae|vqvae"`), `downsample`, `kl`, `vq`, `causal`, `ar`, `phase`,
  `gp`, `schedule`). Controls go in `<div class="arch-controls">` (`button[aria-pressed]`,
  `input[type=range]` + `<output>`); per-step text goes in `.arch-info`.

Use JS only for things that are computed (layer stacks, toy simulations, sliders). To add one:

1. Write `function widgetFoo(fig)` / `renderFoo(svg)` next to the others, using the helpers
   `el(tag, attrs, parent, text)`, `clear`, `rng(seed)` (deterministic), `softmax`,
   `categorical`, `codeColor(k, n)`, `ecgSeries`, `noiseSeries`, `seriesPath`, `renderTraces`.
2. Register it in `WIDGETS` / `DIAGRAMS` at the bottom.
3. Animation loops: `setInterval`/`setTimeout`, stop when `!document.body.contains(svg)`,
   stop auto-play once the user interacts, and honour `reduceMotion` (render the final state).
4. Toy simulations must say they are toys and give the real numbers in a label or caption
   (e.g. "toy: 30 of 625 codes, codebook 12 of 512").
5. Check syntax: `node --check docs/javascripts/architecture.js`.

## Verify

```bash
# one-time: a scratch venv with the docs toolchain + headless Chromium
python -m venv <scratch>/docsenv
<scratch>/docsenv/bin/pip install -r requirements-docs.txt playwright
<scratch>/docsenv/bin/python -m playwright install chromium

<scratch>/docsenv/bin/mkdocs build --strict -d <scratch>/site
<scratch>/docsenv/bin/python .claude/skills/visual-docs/scripts/screenshot_docs.py <scratch>/site <scratch>/shots [visual/foo/ ...]
```

Then **Read every `fig_*_light.png` and a few `*_dark.png`**. Look for: overlapping labels,
arrows crossing text, labels on top of arrows, empty traces, wrong colours in dark mode, and
figures that don't span the content width. Only the GitHub-API 404 is an acceptable error.

## Pitfalls already solved — don't reintroduce

- Material's `figure` styles shrink to content: `.md-typeset figure.arch` forces `display:block; width:auto`.
- `--arch-*` vars must be declared on `[data-md-color-scheme]`, not just `:root`, or dark mode
  inherits light values (and markers outside `.arch` lose colour).
- Loss/gradient figures get tangled fast: put the backward path in its own lane above the
  forward row and route sums along the bottom, rather than drawing curves across boxes.
- Keep small labels off block tops where neighbours also have labels; prefer hover info.
- `pymdown-extensions` must stay `<10.17` (pinned in `requirements-docs.txt`); 12.x breaks
  mkdocstrings and fails `--strict`.
