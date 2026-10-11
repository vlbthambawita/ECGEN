/*
 * Interactive diagrams for the Visual Guide pages (docs/visual/).
 *
 * Static diagrams are inline SVG in the Markdown pages; this script
 *   - injects the shared arrow markers,
 *   - draws synthetic ECG / noise traces into <path data-ecg="x,y,w,h">,
 *   - renders the data-driven diagrams (<svg data-diagram="...">),
 *   - wires up the widgets (<figure data-widget="...">).
 * All numbers shown in the diagrams mirror the defaults in models/ and training/losses/.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, parent, text) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Deterministic PRNG (mulberry32) so traces look the same on every load.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function softmax(logits, temperature) {
    const scaled = logits.map((l) => l / temperature);
    const m = Math.max.apply(null, scaled);
    const e = scaled.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function categorical(p, r) {
    let acc = 0;
    for (let i = 0; i < p.length; i++) {
      acc += p[i];
      if (r < acc) return i;
    }
    return p.length - 1;
  }

  function codeColor(k, n) {
    return "hsl(" + Math.round((k * 360) / n) + ", 65%, 55%)";
  }

  // ------------------------------------------------------------------
  // Shared arrow markers
  // ------------------------------------------------------------------

  function injectDefs() {
    if (document.getElementById("arch-arrow")) return;
    const svg = el("svg", { class: "arch-defs", "aria-hidden": "true" }, document.body);
    const defs = el("defs", {}, svg);
    ["", "-data", "-grad", "-ok", "-loss"].forEach(function (suffix) {
      const m = el("marker", {
        id: "arch-arrow" + suffix,
        viewBox: "0 0 10 10",
        refX: "9",
        refY: "5",
        markerWidth: "7",
        markerHeight: "7",
        orient: "auto-start-reverse",
      }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z" }, m);
    });
  }

  // ------------------------------------------------------------------
  // Synthetic ECG traces
  // ------------------------------------------------------------------

  // Rough per-lead morphology: amplitudes of P, Q, R, S, T (R normalised to ~1 in lead II).
  const LEADS = {
    I: [0.1, -0.05, 0.6, -0.1, 0.22],
    II: [0.14, -0.08, 1.0, -0.2, 0.3],
    III: [0.06, -0.05, 0.45, -0.15, 0.1],
    aVR: [-0.1, 0.05, -0.75, 0.1, -0.25],
    aVL: [0.05, -0.05, 0.35, -0.1, 0.1],
    aVF: [0.1, -0.06, 0.7, -0.15, 0.2],
    V1: [0.06, 0.0, 0.25, -0.9, -0.05],
    V2: [0.07, 0.0, 0.45, -1.0, 0.35],
    V3: [0.08, -0.03, 0.75, -0.6, 0.4],
    V4: [0.08, -0.06, 1.1, -0.3, 0.38],
    V5: [0.08, -0.08, 1.0, -0.2, 0.3],
    V6: [0.08, -0.08, 0.8, -0.1, 0.25],
  };

  function gauss(x, mu, sd) {
    const z = (x - mu) / sd;
    return Math.exp(-z * z);
  }

  function ecgValue(phi, a) {
    return (
      a[0] * gauss(phi, 0.16, 0.03) +
      a[1] * gauss(phi, 0.3, 0.01) +
      a[2] * gauss(phi, 0.325, 0.011) +
      a[3] * gauss(phi, 0.352, 0.012) +
      a[4] * gauss(phi, 0.6, 0.055)
    );
  }

  /* Returns values in roughly [-1, 1] for n samples. */
  function ecgSeries(n, opts) {
    const r = rng(opts.seed || 1);
    const amp = LEADS[opts.lead || "II"] || LEADS.II;
    const beats = opts.beats || 3;
    const offset = r();
    const out = [];
    const wobble = opts.kind === "fake" ? 0.12 : 0.0;
    const noise = opts.kind === "fake" ? 0.05 : 0.008;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * beats + offset;
      const beat = Math.floor(t);
      const br = rng((opts.seed || 1) * 977 + beat);
      const scale = 1 + wobble * (br() - 0.5) * 2;
      const shift = wobble * (br() - 0.5) * 0.3;
      let v = ecgValue(t - beat - shift, amp) * scale;
      v += 0.04 * Math.sin(2 * Math.PI * (i / n) * 0.7 + offset * 6); // baseline wander
      v += noise * (r() * 2 - 1);
      out.push(v);
    }
    return out;
  }

  function noiseSeries(n, seed) {
    const r = rng(seed || 7);
    const out = [];
    for (let i = 0; i < n; i++) out.push(r() * 2 - 1);
    return out;
  }

  function seriesPath(values, x, y, w, h, vmin, vmax) {
    const n = values.length;
    let d = "";
    for (let i = 0; i < n; i++) {
      const px = x + (i / (n - 1)) * w;
      const py = y + h - ((values[i] - vmin) / (vmax - vmin)) * h;
      d += (i ? "L" : "M") + px.toFixed(1) + "," + py.toFixed(1);
    }
    return d;
  }

  function traceSeries(node, w) {
    const kind = node.dataset.kind || "ecg";
    const seed = parseInt(node.dataset.seed || "1", 10);
    const n = Math.max(60, Math.round(w * (kind === "noise" ? 0.8 : 3)));
    if (kind === "noise") return { values: noiseSeries(n, seed), vmin: -1, vmax: 1 };
    const values = ecgSeries(n, {
      seed: seed,
      lead: node.dataset.lead,
      beats: parseFloat(node.dataset.beats || "3"),
      kind: kind,
    });
    return { values: values, vmin: -1.05, vmax: 1.15 };
  }

  function renderTraces(root) {
    root.querySelectorAll("path[data-ecg]").forEach(function (node) {
      const box = node.dataset.ecg.split(",").map(Number);
      const s = traceSeries(node, box[2]);
      node.setAttribute("d", seriesPath(s.values, box[0], box[1], box[2], box[3], s.vmin, s.vmax));
      if (node.classList.contains("draw") && node.getTotalLength) {
        node.style.setProperty("--len", Math.ceil(node.getTotalLength()));
      }
    });
  }

  // ------------------------------------------------------------------
  // Generic hover / tap info: any [data-info] inside a figure with .arch-info
  // ------------------------------------------------------------------

  function wireInfo(fig) {
    const box = fig.querySelector(".arch-info");
    if (!box || fig.dataset.infoWired) return;
    fig.dataset.infoWired = "1";
    const initial = box.innerHTML;
    let active = null;
    function show(node) {
      if (active) active.classList.remove("active");
      active = node;
      node.classList.add("active");
      box.innerHTML = node.dataset.info;
    }
    fig.addEventListener("mouseover", function (e) {
      const node = e.target.closest("[data-info]");
      if (node && fig.contains(node)) show(node);
    });
    fig.addEventListener("click", function (e) {
      const node = e.target.closest("[data-info]");
      if (node && fig.contains(node)) show(node);
    });
    fig.addEventListener("mouseleave", function () {
      if (active) active.classList.remove("active");
      active = null;
      box.innerHTML = initial;
    });
  }

  // ------------------------------------------------------------------
  // Widget: 12-lead input grid (toggle 12 / 8 leads)
  // ------------------------------------------------------------------

  function widgetLeads(fig) {
    const buttons = fig.querySelectorAll("button[data-leads]");
    const shape = fig.querySelector("[data-role=shape]");
    function set(n) {
      fig.querySelectorAll("[data-lead-idx]").forEach(function (g) {
        g.classList.toggle("dim", parseInt(g.dataset.leadIdx, 10) >= n);
      });
      buttons.forEach(function (b) {
        b.setAttribute("aria-pressed", String(b.dataset.leads === String(n)));
      });
      if (shape) shape.textContent = "x : (B, " + n + ", 5000)";
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        set(parseInt(b.dataset.leads, 10));
      });
    });
    set(12);
  }

  // ------------------------------------------------------------------
  // Diagram: Encoder1D / Decoder1D backbone (VAE and VQ-VAE)
  // ------------------------------------------------------------------

  const LEN_H = { 5000: 150, 2500: 108, 1250: 76, 625: 52 };

  function blockWidth(item) {
    if (item.k === "down" || item.k === "up") return 6;
    if (item.k === "tensor") return 14;
    return 7 + item.ch / 12;
  }

  function backboneItems(mode) {
    const vq = mode === "vqvae";
    const lc = vq ? 64 : 8;
    const res = function (cin, cout) {
      return (
        "2 × ResidualBlock1D(" + cin + " → " + cout + "): [Conv1d(k=3) → GroupNorm(8) → SiLU] × 2, plus a " +
        (cin === cout ? "identity" : "1×1 conv") + " shortcut."
      );
    };
    const enc = [
      { k: "tensor", ch: 12, len: 5000, info: "<b>Input x</b> — (B, 12, 5000): 12 leads × 10 s at 500 Hz." },
      { k: "conv", ch: 64, len: 5000, info: "<b>conv_in</b> — Conv1d(12 → 64, k=7, padding=3). Length unchanged." },
      { k: "res", ch: 64, len: 5000, top: "64", info: "<b>Stage 1</b> (mult 1) — " + res(64, 64) },
      { k: "down", ch: 64, len: 2500, info: "<b>Downsample</b> — Conv1d(64 → 64, k=4, stride=2, padding=1): 5000 → 2500." },
      { k: "res", ch: 128, len: 2500, top: "128", info: "<b>Stage 2</b> (mult 2) — " + res(64, 128) },
      { k: "down", ch: 128, len: 1250, info: "<b>Downsample</b> — Conv1d(128 → 128, k=4, stride=2): 2500 → 1250." },
      { k: "res", ch: 256, len: 1250, top: "256", info: "<b>Stage 3</b> (mult 4) — " + res(128, 256) },
      { k: "down", ch: 256, len: 625, info: "<b>Downsample</b> — Conv1d(256 → 256, k=4, stride=2): 1250 → 625. The last stage has no downsample, so 4 stages give 2³ = 8×." },
      { k: "res", ch: 256, len: 625, top: "256", info: "<b>Stage 4</b> (mult 4) — " + res(256, 256) },
      { k: "res", ch: 256, len: 625, top: "mid", info: "<b>Middle</b> — mid_block1, mid_block2: two more ResidualBlock1D(256)." },
      { k: "conv", ch: 2 * lc, len: 625, info: "<b>Output head</b> — GroupNorm(8) → SiLU → Conv1d(256 → " + 2 * lc + ", k=3), then torch.chunk into mean and logvar (" + lc + " channels each)." },
    ];
    const dec = [
      { k: "conv", ch: 256, len: 625, info: "<b>conv_in</b> — Conv1d(" + lc + " → 256, k=3)." },
      { k: "res", ch: 256, len: 625, top: "mid", info: "<b>Middle</b> — two ResidualBlock1D(256)." },
      { k: "res", ch: 256, len: 625, top: "256", info: "<b>Stage 1</b> (reversed mult 4) — " + res(256, 256) },
      { k: "up", ch: 256, len: 1250, info: "<b>Upsample</b> — ConvTranspose1d(256 → 256, k=4, stride=2, padding=1): 625 → 1250." },
      { k: "res", ch: 256, len: 1250, top: "256", info: "<b>Stage 2</b> (mult 4) — " + res(256, 256) },
      { k: "up", ch: 256, len: 2500, info: "<b>Upsample</b> — ConvTranspose1d(k=4, stride=2): 1250 → 2500." },
      { k: "res", ch: 128, len: 2500, top: "128", info: "<b>Stage 3</b> (mult 2) — " + res(256, 128) },
      { k: "up", ch: 128, len: 5000, info: "<b>Upsample</b> — ConvTranspose1d(k=4, stride=2): 2500 → 5000." },
      { k: "res", ch: 64, len: 5000, top: "64", info: "<b>Stage 4</b> (mult 1) — " + res(128, 64) },
      { k: "conv", ch: 64, len: 5000, info: "<b>conv_out</b> — GroupNorm(8) → SiLU → Conv1d(64 → 12, k=7). No output activation." },
      { k: "tensor", ch: 12, len: 5000, info: "<b>Reconstruction x̂</b> — (B, 12, 5000)." },
    ];
    return { enc: enc, dec: dec, lc: lc, vq: vq };
  }

  function drawBlock(svg, item, x, cy) {
    const w = blockWidth(item);
    const h = LEN_H[item.len];
    const g = el("g", { "data-info": item.info, "data-hover": "" }, svg);
    const cls = item.k === "tensor" ? "tensor" : item.cls || "module";
    el("rect", { x: x, y: cy - h / 2, width: w, height: h, rx: 2, class: cls }, g);
    if (item.k === "res") {
      el("line", { x1: x + w / 2, x2: x + w / 2, y1: cy - h / 2 + 3, y2: cy + h / 2 - 3, class: "gridline" }, g);
    }
    if (item.top) el("text", { x: x + w / 2, y: cy - h / 2 - 6, class: "xs c muted" }, g, item.top);
    return w;
  }

  function renderBackbone(svg, mode) {
    clear(svg);
    const spec = backboneItems(mode);
    const cy = 120;
    const gap = 11;
    let x = 14;
    const spans = [];
    function chain(items) {
      const start = x;
      items.forEach(function (item) {
        const w = drawBlock(svg, item, x, cy);
        spans.push({ x0: x, x1: x + w, len: item.len });
        x += w + gap;
      });
      return [start, x - gap];
    }
    const encSpan = chain(spec.enc);

    // Latent section
    const lx = x + 6;
    const latentH = LEN_H[625];
    const half = latentH / 2 - 2;
    const gMu = el("g", { "data-info": spec.vq
      ? "<b>z = mean</b> — (B, 64, 625). VQVAE1D keeps only the first half of the encoder output as the continuous latent."
      : "<b>mean μ</b> — (B, 8, 625).", "data-hover": "" }, svg);
    el("rect", { x: lx, y: cy - half - 2, width: 16, height: half, class: "tensor" }, gMu);
    el("text", { x: lx + 8, y: cy - half - 8, class: "xs c" }, gMu, spec.vq ? "z" : "μ");
    const gLv = el("g", { "data-info": spec.vq
      ? "<b>logvar</b> — computed by Encoder1D but discarded by VQVAE1D.encode()."
      : "<b>logvar</b> — (B, 8, 625), log σ².", "data-hover": "", class: spec.vq ? "ghost" : "" }, svg);
    el("rect", { x: lx, y: cy + 2, width: 16, height: half, class: "tensor" }, gLv);
    el("text", { x: lx + 8, y: cy + half + 16, class: "xs c" }, gLv, spec.vq ? "logvar ✕" : "log σ²");
    spans.push({ x0: lx, x1: lx + 16, len: 625 });

    const nx = lx + 52;
    let after;
    if (spec.vq) {
      const gq = el("g", { "data-info": "<b>VectorQuantizer</b> — codebook nn.Embedding(512, 64). Each of the 625 latent vectors is replaced by its nearest codebook vector (L2); indices (B, 625) are the discrete code.", "data-hover": "" }, svg);
      el("rect", { x: nx - 24, y: cy - 26, width: 48, height: 52, rx: 6, class: "module" }, gq);
      el("text", { x: nx, y: cy - 4, class: "sm c" }, gq, "VQ");
      el("text", { x: nx, y: cy + 12, class: "xs c muted" }, gq, "512×64");
      el("path", { d: "M" + (lx + 18) + "," + (cy - half / 2) + " L" + (nx - 26) + "," + (cy - 8), class: "edge flow" }, svg);
      const gz = el("g", { "data-info": "<b>z_q</b> — (B, 64, 625) quantized latent passed to the decoder.", "data-hover": "" }, svg);
      el("rect", { x: nx + 40, y: cy - latentH / 2, width: 16, height: latentH, class: "tensor" }, gz);
      el("text", { x: nx + 48, y: cy - latentH / 2 - 6, class: "xs c" }, gz, "z_q");
      el("path", { d: "M" + (nx + 26) + "," + cy + " L" + (nx + 38) + "," + cy, class: "edge flow" }, svg);
      spans.push({ x0: nx - 24, x1: nx + 56, len: 625 });
      after = nx + 56;
    } else {
      const gz = el("g", { "data-info": "<b>Reparameterisation</b> — z = μ + exp(0.5·logvar) ⊙ ε, ε ~ N(0, I). Gradients flow to μ and logvar but not to ε.", "data-hover": "" }, svg);
      el("circle", { cx: nx, cy: cy, r: 20, class: "module" }, gz);
      el("text", { x: nx, y: cy + 4, class: "sm c" }, gz, "z");
      el("text", { x: nx, y: cy + 58, class: "xs c muted" }, gz, "ε ~ N(0, I)");
      el("path", { d: "M" + nx + "," + (cy + 46) + " L" + nx + "," + (cy + 22), class: "edge" }, svg);
      el("path", { d: "M" + (lx + 18) + "," + (cy - half / 2) + " L" + (nx - 20) + "," + (cy - 6), class: "edge flow" }, svg);
      el("path", { d: "M" + (lx + 18) + "," + (cy + half / 2) + " L" + (nx - 20) + "," + (cy + 6), class: "edge flow" }, svg);
      const gzz = el("g", { "data-info": "<b>z</b> — (B, 8, 625) sampled latent.", "data-hover": "" }, svg);
      el("rect", { x: nx + 36, y: cy - latentH / 2, width: 12, height: latentH, class: "tensor" }, gzz);
      el("path", { d: "M" + (nx + 21) + "," + cy + " L" + (nx + 34) + "," + cy, class: "edge flow" }, svg);
      spans.push({ x0: nx - 20, x1: nx + 48, len: 625 });
      after = nx + 48;
    }
    x = after + gap + 4;
    const decSpan = chain(spec.dec);

    // Length ruler: merge consecutive spans with the same length.
    const ry = 235;
    const groups = [];
    spans.forEach(function (s) {
      const last = groups[groups.length - 1];
      if (last && last.len === s.len) last.x1 = s.x1;
      else groups.push({ x0: s.x0, x1: s.x1, len: s.len });
    });
    groups.forEach(function (gp) {
      el("path", { d: "M" + (gp.x0 + 1) + "," + (ry - 4) + " V" + ry + " H" + (gp.x1 - 1) + " V" + (ry - 4), class: "edge noarrow" }, svg);
      el("text", { x: (gp.x0 + gp.x1) / 2, y: ry + 13, class: "xs c muted mono" }, svg, String(gp.len));
    });
    el("text", { x: 2, y: ry + 13, class: "xs muted" }, svg, "");

    // Section labels
    [[encSpan, "Encoder1D"], [decSpan, "Decoder1D"]].forEach(function (pair) {
      const s = pair[0];
      el("text", { x: (s[0] + s[1]) / 2, y: 22, class: "lg c" }, svg, pair[1]);
    });
    el("text", { x: lx + 40, y: 22, class: "lg c" }, svg, spec.vq ? "Quantize" : "Sample");
    el("text", { x: (encSpan[0] + decSpan[1]) / 2, y: 268, class: "xs c muted" }, svg,
      "Block width ∝ channels · height ∝ sequence length · numbers above = channels · ruler = length");

    svg.setAttribute("viewBox", "0 0 " + (decSpan[1] + 14) + " 278");
  }

  function widgetBackbone(fig) {
    const svg = fig.querySelector("svg[data-diagram=backbone]");
    const buttons = fig.querySelectorAll("button[data-mode]");
    function set(mode) {
      renderBackbone(svg, mode);
      buttons.forEach(function (b) {
        b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
      });
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () { set(b.dataset.mode); });
    });
    set(fig.dataset.mode || "vae");
  }

  // ------------------------------------------------------------------
  // Widget: downsampling calculator
  // ------------------------------------------------------------------

  function widgetDownsample(fig) {
    const stages = fig.querySelector("input[name=stages]");
    const length = fig.querySelector("select[name=length]");
    const out = fig.querySelector(".arch-info");
    const stagesOut = fig.querySelector("output[name=stages]");
    function update() {
      const s = parseInt(stages.value, 10);
      const L = parseInt(length.value, 10);
      const mults = [1, 2, 4, 4, 8].slice(0, s);
      if (stagesOut) stagesOut.textContent = s + "  (" + mults.join(", ") + ")";
      let latent = L;
      const steps = [L];
      for (let i = 0; i < s - 1; i++) {
        latent = Math.floor(latent / 2); // Conv1d(k=4, stride=2, padding=1)
        steps.push(latent);
      }
      const factor = Math.pow(2, s - 1);
      const recon = latent * factor;
      const ok = recon === L;
      out.innerHTML =
        "Downsample factor <b>2<sup>" + (s - 1) + "</sup> = " + factor + "</b> · lengths " + steps.join(" → ") +
        " · latent length <b>" + latent + "</b> · decoder output <b>" + recon + "</b> " +
        (ok ? "✓ matches the input" : "✗ ≠ " + L + " — the reconstruction would be shorter than the input");
      out.style.borderLeftColor = ok ? "var(--arch-ok)" : "var(--arch-module)";
    }
    stages.addEventListener("input", update);
    length.addEventListener("change", update);
    update();
  }

  // ------------------------------------------------------------------
  // Widget: KL weight (schematic)
  // ------------------------------------------------------------------

  function widgetKL(fig) {
    const svg = fig.querySelector("svg");
    const slider = fig.querySelector("input[type=range]");
    const outp = fig.querySelector("output");
    const W = 760, H = 230, x0 = 40, x1 = W - 40, base = 190;
    const toX = function (z) { return x0 + ((z + 4) / 8) * (x1 - x0); };
    const posts = [
      { m: -2.6, s: 0.25 }, { m: -1.1, s: 0.2 }, { m: 0.6, s: 0.22 }, { m: 2.4, s: 0.3 },
    ];
    clear(svg);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    el("line", { x1: x0, x2: x1, y1: base, y2: base, class: "gridline" }, svg);
    [-3, -2, -1, 0, 1, 2, 3].forEach(function (z) {
      el("text", { x: toX(z), y: base + 14, class: "xs c muted mono" }, svg, String(z));
    });
    function curve(m, s, scale) {
      let d = "";
      for (let i = 0; i <= 200; i++) {
        const z = -4 + (8 * i) / 200;
        const p = Math.exp(-0.5 * Math.pow((z - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI));
        d += (i ? "L" : "M") + toX(z).toFixed(1) + "," + (base - Math.min(p * scale, 170)).toFixed(1);
      }
      return d;
    }
    el("path", { d: curve(0, 1, 150), class: "edge noarrow", "stroke-dasharray": "5 4" }, svg);
    el("text", { x: toX(0), y: base - 150 * 0.399 - 8, class: "xs c muted" }, svg, "prior N(0, 1)");
    const paths = posts.map(function (p, i) {
      return el("path", { class: "trace", style: "stroke:" + codeColor(i * 3, 12) + ";stroke-width:2" }, svg);
    });
    const note = el("text", { x: W / 2, y: 22, class: "sm c" }, svg, "");
    function update() {
      const beta = Math.pow(10, parseFloat(slider.value));
      // Schematic only: how far posteriors are pulled towards the prior.
      const f = 1 / (1 + Math.pow(beta / 1e-3, 0.7));
      posts.forEach(function (p, i) {
        const m = p.m * (0.15 + 0.85 * f);
        const s = p.s + (1 - p.s) * (1 - f);
        paths[i].setAttribute("d", curve(m, s, 26));
      });
      outp.textContent = beta.toExponential(0).replace("e-", "e−");
      note.textContent = f > 0.75 ? "posteriors far apart → sharp reconstructions, gaps in latent space"
        : f > 0.35 ? "balanced" : "posteriors ≈ prior → blurry reconstructions, smooth samples";
    }
    slider.addEventListener("input", update);
    update();
  }

  // ------------------------------------------------------------------
  // Widget: vector quantizer (2-D toy)
  // ------------------------------------------------------------------

  function widgetVQ(fig) {
    const svg = fig.querySelector("svg");
    const btn = fig.querySelector("button[data-action=resample]");
    const W = 760, H = 300;
    const K = 12, T = 8;
    const px0 = 20, py0 = 30, pw = 360, ph = 240;
    let seed = 3;
    let timers = [];
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    function run() {
      timers.forEach(clearTimeout);
      timers = [];
      clear(svg);
      const r = rng(seed++);
      el("rect", { x: px0, y: py0, width: pw, height: ph, class: "plain", rx: 4 }, svg);
      el("text", { x: px0 + pw / 2, y: 20, class: "sm c" }, svg, "latent space (toy 2-D; real: 64-D)");
      const codes = [];
      for (let k = 0; k < K; k++) {
        codes.push([px0 + 20 + r() * (pw - 40), py0 + 20 + r() * (ph - 40)]);
      }
      const zs = [];
      for (let t = 0; t < T; t++) {
        zs.push([px0 + 20 + r() * (pw - 40), py0 + 20 + r() * (ph - 40)]);
      }
      const lines = el("g", {}, svg);
      codes.forEach(function (c, k) {
        const g = el("g", { transform: "translate(" + c[0] + "," + c[1] + ")" }, svg);
        el("path", { d: "M-6,-6 L6,6 M-6,6 L6,-6", style: "stroke:" + codeColor(k, K) + ";stroke-width:2.5" }, g);
        el("text", { x: 8, y: -6, class: "xs mono muted" }, g, "e" + k);
      });
      const nearest = zs.map(function (z) {
        let best = 0, bd = Infinity;
        codes.forEach(function (c, k) {
          const d = Math.pow(c[0] - z[0], 2) + Math.pow(c[1] - z[1], 2);
          if (d < bd) { bd = d; best = k; }
        });
        return best;
      });
      const dots = zs.map(function (z, t) {
        const g = el("g", { style: "transform: translate(" + z[0] + "px," + z[1] + "px); transition: transform 0.9s ease-in-out" }, svg);
        el("circle", { r: 7, class: "tensor" }, g);
        el("text", { y: 3.5, class: "xs c mono" }, g, String(t));
        return g;
      });

      // Right: index sequence
      const sx = 420, sy = 70, cw = 38;
      el("text", { x: sx, y: 50, class: "sm" }, svg, "z  (B, 64, 625)  →  one vector per time step t");
      el("text", { x: sx, y: 170, class: "sm" }, svg, "indices  (B, 625)");
      const cells = [];
      for (let t = 0; t < T; t++) {
        el("rect", { x: sx + t * cw, y: sy, width: cw - 4, height: 30, class: "tensor", rx: 3 }, svg);
        el("text", { x: sx + t * cw + (cw - 4) / 2, y: sy + 19, class: "xs c mono" }, svg, "z" + t);
        el("rect", { x: sx + t * cw, y: sy + 110, width: cw - 4, height: 30, class: "plain", rx: 3 }, svg);
        const bg = el("rect", { x: sx + t * cw, y: sy + 110, width: cw - 4, height: 30, rx: 3, style: "fill:transparent" }, svg);
        const tx = el("text", { x: sx + t * cw + (cw - 4) / 2, y: sy + 129, class: "sm c mono" }, svg, "");
        cells.push([bg, tx]);
      }
      el("text", { x: sx + T * cw + 4, y: sy + 129, class: "sm mono muted" }, svg, "…");
      el("text", { x: sx, y: 255, class: "xs muted" }, svg, "k*ₜ = argminₖ ‖zₜ − eₖ‖₂ ,   z_q,ₜ = e_{k*ₜ}");
      el("text", { x: sx, y: 272, class: "xs muted" }, svg, "decoder input: z + (z_q − z).detach()  (straight-through)");

      const step = reduceMotion ? 0 : 1;
      timers.push(setTimeout(function () {
        zs.forEach(function (z, t) {
          const c = codes[nearest[t]];
          el("line", { x1: z[0], y1: z[1], x2: c[0], y2: c[1], style: "stroke:" + codeColor(nearest[t], K) + ";stroke-dasharray:3 3;stroke-width:1.5" }, lines);
        });
      }, 900 * step));
      timers.push(setTimeout(function () {
        dots.forEach(function (g, t) {
          const c = codes[nearest[t]];
          g.style.transform = "translate(" + c[0] + "px," + c[1] + "px)";
          g.querySelector("circle").setAttribute("style", "fill:" + codeColor(nearest[t], K));
        });
      }, 2000 * step));
      nearest.forEach(function (k, t) {
        timers.push(setTimeout(function () {
          cells[t][0].setAttribute("style", "fill:" + codeColor(k, K));
          cells[t][1].textContent = String(k);
          cells[t][1].setAttribute("style", "fill:#111;font-weight:700");
        }, (3100 + t * 180) * step));
      });
      if (!reduceMotion) timers.push(setTimeout(run, 7500));
    }
    if (btn) btn.addEventListener("click", run);
    run();
  }

  // ------------------------------------------------------------------
  // Widget: causal dilated stack / receptive field (toy)
  // ------------------------------------------------------------------

  function widgetCausal(fig) {
    const svg = fig.querySelector("svg");
    const info = fig.querySelector(".arch-info");
    const N = 24;
    const dil = [1, 2, 4];
    const rows = ["codes c", "shift right", "d = 1", "d = 2", "d = 4", "logits"];
    const W = 780, x0 = 110, dx = (W - x0 - 20) / (N - 1);
    const rowY = function (r) { return 270 - r * 46; };
    svg.setAttribute("viewBox", "0 0 " + W + " 300");
    clear(svg);
    // parents[r][i] = list of [r-1, j]
    const parents = [];
    for (let r = 0; r < rows.length; r++) {
      parents.push([]);
      for (let i = 0; i < N; i++) {
        let p = [];
        if (r === 1) p = i >= 1 ? [[0, i - 1]] : [];
        else if (r >= 2 && r <= 4) {
          const d = dil[r - 2];
          p = [[r - 1, i]];
          if (i - d >= 0) p.push([r - 1, i - d]);
        } else if (r === 5) p = [[4, i]];
        parents[r].push(p);
      }
    }
    const edgeEls = {};
    const nodeEls = [];
    for (let r = 1; r < rows.length; r++) {
      for (let i = 0; i < N; i++) {
        parents[r][i].forEach(function (pj) {
          const e = el("line", { x1: x0 + pj[1] * dx, y1: rowY(pj[0]), x2: x0 + i * dx, y2: rowY(r), class: "rf-edge" }, svg);
          edgeEls[r + ":" + i + ":" + pj[0] + ":" + pj[1]] = e;
        });
      }
    }
    for (let r = 0; r < rows.length; r++) {
      el("text", { x: x0 - 18, y: rowY(r) + 4, class: "sm r" + (r >= 2 && r <= 4 ? " mono" : "") }, svg, rows[r]);
      nodeEls.push([]);
      for (let i = 0; i < N; i++) {
        const isZero = r === 1 && i === 0;
        const c = el("circle", {
          cx: x0 + i * dx, cy: rowY(r), r: r === 0 || r === 5 ? 7 : 5,
          class: isZero ? "plain" : r === 0 ? "tensor" : r === 5 ? "loss" : "module",
        }, svg);
        if (r === 5) {
          c.setAttribute("data-t", String(i));
          c.style.cursor = "pointer";
        }
        nodeEls[r].push(c);
      }
    }
    el("text", { x: x0, y: rowY(1) - 10, class: "xs c muted" }, svg, "0");
    el("text", { x: x0 + (N - 1) * dx, y: 292, class: "xs c muted" }, svg, "t →");
    const target = el("rect", { y: rowY(0) - 12, width: 24, height: 24, rx: 4, style: "fill:none;stroke:var(--arch-loss);stroke-width:2.5" }, svg);

    function select(t) {
      Object.keys(edgeEls).forEach(function (k) { edgeEls[k].classList.remove("on"); });
      nodeEls.forEach(function (row) {
        row.forEach(function (n) { n.classList.remove("hl-node"); n.classList.remove("dim"); });
      });
      const seen = {};
      const stack = [[5, t]];
      while (stack.length) {
        const cur = stack.pop();
        const key = cur[0] + ":" + cur[1];
        if (seen[key]) continue;
        seen[key] = true;
        nodeEls[cur[0]][cur[1]].classList.add("hl-node");
        parents[cur[0]][cur[1]].forEach(function (pj) {
          edgeEls[cur[0] + ":" + cur[1] + ":" + pj[0] + ":" + pj[1]].classList.add("on");
          stack.push(pj);
        });
      }
      nodeEls[0].forEach(function (n, i) { if (i >= t) n.classList.add("dim"); });
      nodeEls[0][t].classList.remove("hl-node");
      target.setAttribute("x", String(x0 + t * dx - 12));
      const lo = Math.max(0, t - 8);
      info.innerHTML = "logits at <b>t = " + t + "</b> see codes <b>" + (t === 0 ? "none (only the zero pad)" : lo + " … " + (t - 1)) +
        "</b> and are trained with cross-entropy to predict code <b>c<sub>" + t + "</sub></b> (boxed). Receptive field of this toy stack: 1 + 1 + 2 + 4 = <b>8</b> codes; the real prior (10 layers, d = 1 … 512) sees <b>1024</b> ≥ 625.";
    }
    let t = 13;
    let auto = !reduceMotion;
    svg.addEventListener("mouseover", function (e) {
      const tt = e.target.getAttribute && e.target.getAttribute("data-t");
      if (tt !== null && tt !== undefined) { auto = false; select(parseInt(tt, 10)); }
    });
    svg.addEventListener("click", function (e) {
      const tt = e.target.getAttribute && e.target.getAttribute("data-t");
      if (tt !== null && tt !== undefined) { auto = false; select(parseInt(tt, 10)); }
    });
    select(t);
    const timer = setInterval(function () {
      if (!auto || !document.body.contains(svg)) { clearInterval(timer); return; }
      t = (t + 1) % N;
      select(t);
    }, 1100);
  }

  // ------------------------------------------------------------------
  // Widget: autoregressive sampling with temperature
  // ------------------------------------------------------------------

  function widgetAR(fig) {
    const svg = fig.querySelector("svg");
    const temp = fig.querySelector("input[name=temperature]");
    const tempOut = fig.querySelector("output[name=temperature]");
    const playBtn = fig.querySelector("button[data-action=play]");
    const restartBtn = fig.querySelector("button[data-action=restart]");
    const N = 30, K = 12;
    const W = 760, H = 300;
    const sx = 20, cw = (W - 40) / N;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    el("text", { x: sx, y: 18, class: "sm" }, svg, "codes being sampled (toy: 30 of 625, codebook 12 of 512)");
    const cells = [];
    for (let i = 0; i < N; i++) {
      const rct = el("rect", { x: sx + i * cw, y: 28, width: cw - 3, height: 26, rx: 3, class: "plain" }, svg);
      const txt = el("text", { x: sx + i * cw + (cw - 3) / 2, y: 45, class: "xs c mono" }, svg, "");
      cells.push([rct, txt]);
    }
    const cursor = el("path", { class: "edge loss" }, svg);
    el("text", { x: 20, y: 98, class: "sm" }, svg, "softmax(logits / T) at the current position");
    const bx = 20, by = 240, bw = 26, bh = 120;
    const bars = [];
    for (let k = 0; k < K; k++) {
      const rct = el("rect", { x: bx + k * (bw + 6), width: bw, y: by, height: 0, style: "fill:" + codeColor(k, K) + ";opacity:0.55" }, svg);
      el("text", { x: bx + k * (bw + 6) + bw / 2, y: by + 14, class: "xs c mono muted" }, svg, String(k));
      bars.push(rct);
    }
    el("line", { x1: bx - 4, x2: bx + K * (bw + 6), y1: by, y2: by, class: "gridline" }, svg);
    // Decoder + output
    el("rect", { x: 450, y: 150, width: 90, height: 44, rx: 6, class: "frozen" }, svg);
    el("text", { x: 495, y: 170, class: "sm c" }, svg, "VQ-VAE");
    el("text", { x: 495, y: 185, class: "xs c muted" }, svg, "decode (frozen)");
    el("text", { x: 444, y: 140, class: "xs r muted" }, svg, "all codes →");
    el("path", { d: "M420,172 L448,172", class: "edge flow" }, svg);
    el("path", { d: "M542,172 L566,172", class: "edge flow" }, svg);
    const out = el("path", { "data-ecg": "570,135,170,74", "data-seed": "11", "data-lead": "II", "data-beats": "2.5", class: "trace" }, svg);
    const outLabel = el("text", { x: 655, y: 228, class: "xs c muted" }, svg, "ECG (B, 12, 5000)");
    renderTraces(svg);

    let codes = [];
    let pos = 0;
    let playing = !reduceMotion;
    let seed = 1;
    let timer = null;

    function logitsAt(i) {
      // Toy "learned" conditional: prefers codes near the previous code.
      const r = rng(seed * 131 + i);
      const prev = i ? codes[i - 1] : 0;
      const l = [];
      for (let k = 0; k < K; k++) {
        const dist = Math.min(Math.abs(k - prev), K - Math.abs(k - prev));
        l.push(2.2 - 0.9 * dist + 1.2 * r());
      }
      return l;
    }
    function draw(p, chosen) {
      p.forEach(function (v, k) {
        bars[k].setAttribute("y", String(by - v * bh));
        bars[k].setAttribute("height", String(v * bh));
        bars[k].style.opacity = k === chosen ? "1" : "0.45";
      });
    }
    function reset() {
      codes = [];
      pos = 0;
      seed++;
      cells.forEach(function (c) {
        c[0].setAttribute("class", "plain");
        c[0].removeAttribute("style");
        c[1].textContent = "";
      });
      out.style.opacity = "0.15";
      outLabel.style.opacity = "0.3";
    }
    function step() {
      if (pos >= N) {
        out.style.opacity = "1";
        outLabel.style.opacity = "1";
        cursor.setAttribute("d", "");
        if (playing) timer = setTimeout(function () { reset(); step(); }, 2600);
        return;
      }
      const T = parseFloat(temp.value);
      const p = softmax(logitsAt(pos), T);
      const k = categorical(p, rng(seed * 7919 + pos)());
      codes.push(k);
      draw(p, k);
      const c = cells[pos];
      c[0].setAttribute("class", "");
      c[0].setAttribute("style", "fill:" + codeColor(k, K) + ";opacity:0.85");
      c[1].textContent = String(k);
      const cx = sx + pos * cw + (cw - 3) / 2;
      cursor.setAttribute("d", "M" + cx + ",80 L" + cx + ",58");
      pos++;
      if (playing) timer = setTimeout(step, 380);
    }
    function setPlaying(v) {
      playing = v;
      playBtn.textContent = v ? "Pause" : "Play";
      clearTimeout(timer);
      if (v) step();
    }
    temp.addEventListener("input", function () {
      tempOut.textContent = parseFloat(temp.value).toFixed(1);
      if (pos > 0 && pos <= N) draw(softmax(logitsAt(pos - 1 < 0 ? 0 : pos - 1), parseFloat(temp.value)), codes[pos - 1]);
    });
    playBtn.addEventListener("click", function () { setPlaying(!playing); });
    restartBtn.addEventListener("click", function () { clearTimeout(timer); reset(); if (playing) step(); else step(); });
    tempOut.textContent = parseFloat(temp.value).toFixed(1);
    reset();
    if (reduceMotion) { while (pos < N) step(); }
    setPlaying(playing);
  }

  // ------------------------------------------------------------------
  // Diagram: Pulse2Pulse generator (U-Net)
  // ------------------------------------------------------------------

  function renderUNet(svg) {
    clear(svg);
    const W = 760, H = 420;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const lx = 215, rx = 545, y0 = 98, dy = 42;
    const bw = function (len) { return 52 + 14 * Math.log10(len); };
    const enc = [
      { name: "input z", ch: 8, len: 5000, cls: "tensor", info: "<b>Noise z</b> — (B, 8, 5000), uniform in [−1, 1]. Same shape as an ECG, not a small latent vector." },
      { name: "conv_1", ch: 10, len: 2500, s: 2, info: "<b>conv_1</b> — Conv1d(8 → 10, k=25, stride=2, padding=12) + LeakyReLU. Not used as a skip." },
      { name: "conv_2", ch: 25, len: 500, s: 5, info: "<b>conv_2</b> — Conv1d(10 → 25, k=25, stride=5) + LeakyReLU." },
      { name: "conv_3", ch: 50, len: 250, s: 2, info: "<b>conv_3</b> — Conv1d(25 → 50, k=25, stride=2) + LeakyReLU." },
      { name: "conv_4", ch: 150, len: 50, s: 5, info: "<b>conv_4</b> — Conv1d(50 → 150, k=25, stride=5) + LeakyReLU." },
      { name: "conv_5", ch: 250, len: 10, s: 5, info: "<b>conv_5</b> — Conv1d(150 → 250, k=25, stride=5) + LeakyReLU." },
    ];
    const dec = [
      { name: "deconv_7", ch: 8, len: 5000, cls: "tensor", up: 2, info: "<b>deconv_7</b> — Upsample ×2 → zero-pad 12 → Conv1d(10 → 8, k=25) → <b>tanh</b>. Output (B, 8, 5000) in [−1, 1]." },
      { name: "deconv_6", ch: 10, len: 2500, up: 5, info: "<b>deconv_6</b> — concat(x, conv_2) (25 + 25 = 50 ch) → Upsample ×5 → Conv1d(50 → 10, k=25) + ReLU." },
      { name: "deconv_5", ch: 25, len: 500, up: 2, info: "<b>deconv_5</b> — concat(x, conv_3) (50 + 50 = 100 ch) → Upsample ×2 → Conv1d(100 → 25, k=25) + ReLU." },
      { name: "deconv_3", ch: 50, len: 250, up: 5, info: "<b>deconv_3</b> — concat(x, conv_4) (150 + 150 = 300 ch) → Upsample ×5 → Conv1d(300 → 50, k=25) + ReLU." },
      { name: "deconv_2", ch: 150, len: 50, up: 5, info: "<b>deconv_2</b> — concat(x, conv_5) (250 + 250 = 500 ch) → Upsample ×5 → Conv1d(500 → 150, k=25) + ReLU." },
      { name: "deconv_1", ch: 250, len: 10, up: 5, info: "<b>deconv_1</b> — Upsample ×5 → zero-pad 12 → Conv1d(250 → 250, k=25) + ReLU." },
    ];
    function bar(cx, y, item, side) {
      const w = bw(item.len);
      const g = el("g", { "data-info": item.info, "data-hover": "" }, svg);
      el("rect", { x: cx - w / 2, y: y - 10, width: w, height: 20, rx: 3, class: item.cls || "module" }, g);
      el("text", { x: cx, y: y + 4, class: "xs c mono" }, g, item.ch + " × " + item.len);
      el("text", { x: side < 0 ? cx - w / 2 - 8 : cx + w / 2 + 8, y: y + 4, class: "sm mono " + (side < 0 ? "r" : "") }, g, item.name);
      return w;
    }
    enc.forEach(function (item, i) {
      const y = y0 + i * dy;
      bar(lx, y, item, -1);
      if (i > 0) {
        el("path", { d: "M" + lx + "," + (y - dy + 11) + " L" + lx + "," + (y - 12), class: "edge flow" }, svg);
        el("text", { x: lx + 8 + bw(enc[i - 1].len) / 2 - 40, y: y - dy / 2 + 4, class: "xs muted" }, svg, "↓ s=" + item.s);
      }
    });
    dec.forEach(function (item, i) {
      const y = y0 + i * dy;
      bar(rx, y, item, 1);
      if (i < dec.length - 1) {
        el("path", { d: "M" + rx + "," + (y + dy - 11) + " L" + rx + "," + (y + 12), class: "edge flow" }, svg);
        el("text", { x: rx - 8 - bw(item.len) / 2 + 40, y: y + dy / 2 + 4, class: "xs r muted" }, svg, "↑ ×" + item.up);
      }
    });
    // Bottleneck conv_6
    const by = y0 + 6 * dy;
    el("text", { x: (lx + rx) / 2, y: by + 26, class: "sm c mono" }, svg, "conv_6");
    bar((lx + rx) / 2, by, { name: "", ch: 250, len: 2, info: "<b>conv_6</b> — Conv1d(250 → 250, k=25, stride=5) + LeakyReLU: the bottleneck, length 2." }, 1);
    el("path", { d: "M" + lx + "," + (y0 + 5 * dy + 11) + " Q" + lx + "," + by + " " + ((lx + rx) / 2 - 42) + "," + by, class: "edge flow" }, svg);
    el("path", { d: "M" + ((lx + rx) / 2 + 42) + "," + by + " Q" + rx + "," + by + " " + rx + "," + (y0 + 5 * dy + 12), class: "edge flow" }, svg);
    el("text", { x: rx + 12, y: by - 8, class: "xs muted" }, svg, "↑ ×5 (deconv_1)");
    // Skips: conv_k joins the decoder at the same length
    [[5, "conv_5"], [4, "conv_4"], [3, "conv_3"], [2, "conv_2"]].forEach(function (pair) {
      const i = pair[0];
      const y = y0 + i * dy;
      const xa = lx + bw(enc[i].len) / 2 + 2;
      const xb = rx - bw(dec[i].len) / 2 - 18;
      el("path", { d: "M" + xa + "," + y + " L" + xb + "," + y, class: "edge skip" }, svg);
      el("text", { x: xb + 8, y: y + 4, class: "sm c" }, svg, "⊕");
    });
    el("text", { x: (lx + rx) / 2, y: y0 + 2 * dy - 14, class: "xs c muted" }, svg, "skip: concat on channels");
    el("text", { x: (lx + rx) / 2, y: y0 + 1 * dy + 4, class: "xs c muted" }, svg, "(conv_1: no skip)");
    // Traces
    el("path", { "data-ecg": (lx - 70) + ",14,140,52", "data-kind": "noise", "data-seed": "4", class: "trace noise" }, svg);
    el("text", { x: lx, y: 80, class: "xs c muted" }, svg, "U(−1, 1) noise");
    el("path", { "data-ecg": (rx - 70) + ",14,140,52", "data-seed": "21", "data-lead": "II", class: "trace fake draw" }, svg);
    el("text", { x: rx, y: 80, class: "xs c muted" }, svg, "generated ECG");
    el("text", { x: 30, y: 40, class: "lg" }, svg, "Encoder");
    el("text", { x: W - 30, y: 40, class: "lg r" }, svg, "Decoder");
    renderTraces(svg);
  }

  // ------------------------------------------------------------------
  // Diagram: Pulse2Pulse critic
  // ------------------------------------------------------------------

  function renderCritic(svg) {
    clear(svg);
    const W = 780, H = 250, cy = 115;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const layers = [
      ["conv1", 8, 50, 2499, 2], ["conv2", 50, 100, 1249, 2], ["conv3", 100, 250, 624, 2], ["conv4", 250, 500, 311, 2],
      ["conv5", 500, 1000, 78, 4], ["conv6", 1000, 1250, 19, 4], ["conv7", 1250, 5000, 5, 4],
    ];
    const hOf = function (len) { return 24 + 30 * Math.log10(len); };
    const wOf = function (ch) { return 6 + 3.2 * Math.log2(ch / 8 + 1); };
    let x = 16;
    const gi = el("g", { "data-info": "<b>Input</b> — real ECG or generated ECG, (B, 8, 5000).", "data-hover": "" }, svg);
    el("rect", { x: x, y: cy - hOf(5000) / 2, width: 14, height: hOf(5000), class: "tensor" }, gi);
    el("text", { x: x + 7, y: cy - hOf(5000) / 2 - 20, class: "xs c mono muted" }, gi, "8×5000");
    x += 26;
    layers.forEach(function (L, i) {
      const h = hOf(L[3]), w = wOf(L[2]);
      const g = el("g", { "data-info": "<b>" + L[0] + "</b> — Conv1d(" + L[1] + " → " + L[2] + ", k=25, stride=" + L[4] + ", padding=11) + LeakyReLU(0.2) → (B, " + L[2] + ", " + L[3] + ")" + (i < 6 ? ", then PhaseShuffle(2)." : ". No phase shuffle after the last conv."), "data-hover": "" }, svg);
      el("path", { d: "M" + (x - 10) + "," + cy + " L" + (x - 1) + "," + cy, class: "edge" }, svg);
      el("rect", { x: x, y: cy - h / 2, width: w, height: h, rx: 2, class: "module" }, g);
      el("text", { x: x + w / 2, y: cy - h / 2 - 6, class: "xs c mono" }, g, L[0]);
      el("text", { x: x + w / 2, y: cy + h / 2 + 14, class: "xs c mono muted" }, g, L[2] + "×" + L[3]);
      x += w + 12;
      if (i < 6) {
        const gp = el("g", { "data-info": "<b>PhaseShuffle(2)</b> — shifts each example by a random k ∈ {−2, …, 2} samples, reflection-padding the gap (see the demo below).", "data-hover": "" }, svg);
        el("rect", { x: x, y: cy - 7, width: 14, height: 14, transform: "rotate(45 " + (x + 7) + " " + cy + ")", class: "ok" }, gp);
        x += 30;
      }
    });
    const gf = el("g", { "data-info": "<b>Flatten + Linear</b> — 5000 × 5 = 25 000 features → nn.Linear(25000, 1), created lazily on the first forward pass. No sigmoid: the critic outputs an unbounded score.", "data-hover": "" }, svg);
    el("path", { d: "M" + (x - 10) + "," + cy + " L" + (x + 4) + "," + cy, class: "edge" }, svg);
    el("rect", { x: x + 6, y: cy - 22, width: 56, height: 44, rx: 6, class: "module" }, gf);
    el("text", { x: x + 34, y: cy - 3, class: "xs c" }, gf, "flatten");
    el("text", { x: x + 34, y: cy + 11, class: "xs c" }, gf, "Linear→1");
    el("path", { d: "M" + (x + 64) + "," + cy + " L" + (x + 84) + "," + cy, class: "edge flow" }, svg);
    const gs = el("g", { "data-info": "<b>Score D(x)</b> — (B, 1). Higher means “more real”. Only differences of scores matter (Wasserstein estimate).", "data-hover": "" }, svg);
    el("circle", { cx: x + 100, cy: cy, r: 15, class: "loss" }, gs);
    el("text", { x: x + 100, y: cy + 4, class: "xs c" }, gs, "D(x)");
    el("text", { x: 16, y: 220, class: "xs muted" }, svg, "◇ PhaseShuffle   ·   bar height ∝ log(length), width ∝ log(channels)   ·   ~204 M parameters at model_size = 50, mostly conv7");
    svg.setAttribute("viewBox", "0 0 " + Math.max(W, x + 130) + " " + H);
  }

  // ------------------------------------------------------------------
  // Widget: PhaseShuffle demo
  // ------------------------------------------------------------------

  function widgetPhase(fig) {
    const svg = fig.querySelector("svg");
    const W = 760, H = 200, n = 36, x0 = 60, bw = (W - x0 - 20) / n;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const src = ecgSeries(n, { seed: 5, lead: "II", beats: 1.1 }).map(function (v) { return v * 0.9 + 0.1; });
    function row(y, label) {
      el("text", { x: x0 - 10, y: y - 20, class: "sm r" }, svg, label);
      el("line", { x1: x0, x2: W - 20, y1: y, y2: y, class: "gridline" }, svg);
      const bars = [];
      for (let i = 0; i < n; i++) bars.push(el("rect", { x: x0 + i * bw + 1, width: bw - 2, class: "tensor" }, svg));
      return bars;
    }
    const top = row(85, "x");
    const bot = row(180, "shuffled");
    const kText = el("text", { x: W - 20, y: 112, class: "sm r mono" }, svg, "");
    function set(bars, vals, padded, y) {
      bars.forEach(function (b, i) {
        const v = vals[i];
        const h = Math.abs(v) * 60;
        b.setAttribute("y", String(v >= 0 ? y - h : y));
        b.setAttribute("height", String(Math.max(h, 1)));
        b.setAttribute("class", padded.indexOf(i) >= 0 ? "loss" : "tensor");
      });
    }
    set(top, src, [], 85);
    function shuffle(k) {
      const out = [];
      const padded = [];
      for (let i = 0; i < n; i++) {
        if (k > 0) {
          if (i < k) { out.push(src[k - i]); padded.push(i); } else out.push(src[i - k]);
        } else if (k < 0) {
          const m = -k;
          if (i < n - m) out.push(src[i + m]);
          else { out.push(src[n - 2 - (i - (n - m))]); padded.push(i); }
        } else out.push(src[i]);
      }
      set(bot, out, padded, 180);
      kText.textContent = "k = " + (k > 0 ? "+" : "") + k + (k ? "  (orange = reflection padding)" : "  (no shift)");
    }
    const ks = [2, -1, 0, -2, 1];
    let i = 0;
    shuffle(ks[0]);
    if (reduceMotion) return;
    const timer = setInterval(function () {
      if (!document.body.contains(svg)) { clearInterval(timer); return; }
      i = (i + 1) % ks.length;
      shuffle(ks[i]);
    }, 1500);
  }

  // ------------------------------------------------------------------
  // Widget: WGAN-GP interpolation
  // ------------------------------------------------------------------

  function widgetGP(fig) {
    const svg = fig.querySelector("svg");
    const slider = fig.querySelector("input[name=alpha]");
    const outp = fig.querySelector("output[name=alpha]");
    const W = 760, H = 250, n = 500;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const real = ecgSeries(n, { seed: 2, lead: "II", beats: 3 });
    const fake = ecgSeries(n, { seed: 9, lead: "V2", beats: 3.6, kind: "fake" });
    const rows = [
      [30, "real x", "trace"], [110, "x̂ = αx + (1−α)G(z)", "trace mix"], [190, "fake G(z)", "trace fake"],
    ];
    const paths = rows.map(function (r) {
      el("text", { x: 180, y: r[0] + 30, class: "sm r" }, svg, r[1]);
      return el("path", { class: r[2], style: "stroke-width:1.6" }, svg);
    });
    paths[0].setAttribute("d", seriesPath(real, 195, 10, 540, 52, -1.1, 1.2));
    paths[2].setAttribute("d", seriesPath(fake, 195, 170, 540, 52, -1.1, 1.2));
    function update() {
      const a = parseFloat(slider.value);
      const mix = real.map(function (v, i) { return a * v + (1 - a) * fake[i]; });
      paths[1].setAttribute("d", seriesPath(mix, 195, 90, 540, 52, -1.1, 1.2));
      outp.textContent = a.toFixed(2);
    }
    let auto = !reduceMotion;
    slider.addEventListener("input", function () { auto = false; update(); });
    update();
    const r = rng(17);
    const timer = setInterval(function () {
      if (!auto || !document.body.contains(svg)) { clearInterval(timer); return; }
      slider.value = r().toFixed(2);
      update();
    }, 1600);
  }

  // ------------------------------------------------------------------
  // Widget: n_critic schedule
  // ------------------------------------------------------------------

  function widgetSchedule(fig) {
    const svg = fig.querySelector("svg");
    const slider = fig.querySelector("input[name=ncritic]");
    const outp = fig.querySelector("output[name=ncritic]");
    const info = fig.querySelector(".arch-info");
    const N = 20, W = 760, H = 150, x0 = 150, cw = (W - x0 - 20) / N;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    let cur = 0;
    let cellsD = [], cellsG = [], cursor;
    function build() {
      clear(svg);
      cellsD = []; cellsG = [];
      const nc = parseInt(slider.value, 10);
      outp.textContent = String(nc);
      el("text", { x: x0 - 12, y: 52, class: "sm r" }, svg, "critic step (opt_d)");
      el("text", { x: x0 - 12, y: 96, class: "sm r" }, svg, "generator step (opt_g)");
      el("text", { x: x0 - 12, y: 20, class: "xs r muted" }, svg, "batch_idx");
      for (let i = 0; i < N; i++) {
        el("text", { x: x0 + i * cw + cw / 2, y: 20, class: "xs c mono muted" }, svg, String(i));
        cellsD.push(el("rect", { x: x0 + i * cw + 2, y: 36, width: cw - 4, height: 24, rx: 3, class: "module" }, svg));
        const g = (i + 1) % nc === 0;
        cellsG.push(el("rect", { x: x0 + i * cw + 2, y: 80, width: cw - 4, height: 24, rx: 3, class: g ? "ok" : "plain" }, svg));
      }
      cursor = el("rect", { y: 30, width: cw, height: 80, rx: 4, style: "fill:none;stroke:var(--arch-loss);stroke-width:2.5" }, svg);
      el("text", { x: x0, y: 135, class: "xs muted" }, svg, "Every batch updates the critic; the generator is updated when (batch_idx + 1) % n_critic == 0.");
      move(cur);
    }
    function move(i) {
      const nc = parseInt(slider.value, 10);
      cursor.setAttribute("x", String(x0 + i * cw));
      const g = (i + 1) % nc === 0;
      info.innerHTML = "<b>batch_idx = " + i + "</b>: L<sub>D</sub> = D(G(z)) − D(x) + GP → opt_d.step()" +
        (g ? "; then fresh z, L<sub>G</sub> = −D(G(z)) → <b>opt_g.step()</b>" : "; generator skipped (logged g_loss = 0)");
    }
    slider.addEventListener("input", build);
    build();
    if (reduceMotion) return;
    const timer = setInterval(function () {
      if (!document.body.contains(svg)) { clearInterval(timer); return; }
      cur = (cur + 1) % N;
      move(cur);
    }, 900);
  }

  // ------------------------------------------------------------------
  // Diffusion / flow-matching helpers
  // ------------------------------------------------------------------

  // Standard-normal PRNG (Box–Muller on top of rng()).
  function gaussRng(seed) {
    const r = rng(seed);
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0;
      while (u === 0) u = r();
      const v = r();
      const m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
  }

  function gaussSeries(n, seed) {
    const g = gaussRng(seed);
    const out = [];
    for (let i = 0; i < n; i++) out.push(g());
    return out;
  }

  function smooth(values, win) {
    const out = [];
    for (let i = 0; i < values.length; i++) {
      let s = 0, c = 0;
      for (let j = i - win; j <= i + win; j++) {
        if (j >= 0 && j < values.length) { s += values[j]; c++; }
      }
      out.push(s / c);
    }
    return out;
  }

  // alphas_cumprod exactly as ecgen.processes.ddpm._schedule.
  function alphasCumprod(name, T) {
    const ac = [];
    let p = 1;
    if (name === "linear") {
      for (let i = 0; i < T; i++) {
        p *= 1 - (1e-4 + ((0.02 - 1e-4) * i) / (T - 1));
        ac.push(p);
      }
      return ac;
    }
    const s = 0.008;
    const f = function (x) { return Math.pow(Math.cos((((x / T) + s) / (1 + s)) * Math.PI / 2), 2); };
    const f0 = f(0);
    for (let i = 0; i < T; i++) {
      const beta = Math.min(1 - (f(i + 1) / f0) / (f(i) / f0), 0.999);
      p *= 1 - beta;
      ac.push(p);
    }
    return ac;
  }

  // Min-SNR weight for v-prediction (DDPMProcess.training_pair).
  function minSnrV(ac, gamma) {
    const snr = ac / (1 - ac);
    return Math.min(snr, gamma) / (snr + 1);
  }

  // DDPMProcess.timesteps: evenly spaced, rounded, unique, descending.
  function ddimTimesteps(steps, T) {
    steps = Math.max(1, Math.min(steps, T));
    const seen = {};
    const out = [];
    for (let i = 0; i < steps; i++) {
      const v = steps === 1 ? 0 : Math.round(((T - 1) * i) / (steps - 1));
      if (!seen[v]) { seen[v] = true; out.push(v); }
    }
    return out.sort(function (a, b) { return b - a; });
  }

  function fmt(v, d) { return v.toFixed(d === undefined ? 2 : d); }

  // ------------------------------------------------------------------
  // Widget: forward noising with the cosine / linear schedule
  // ------------------------------------------------------------------

  function widgetNoising(fig) {
    const svg = fig.querySelector("svg");
    const slider = fig.querySelector("input[name=t]");
    const outp = fig.querySelector("output[name=t]");
    const info = fig.querySelector(".arch-info");
    const buttons = fig.querySelectorAll("button[data-schedule]");
    const W = 760, H = 270, T = 1000, n = 400;
    const tx = 100, tw = 290, rowH = 64, rows = [18, 100, 182];
    const vmin = -3.4, vmax = 3.4;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const sched = { cosine: alphasCumprod("cosine", T), linear: alphasCumprod("linear", T) };
    let name = "cosine";
    const x0 = ecgSeries(n, { seed: 2, lead: "II", beats: 3 }).map(function (v) { return 1.7 * v; });
    const eps = gaussSeries(n, 5);

    [["x₀", "data (schematic)"], ["ε", "N(0, I)"], ["x_t", "noisy input"]].forEach(function (lab, i) {
      el("rect", { x: tx, y: rows[i], width: tw, height: rowH, rx: 3, class: "plain" }, svg);
      el("text", { x: tx - 10, y: rows[i] + rowH / 2, class: "sm r" }, svg, lab[0]);
      el("text", { x: tx - 10, y: rows[i] + rowH / 2 + 13, class: "xs r muted" }, svg, lab[1]);
    });
    el("path", { d: seriesPath(x0, tx + 2, rows[0] + 2, tw - 4, rowH - 4, vmin, vmax), class: "trace" }, svg);
    el("path", { d: seriesPath(eps, tx + 2, rows[1] + 2, tw - 4, rowH - 4, vmin, vmax), class: "trace noise" }, svg);
    const xtPath = el("path", { class: "trace mix" }, svg);
    const ca = el("text", { x: tx + tw + 10, y: rows[0] + rowH / 2 + 4, class: "sm mono" }, svg, "");
    const cb = el("text", { x: tx + tw + 10, y: rows[1] + rowH / 2 + 4, class: "sm mono" }, svg, "");
    el("text", { x: tx + tw + 10, y: rows[2] + rowH / 2 + 4, class: "sm mono" }, svg, "= sum");

    // Right: schedule plot
    const px = 548, py = 26, pw = 192, ph = 190;
    el("rect", { x: px, y: py, width: pw, height: ph, class: "plain" }, svg);
    [0, 0.5, 1].forEach(function (v) {
      el("text", { x: px - 6, y: py + ph - v * ph + 4, class: "xs r mono muted" }, svg, String(v));
    });
    [0, 500, 999].forEach(function (v) {
      el("text", { x: px + (v / (T - 1)) * pw, y: py + ph + 13, class: "xs c mono muted" }, svg, String(v));
    });
    el("text", { x: px + pw / 2, y: py + ph + 28, class: "xs c muted" }, svg, "step t");
    function curve(fn) {
      let d = "";
      for (let i = 0; i < T; i += 5) {
        d += (i ? "L" : "M") + (px + (i / (T - 1)) * pw).toFixed(1) + "," + (py + ph - fn(i) * ph).toFixed(1);
      }
      return d;
    }
    const pathLin = el("path", { class: "edge noarrow", style: "stroke-dasharray:4 3" }, svg);
    const pathCos = el("path", { class: "trace", style: "stroke-width:2" }, svg);
    const pathW = el("path", { class: "trace", style: "stroke:var(--arch-loss);stroke-width:2" }, svg);
    const marker = el("line", { y1: py, y2: py + ph, style: "stroke:var(--arch-grad);stroke-width:1.5" }, svg);
    const dot = el("circle", { r: 4, style: "fill:var(--arch-data)" }, svg);
    const dotW = el("circle", { r: 4, style: "fill:var(--arch-loss)" }, svg);
    const lx = px + pw - 8;
    el("text", { x: lx, y: py + 16, class: "xs r", style: "fill:var(--arch-data)" }, svg, "ᾱ_t (selected schedule)");
    el("text", { x: lx, y: py + 30, class: "xs r muted" }, svg, "- - other schedule");
    el("text", { x: lx, y: py + 44, class: "xs r", style: "fill:var(--arch-loss)" }, svg, "min-SNR weight (v, γ = 5)");

    function update() {
      const t = parseInt(slider.value, 10);
      const ac = sched[name][t];
      const a = Math.sqrt(ac), s = Math.sqrt(1 - ac);
      const xt = x0.map(function (v, i) { return a * v + s * eps[i]; });
      xtPath.setAttribute("d", seriesPath(xt, tx + 2, rows[2] + 2, tw - 4, rowH - 4, vmin, vmax));
      ca.textContent = "× √ᾱ = " + fmt(a);
      cb.textContent = "× √(1−ᾱ) = " + fmt(s);
      outp.textContent = String(t);
      const other = name === "cosine" ? "linear" : "cosine";
      pathCos.setAttribute("d", curve(function (i) { return sched[name][i]; }));
      pathLin.setAttribute("d", curve(function (i) { return sched[other][i]; }));
      pathW.setAttribute("d", curve(function (i) { return minSnrV(sched[name][i], 5); }));
      const mx = px + (t / (T - 1)) * pw;
      marker.setAttribute("x1", mx); marker.setAttribute("x2", mx);
      dot.setAttribute("cx", mx); dot.setAttribute("cy", py + ph - ac * ph);
      const w = minSnrV(ac, 5);
      dotW.setAttribute("cx", mx); dotW.setAttribute("cy", py + ph - w * ph);
      const snr = ac / (1 - ac);
      info.innerHTML = "<b>t = " + t + "</b> (" + name + ") · ᾱ<sub>t</sub> = " + fmt(ac, 3) +
        " · x<sub>t</sub> = " + fmt(a) + "·x<sub>0</sub> + " + fmt(s) + "·ε · SNR = ᾱ/(1−ᾱ) = " +
        (snr > 1000 ? snr.toExponential(1) : fmt(snr)) + " · loss weight min(SNR, 5)/(SNR + 1) = <b>" + fmt(w) + "</b>";
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        name = b.dataset.schedule;
        buttons.forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
        update();
      });
    });
    buttons.forEach(function (o) { o.setAttribute("aria-pressed", String(o.dataset.schedule === name)); });
    let auto = !reduceMotion;
    slider.addEventListener("input", function () { auto = false; update(); });
    update();
    let dir = 1;
    const timer = setInterval(function () {
      if (!auto || !document.body.contains(svg)) { clearInterval(timer); return; }
      let t = parseInt(slider.value, 10) + dir * 37;
      if (t > 999) { t = 999; dir = -1; }
      if (t < 0) { t = 0; dir = 1; }
      slider.value = String(t);
      update();
    }, 220);
  }

  // ------------------------------------------------------------------
  // Diagram: UNet1d (diffusion / flow backbone)
  // ------------------------------------------------------------------

  function renderUNet1d(svg) {
    clear(svg);
    const W = 760, H = 500;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const LX = 210, RX = 550, MX = (LX + RX) / 2;
    const ys = [116, 178, 240, 302], midY = 364, bh = 32;
    const bw = function (ch) { return 120 + (ch / 64) * 14; };
    const lens = [5000, 2500, 1250, 625];
    const widths = [64, 128, 256, 256];
    const ins = [[64, 64], [64, 128], [128, 256], [256, 256]];
    const upIns = [[64 + 64, 64], [128 + 64, 128], [256 + 128, 256], [256 + 256, 256]];
    function block(cx, y, ch, text, info, cls) {
      const w = bw(ch);
      const g = el("g", { "data-info": info, "data-hover": "" }, svg);
      el("rect", { x: cx - w / 2, y: y - bh / 2, width: w, height: bh, rx: 4, class: cls || "module" }, g);
      el("text", { x: cx, y: y + 4, class: "sm c" }, g, text);
      return w;
    }
    // input / output traces
    el("path", { "data-ecg": (LX - 70) + ",4,140,36", "data-kind": "noise", "data-seed": "4", class: "trace mix" }, svg);
    el("text", { x: LX - 80, y: 20, class: "sm r" }, svg, "x_t");
    el("text", { x: LX - 80, y: 34, class: "xs r mono muted" }, svg, "(B, 12, 5000)");
    el("path", { "data-ecg": (RX - 70) + ",4,140,36", "data-kind": "noise", "data-seed": "9", class: "trace fake" }, svg);
    el("text", { x: RX + 80, y: 20, class: "sm" }, svg, "v̂ (or ε̂, x̂₀)");
    el("text", { x: RX + 80, y: 34, class: "xs mono muted" }, svg, "(B, 12, 5000)");
    // conv_in / conv_out
    const gi = el("g", { "data-info": "<b>conv_in</b> — Conv1d(12 + dense_channels → 64, k=3, padding=1). A dense condition (e.g. masked_signal: [x·m, m], 24 channels) is concatenated to x_t on the channel axis first.", "data-hover": "" }, svg);
    el("rect", { x: LX - 60, y: 52, width: 120, height: 24, rx: 4, class: "module" }, gi);
    el("text", { x: LX, y: 68, class: "xs c" }, gi, "[x_t ; c.dense] → conv_in");
    el("path", { d: "M" + LX + ",42 L" + LX + ",50", class: "edge flow" }, svg);
    el("path", { d: "M" + LX + ",78 L" + LX + "," + (ys[0] - bh / 2 - 2), class: "edge flow" }, svg);
    const go = el("g", { "data-info": "<b>Output head</b> — GroupNorm(8, 64) → SiLU → Conv1d(64 → 12, k=3). conv_out is <b>zero-initialised</b>, so a fresh network outputs 0 everywhere. No output activation: the target (v, ε or x₀) is unbounded.", "data-hover": "" }, svg);
    el("rect", { x: RX - 70, y: 52, width: 140, height: 24, rx: 4, class: "module" }, go);
    el("text", { x: RX, y: 68, class: "xs c" }, go, "GN · SiLU · conv_out (zero)");
    el("path", { d: "M" + RX + "," + (ys[0] - bh / 2 - 2) + " L" + RX + ",78", class: "edge flow" }, svg);
    el("path", { d: "M" + RX + ",50 L" + RX + ",42", class: "edge flow" }, svg);

    for (let l = 0; l < 4; l++) {
      const y = ys[l];
      const attn = l === 3;
      const resTxt = function (pair) {
        return "ResBlock(" + pair[0] + " → " + pair[1] + "): GN → SiLU → Conv(k=3) → <b>AdaGN(emb)</b> → SiLU → Dropout → Conv(k=3, zero-init), plus a " +
          (pair[0] === pair[1] ? "identity" : "1×1 conv") + " shortcut.";
      };
      const attnTxt = attn ? " After each ResBlock: SelfAttention1d(256, 4 heads) and, when the model has token conditions, CrossAttention1d to the condition tokens." : "";
      block(LX, y, widths[l], (attn ? "2 × (ResBlock + attn)" : "2 × ResBlock") + " · " + widths[l],
        "<b>Down level " + l + "</b> (mult " + [1, 2, 4, 4][l] + ", length " + lens[l] + ") — 2 × " + resTxt(ins[l]) + attnTxt);
      block(RX, y, widths[l], (attn ? "3 × (ResBlock + attn)" : "3 × ResBlock") + " · " + widths[l],
        "<b>Up level " + l + "</b> (length " + lens[l] + ") — num_res_blocks + 1 = 3 ResBlocks, each on concat(h, skip). First block: " + resTxt(upIns[l]) + attnTxt);
      el("text", { x: 14, y: y + 4, class: "xs mono muted" }, svg, "L = " + lens[l]);
      // skips
      const xa = LX + bw(widths[l]) / 2 + 2, xb = RX - bw(widths[l]) / 2 - 3;
      el("path", { d: "M" + xa + "," + y + " L" + xb + "," + y, class: "edge skip" }, svg);
      el("text", { x: MX, y: y - 5, class: "xs c muted" }, svg, "3 skips (concat)");
      if (l < 3) {
        el("path", { d: "M" + LX + "," + (y + bh / 2 + 1) + " L" + LX + "," + (ys[l + 1] - bh / 2 - 2), class: "edge flow" }, svg);
        const gd = el("g", { "data-info": "<b>Downsample</b> — Conv1d(" + widths[l] + " → " + widths[l] + ", k=3, stride=2, padding=1): " + lens[l] + " → " + lens[l + 1] + ".", "data-hover": "" }, svg);
        el("text", { x: LX + 8, y: (y + ys[l + 1]) / 2 + 4, class: "xs muted" }, gd, "↓ conv s=2");
        el("path", { d: "M" + RX + "," + (ys[l + 1] - bh / 2 - 1) + " L" + RX + "," + (y + bh / 2 + 2), class: "edge flow" }, svg);
        const gu = el("g", { "data-info": "<b>Upsample</b> — F.interpolate(×2, nearest) → Conv1d(" + widths[l + 1] + " → " + widths[l + 1] + ", k=3): " + lens[l + 1] + " → " + lens[l] + ".", "data-hover": "" }, svg);
        el("text", { x: RX - 8, y: (y + ys[l + 1]) / 2 + 4, class: "xs r muted" }, gu, "nearest ×2 + conv ↑");
      }
    }
    // middle
    const mw = block(MX, midY, 256, "mid: 2 × (ResBlock + attn) · 256",
      "<b>Middle</b> — 2 × [ResBlock(256 → 256) → SelfAttention1d → CrossAttention1d (with tokens)] at length 625. Self-attention is always on here.");
    el("text", { x: 14, y: midY + 4, class: "xs mono muted" }, svg, "L = 625");
    el("path", { d: "M" + LX + "," + (ys[3] + bh / 2 + 1) + " Q" + LX + "," + midY + " " + (MX - mw / 2 - 2) + "," + midY, class: "edge flow" }, svg);
    el("path", { d: "M" + (MX + mw / 2 + 1) + "," + midY + " Q" + RX + "," + midY + " " + RX + "," + (ys[3] + bh / 2 + 2), class: "edge flow" }, svg);
    // time / condition embedding panel
    const py = 410;
    const ga = el("g", { "data-info": "<b>TimestepEmbedding(64)</b> — sinusoidal embedding of t (64-d) → Linear(64, 256) → SiLU → Linear(256, 256). DDPM passes the integer step 0…999; rectified flow passes 1000·t.", "data-hover": "" }, svg);
    el("rect", { x: 20, y: py, width: 200, height: 44, rx: 6, class: "module" }, ga);
    el("text", { x: 120, y: py + 18, class: "sm c" }, ga, "t → sin(64) → MLP");
    el("text", { x: 120, y: py + 34, class: "xs c mono muted" }, ga, "emb (B, 256)");
    el("path", { d: "M222," + (py + 22) + " L252," + (py + 22), class: "edge flow" }, svg);
    const gb = el("g", { "data-info": "<b>AddToTimeEmb</b> — emb + Linear(cond_dim → 256)(c.vector). The Linear is zero-initialised, so a new conditional model starts out identical to an unconditional one. Only built when the conditioner has a vector slot.", "data-hover": "" }, svg);
    el("rect", { x: 254, y: py, width: 200, height: 44, rx: 6, class: "module" }, gb);
    el("text", { x: 354, y: py + 18, class: "sm c" }, gb, "+ Linear(c.vector)");
    el("text", { x: 354, y: py + 34, class: "xs c muted" }, gb, "AddToTimeEmb (zero-init)");
    const gc = el("g", { "data-info": "<b>Condition tokens</b> — c.tokens (B, N, token_dim) with a padding mask. Each CrossAttention1d: x + out(attn(GN(x) as queries, tokens as keys/values)), out zero-initialised. Only at attn_levels and in the middle block.", "data-hover": "" }, svg);
    el("rect", { x: 486, y: py, width: 254, height: 44, rx: 6, class: "tensor" }, gc);
    el("text", { x: 613, y: py + 18, class: "sm c" }, gc, "c.tokens (B, N, 256)");
    el("text", { x: 613, y: py + 34, class: "xs c muted" }, gc, "cross-attention at level 3 + mid");
    el("text", { x: 380, y: py + 70, class: "xs c muted" }, svg,
      "emb → AdaGN (scale, shift) in all 22 ResBlocks · c.dense → concatenated to x_t · c.tokens → cross-attention");
    renderTraces(svg);
  }

  // ------------------------------------------------------------------
  // Widget: classifier-free guidance
  // ------------------------------------------------------------------

  function widgetCFG(fig) {
    const svg = fig.querySelector("svg");
    const slider = fig.querySelector("input[name=w]");
    const outp = fig.querySelector("output[name=w]");
    const info = fig.querySelector(".arch-info");
    const W = 760, H = 250, n = 400;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const cond = ecgSeries(n, { seed: 3, lead: "V3", beats: 2 }).map(function (v) { return 1.3 * v; });
    const unc = smooth(ecgSeries(n, { seed: 3, lead: "II", beats: 2 }), 6).map(function (v) { return 0.7 * v; });
    // Left: toy vector picture
    el("rect", { x: 10, y: 10, width: 230, height: 230, rx: 4, class: "plain" }, svg);
    el("text", { x: 125, y: 28, class: "xs c muted" }, svg, "network outputs (toy 2-D)");
    const U = [50, 196], C = [96, 160];
    el("line", { x1: U[0], y1: U[1], x2: U[0] + 4.3 * (C[0] - U[0]), y2: U[1] + 4.3 * (C[1] - U[1]), class: "gridline", style: "stroke-dasharray:3 3" }, svg);
    const arrow = el("path", { class: "edge", style: "stroke:var(--arch-ok);stroke-width:2;marker-end:url(#arch-arrow-ok)" }, svg);
    el("circle", { cx: U[0], cy: U[1], r: 6, class: "frozen" }, svg);
    el("text", { x: U[0] + 10, y: U[1] + 16, class: "xs" }, svg, "u (null condition)");
    el("circle", { cx: C[0], cy: C[1], r: 6, class: "tensor" }, svg);
    el("text", { x: C[0] + 10, y: C[1] + 12, class: "xs" }, svg, "c (condition)");
    const gdot = el("circle", { r: 6, class: "ok" }, svg);
    const glab = el("text", { class: "xs" }, svg, "");
    // Right: traces
    const tx = 340, tw = 405, rows = [[8, "u", "uncond. pass", "trace noise"], [88, "c", "cond. pass", "trace"], [168, "guided", "u + w(c − u)", "trace mix"]];
    const vmin = -1.6, vmax = 2.2, rh = 70;
    const paths = rows.map(function (r) {
      el("rect", { x: tx, y: r[0], width: tw, height: rh, rx: 3, class: "plain" }, svg);
      el("text", { x: tx - 8, y: r[0] + rh / 2, class: "sm r" }, svg, r[1]);
      el("text", { x: tx - 8, y: r[0] + rh / 2 + 13, class: "xs r muted" }, svg, r[2]);
      return el("path", { class: r[3], style: "stroke-width:1.6" }, svg);
    });
    paths[0].setAttribute("d", seriesPath(unc, tx + 2, rows[0][0] + 2, tw - 4, rh - 4, vmin, vmax));
    paths[1].setAttribute("d", seriesPath(cond, tx + 2, rows[1][0] + 2, tw - 4, rh - 4, vmin, vmax));
    function update() {
      const w = parseFloat(slider.value);
      outp.textContent = w.toFixed(1);
      const g = unc.map(function (u, i) { return Math.max(vmin, Math.min(vmax, u + w * (cond[i] - u))); });
      paths[2].setAttribute("d", seriesPath(g, tx + 2, rows[2][0] + 2, tw - 4, rh - 4, vmin, vmax));
      const gx = U[0] + w * (C[0] - U[0]), gy = U[1] + w * (C[1] - U[1]);
      gdot.setAttribute("cx", gx); gdot.setAttribute("cy", gy);
      arrow.setAttribute("d", w > 0.15 ? "M" + U[0] + "," + U[1] + " L" + (gx - (C[0] - U[0]) * 0.12) + "," + (gy - (C[1] - U[1]) * 0.12) : "");
      glab.setAttribute("x", gx - 10); glab.setAttribute("y", gy - 10);
      glab.setAttribute("class", "xs r");
      glab.textContent = "w = " + w.toFixed(1);
      const passes = w === 1 ? "1 network pass per step (guidance off)" : "2 network passes per step (conditional + null)";
      info.innerHTML = "<b>w = " + w.toFixed(1) + "</b> — " +
        (w === 0 ? "unconditional output only" : w === 1 ? "plain conditional model" : w < 1 ? "between unconditional and conditional" : "extrapolates past the conditional output, away from the unconditional one") +
        " · " + passes + ".";
    }
    let auto = !reduceMotion;
    slider.addEventListener("input", function () { auto = false; update(); });
    update();
    const seq = [1, 2, 3, 4, 3, 2, 1, 0];
    let k = 0;
    const timer = setInterval(function () {
      if (!auto || !document.body.contains(svg)) { clearInterval(timer); return; }
      k = (k + 1) % seq.length;
      slider.value = String(seq[k]);
      update();
    }, 1400);
  }

  // ------------------------------------------------------------------
  // Widget: DDIM sampling loop (toy denoiser) with inpainting modes
  // ------------------------------------------------------------------

  function widgetDDIM(fig) {
    const svg = fig.querySelector("svg");
    const stepsIn = fig.querySelector("input[name=steps]");
    const stepsOut = fig.querySelector("output[name=steps]");
    const etaIn = fig.querySelector("input[name=eta]");
    const etaOut = fig.querySelector("output[name=eta]");
    const modeBtns = fig.querySelectorAll("button[data-inpaint]");
    const playBtn = fig.querySelector("button[data-action=play]");
    const restartBtn = fig.querySelector("button[data-action=restart]");
    const info = fig.querySelector(".arch-info");
    const W = 760, H = 270, T = 1000, n = 500;
    const AC = alphasCumprod("cosine", T);
    const tx = 80, tw = 660, rh = 80, rows = [52, 156];
    const vmin = -3.6, vmax = 3.6;
    const target = ecgSeries(n, { seed: 6, lead: "II", beats: 3 }).map(function (v) { return 1.6 * v; });
    const coarse = smooth(target, 22);
    const nObs = Math.round(n * 0.35);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    // timeline
    el("text", { x: tx - 10, y: 22, class: "xs r muted" }, svg, "t: 999");
    el("text", { x: tx + tw + 4, y: 22, class: "xs muted" }, svg, "0");
    el("line", { x1: tx, x2: tx + tw, y1: 18, y2: 18, class: "gridline" }, svg);
    const ticks = el("g", {}, svg);
    const cursor = el("circle", { r: 5, cy: 18, style: "fill:var(--arch-loss)" }, svg);
    const mask = el("rect", { x: tx, y: rows[0] - 6, width: (tw * nObs) / n, height: rows[1] + rh - rows[0] + 12, rx: 3, style: "fill:var(--arch-loss-fill);stroke:var(--arch-loss);stroke-dasharray:4 3" }, svg);
    const maskLab = el("text", { x: tx + (tw * nObs) / n / 2, y: rows[0] - 10, class: "xs c", style: "fill:var(--arch-loss)" }, svg, "observed: mask = 1");
    [["x_t", "current sample"], ["x̂₀", "predicted clean"]].forEach(function (lab, i) {
      el("rect", { x: tx, y: rows[i], width: tw, height: rh, rx: 3, class: "plain", style: "fill-opacity:0.6" }, svg);
      el("text", { x: tx - 10, y: rows[i] + rh / 2, class: "sm r" }, svg, lab[0]);
      el("text", { x: tx - 10, y: rows[i] + rh / 2 + 13, class: "xs r muted" }, svg, lab[1]);
    });
    const pX = el("path", { class: "trace mix" }, svg);
    const pX0 = el("path", { class: "trace" }, svg);
    el("text", { x: W / 2, y: 262, class: "xs c muted" }, svg, "toy denoiser that knows the answer (real: UNet1d, 12 × 5000) · cosine schedule, v-prediction converted to (x̂₀, ε̂)");

    let mode = "none", ts = [], i = 0, x = [], calls = 0, playing = !reduceMotion, timer = null, seed = 1;
    let g = gaussRng(seed);

    function predictX0(xt, t) {
      const ac = AC[t];
      return target.map(function (v, k) { return ac * v + (1 - ac) * 0.5 * coarse[k]; });
    }
    function qSample(t) {
      const a = Math.sqrt(AC[t]), s = Math.sqrt(1 - AC[t]);
      return target.map(function (v) { return a * v + s * g(); });
    }
    function replace(xs, tPrev) {
      if (mode === "none") return xs;
      const known = tPrev < 0 ? target : qSample(tPrev);
      return xs.map(function (v, k) { return k < nObs ? known[k] : v; });
    }
    function ddimStep(xs, t, tPrev, eta) {
      const x0 = predictX0(xs, t);
      calls++;
      const ac = AC[t];
      const eps = xs.map(function (v, k) { return (v - Math.sqrt(ac) * x0[k]) / Math.sqrt(1 - ac); });
      if (tPrev < 0) return { x: x0, x0: x0 };
      const acp = AC[tPrev];
      const sigma = eta * Math.sqrt(Math.max(0, ((1 - acp) / (1 - ac)) * (1 - ac / acp)));
      const dir = Math.sqrt(Math.max(0, 1 - acp - sigma * sigma));
      const out = x0.map(function (v, k) { return Math.sqrt(acp) * v + dir * eps[k] + (eta > 0 ? sigma * g() : 0); });
      return { x: out, x0: x0 };
    }
    function renoise(xs, tFrom, tTo) {
      const r = AC[tTo] / AC[tFrom];
      return xs.map(function (v) { return Math.sqrt(r) * v + Math.sqrt(1 - r) * g(); });
    }
    function drawTicks() {
      clear(ticks);
      ts.forEach(function (t) {
        el("line", { x1: tx + ((999 - t) / 999) * tw, x2: tx + ((999 - t) / 999) * tw, y1: 13, y2: 23, style: "stroke:var(--arch-muted);stroke-width:1" }, ticks);
      });
    }
    function reset() {
      seed++;
      g = gaussRng(seed);
      ts = ddimTimesteps(parseInt(stepsIn.value, 10), T);
      i = 0; calls = 0;
      x = gaussSeries(n, seed * 13);
      drawTicks();
      pX.setAttribute("d", seriesPath(x, tx + 2, rows[0] + 2, tw - 4, rh - 4, vmin, vmax));
      pX0.setAttribute("d", "");
      cursor.setAttribute("cx", tx);
      const show = mode !== "none";
      mask.style.display = show ? "" : "none";
      maskLab.style.display = show ? "" : "none";
      report(999, null);
    }
    function report(t, tPrev) {
      const eta = parseFloat(etaIn.value);
      const jumps = mode === "repaint" ? 5 : 1;
      const total = jumps * (ts.length - 1) + 1;
      info.innerHTML = (tPrev === null ? "start: x<sub>T</sub> ~ N(0, I)" :
        "step <b>" + i + " / " + ts.length + "</b>: t = " + t + " → " + (tPrev < 0 ? "x̂₀ (last step returns the prediction)" : "t′ = " + tPrev)) +
        " · η = " + eta.toFixed(1) + (eta === 0 ? " (deterministic DDIM)" : eta === 1 ? " (ancestral)" : "") +
        " · network calls " + calls + " / " + total + (mode === "repaint" ? " (RePaint: 5 jumps per step)" : "") +
        " — ×2 with classifier-free guidance.";
    }
    function step() {
      if (i >= ts.length) {
        if (playing) timer = setTimeout(function () { reset(); step(); }, 2200);
        return;
      }
      const t = ts[i], tPrev = i + 1 < ts.length ? ts[i + 1] : -1;
      const eta = parseFloat(etaIn.value);
      const jumps = mode === "repaint" && tPrev >= 0 ? 5 : 1;
      let res;
      for (let j = 0; j < jumps; j++) {
        res = ddimStep(x, t, tPrev, eta);
        res.x = replace(res.x, tPrev);
        x = j < jumps - 1 ? renoise(res.x, tPrev, t) : res.x;
      }
      i++;
      pX.setAttribute("d", seriesPath(x, tx + 2, rows[0] + 2, tw - 4, rh - 4, vmin, vmax));
      pX0.setAttribute("d", seriesPath(res.x0, tx + 2, rows[1] + 2, tw - 4, rh - 4, vmin, vmax));
      cursor.setAttribute("cx", tx + ((999 - Math.max(tPrev, 0)) / 999) * tw);
      report(t, tPrev);
      if (playing) timer = setTimeout(step, Math.max(60, 4000 / ts.length));
    }
    function setPlaying(v) {
      playing = v;
      playBtn.textContent = v ? "Pause" : "Play";
      clearTimeout(timer);
      if (v) step();
    }
    function restart() { clearTimeout(timer); reset(); if (playing) step(); else { while (i < ts.length) step(); } }
    stepsIn.addEventListener("input", function () { stepsOut.textContent = stepsIn.value; restart(); });
    etaIn.addEventListener("input", function () { etaOut.textContent = parseFloat(etaIn.value).toFixed(1); restart(); });
    modeBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        mode = b.dataset.inpaint;
        modeBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
        restart();
      });
    });
    modeBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o.dataset.inpaint === mode)); });
    playBtn.addEventListener("click", function () { setPlaying(!playing); });
    restartBtn.addEventListener("click", restart);
    stepsOut.textContent = stepsIn.value;
    etaOut.textContent = parseFloat(etaIn.value).toFixed(1);
    reset();
    if (!playing) { while (i < ts.length) step(); }
    setPlaying(playing);
    const watch = setInterval(function () {
      if (!document.body.contains(svg)) { clearTimeout(timer); clearInterval(watch); }
    }, 2000);
  }

  // ------------------------------------------------------------------
  // Widget: rectified-flow paths and Euler / Heun steps (toy 2-D)
  // ------------------------------------------------------------------

  function widgetFlow(fig) {
    const svg = fig.querySelector("svg");
    const tIn = fig.querySelector("input[name=t]");
    const tOut = fig.querySelector("output[name=t]");
    const stepsIn = fig.querySelector("input[name=steps]");
    const stepsOut = fig.querySelector("output[name=steps]");
    const pairBtns = fig.querySelectorAll("button[data-pairing]");
    const solverBtns = fig.querySelectorAll("button[data-solver]");
    const info = fig.querySelector(".arch-info");
    const W = 760, H = 350, P = 300, D = 4;
    const panels = [{ x: 40, y: 36 }, { x: 420, y: 36 }];
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    clear(svg);
    const toS = function (pn, p) {
      return [pn.x + ((p[0] + D) / (2 * D)) * P, pn.y + ((D - p[1]) / (2 * D)) * P];
    };
    // toy data: mixture of three clusters; noise N(0, I)
    const mus = [[2.2, 2.0], [2.4, -1.9], [-2.4, 2.1]], sd = 0.35;
    const K = 18;
    const gN = gaussRng(31), gD = gaussRng(47);
    const x0s = [], x1s = [];
    for (let k = 0; k < K; k++) {
      x0s.push([gN(), gN()]);
      const m = mus[k % 3];
      x1s.push([m[0] + sd * gD(), m[1] + sd * gD()]);
    }
    function cost(a, b) { return Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2); }
    // random pairing = index order (x1 clusters interleaved, x0 random); OT via 2-opt swaps
    const randPerm = [];
    for (let k = 0; k < K; k++) randPerm.push(k);
    const otPerm = randPerm.slice();
    let improved = true;
    while (improved) {
      improved = false;
      for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
          const now = cost(x1s[a], x0s[otPerm[a]]) + cost(x1s[b], x0s[otPerm[b]]);
          const sw = cost(x1s[a], x0s[otPerm[b]]) + cost(x1s[b], x0s[otPerm[a]]);
          if (sw < now - 1e-9) { const tmp = otPerm[a]; otPerm[a] = otPerm[b]; otPerm[b] = tmp; improved = true; }
        }
      }
    }
    let pairing = "random", solver = "euler";

    function frame(pn, title, sub) {
      el("rect", { x: pn.x, y: pn.y, width: P, height: P, rx: 4, class: "plain" }, svg);
      el("text", { x: pn.x + P / 2, y: pn.y - 18, class: "sm c" }, svg, title);
      el("text", { x: pn.x + P / 2, y: pn.y - 5, class: "xs c muted" }, svg, sub);
      mus.forEach(function (m) {
        const c = toS(pn, m);
        el("circle", { cx: c[0], cy: c[1], r: (sd * 2.4 * P) / (2 * D), style: "fill:var(--arch-data-fill);stroke:var(--arch-data);stroke-width:1;opacity:0.8" }, svg);
      });
      const c0 = toS(pn, [0, 0]);
      el("circle", { cx: c0[0], cy: c0[1], r: (1.6 * P) / (2 * D), style: "fill:none;stroke:var(--arch-frozen);stroke-dasharray:3 3" }, svg);
    }
    frame(panels[0], "Training: straight paths x₀ → x₁", "x_t = t·x₁ + (1 − t)·x₀ (σ_min = 0)");
    frame(panels[1], "Sampling: integrate the learned v(x, t)", "dashed = exact ODE · dots = solver steps");
    el("text", { x: panels[0].x + 6, y: panels[0].y + P - 8, class: "xs muted" }, svg, "grey ring: noise N(0, I) · blue: data");
    const gA = el("g", {}, svg);
    const gB = el("g", {}, svg);

    // exact marginal velocity for Gaussian-mixture data and sigma_min = 0
    function velocity(x, t) {
      const tc = Math.min(t, 0.999);
      const varT = tc * tc * sd * sd + (1 - tc) * (1 - tc);
      let wsum = 0;
      const ws = [], vs = [];
      mus.forEach(function (m) {
        const dx = [x[0] - tc * m[0], x[1] - tc * m[1]];
        const w = Math.exp(-(dx[0] * dx[0] + dx[1] * dx[1]) / (2 * varT));
        const e1 = [m[0] + ((tc * sd * sd) / varT) * dx[0], m[1] + ((tc * sd * sd) / varT) * dx[1]];
        const e0 = [((1 - tc) / varT) * dx[0], ((1 - tc) / varT) * dx[1]];
        ws.push(w); vs.push([e1[0] - e0[0], e1[1] - e0[1]]);
        wsum += w;
      });
      const v = [0, 0];
      ws.forEach(function (w, k) { v[0] += (w / wsum) * vs[k][0]; v[1] += (w / wsum) * vs[k][1]; });
      return v;
    }
    function integrate(x, steps, how) {
      const pts = [x.slice()];
      let calls = 0;
      for (let i = 0; i < steps; i++) {
        const t = i / steps, h = 1 / steps;
        const d = velocity(x, t); calls++;
        if (how === "heun" && i < steps - 1) {
          const xe = [x[0] + h * d[0], x[1] + h * d[1]];
          const d2 = velocity(xe, t + h); calls++;
          x = [x[0] + h * 0.5 * (d[0] + d2[0]), x[1] + h * 0.5 * (d[1] + d2[1])];
        } else {
          x = [x[0] + h * d[0], x[1] + h * d[1]];
        }
        pts.push(x.slice());
      }
      return { pts: pts, calls: calls };
    }
    const starts = [[-0.9, 0.4], [0.3, -1.1], [1.0, 0.9], [-0.4, -0.3], [0.2, 1.4], [-1.3, -0.9], [1.3, -0.2]];
    const exact = starts.map(function (s) { return integrate(s.slice(), 400, "euler").pts; });

    function poly(pn, pts) {
      return pts.map(function (p, i) { const s = toS(pn, p); return (i ? "L" : "M") + s[0].toFixed(1) + "," + s[1].toFixed(1); }).join("");
    }
    function drawA() {
      clear(gA);
      const t = parseFloat(tIn.value);
      const perm = pairing === "ot" ? otPerm : randPerm;
      let total = 0;
      for (let k = 0; k < K; k++) {
        const a = x0s[perm[k]], b = x1s[k];
        total += Math.sqrt(cost(a, b));
        const sa = toS(panels[0], a), sb = toS(panels[0], b);
        el("line", { x1: sa[0], y1: sa[1], x2: sb[0], y2: sb[1], style: "stroke:var(--arch-muted);stroke-width:0.8;opacity:0.7" }, gA);
        el("circle", { cx: sa[0], cy: sa[1], r: 2.5, style: "fill:var(--arch-frozen)" }, gA);
        el("circle", { cx: sb[0], cy: sb[1], r: 2.5, style: "fill:var(--arch-data)" }, gA);
        const p = [t * b[0] + (1 - t) * a[0], t * b[1] + (1 - t) * a[1]];
        const sp = toS(panels[0], p);
        el("circle", { cx: sp[0], cy: sp[1], r: 4, class: "ok" }, gA);
        if (k === 0) {
          const v = [b[0] - a[0], b[1] - a[1]];
          const tip = toS(panels[0], [p[0] + 0.28 * v[0], p[1] + 0.28 * v[1]]);
          el("path", { d: "M" + sp[0] + "," + sp[1] + " L" + tip[0] + "," + tip[1], class: "edge", style: "stroke:var(--arch-loss);stroke-width:2;marker-end:url(#arch-arrow-loss)" }, gA);
        }
      }
      return total / K;
    }
    function drawB() {
      clear(gB);
      const steps = parseInt(stepsIn.value, 10);
      let err = 0, calls = 0;
      starts.forEach(function (s, k) {
        el("path", { d: poly(panels[1], exact[k]), class: "edge noarrow", style: "stroke-dasharray:3 3;stroke-width:1.2" }, gB);
        const r = integrate(s.slice(), steps, solver);
        calls = r.calls;
        el("path", { d: poly(panels[1], r.pts), class: "trace mix", style: "stroke-width:1.6" }, gB);
        r.pts.forEach(function (p, i) {
          const sp = toS(panels[1], p);
          el("circle", { cx: sp[0], cy: sp[1], r: i === 0 ? 3.5 : 2.6, style: i === 0 ? "fill:var(--arch-frozen)" : "fill:var(--arch-ok)" }, gB);
        });
        const e = exact[k][exact[k].length - 1], f = r.pts[r.pts.length - 1];
        err += Math.sqrt(cost(e, f));
      });
      return { err: err / starts.length, calls: calls, steps: steps };
    }
    function update() {
      tOut.textContent = parseFloat(tIn.value).toFixed(2);
      stepsOut.textContent = stepsIn.value;
      const len = drawA();
      const b = drawB();
      info.innerHTML = "Left: " + (pairing === "ot" ? "minibatch OT pairing" : "random pairing") + ", mean path length " + fmt(len) +
        "; the orange arrow is the training target x₁ − x₀ (constant along the path). Right: <b>" + b.steps + " " + solver +
        " step" + (b.steps > 1 ? "s" : "") + "</b> = " + b.calls + " network calls; mean endpoint error vs the exact ODE " + fmt(b.err) + ".";
    }
    pairBtns.forEach(function (bt) {
      bt.addEventListener("click", function () {
        pairing = bt.dataset.pairing;
        pairBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o === bt)); });
        update();
      });
    });
    solverBtns.forEach(function (bt) {
      bt.addEventListener("click", function () {
        solver = bt.dataset.solver;
        solverBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o === bt)); });
        update();
      });
    });
    pairBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o.dataset.pairing === pairing)); });
    solverBtns.forEach(function (o) { o.setAttribute("aria-pressed", String(o.dataset.solver === solver)); });
    let auto = !reduceMotion;
    tIn.addEventListener("input", function () { auto = false; update(); });
    stepsIn.addEventListener("input", update);
    update();
    let t = parseFloat(tIn.value);
    const timer = setInterval(function () {
      if (!auto || !document.body.contains(svg)) { clearInterval(timer); return; }
      t = t >= 1 ? 0 : Math.min(1, t + 0.04);
      tIn.value = String(t);
      tOut.textContent = t.toFixed(2);
      drawA();
    }, 120);
  }

  // ------------------------------------------------------------------
  // Diagram: training-time distribution for rectified flow
  // ------------------------------------------------------------------

  function renderTSampling(svg) {
    clear(svg);
    const W = 760, H = 190, px = 60, py = 20, pw = 640, ph = 130;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const ymax = 2.0;
    el("rect", { x: px, y: py, width: pw, height: ph, class: "plain" }, svg);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el("text", { x: px + v * pw, y: py + ph + 14, class: "xs c mono muted" }, svg, String(v));
    });
    [0, 1, 2].forEach(function (v) {
      el("text", { x: px - 6, y: py + ph - (v / ymax) * ph + 4, class: "xs r mono muted" }, svg, String(v));
    });
    el("text", { x: px + pw / 2, y: py + ph + 30, class: "xs c muted" }, svg, "t  (0 = noise, 1 = data)");
    el("text", { x: px - 34, y: py + ph / 2, class: "xs c muted", transform: "rotate(-90 " + (px - 34) + " " + (py + ph / 2) + ")" }, svg, "density");
    const yU = py + ph - (1 / ymax) * ph;
    el("path", { d: "M" + px + "," + yU + " L" + (px + pw) + "," + yU, class: "edge noarrow", style: "stroke-dasharray:5 4" }, svg);
    let d = "";
    for (let i = 1; i < 400; i++) {
      const t = i / 400;
      const z = Math.log(t / (1 - t));
      const p = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * t * (1 - t));
      d += (i > 1 ? "L" : "M") + (px + t * pw).toFixed(1) + "," + (py + ph - Math.min(p / ymax, 1) * ph).toFixed(1);
    }
    el("path", { d: d, class: "trace", style: "stroke-width:2;fill:var(--arch-data-fill);fill-opacity:0.5" }, svg);
    el("text", { x: px + 8, y: yU - 6, class: "xs muted" }, svg, "uniform  (t_sampling: uniform)");
    el("text", { x: px + pw * 0.5 + 70, y: py + 30, class: "xs", style: "fill:var(--arch-data)" }, svg, "logit-normal(0, 1) = sigmoid(N(0, 1))  — default");
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------

  const WIDGETS = {
    leads: widgetLeads,
    backbone: widgetBackbone,
    downsample: widgetDownsample,
    kl: widgetKL,
    vq: widgetVQ,
    causal: widgetCausal,
    ar: widgetAR,
    phase: widgetPhase,
    gp: widgetGP,
    schedule: widgetSchedule,
    noising: widgetNoising,
    cfg: widgetCFG,
    ddim: widgetDDIM,
    flowpaths: widgetFlow,
  };
  const DIAGRAMS = { unet: renderUNet, critic: renderCritic, unet1d: renderUNet1d, tsampling: renderTSampling };

  function init() {
    if (!document.querySelector(".arch")) return;
    injectDefs();
    renderTraces(document);
    document.querySelectorAll("svg[data-diagram]").forEach(function (svg) {
      const fn = DIAGRAMS[svg.dataset.diagram];
      if (fn) fn(svg);
    });
    document.querySelectorAll("figure[data-widget]").forEach(function (fig) {
      const fn = WIDGETS[fig.dataset.widget];
      if (fn && !fig.dataset.ready) {
        fig.dataset.ready = "1";
        fn(fig);
      }
    });
    document.querySelectorAll("figure.arch").forEach(wireInfo);
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
