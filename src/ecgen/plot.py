"""
ecgen.plot — ECG visualisation utilities.

Three D3/SVG output formats
----------------------------
``"html"``  — self-contained interactive HTML (D3 v7, pan/zoom, save-SVG, print-PDF buttons).
``"svg"``   — static SVG, no JavaScript, suitable for documents / reports.
``"pdf"``   — static PDF via cairosvg  (``pip install 'ecgen[plot]'``).

Matplotlib PNG output
---------------------
``plot_generated_single`` — all leads of one sample as stacked subplots → PNG.
``plot_generated_batch``  — multiple samples side by side → PNG.

Quick start
-----------
    from ecgen.plot import plot_ecg, plot_generated_single, plot_generated_batch
    import numpy as np

    ecg = np.random.randn(8, 5000).astype("float32") * 0.3

    # Interactive browser plot
    plot_ecg(ecg, output_path="ecg.html", title="Generated ECG")

    # Static SVG
    plot_ecg(ecg, output_path="ecg.svg", format="svg")

    # PDF (requires cairosvg)
    plot_ecg(ecg, output_path="ecg.pdf", format="pdf")

    # Matplotlib PNG (single sample)
    fig = plot_generated_single(ecg, sampling_rate=500)
    fig.savefig("sample.png", dpi=150)

    # Matplotlib PNG (batch)
    batch = np.random.randn(4, 8, 5000).astype("float32") * 0.3
    fig = plot_generated_batch(batch, num_of_plots=4)
    fig.savefig("batch.png", dpi=150)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# ── default lead names (Pulse2Pulse PTB-XL order) ──────────────────────────
LEAD_NAMES_DEFAULT: List[str] = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── standard ECG paper parameters ──────────────────────────────────────────
_MM_S_DEFAULT: float = 25.0   # paper speed  mm/s
_MM_MV_DEFAULT: float = 10.0  # gain         mm/mV

# Screen: use 96 dpi → 1 mm ≈ 3.7795 px
_PX_PER_MM: float = 3.7795

# Layout (all in mm)
_LEAD_H_MM: float = 28.0   # height of each lead strip
_LEAD_GAP_MM: float = 3.0  # vertical gap between strips
_MARGIN_MM = dict(top=12, right=8, bottom=10, left=18)

# ECG paper colours
_C_BG = "#FFF5F5"        # paper background (very light pink)
_C_MINOR = "#FFBCBC"     # 1 mm grid lines
_C_MAJOR = "#FF7F7F"     # 5 mm grid lines
_C_TRACE = "#1a1a1a"     # signal
_C_LABEL = "#333333"     # lead labels
_C_BASELINE = "#FFB0B0"  # 0 mV dashed reference


# ═══════════════════════════════════════════════════════════════════════════
#   Public API
# ═══════════════════════════════════════════════════════════════════════════

def plot_ecg(
    ecg: "np.ndarray",
    *,
    sample_rate: int = 500,
    lead_names: Optional[List[str]] = None,
    title: str = "ECG",
    output_path: Optional[Union[str, Path]] = None,
    format: str = "html",
    paper_speed: float = _MM_S_DEFAULT,
    amplitude_scale: float = _MM_MV_DEFAULT,
) -> Optional[str]:
    """Render ECG signals as an interactive D3 HTML, static SVG, or PDF.

    Parameters
    ----------
    ecg : np.ndarray
        Shape ``(n_leads, n_samples)`` or ``(n_samples, n_leads)`` — auto-detected.
    sample_rate : int
        Hz (default 500).
    lead_names : list[str], optional
        One label per lead.  Defaults to ``["I","II","V1"…"V6"]``.
    title : str
        Shown in the toolbar and as the SVG/PDF title.
    output_path : str or Path, optional
        File to write.  If *None*, the HTML/SVG string is returned instead.
        For ``format="pdf"`` an ``output_path`` is required.
    format : str
        ``"html"`` | ``"svg"`` | ``"pdf"``
    paper_speed : float
        mm/s (default 25).
    amplitude_scale : float
        mm/mV (default 10).

    Returns
    -------
    ``str`` (HTML or SVG content) when ``output_path`` is *None*, else ``None``.
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim == 2 and ecg.shape[0] > ecg.shape[1]:
        ecg = ecg.T  # (n_samples, n_leads) → (n_leads, n_samples)
    if ecg.ndim != 2:
        raise ValueError(f"ecg must be 2-D, got shape {ecg.shape}")

    n_leads = ecg.shape[0]
    if lead_names is None:
        lead_names = (
            LEAD_NAMES_DEFAULT[:n_leads]
            if n_leads <= len(LEAD_NAMES_DEFAULT)
            else [f"L{i + 1}" for i in range(n_leads)]
        )
    elif len(lead_names) != n_leads:
        raise ValueError(f"lead_names has {len(lead_names)} entries but ecg has {n_leads} leads")

    if format == "html":
        content = _build_html(ecg, sample_rate, lead_names, title, paper_speed, amplitude_scale)
        ext = ".html"
    elif format == "svg":
        content = _build_svg(ecg, sample_rate, lead_names, title, paper_speed, amplitude_scale)
        ext = ".svg"
    elif format == "pdf":
        if output_path is None:
            raise ValueError("output_path is required when format='pdf'.")
        svg_str = _build_svg(ecg, sample_rate, lead_names, title, paper_speed, amplitude_scale)
        _save_pdf(svg_str, Path(output_path))
        return None
    else:
        raise ValueError(f"Unknown format '{format}'. Choose 'html', 'svg', or 'pdf'.")

    if output_path is None:
        return content

    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(ext)
    path.write_text(content, encoding="utf-8")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#   HTML — interactive D3 v7
# ═══════════════════════════════════════════════════════════════════════════

def _build_html(
    ecg: np.ndarray,
    sample_rate: int,
    lead_names: List[str],
    title: str,
    paper_speed: float,
    amplitude_scale: float,
) -> str:
    n_leads, n_samples = ecg.shape

    # Round to 5 decimal places so JSON doesn't explode in size
    leads_json = json.dumps(
        [{"name": name, "data": [round(float(v), 5) for v in ecg[i]]}
         for i, name in enumerate(lead_names)]
    )
    config_json = json.dumps({
        "sampleRate": sample_rate,
        "paperSpeed": paper_speed,
        "amplitudeScale": amplitude_scale,
        "title": title,
        "nLeads": n_leads,
        "nSamples": n_samples,
        "pxPerMm": _PX_PER_MM,
        "leadHeightMm": _LEAD_H_MM,
        "leadGapMm": _LEAD_GAP_MM,
        "marginMm": _MARGIN_MM,
        "colors": {
            "bg": _C_BG, "minor": _C_MINOR, "major": _C_MAJOR,
            "trace": _C_TRACE, "label": _C_LABEL, "baseline": _C_BASELINE,
        },
    })

    return _HTML_TEMPLATE.replace("__LEADS_JSON__", leads_json).replace(
        "__CONFIG_JSON__", config_json
    ).replace("__TITLE__", title)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#d8d8d8;font-family:'Courier New',Courier,monospace;overflow-x:hidden}

  #toolbar{
    background:#2b2b2b;color:#eee;padding:8px 16px;
    display:flex;align-items:center;gap:10px;
    position:sticky;top:0;z-index:100;user-select:none;
    box-shadow:0 2px 6px rgba(0,0,0,.4);
  }
  #toolbar h1{font-size:13px;font-weight:bold;flex:1;letter-spacing:.5px}
  #toolbar .meta{font-size:11px;color:#aaa;white-space:nowrap}
  #toolbar button{
    background:#444;color:#eee;border:1px solid #666;
    padding:4px 11px;cursor:pointer;font-size:11px;
    border-radius:3px;font-family:inherit;white-space:nowrap;
  }
  #toolbar button:hover{background:#666}

  #ecg-wrap{
    padding:18px 24px;overflow:auto;
    background:#d8d8d8;min-height:calc(100vh - 36px);
  }

  #ecg-svg{
    display:block;background:#FFF5F5;
    box-shadow:0 2px 12px rgba(0,0,0,.3);
    cursor:grab;
  }
  #ecg-svg:active{cursor:grabbing}

  .lead-label{
    font-family:'Courier New',Courier,monospace;
    font-size:11px;font-weight:bold;fill:#333;
  }
  .ecg-trace{fill:none;stroke:#1a1a1a;stroke-linejoin:round;stroke-linecap:round}
  .baseline{fill:none;stroke:#FFB0B0;stroke-dasharray:3,5}
  .sep-line{stroke:#FF7F7F;fill:none}
  .sec-tick{stroke:#FF4040;opacity:.45}
  .sec-label{font-family:'Courier New',Courier,monospace;font-size:9px;fill:#888}

  #zoom-hint{
    font-size:11px;color:#888;white-space:nowrap;
    pointer-events:none;
  }

  @media print{
    #toolbar{display:none}
    body{background:white}
    #ecg-wrap{padding:0}
    #ecg-svg{box-shadow:none}
  }
</style>
</head>
<body>

<div id="toolbar">
  <h1 id="plot-title">__TITLE__</h1>
  <span class="meta" id="meta-info"></span>
  <span id="zoom-hint">scroll / pinch to zoom · drag to pan</span>
  <button onclick="resetZoom()">↺ Reset</button>
  <button onclick="saveSVG()">⤓ SVG</button>
  <button onclick="window.print()">🖨 Print / PDF</button>
</div>

<div id="ecg-wrap">
  <svg id="ecg-svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function(){
"use strict";

const ECG  = __LEADS_JSON__;
const CFG  = __CONFIG_JSON__;

const SR   = CFG.sampleRate;
const PS   = CFG.paperSpeed;    // mm/s
const AS   = CFG.amplitudeScale; // mm/mV
const PX   = CFG.pxPerMm;       // px per mm

const SAMPLES_PER_MM = SR / PS;
const PX_PER_SAMPLE  = PX / SAMPLES_PER_MM;
const PX_PER_MV      = AS * PX;

const N_LEADS   = ECG.length;
const N_SAMPLES = ECG[0].data.length;

const LEAD_H   = CFG.leadHeightMm * PX;
const LEAD_GAP = CFG.leadGapMm * PX;
const ML = CFG.marginMm.left  * PX;
const MT = CFG.marginMm.top   * PX;
const MR = CFG.marginMm.right * PX;
const MB = CFG.marginMm.bottom* PX;

const PLOT_W = N_SAMPLES * PX_PER_SAMPLE;
const PLOT_H = N_LEADS * (LEAD_H + LEAD_GAP) - LEAD_GAP;
const SVG_W  = ML + PLOT_W + MR;
const SVG_H  = MT + PLOT_H + MB;

// Update meta bar
document.getElementById("meta-info").textContent =
  `${SR} Hz · ${PS} mm/s · ${AS} mm/mV · ${(N_SAMPLES/SR).toFixed(1)}s`;

// ── SVG root ──────────────────────────────────────────────────────────────
const svg = d3.select("#ecg-svg")
  .attr("width",   SVG_W)
  .attr("height",  SVG_H)
  .attr("viewBox", `0 0 ${SVG_W} ${SVG_H}`);

// ── defs ──────────────────────────────────────────────────────────────────
const defs = svg.append("defs");

// ECG paper grid patterns (tiled, both use userSpaceOnUse)
const minorPx = 1 * PX;   // 1 mm
const majorPx = 5 * PX;   // 5 mm

const minorPat = defs.append("pattern")
  .attr("id","minor-grid")
  .attr("patternUnits","userSpaceOnUse")
  .attr("width", minorPx).attr("height", minorPx);
minorPat.append("path")
  .attr("d", `M ${minorPx} 0 L 0 0 0 ${minorPx}`)
  .attr("fill","none").attr("stroke", CFG.colors.minor).attr("stroke-width","0.5");

const majorPat = defs.append("pattern")
  .attr("id","major-grid")
  .attr("patternUnits","userSpaceOnUse")
  .attr("width", majorPx).attr("height", majorPx);
majorPat.append("rect")
  .attr("width", majorPx).attr("height", majorPx)
  .attr("fill","url(#minor-grid)");
majorPat.append("path")
  .attr("d", `M ${majorPx} 0 L 0 0 0 ${majorPx}`)
  .attr("fill","none").attr("stroke", CFG.colors.major).attr("stroke-width","1");

// Clip path for the entire plot area
defs.append("clipPath").attr("id","plot-clip")
  .append("rect").attr("width", PLOT_W).attr("height", PLOT_H + 10);

// ── lead labels (outside zoom group — never scale) ────────────────────────
const labelsG = svg.append("g")
  .attr("transform", `translate(0, ${MT})`);

ECG.forEach((lead, i) => {
  const cy = i * (LEAD_H + LEAD_GAP) + LEAD_H / 2;
  labelsG.append("text")
    .attr("class","lead-label")
    .attr("x", ML - 5)
    .attr("y", cy + 4)
    .attr("text-anchor","end")
    .text(lead.name);
});

// ── zoom wrapper (clip applied here) ────────────────────────────────────
const clipWrapper = svg.append("g")
  .attr("transform", `translate(${ML}, ${MT})`)
  .attr("clip-path","url(#plot-clip)");

// ── zoom content group (zoom transform applied to this g) ────────────────
const contentG = clipWrapper.append("g").attr("id","zoom-content");

// Paper background
contentG.append("rect")
  .attr("x",0).attr("y",0)
  .attr("width", PLOT_W).attr("height", PLOT_H)
  .attr("fill", CFG.colors.bg);

// Grid overlay
contentG.append("rect")
  .attr("x",0).attr("y",0)
  .attr("width", PLOT_W).attr("height", PLOT_H)
  .attr("fill","url(#major-grid)");

// ── xScale (sample index → px) ───────────────────────────────────────────
const xScale = d3.scaleLinear()
  .domain([0, N_SAMPLES - 1])
  .range([0, PLOT_W]);

// ── lead groups (traces + baselines + separators) ─────────────────────────
const leadGs = contentG.selectAll(".lead-g")
  .data(ECG).enter()
  .append("g").attr("class","lead-g")
  .attr("transform", (_, i) => `translate(0, ${i * (LEAD_H + LEAD_GAP)})`);

// Separator line (top of each strip)
leadGs.append("line").attr("class","sep-line")
  .attr("x1",0).attr("x2", PLOT_W)
  .attr("y1",0).attr("y2",0)
  .attr("stroke-width","1");

// 0 mV baseline
leadGs.append("line").attr("class","baseline")
  .attr("x1",0).attr("x2", PLOT_W)
  .attr("y1", LEAD_H/2).attr("y2", LEAD_H/2)
  .attr("stroke-width","0.5");

// ECG trace
const tracePathGen = (xSc) => d3.line()
  .x((_, i) => xSc(i))
  .y(v   => LEAD_H/2 - v * PX_PER_MV)
  .defined(v => isFinite(v));

leadGs.append("path").attr("class","ecg-trace")
  .attr("stroke", CFG.colors.trace)
  .attr("stroke-width","1.5")
  .datum(d => d.data)
  .attr("d", tracePathGen(xScale));

// ── second tick marks (above all leads) ──────────────────────────────────
const tickG = contentG.append("g").attr("class","ticks");
const nSec  = Math.floor(N_SAMPLES / SR) + 1;
const secData = Array.from({length: nSec}, (_, s) => s);

tickG.selectAll(".sec-tick")
  .data(secData).enter()
  .append("line").attr("class","sec-tick")
  .attr("x1", s => xScale(s * SR)).attr("x2", s => xScale(s * SR))
  .attr("y1", -5).attr("y2", PLOT_H)
  .attr("stroke-width", s => s % 5 === 0 ? 1 : 0.4);

tickG.selectAll(".sec-label")
  .data(secData).enter()
  .append("text").attr("class","sec-label")
  .attr("x", s => xScale(s * SR) + 2)
  .attr("y", -6)
  .text(s => `${s}s`);

// Bottom border
contentG.append("line")
  .attr("x1",0).attr("x2", PLOT_W)
  .attr("y1", PLOT_H).attr("y2", PLOT_H)
  .attr("stroke", CFG.colors.major).attr("stroke-width","1");

// ── zoom behaviour ────────────────────────────────────────────────────────
let currentZoom = d3.zoomIdentity;

const zoom = d3.zoom()
  .scaleExtent([0.3, 80])
  .translateExtent([[0, 0], [SVG_W, SVG_H]])
  .on("zoom", (event) => {
    currentZoom = event.transform;
    const nx = currentZoom.rescaleX(xScale);

    // Redraw traces
    leadGs.selectAll(".ecg-trace")
      .attr("d", function() {
        return tracePathGen(nx)(d3.select(this).datum());
      });

    // Shift tick marks
    tickG.selectAll(".sec-tick")
      .attr("x1", s => nx(s * SR))
      .attr("x2", s => nx(s * SR));
    tickG.selectAll(".sec-label")
      .attr("x", s => nx(s * SR) + 2);

    // Update grid pattern phase so grid tracks the signal
    const k = currentZoom.k;
    const tx = currentZoom.x;
    const minSz = minorPx * k;
    const majSz = majorPx * k;
    const txMod = ((tx % majSz) + majSz) % majSz;
    d3.select("#minor-grid")
      .attr("width", minSz).attr("height", minSz)
      .attr("x", ((tx % minSz) + minSz) % minSz)
      .select("path")
        .attr("d", `M ${minSz} 0 L 0 0 0 ${minSz}`);
    d3.select("#major-grid")
      .attr("width", majSz).attr("height", majSz)
      .attr("x", txMod)
      .select("path")
        .attr("d", `M ${majSz} 0 L 0 0 0 ${majSz}`);
    d3.select("#major-grid rect")
      .attr("width", majSz).attr("height", majSz);
  });

svg.call(zoom);

// ── toolbar actions ───────────────────────────────────────────────────────
window.resetZoom = () => {
  svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity);
};

window.saveSVG = () => {
  // Reset zoom before export so the SVG captures the full signal
  svg.call(zoom.transform, d3.zoomIdentity);
  setTimeout(() => {
    const svgEl  = document.getElementById("ecg-svg");
    const serial = new XMLSerializer();
    let   svgStr = serial.serializeToString(svgEl);
    if (!svgStr.match(/^<svg[^>]+xmlns=/)) {
      svgStr = svgStr.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    const blob = new Blob([svgStr], {type:"image/svg+xml;charset=utf-8"});
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = (CFG.title || "ecg") + ".svg";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }, 350);
};

})();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
#   SVG — pure Python, no JavaScript
# ═══════════════════════════════════════════════════════════════════════════

def _build_svg(
    ecg: np.ndarray,
    sample_rate: int,
    lead_names: List[str],
    title: str,
    paper_speed: float,
    amplitude_scale: float,
) -> str:
    n_leads, n_samples = ecg.shape

    samples_per_mm = sample_rate / paper_speed
    px_per_sample  = _PX_PER_MM / samples_per_mm
    px_per_mv      = amplitude_scale * _PX_PER_MM

    lead_h   = _LEAD_H_MM  * _PX_PER_MM
    lead_gap = _LEAD_GAP_MM * _PX_PER_MM
    ml = _MARGIN_MM["left"]   * _PX_PER_MM
    mt = _MARGIN_MM["top"]    * _PX_PER_MM
    mr = _MARGIN_MM["right"]  * _PX_PER_MM
    mb = _MARGIN_MM["bottom"] * _PX_PER_MM

    plot_w  = n_samples * px_per_sample
    plot_h  = n_leads * (lead_h + lead_gap) - lead_gap
    svg_w   = ml + plot_w + mr
    svg_h   = mt + plot_h + mb + 20  # extra for title

    minor_px = 1 * _PX_PER_MM
    major_px = 5 * _PX_PER_MM

    p = minor_px  # shorthand
    q = major_px

    out: List[str] = []
    a = out.append  # shorthand

    a(f'<svg xmlns="http://www.w3.org/2000/svg"'
      f' width="{svg_w:.1f}" height="{svg_h:.1f}"'
      f' viewBox="0 0 {svg_w:.1f} {svg_h:.1f}">')

    # ── defs ────────────────────────────────────────────────────────────
    a("<defs>")
    a(f'  <pattern id="mgrid" patternUnits="userSpaceOnUse"'
      f' width="{p:.3f}" height="{p:.3f}">')
    a(f'    <path d="M {p:.3f} 0 L 0 0 0 {p:.3f}"'
      f' fill="none" stroke="{_C_MINOR}" stroke-width="0.5"/>')
    a(f'  </pattern>')
    a(f'  <pattern id="Mgrid" patternUnits="userSpaceOnUse"'
      f' width="{q:.3f}" height="{q:.3f}">')
    a(f'    <rect width="{q:.3f}" height="{q:.3f}" fill="url(#mgrid)"/>')
    a(f'    <path d="M {q:.3f} 0 L 0 0 0 {q:.3f}"'
      f' fill="none" stroke="{_C_MAJOR}" stroke-width="1"/>')
    a(f'  </pattern>')
    for i in range(n_leads):
        a(f'  <clipPath id="cp{i}">')
        a(f'    <rect x="0" y="0" width="{plot_w:.1f}" height="{lead_h:.1f}"/>')
        a(f'  </clipPath>')
    a("</defs>")

    # ── title ────────────────────────────────────────────────────────────
    tx = svg_w / 2
    a(f'<text x="{tx:.1f}" y="14" text-anchor="middle"'
      f' font-family="Courier New,Courier,monospace" font-size="13" font-weight="bold"'
      f' fill="{_C_LABEL}">{title}</text>')
    dur_s = n_samples / sample_rate
    meta  = f"{sample_rate} Hz · {paper_speed} mm/s · {amplitude_scale} mm/mV · {dur_s:.1f}s"
    a(f'<text x="{tx:.1f}" y="26" text-anchor="middle"'
      f' font-family="Courier New,Courier,monospace" font-size="9" fill="#999">{meta}</text>')

    # ── main group ───────────────────────────────────────────────────────
    a(f'<g transform="translate({ml:.1f},{mt:.1f})">')

    # paper background
    a(f'<rect width="{plot_w:.1f}" height="{plot_h:.1f}" fill="{_C_BG}"/>')
    # grid overlay
    a(f'<rect width="{plot_w:.1f}" height="{plot_h:.1f}" fill="url(#Mgrid)"/>')

    # second ticks and labels
    n_sec = int(n_samples / sample_rate) + 1
    for s in range(n_sec):
        x  = s * sample_rate * px_per_sample
        sw = "0.9" if s % 5 == 0 else "0.35"
        sc = "#FF4040" if s % 5 == 0 else "#FFB0B0"
        a(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="-4" y2="{plot_h:.1f}"'
          f' stroke="{sc}" stroke-width="{sw}" opacity="0.5"/>')
        a(f'<text x="{x+1.5:.1f}" y="-5"'
          f' font-family="Courier New,Courier,monospace" font-size="8" fill="#888">{s}s</text>')

    # ── per-lead content ─────────────────────────────────────────────────
    for i, (name, lead_data) in enumerate(zip(lead_names, ecg)):
        y0       = i * (lead_h + lead_gap)
        baseline = y0 + lead_h / 2

        # lead label (left of plot area, not clipped)
        a(f'<text x="-4" y="{baseline + 4:.1f}" text-anchor="end"'
          f' font-family="Courier New,Courier,monospace" font-size="11" font-weight="bold"'
          f' fill="{_C_LABEL}">{name}</text>')

        # separator + baseline inside clip
        a(f'<g clip-path="url(#cp{i})" transform="translate(0,{y0:.1f})">')
        a(f'  <line x1="0" x2="{plot_w:.1f}" y1="0" y2="0"'
          f' stroke="{_C_MAJOR}" stroke-width="0.8"/>')
        a(f'  <line x1="0" x2="{plot_w:.1f}" y1="{lead_h/2:.1f}" y2="{lead_h/2:.1f}"'
          f' stroke="{_C_BASELINE}" stroke-width="0.5" stroke-dasharray="3,5"/>')

        # ECG trace as a single <polyline>
        pts: List[str] = []
        half = lead_h / 2
        for j in range(n_samples):
            x  = j * px_per_sample
            y  = half - float(lead_data[j]) * px_per_mv
            y  = max(1.0, min(lead_h - 1.0, y))  # soft clip
            pts.append(f"{x:.2f},{y:.2f}")

        # Write points in chunks to avoid one enormous line
        chunk = 500
        pts_str = " ".join(pts)
        a(f'  <polyline points="{pts_str}"'
          f' fill="none" stroke="{_C_TRACE}" stroke-width="1.2"'
          f' stroke-linejoin="round" stroke-linecap="round"/>')
        a("</g>")

    # bottom border
    a(f'<line x1="0" x2="{plot_w:.1f}" y1="{plot_h:.1f}" y2="{plot_h:.1f}"'
      f' stroke="{_C_MAJOR}" stroke-width="1"/>')

    a("</g>")  # close main group
    a("</svg>")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════
#   Matplotlib PNG helpers  (mirrors pulse2pulse_train.py run_inference)
# ═══════════════════════════════════════════════════════════════════════════

def plot_generated_single(
    fake_ecg: "np.ndarray",
    sampling_rate: int = 500,
    title: str = "Generated ECG",
) -> "matplotlib.figure.Figure":
    """Plot all leads of a single generated ECG as stacked subplots.

    Parameters
    ----------
    fake_ecg : np.ndarray
        Shape ``(n_leads, T)``.
    sampling_rate : int
        Samples per second — used to label the x-axis in seconds.
    title : str
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fake_ecg = np.asarray(fake_ecg)
    num_leads, T = fake_ecg.shape
    t = np.arange(T) / sampling_rate

    fig, axs = plt.subplots(num_leads, 1, figsize=(14, num_leads * 1.6), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i in range(num_leads):
        label = LEAD_NAMES_DEFAULT[i] if i < len(LEAD_NAMES_DEFAULT) else f"Ch{i}"
        axs[i].plot(t, fake_ecg[i], linewidth=0.8)
        axs[i].set_ylabel(label, fontsize=9)
        axs[i].grid(True, alpha=0.3)
        axs[i].tick_params(labelsize=7)

    axs[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


def plot_generated_batch(
    fake_ecgs: "np.ndarray",
    num_of_plots: int = 4,
    sampling_rate: int = 500,
    title: str = "Generated ECG batch",
) -> "matplotlib.figure.Figure":
    """Plot multiple generated ECG samples, each sample as a column of lead subplots.

    Parameters
    ----------
    fake_ecgs : np.ndarray
        Shape ``(N, n_leads, T)``.
    num_of_plots : int
        How many samples from the batch to include (at most ``N``).
    sampling_rate : int
        Samples per second.
    title : str
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fake_ecgs = np.asarray(fake_ecgs)
    N = min(fake_ecgs.shape[0], num_of_plots)
    num_leads = fake_ecgs.shape[1]
    T = fake_ecgs.shape[2]
    t = np.arange(T) / sampling_rate

    fig, axs = plt.subplots(num_leads, N, figsize=(5 * N, num_leads * 1.4), sharex=True)
    if N == 1:
        axs = axs[:, np.newaxis]

    for col in range(N):
        axs[0, col].set_title(f"Sample {col + 1}", fontsize=9)
        for row in range(num_leads):
            axs[row, col].plot(t, fake_ecgs[col, row], linewidth=0.7)
            axs[row, col].grid(True, alpha=0.3)
            axs[row, col].tick_params(labelsize=6)
            if col == 0:
                label = LEAD_NAMES_DEFAULT[row] if row < len(LEAD_NAMES_DEFAULT) else f"Ch{row}"
                axs[row, col].set_ylabel(label, fontsize=8)

    for col in range(N):
        axs[-1, col].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#   PDF — SVG → cairosvg
# ═══════════════════════════════════════════════════════════════════════════

def _save_pdf(svg_str: str, path: Path) -> None:
    try:
        import cairosvg
    except ImportError as exc:
        raise ImportError(
            "cairosvg is required for PDF output. "
            "Install with: pip install 'ecgen[plot]'"
        ) from exc

    cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"), write_to=str(path))
