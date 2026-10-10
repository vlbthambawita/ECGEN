from ecgen import generate

# ── 1. CSV output (one file per sample, with lead header row) ───────────────
#generate(
#    model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
#    n_samples=10,
#    output_dir="outputs/csv/",
#    format="csv",
#    header=True,           # include I, II, V1–V6 header row
#    denorm=1.0,
#)
# Output: outputs/csv/sample_01.csv … sample_10.csv
# Each CSV: 5000 rows × 8 columns (I, II, V1, V2, V3, V4, V5, V6)

# ── 2. NPY output (all samples in one array) ────────────────────────────────
#generate(
#    model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
#    n_samples=100,
#    output_dir="outputs/npy/",
#    format="npy",
#    denorm=1.0,
#)
# Output: outputs/npy/generated_ecgs.npy  — shape (100, 8, 5000)

# ── 3. NPY + matplotlib PNG plots (mirrors run_inference in pulse2pulse_train.py) ──
generate(
    model_path="/work/vajira/DL2026/Pulse2Pulse/output/pulse2pulse_exp_ptbxl_full/checkpoints/pulse2pulse_exp_ptbxl_full_epoch:950.pt",
    n_samples=10,
    output_dir="outputs/npy_with_plots/",
    format="npy",
    ecgplot=True,
    plot_format="png",     # matplotlib PNG (default)
    num_plot_samples=4,    # save 4 individual PNGs + one batch overview PNG
    denorm=1.0,
)
# Output:
#   outputs/npy_with_plots/generated_ecgs.npy
#   outputs/npy_with_plots/plots/generated_sample_0000.png … _0003.png
#   outputs/npy_with_plots/plots/generated_batch_overview.png

# ── 4. CSV + interactive HTML plots ─────────────────────────────────────────
generate(
    model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
    n_samples=5,
    output_dir="outputs/csv_html/",
    format="csv",
    ecgplot=True,
    plot_format="html",    # interactive D3 (pan/zoom)
    denorm=1.0,
)
# Output:
#   outputs/csv_html/sample_1.csv … sample_5.csv
#   outputs/csv_html/plots/sample_1.html … sample_5.html

# ── 5. CSV + PDF plots  (pip install 'ecgen[plot]') ─────────────────────────
# generate(
#     model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
#     n_samples=5,
#     output_dir="outputs/csv_pdf/",
#     format="csv",
#     ecgplot=True,
#     plot_format="pdf",
#     denorm=1.0,
# )
