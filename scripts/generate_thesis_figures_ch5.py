"""Generate Chapter 5 (Results) bar-chart figures for the thesis.

Produces four figures, each as a vector PDF (for ``\\includegraphics``)
and a 150-DPI PNG (for quick preview / Word):

    img/F5_1_ocr_engine_comparison.{pdf,png}
    img/F5_2_visual_variance_leaderboard.{pdf,png}
    img/F5_3_audio_summarization_leaderboard.{pdf,png}
    img/F5_4_ocr_vs_vlm_per_slice.{pdf,png}

Source values are read literally from the project's evaluation JSONs
(provenance is cited inline in each builder function below):

    results/validation_report.json           (F5.4 OCR rows)
    results/vlm_evaluation.json              (F5.4 VLM rows)
    results/vlm_opensource_evaluation.json   (cross-checked in
                                              ch5_results.md §2.4)
    results/variance_report.json             (F5.2)
    results/asr_summary_evaluation.json      (F5.3)
    ocr_comparison/output/comparison_report.json  (F5.1)

Style matches the four Chapter-3 diagrams in img/F3_*.pdf:
  - serif font (Times / Liberation Serif) for LaTeX-default look
  - restrained palette: slate-blue accent + grayscale, no other hues
  - no top/right spines, white background, no shadows / gradients
  - 150 DPI PNG; PDF is vector

Run:
    python scripts/generate_thesis_figures_ch5.py

Idempotent: re-running overwrites both formats cleanly.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).parent.parent.resolve()
IMG_DIR = PROJECT_DIR / "img"
IMG_DIR.mkdir(exist_ok=True)

# --- shared style -----------------------------------------------------------

DPI = 150
ACCENT = "#324A6D"          # slate-blue (matches F3 figures)
ACCENT_MID = "#7895B5"      # mid slate
ACCENT_LIGHT = "#C4D2E0"    # light slate
ACCENT_VLIGHT = "#E6ECF4"   # very light slate (matches F3 fills)
GRAY_DARK = "#444444"
GRAY_MID = "#888888"
GRAY_LIGHT = "#CCCCCC"
INK = "#222222"

# Try Times first; matplotlib falls back through this list if Times isn't
# installed (most Linux/macOS distros ship Liberation Serif or DejaVu Serif).
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                   "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRAY_DARK,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def _new_fig(width_in: float, height_in: float) -> tuple[plt.Figure, plt.Axes]:
    """Return a (fig, ax) with the shared style applied: white background,
    top + right spines removed."""
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_DARK)
    ax.spines["bottom"].set_color(GRAY_DARK)
    ax.tick_params(colors=INK, length=4, width=0.8)
    return fig, ax


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf = IMG_DIR / f"{stem}.pdf"
    png = IMG_DIR / f"{stem}.png"
    for p in (pdf, png):
        if p.exists():
            p.unlink()
    # Tight layout + a small uniform pad so labels never clip.
    fig.tight_layout(pad=0.6)
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, format="png", bbox_inches="tight", pad_inches=0.05,
                dpi=DPI)
    plt.close(fig)
    return pdf, png


def _png_dims(png_path: Path) -> tuple[int, int]:
    with open(png_path, "rb") as f:
        f.seek(16)
        w, h = struct.unpack(">II", f.read(8))
    return w, h


# --- F5.1 — OCR engine comparison ------------------------------------------
# Source: ocr_comparison/output/comparison_report.json
#   Pure engine F1 = comparison_report.txt §"PURE ENGINES" / "P@70 F1@70"
#   Augmented F1   = comparison_report.txt §"ENGINES + TESSERACT DASH"
# Values below are mean over slices A and B (slice values from .txt report).

OCR_ENGINES = [
    # (engine label, F1 pure, F1 dash-augmented)
    ("Tesseract",      85.35,  85.40),   # (84.2+86.5)/2, (86.3+84.5)/2
    ("EasyOCR",         0.00,  91.25),   # 0.0, (89.8+92.7)/2
    ("PaddleOCR",       8.45,  22.95),   # (6.1+10.8)/2, (17.9+28.0)/2
    ("docTR",          63.85,  80.25),   # (69.3+58.4)/2, (80.7+79.8)/2
    ("CRAFT+TrOCR",     0.00,  29.90),   # 0.0, (28.9+30.9)/2
]


def build_f5_1() -> tuple[Path, Path]:
    fig, ax = _new_fig(7.5, 4.6)

    labels = [e[0] for e in OCR_ENGINES]
    pure = [e[1] for e in OCR_ENGINES]
    augmented = [e[2] for e in OCR_ENGINES]

    x = list(range(len(labels)))
    bar_w = 0.38

    bars_pure = ax.bar([xi - bar_w / 2 for xi in x], pure, bar_w,
                       label="Engine alone",
                       color=GRAY_LIGHT, edgecolor=GRAY_DARK, linewidth=0.8)
    bars_aug = ax.bar([xi + bar_w / 2 for xi in x], augmented, bar_w,
                      label="Engine + Tesseract dash augmentation",
                      color=ACCENT, edgecolor=ACCENT, linewidth=0.8)

    # Value labels above each bar.
    for bars, values in [(bars_pure, pure), (bars_aug, augmented)]:
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                    f"{v:.1f}" if v > 0 else "0",
                    ha="center", va="bottom", fontsize=9, color=INK)

    # Highlight the production choice (EasyOCR + dash) with a tiny annotation.
    easy_aug_x = x[labels.index("EasyOCR")] + bar_w / 2
    easy_aug_y = augmented[labels.index("EasyOCR")]
    ax.annotate("production choice",
                xy=(easy_aug_x, easy_aug_y),
                xytext=(easy_aug_x, easy_aug_y + 8),
                ha="center", va="bottom", fontsize=9, color=ACCENT,
                style="italic",
                arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("F1 @ 70 % fuzzy-match threshold (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02),
              ncol=2, frameon=False)

    return _save(fig, "F5_1_ocr_engine_comparison")


# --- F5.2 — Visual-path summarization variance leaderboard -----------------
# Source: results/variance_report.json
#   models[i].{model, bertscore_f1_mean, bertscore_f1_std, n_successful_runs}
# (values multiplied by 100 in the chart for readability)

VISUAL_VARIANCE = [
    # (display, mean %, std %, n_successful, was_attempted)
    ("Gemini 2.5 Flash",            86.47, 0.09, 2, True),
    ("Llama 4 Scout 17B (Groq)",    85.46, 0.15, 3, True),
    ("Command-R (Cohere)",          85.09, 0.33, 3, True),
    ("Llama 3.3 70B (Groq)",        85.00, 0.32, 3, True),
    ("Qwen3 32B (Groq)",            84.44, 0.15, 3, True),
    ("Llama 3.1 8B (Groq)",         84.26, 0.62, 3, True),
    ("Llama 3 8B (HuggingFace)",    83.94, 1.07, 3, True),
    ("Llama 3.2 3B (Ollama local)", 83.81, 0.70, 2, True),
    ("Llama 3.1 8B (Ollama local)",  0.0,  0.0, 0, True),  # failed
]


def build_f5_2() -> tuple[Path, Path]:
    fig, ax = _new_fig(7.5, 5.4)

    # Models are already sorted descending by mean (failure last).
    # In a horizontal bar chart with `barh`, the FIRST item in the list is
    # drawn at the BOTTOM of the y-axis. We want highest mean at the TOP,
    # so reverse before plotting and invert later.
    rows = list(VISUAL_VARIANCE)
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    ns = [r[3] for r in rows]

    y_pos = list(range(len(rows)))

    # Plot all successful rows as bars. For n=0 (failed), just place an
    # italic label where the bar would have started — no horizontal rule
    # (the empty row is its own signal that nothing rendered).
    for i, (label, mean, std, n, _) in enumerate(rows):
        if n == 0:
            ax.text(82.1, i, "n = 0 (failed)",
                    va="center", ha="left", fontsize=9,
                    color=GRAY_MID, style="italic", zorder=3)
            continue
        # Solid bar
        ax.barh(i, mean, height=0.62,
                color=ACCENT, edgecolor=ACCENT, linewidth=0.6,
                xerr=std, capsize=3,
                error_kw=dict(ecolor=GRAY_DARK, elinewidth=0.9,
                              capthick=0.9))
        # Value label at end of bar
        suffix = "" if n == 3 else f"   (n = {n})"
        ax.text(mean + std + 0.10, i,
                f"{mean:.2f} ± {std:.2f}{suffix}",
                va="center", ha="left", fontsize=9, color=INK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top model on top

    ax.set_xlim(82, 88)
    ax.set_xticks([82, 83, 84, 85, 86, 87, 88])
    ax.set_xlabel("BERTScore F1 (%) — mean ± std across 3 runs")

    # Subtle grid behind bars to ease reading on a zoomed x-axis.
    ax.grid(axis="x", color=GRAY_LIGHT, linestyle="-", linewidth=0.5,
            alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    return _save(fig, "F5_2_visual_variance_leaderboard")


# --- F5.3 — Audio-path summarization leaderboard ---------------------------
# Source: results/asr_summary_evaluation.json
#   results[i].{display_name, bertscore_f1}
# L1 coverage values: PROGRESS_REPORT.md §7 Level-1 coverage table.

AUDIO_RESULTS = [
    # (display, BERT-F1 (0-1 scale), L1 coverage str, partial?, failed?)
    ("Command-R (Cohere)",          0.7779, "59/59",  False, False),
    ("Qwen3 32B (Groq)",            0.7695, "59/59",  False, False),
    ("Llama 3.2 3B (Ollama local)", 0.7684, "59/59",  False, False),
    ("Llama 3.3 70B (Groq)",        0.7678, "36/59",  True,  False),
    ("Llama 4 Scout 17B (Groq)",    0.7673, "59/59",  False, False),
    ("Llama 3.1 8B (Groq)",         0.7669, "59/59",  False, False),
    ("Gemini 2.5 Flash",            0.7666, "21/59",  True,  False),
    ("Llama 3 8B (HuggingFace)",    0.7580, "59/59",  False, False),
    ("Llama 3.1 8B (Ollama local)", 0.0,    "4/59",   False, True),
]


def build_f5_3() -> tuple[Path, Path]:
    fig, ax = _new_fig(7.5, 5.4)

    rows = list(AUDIO_RESULTS)
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]

    y_pos = list(range(len(rows)))

    for i, (label, mean, cov, partial, failed) in enumerate(rows):
        if failed:
            ax.text(0.7510, i, f"n/a — {cov} L1, failed",
                    va="center", ha="left", fontsize=9,
                    color=GRAY_MID, style="italic", zorder=3)
            continue
        if partial:
            # Hatched bar — same accent colour, lighter fill, dashed hatch.
            ax.barh(i, mean, height=0.62,
                    color=ACCENT_LIGHT, edgecolor=ACCENT, linewidth=0.8,
                    hatch="///")
        else:
            ax.barh(i, mean, height=0.62,
                    color=ACCENT, edgecolor=ACCENT, linewidth=0.6)
        # Value label + coverage suffix
        cov_note = f"   ({cov} L1)" if partial else ""
        ax.text(mean + 0.0008, i,
                f"{mean:.4f}{cov_note}",
                va="center", ha="left", fontsize=9, color=INK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlim(0.75, 0.79)
    ax.set_xticks([0.75, 0.76, 0.77, 0.78, 0.79])
    ax.set_xlabel("BERTScore F1 (vs. 2 212-word audio reference)")

    ax.grid(axis="x", color=GRAY_LIGHT, linestyle="-", linewidth=0.5,
            alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Legend explaining solid vs hatched.
    legend_handles = [
        Patch(facecolor=ACCENT, edgecolor=ACCENT,
              label="full Level-1 coverage (59/59 chunks)"),
        Patch(facecolor=ACCENT_LIGHT, edgecolor=ACCENT, hatch="///",
              label="partial Level-1 coverage  (tentative)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), frameon=False)

    return _save(fig, "F5_3_audio_summarization_leaderboard")


# --- F5.4 — OCR vs VLM head-to-head (per-slice P / R / F1) ----------------
# Source rows:
#   OCR (LLM-cleaned, slice_{A,B}): results/validation_report.json
#       results[3].thresholds["70"]  → slice_A
#       results[4].thresholds["70"]  → slice_B
#   VLM (Gemini raw, slice_{A,B}):  results/vlm_evaluation.json
#       per_run[0].thresholds["70"]  → variant=raw  gt_slice=slice_A
#       per_run[2].thresholds["70"]  → variant=raw  gt_slice=slice_B

OCR_VLM_DATA = {
    # slice: { system: (P%, R%, F1%) }
    "Slice A": {
        "OCR (LLM-cleaned)": (63.16, 88.89, 73.85),
        "VLM (Gemini, raw)": (48.00, 100.00, 64.86),
    },
    "Slice B": {
        "OCR (LLM-cleaned)": (65.79, 89.29, 75.76),
        "VLM (Gemini, raw)": (39.00, 100.00, 56.12),
    },
}

METRIC_ORDER = ["Precision", "Recall", "F1"]
# Within each system: light = P, mid = R, dark = F1.
OCR_TONES = [ACCENT_LIGHT, ACCENT_MID, ACCENT]      # OCR shaded in slate
VLM_TONES = [GRAY_LIGHT, GRAY_MID, GRAY_DARK]        # VLM in grayscale


def build_f5_4() -> tuple[Path, Path]:
    # A touch taller to accommodate two stacked legend rows above the axes.
    fig, ax = _new_fig(8.0, 5.0)

    slices = list(OCR_VLM_DATA.keys())
    systems = ["OCR (LLM-cleaned)", "VLM (Gemini, raw)"]

    n_metrics = len(METRIC_ORDER)
    n_systems = len(systems)
    bars_per_group = n_metrics * n_systems
    bar_w = 0.13
    group_width = bars_per_group * bar_w
    group_gap = 0.45    # space between Slice A and Slice B groups

    # Build x positions: each slice gets a centred block of 6 bars,
    # with the requested order P / R / F1 × OCR / VLM repeated by
    # metric (e.g. P_OCR, P_VLM, R_OCR, R_VLM, F1_OCR, F1_VLM) so the
    # tone-gradient (light → dark) flows naturally left → right and OCR
    # vs VLM are paired by metric.
    bar_positions = []     # list[(x, value, color, edgecolor, metric, system)]
    group_centres = []
    cur_x = 0.0
    for slice_label in slices:
        block_start = cur_x
        for m_idx, metric in enumerate(METRIC_ORDER):
            for s_idx, sys_label in enumerate(systems):
                p, r, f = OCR_VLM_DATA[slice_label][sys_label]
                value = (p, r, f)[m_idx]
                tones = OCR_TONES if sys_label.startswith("OCR") else VLM_TONES
                color = tones[m_idx]
                # OCR gets a slate edge, VLM a gray edge — keeps the system
                # distinction readable even when the tone is the same
                # nominal lightness.
                edge = ACCENT if sys_label.startswith("OCR") else GRAY_DARK
                bar_positions.append((cur_x, value, color, edge,
                                       metric, sys_label))
                cur_x += bar_w
        block_end = cur_x
        group_centres.append((block_start + block_end) / 2)
        cur_x += group_gap

    # Draw bars + value labels.
    for x, v, c, e, _metric, _sys in bar_positions:
        ax.bar(x, v, bar_w, color=c, edgecolor=e, linewidth=0.6,
               align="edge")
        ax.text(x + bar_w / 2, v + 1.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(slices)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Per-slice value at 70 % fuzzy threshold (%)")

    # One combined legend with all six (system × metric) swatches, laid
    # out as a 2-row × 3-col grid above the axes.
    # matplotlib fills legends COLUMN-major, so we interleave the handles
    # [OCR-P, VLM-P, OCR-R, VLM-R, OCR-F1, VLM-F1] to get a visually
    # row-major matrix: row 1 = OCR (slate tones), row 2 = VLM (gray
    # tones); columns = Precision / Recall / F1.
    combined_handles = [
        Patch(facecolor=OCR_TONES[0], edgecolor=ACCENT,
              label="OCR — Precision"),
        Patch(facecolor=VLM_TONES[0], edgecolor=GRAY_DARK,
              label="VLM — Precision"),
        Patch(facecolor=OCR_TONES[1], edgecolor=ACCENT,
              label="OCR — Recall"),
        Patch(facecolor=VLM_TONES[1], edgecolor=GRAY_DARK,
              label="VLM — Recall"),
        Patch(facecolor=OCR_TONES[2], edgecolor=ACCENT,
              label="OCR — F1"),
        Patch(facecolor=VLM_TONES[2], edgecolor=GRAY_DARK,
              label="VLM — F1"),
    ]
    ax.legend(handles=combined_handles, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False,
              columnspacing=1.6, handlelength=1.5)

    return _save(fig, "F5_4_ocr_vs_vlm_per_slice")


# --- driver -----------------------------------------------------------------

FIGURES = [
    ("F5.1  OCR engine comparison (5 engines × {alone, +dash augmentation}); "
     "EasyOCR+dash is the production choice at 91.3 % mean F1.",
     build_f5_1),
    ("F5.2  Visual-path summarisation BERT-F1 leaderboard with 3-run "
     "variance error bars; Ollama Llama 3.1 8B failed (n=0).",
     build_f5_2),
    ("F5.3  Audio-path summarisation BERT-F1 leaderboard; hatched bars flag "
     "two models with partial Level-1 coverage (tentative result).",
     build_f5_3),
    ("F5.4  OCR vs VLM head-to-head per slice — Precision, Recall, F1 at the "
     "70 % threshold; OCR wins F1 by 9–20 pp despite VLM's 100 % recall.",
     build_f5_4),
]


def main() -> int:
    print(f"Rendering {len(FIGURES)} figures into {IMG_DIR} at {DPI} DPI:\n")
    rendered = []
    for desc, builder in FIGURES:
        pdf, png = builder()
        w_px, h_px = _png_dims(png)
        w_in, h_in = w_px / DPI, h_px / DPI
        w_cm, h_cm = w_in * 2.54, h_in * 2.54
        rendered.append((pdf, png, w_in, h_in, w_cm, h_cm, desc))

    # Path list + dimensions
    print("Files:")
    for pdf, png, w_in, h_in, w_cm, h_cm, _desc in rendered:
        print(f"  {pdf.name}  ({pdf.stat().st_size:,} B)")
        print(f"  {png.name}  ({png.stat().st_size:,} B)  "
              f"{w_in:.2f} x {h_in:.2f} in   ({w_cm:.1f} x {h_cm:.1f} cm)")

    print()
    print("Summary:")
    for _pdf, _png, _wi, _hi, _wc, _hc, desc in rendered:
        # The description string holds the figure number + 1-line summary,
        # already shaped like the requested ASCII summary.
        print(f"  {desc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
