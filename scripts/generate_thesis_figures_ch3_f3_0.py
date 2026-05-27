"""Generate F3_0 — Frame Sampling & Ticker Cropping process diagram.

Produces:
    img/F3_0_frame_sampling.pdf
    img/F3_0_frame_sampling.png

Layout (left → right):
    [8 video frames] → [single frame + ticker highlight] → [3 ticker strips]

Run:
    python scripts/generate_thesis_figures_ch3_f3_0.py
"""
from __future__ import annotations

import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

PROJECT_DIR = Path(__file__).parent.parent.resolve()
IMG_DIR = PROJECT_DIR / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── color palette (matches graphviz thesis figures) ──────────────────────────
BLUE_FILL    = "#D6E8F7"
BLUE_BORDER  = "#2E75B6"
AMBER_FILL   = "#FFF3CD"
AMBER_BORDER = "#E6A817"
TEAL_FILL    = "#E0F7F7"
TEAL_BORDER  = "#2E9FA0"
GRAY_FILL    = "#EEEEEE"
GRAY_BORDER  = "#AAAAAA"
INK          = "#1A1A1A"
ANNOT_COLOR  = "#666666"
ANNOT_LINE   = "#BBBBBB"

# Pick best available serif font
_avail = {f.name for f in fm.fontManager.ttflist}
if "Times New Roman" in _avail:
    SERIF = "Times New Roman"
elif "DejaVu Serif" in _avail:
    SERIF = "DejaVu Serif"
else:
    SERIF = "serif"

plt.rcParams.update({"font.family": SERIF, "text.color": INK})

# ── figure setup ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 9.4, 3.5          # inches — reduced height, less blank space
DPI = 150

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Data space = inches (xlim and ylim match figure dimensions exactly → 1:1 mapping)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── drawing helpers ───────────────────────────────────────────────────────────

def rect(x, y, w, h, fill, border, lw=1.0, zorder=2, **kw):
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h,
        linewidth=lw, edgecolor=border, facecolor=fill, zorder=zorder, **kw))


def bold_arrow(x0, y, x1, color=BLUE_BORDER, lw=1.8):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14))


def txt(x, y, s, size=9.5, color=INK, ha="center", va="bottom", **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, **kw)


def dashed_annot(xy_tip, xytext, label, fsize=7.5):
    ax.annotate(label, xy=xy_tip, xytext=xytext,
        ha="left", va="center", fontsize=fsize, color=ANNOT_COLOR,
        arrowprops=dict(arrowstyle="-", color=ANNOT_LINE, lw=0.7,
                        linestyle=(0, (4, 2))))


# ── layout constants ──────────────────────────────────────────────────────────
Y_MID = FIG_H / 2   # = 1.75 — vertically centered in the new shorter figure

# PART 1 — 8 small video frames (16:9 ratio, wider than before)
FW   = 0.42         # frame width in data units (= inches at 1:1 mapping)
FH   = FW * 9 / 16  # ≈ 0.236  (proper 16:9 physical aspect)
FGAP = 0.038
SAMPLED = {0, 5}    # frames 1 and 6 (every 5th, 0-indexed)

f_x0 = 0.20
f_total_w = 8 * FW + 7 * FGAP   # = 3.626

# PART 2 — arrow 1
a1_x0 = f_x0 + f_total_w + 0.13
a1_len = 0.44
a1_x1  = a1_x0 + a1_len         # ≈ 4.40

# PART 3 — single frame (larger, 16:9)
BW = 1.44
BH = BW * 9 / 16                 # ≈ 0.810
b_x0 = a1_x1 + 0.12             # ≈ 4.52
b_y0 = Y_MID - BH / 2           # ≈ 1.345

# Ticker strip geometry (x: 14–96%, y: bottom 12%)
tx0 = b_x0 + 0.14 * BW
tw  = (0.96 - 0.14) * BW
th  = 0.12 * BH
ty0 = b_y0                       # flush with frame bottom

# PART 4 — arrow 2  (tighter gap — annotation only needs ~0.38 in)
a2_x0 = b_x0 + BW + 0.45
a2_len = 0.44
a2_x1  = a2_x0 + a2_len         # ≈ 6.85

# PART 5 — 3 stacked ticker strips
SW     = 9.4 - a2_x1 - 0.10 - 0.17   # remaining width (≈ 2.18 in)
SH     = 0.23
SGAP   = 0.10
N_S    = 3
s_x0   = a2_x1 + 0.10
s_htot = N_S * SH + (N_S - 1) * SGAP  # = 0.89
s_y0   = Y_MID - s_htot / 2           # centered on Y_MID

# ── PART 1: 8 video frames ───────────────────────────────────────────────────

txt(f_x0 + f_total_w / 2, Y_MID + FH / 2 + 0.52,
    "Source Video Frames", size=10.5, fontweight="bold")
txt(f_x0 + f_total_w / 2, Y_MID + FH / 2 + 0.28,
    "30 fps input", size=8.5, color=ANNOT_COLOR)

for i in range(8):
    x = f_x0 + i * (FW + FGAP)
    y = Y_MID - FH / 2
    fill   = BLUE_FILL   if i in SAMPLED else GRAY_FILL
    bcolor = BLUE_BORDER if i in SAMPLED else GRAY_BORDER
    lw     = 1.5 if i in SAMPLED else 0.6
    rect(x, y, FW, FH, fill, bcolor, lw=lw)
    txt(x + FW / 2, y - 0.12, f"frame {i+1}", size=7.5, color=INK)
    if i in SAMPLED:
        # filled diamond marker — avoids unicode glyph fallback
        ax.plot(x + FW / 2, y + FH + 0.07, "D",
                color=BLUE_BORDER, markersize=4.5,
                markerfacecolor=BLUE_BORDER, zorder=4)

txt(f_x0 + f_total_w / 2, Y_MID - FH / 2 - 0.32,
    "sample every 5th frame  →  6 fps effective",
    size=8.0, color="#555555", style="italic")

# ── PART 2: arrow "crop ticker region" ───────────────────────────────────────

bold_arrow(a1_x0, Y_MID, a1_x1)
txt((a1_x0 + a1_x1) / 2, Y_MID + 0.10,
    "crop\nticker\nregion",
    size=8, color=BLUE_BORDER, multialignment="center")

# ── PART 3: single sampled frame with ticker highlight ────────────────────────

txt(b_x0 + BW / 2, b_y0 + BH + 0.16,
    "Single sampled frame", size=10.5, fontweight="bold")

# Frame body
rect(b_x0, b_y0, BW, BH, "#F7F7F7", BLUE_BORDER, lw=1.3)

# Subtle "TV content" band above ticker
rect(b_x0 + 0.04, b_y0 + th + 0.04, BW - 0.08, BH - th - 0.08,
     "#EEF4FA", "none", lw=0)

# Ticker strip: amber highlight
rect(tx0, ty0, tw, th, AMBER_FILL, AMBER_BORDER, lw=1.3, zorder=3)
txt(tx0 + tw / 2, ty0 + th / 2,
    "ticker region   1049 × 87 px",
    size=7.0, color="#7A5200", fontweight="bold", va="center")

# Right annotation: y range
ann_rx = b_x0 + BW + 0.05
dashed_annot(
    xy_tip=(ann_rx, ty0 + th / 2),
    xytext=(ann_rx + 0.30, ty0 + th / 2),
    label="y: 88–100%",
)

# Bottom annotation: x range
ax.annotate("x: 14–96%",
    xy=(tx0 + tw / 2, ty0),
    xytext=(tx0 + tw / 2, ty0 - 0.24),
    ha="center", va="top", fontsize=7.5, color=ANNOT_COLOR,
    arrowprops=dict(arrowstyle="-", color=ANNOT_LINE, lw=0.7,
                    linestyle=(0, (4, 2))))

# ── PART 4: arrow "→ panorama stitching" ─────────────────────────────────────

bold_arrow(a2_x0, Y_MID, a2_x1)
txt((a2_x0 + a2_x1) / 2, Y_MID + 0.10,
    "panorama\nstitching",
    size=8, color=BLUE_BORDER, multialignment="center")

# ── PART 5: 3 consecutive ticker strips ──────────────────────────────────────

txt(s_x0 + SW / 2, s_y0 + s_htot + 0.17,
    "Consecutive Ticker Crops", size=10.5, fontweight="bold")

TICKER_TEXTS = [
    "IRAN HORMUZ WILL REOPEN…",
    "TEHRAN ENVOY ARRIVES IN…",
    "UN TALKS RESUME AFTER…",
]

for i in range(N_S):
    sy = s_y0 + i * (SH + SGAP)
    rect(s_x0, sy, SW, SH, TEAL_FILL, TEAL_BORDER, lw=0.9)
    ax.text(s_x0 + 0.10, sy + SH / 2, TICKER_TEXTS[i],
            ha="left", va="center", fontsize=7.5, color="#1A4A4A",
            fontfamily="monospace", zorder=3)

# Overlap indicator: horizontal ↔ arrow in gap between strips 0 and 1
ov_y  = s_y0 + SH + SGAP / 2
ov_x0 = s_x0 + SW * 0.52
ov_x1 = s_x0 + SW * 0.88
ax.annotate("", xy=(ov_x1, ov_y), xytext=(ov_x0, ov_y),
    arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.9,
                    mutation_scale=8))
txt((ov_x0 + ov_x1) / 2, ov_y + 0.05,
    "overlap", size=7, color="#777777")

txt(s_x0 + SW / 2, s_y0 - 0.15,
    "→ stitched into panorama", size=8.5, color="#555555", style="italic")

# ── outer border ─────────────────────────────────────────────────────────────

ax.add_patch(mpatches.FancyBboxPatch(
    (0.04, 0.04), FIG_W - 0.08, FIG_H - 0.08,
    boxstyle="square,pad=0",
    linewidth=0.7, edgecolor="#CCCCCC", facecolor="none",
    zorder=10, transform=ax.transData,
))

# ── save ─────────────────────────────────────────────────────────────────────

pdf_path = IMG_DIR / "F3_0_frame_sampling.pdf"
png_path = IMG_DIR / "F3_0_frame_sampling.png"

for p in (pdf_path, png_path):
    if p.exists():
        p.unlink()

fig.savefig(str(pdf_path), format="pdf", dpi=DPI,
            bbox_inches="tight", facecolor="white")
fig.savefig(str(png_path), format="png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.close(fig)


def _png_dims(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        f.seek(16)
        w, h = struct.unpack(">II", f.read(8))
    return w, h


w, h = _png_dims(png_path)
print(f"PDF : {pdf_path.name}  ({pdf_path.stat().st_size:,} B)")
print(f"PNG : {png_path.name}  {w}x{h} px  "
      f"({w/DPI:.2f} x {h/DPI:.2f} in  /  "
      f"{w/DPI*2.54:.1f} x {h/DPI*2.54:.1f} cm)")
