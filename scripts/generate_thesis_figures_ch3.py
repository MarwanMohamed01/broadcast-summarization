"""Generate Chapter 3 (Methodology) flow diagrams for the thesis.

Produces four figures, in both PDF (for ``\\includegraphics``) and PNG
(for quick preview / Word):

    img/F3_1_visual_pipeline_ocr.{pdf,png}
    img/F3_2_visual_pipeline_vlm.{pdf,png}
    img/F3_3_audio_pipeline.{pdf,png}
    img/F3_4_summarization_hierarchy.{pdf,png}

Layout — snake / zig-zag for the three linear pipelines:
  - Each row has 3 nodes; rows alternate direction (L→R, R→L, …).
  - Within a row, ``rank=same`` keeps the nodes on one horizontal band.
  - Invisible high-weight edges enforce the visual L→R declaration order.
  - The real flow edges are drawn with ``constraint=false`` so they
    paint the visible arrows without fighting the layout.
  - Cross-row "drop" edges (the only ones with ``constraint=true``)
    connect the end of one row to the start of the next and create the
    rank separation that makes the snake stack vertically.
  - Side branches (ground truth → evaluation) attach perpendicular to
    the main flow.

F3.4 uses a more compact two-row snake of its own because the
hierarchical summarisation pipeline has a natural fan-out shape rather
than a long linear chain.

Style:
  - Times-Roman, restrained grayscale + slate-blue accent reserved for
    the evaluation band.
  - Node fontsize 14 pt so the text remains readable when the figure
    is scaled to ~16 cm text width in LaTeX.

Shape vocabulary (consistent across all four figures):
    box (rounded) = processing stage
    cylinder      = on-disk data store
    parallelogram = pipeline input or final output

Run:
    python scripts/generate_thesis_figures_ch3.py

Requirements:
  - Python ``graphviz`` package (``pip install graphviz``)
  - Graphviz binary on PATH (``dot``)

Idempotent: re-running overwrites the PDFs/PNGs cleanly. Intermediate
``.gv`` source files are removed.
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

try:
    import graphviz
except ImportError:
    sys.stderr.write(
        "ERROR: the `graphviz` Python package is not installed.\n"
        "       Install with: pip install graphviz\n"
    )
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "ERROR: Pillow is not installed. Install with: pip install Pillow\n"
    )
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent.parent.resolve()
IMG_DIR = PROJECT_DIR / "img"
IMG_DIR.mkdir(exist_ok=True)

# --- shared style -----------------------------------------------------------

FONT = "Times-Roman"
INK = "#1A1A1A"
LINE = "#2E75B6"
FILL_STAGE = "#D6E8F7"
FILL_DATA = "#D8EDD8"
FILL_IO = "#FFF3CD"
ACCENT = "#7B2D8B"
ACCENT_FILL = "#EDE0F5"

DPI = "150"
NODE_FONTSIZE = "14"
EDGE_FONTSIZE = "11"

GRAPH_ATTR = {
    "rankdir": "TB",
    "splines": "true",
    "nodesep": "0.15",
    "ranksep": "0.30",
    "bgcolor": "white",
    "fontname": FONT,
    "fontsize": NODE_FONTSIZE,
    "margin": "0.08",
    "dpi": DPI,
}
NODE_ATTR = {
    "fontname": FONT,
    "fontsize": NODE_FONTSIZE,
    "color": LINE,
    "fontcolor": INK,
    "penwidth": "1.1",
    "style": "filled",
    "fillcolor": FILL_STAGE,
    "margin": "0.10,0.07",
}
EDGE_ATTR = {
    "fontname": FONT,
    "fontsize": EDGE_FONTSIZE,
    "color": LINE,
    "fontcolor": INK,
    "arrowsize": "0.7",
    "penwidth": "1.0",
}


def _new_graph(name: str) -> graphviz.Digraph:
    g = graphviz.Digraph(name=name, format="pdf")
    g.attr(**GRAPH_ATTR)
    g.attr("node", **NODE_ATTR)
    g.attr("edge", **EDGE_ATTR)
    return g


def _stage(g, node_id, label, accent=False):
    g.node(
        node_id, label,
        shape="box",
        style="filled,rounded",
        fillcolor=ACCENT_FILL if accent else FILL_STAGE,
        color=ACCENT if accent else LINE,
        fontcolor=ACCENT if accent else INK,
    )


def _data(g, node_id, label):
    g.node(node_id, label, shape="cylinder",
           style="filled", fillcolor=FILL_DATA, color="#4CAF50")


def _io(g, node_id, label, accent=False):
    g.node(
        node_id, label,
        shape="parallelogram",
        style="filled",
        fillcolor=ACCENT_FILL if accent else FILL_IO,
        color=ACCENT if accent else "#E6A817",
        fontcolor=ACCENT if accent else INK,
    )


# --- snake-row helpers ------------------------------------------------------

def _snake_row(g: graphviz.Digraph,
               visual_ids: list[str],
               flow_edges: list[tuple] = ()) -> None:
    """Lay ``visual_ids`` out as a single horizontal rank, in the given
    visual left-to-right order. ``flow_edges`` is a list of
    ``(src, dst[, attrs_dict])`` for the *real* arrows within the row;
    they are drawn with ``constraint=false`` so they don't fight the
    layout (the invisible chain established here owns that).

    Within-row edge labels render correctly with this scheme.
    """
    with g.subgraph(name=f"row_{visual_ids[0]}") as s:
        s.attr(rank="same")
        for nid in visual_ids:
            s.node(nid)
    # Invisible high-weight chain to pin visual L→R order.
    for a, b in zip(visual_ids, visual_ids[1:]):
        g.edge(a, b, style="invis", weight="100")
    # Real arrows for the data flow (do not influence ranking).
    for spec in flow_edges:
        src, dst, *rest = spec
        attrs = dict(rest[0]) if rest else {}
        attrs.setdefault("constraint", "false")
        g.edge(src, dst, **attrs)


def _drop(g: graphviz.Digraph, src: str, dst: str,
          **attrs) -> None:
    """A cross-row 'drop' edge — has ``constraint=true`` (default) so it
    creates the rank separation that stacks the snake vertically."""
    g.edge(src, dst, **attrs)


# --- F3.1 — Visual Pipeline (OCR variant) ----------------------------------
# Snake, 3 nodes per row, 4 rows.
# Row 1 (L→R):  Source video → Frame sampling → Scroll detection
# Row 2 (R→L):  drop into Panorama stitching → Multi-engine OCR → Segmentation
# Row 3 (L→R):  drop into Raw headlines → Re-segmentation → LLM cleanup
# Row 4 (R→L):  drop into Cleaned headlines → Evaluation → Report
#               (Ground truth attaches perpendicular to Evaluation)
# ---------------------------------------------------------------------------

def build_f3_1_visual_ocr() -> graphviz.Digraph:
    g = _new_graph("F3_1_visual_pipeline_ocr")

    # Row 1
    _io(g,    "video",   "Source video")
    _stage(g, "frames",  "Frame sampling\n+ ticker crop")
    _stage(g, "scroll",  "Scroll\ndetection")
    _snake_row(g,
        ["video", "frames", "scroll"],
        flow_edges=[
            ("video", "frames"),
            ("frames", "scroll"),
        ])

    # Row 2 (R→L visually: rightmost is "stitch", flow stitch→ocr→segment)
    _stage(g, "segment", "Headline\nsegmentation")
    _stage(g, "ocr",     "Multi-engine\nOCR")
    _stage(g, "stitch",  "Panorama\nstitching")
    _snake_row(g,
        ["segment", "ocr", "stitch"],
        flow_edges=[
            ("stitch", "ocr"),
            ("ocr",    "segment"),
        ])
    _drop(g, "scroll", "stitch")

    # Row 3 (L→R: raw → resegment → clean)
    _data(g,  "raw",       "Raw headlines")
    _stage(g, "resegment", "Headline\nre-segmentation")
    _stage(g, "clean",     "LLM-assisted\ntext cleanup")
    _snake_row(g,
        ["raw", "resegment", "clean"],
        flow_edges=[
            ("raw",       "resegment"),
            ("resegment", "clean"),
        ])
    _drop(g, "segment", "raw")

    # Row 4 (R→L visually: rightmost is "cleaned", flow cleaned→eval→report)
    _io(g,    "report",  "Evaluation\nreport",   accent=True)
    _stage(g, "eval",    "Evaluation\n(P, R, F1,\nCER, WER)", accent=True)
    _io(g,    "cleaned", "Cleaned\nheadlines", accent=True)
    _snake_row(g,
        ["report", "eval", "cleaned"],
        flow_edges=[
            ("cleaned", "eval",   {"color": ACCENT, "fontcolor": ACCENT}),
            ("eval",    "report", {"color": ACCENT}),
        ])
    _drop(g, "clean", "cleaned", color=ACCENT)

    # Ground truth — perpendicular side branch from below into Evaluation.
    _data(g, "gt", "Ground truth\nheadlines")
    _drop(g, "gt", "eval", color=ACCENT, style="dashed")

    return g


# --- F3.2 — Visual Pipeline (VLM variant) ----------------------------------
# Snake, 3 nodes per row, 4 rows.
# Row 1 (L→R):  Source video → Frame sampling → Scroll detection
# Row 2 (R→L):  Tiling → VLM call → JSON parsing
#                 (drops in from "Panorama stitching" which is on row 2 right)
# Row 3 (L→R):  Cross-tile aggregation → Consolidated headlines → Evaluation
# (Ground truth attaches perpendicular to Evaluation.)
# ---------------------------------------------------------------------------

def build_f3_2_visual_vlm() -> graphviz.Digraph:
    g = _new_graph("F3_2_visual_pipeline_vlm")

    # Row 1
    _io(g,    "video",  "Source video")
    _stage(g, "frames", "Frame sampling\n+ ticker crop")
    _stage(g, "scroll", "Scroll\ndetection")
    _snake_row(g,
        ["video", "frames", "scroll"],
        flow_edges=[
            ("video",  "frames"),
            ("frames", "scroll"),
        ])

    # Row 2 (R→L: stitch → tile → vlm)
    _stage(g, "vlm",    "Vision-\nlanguage model")
    _stage(g, "tile",   "Panorama\ntiling")
    _stage(g, "stitch", "Panorama\nstitching")
    _snake_row(g,
        ["vlm", "tile", "stitch"],
        flow_edges=[
            ("stitch", "tile"),
            ("tile",   "vlm"),
        ])
    _drop(g, "scroll", "stitch")

    # Row 3 (L→R: parse → agg → combined)
    _stage(g, "parse",    "Response\nparsing")
    _stage(g, "agg",      "Cross-tile\naggregation")
    _io(g,    "combined", "Consolidated\nheadlines", accent=True)
    _snake_row(g,
        ["parse", "agg", "combined"],
        flow_edges=[
            ("parse", "agg"),
            ("agg",   "combined", {"color": ACCENT, "fontcolor": ACCENT}),
        ])
    _drop(g, "vlm", "parse")

    # Row 4 (R→L: combined → eval → report)
    _io(g,    "report", "Evaluation\nreport",            accent=True)
    _stage(g, "eval",   "Evaluation\n(P, R, F1, CER)",   accent=True)
    _data(g,  "gt",     "Ground truth\nheadlines")
    _snake_row(g,
        ["report", "eval", "gt"],
        flow_edges=[
            ("gt",   "eval",   {"color": ACCENT, "style": "dashed"}),
            ("eval", "report", {"color": ACCENT}),
        ])
    _drop(g, "combined", "eval", color=ACCENT)

    return g


# --- F3.3 — Audio Pipeline --------------------------------------------------
# Snake, 3 nodes per row, 3 rows.
# Row 1 (L→R):  Source video → Audio extraction → Audio waveform
# Row 2 (R→L):  Consolidation → Whisper transcription → 20-min chunking
# Row 3 (L→R):  Transcript → Evaluation → Report
# (Ground truth attaches perpendicular to Evaluation.)
# ---------------------------------------------------------------------------

def build_f3_3_audio() -> graphviz.Digraph:
    g = _new_graph("F3_3_audio_pipeline")

    # Row 1
    _io(g,    "video",   "Source video")
    _stage(g, "extract", "Audio track\nextraction")
    _data(g,  "wav",     "Audio\nwaveform")
    _snake_row(g,
        ["video", "extract", "wav"],
        flow_edges=[
            ("video",   "extract"),
            ("extract", "wav"),
        ])

    # Row 2 (R→L: chunk → whisper → consolidate)
    _stage(g, "consolidate", "Segment\nconsolidation")
    _stage(g, "whisper",     "ASR\ntranscription")
    _stage(g, "chunk",       "Long-form\nchunking")
    _snake_row(g,
        ["consolidate", "whisper", "chunk"],
        flow_edges=[
            ("chunk",   "whisper"),
            ("whisper", "consolidate"),
        ])
    _drop(g, "wav", "chunk")

    # Row 3 (L→R: transcript → eval → report)
    _io(g,    "transcript", "Transcript", accent=True)
    _stage(g, "eval",       "Transcription\nevaluation\n(WER, CER)", accent=True)
    _io(g,    "report",     "Evaluation\nreport", accent=True)
    _snake_row(g,
        ["transcript", "eval", "report"],
        flow_edges=[
            ("transcript", "eval",   {"color": ACCENT, "fontcolor": ACCENT}),
            ("eval",       "report", {"color": ACCENT}),
        ])
    _drop(g, "consolidate", "transcript", color=ACCENT)

    # Ground truth — perpendicular side branch from below into Evaluation.
    _data(g, "gt", "Ground truth\ntranscript")
    _drop(g, "gt", "eval", color=ACCENT, style="dashed")

    return g


# --- F3.4 — Hierarchical Summarisation --------------------------------------
# A two-row LR snake (more compact than the original tall column) with a
# perpendicular side annotation for the variance experiment.
# Row 1 (L→R):  Pipeline input → Chunking → Per-chunk text → Level-1 LLMs
# Row 2 (R→L):  Report ← Evaluation ← Per-model summary ← Level-2 LLMs
#                              (per-chunk summaries cylinder sits between rows)
# ---------------------------------------------------------------------------

def build_f3_4_summarization() -> graphviz.Digraph:
    """Clean snake layout, single linear main flow:

      input -> chunking -> per-chunk text -> L1 -> per-chunk summaries ->
               L2 -> per-model summary -> evaluation -> report

    Plus a single dashed variance annotation attached to L1, and a
    reference-summary cylinder attached perpendicular to evaluation
    (mirrors Ground truth in F3.1-F3.3).
    """
    g = _new_graph("F3_4_summarization_hierarchy")

    # Row 1 (L→R, 3): input → chunking → per-chunk text
    _io(g,    "input",  "Pipeline input\n(transcript or\nheadlines)")
    _stage(g, "chunk",  "Time-based\nchunking")
    _data(g,  "chunks", "Per-chunk text")
    _snake_row(g,
        ["input", "chunk", "chunks"],
        flow_edges=[
            ("input", "chunk"),
            ("chunk", "chunks"),
        ])

    # Row 2 (R→L, 3): drop from "chunks" (rightmost of row 1) down to L1
    # (rightmost of row 2). Flow then goes L1 → per-chunk summaries →
    # L2, with L2 on the left so its drop continues into row 3 (also left).
    _stage(g, "l2",
           "Level 2 summarisation\n(synthesis,\n9 LLMs in parallel)",
           accent=True)
    _data(g, "l1_out", "Per-chunk\nsummaries")
    _stage(g, "l1",
           "Level 1 summarisation\n(per-chunk,\n9 LLMs in parallel)",
           accent=True)
    _snake_row(g,
        ["l2", "l1_out", "l1"],
        flow_edges=[
            ("l1",     "l1_out",
             {"color": ACCENT, "fontcolor": ACCENT}),
            ("l1_out", "l2",
             {"label": "model summaries",
              "color": ACCENT, "fontcolor": ACCENT}),
        ])
    _drop(g, "chunks", "l1", label="chunk text",
          color=ACCENT, fontcolor=ACCENT)

    # Row 3 (L→R, 3): drop from L2 (leftmost of row 2) down to per-model
    # summary, then evaluation, then report.
    _io(g,    "summary", "Per-model\nfinal summary", accent=True)
    _stage(g, "eval",    "Evaluation\n(ROUGE +\nBERTScore)", accent=True)
    _io(g,    "report",  "Summarisation\nreport", accent=True)
    _snake_row(g,
        ["summary", "eval", "report"],
        flow_edges=[
            ("summary", "eval",
             {"color": ACCENT, "fontcolor": ACCENT}),
            ("eval",    "report",
             {"color": ACCENT}),
        ])
    _drop(g, "l2", "summary", color=ACCENT)

    # Reference summary — perpendicular side branch into Evaluation,
    # mirroring the GT cylinder in F3.1/F3.2/F3.3.
    _data(g, "ref", "Reference\nsummary")
    _drop(g, "ref", "eval", color=ACCENT, style="dashed")

    # Variance annotation — single small note, single dashed edge
    # attached to L1 (a point between the L1 and L2 nodes).
    g.node("variance",
           "Each summarisation stage\nrepeated 3x per model\n"
           "for variance analysis",
           shape="note", style="filled",
           fillcolor="#FDE8D0", color="#E07B2A", fontcolor="#E07B2A",
           fontname=FONT, fontsize="11")
    g.edge("variance", "l1",
           style="dashed", color=ACCENT,
           constraint="false", arrowhead="none")

    return g


# --- driver -----------------------------------------------------------------

FIGURES = [
    ("F3_1_visual_pipeline_ocr",     build_f3_1_visual_ocr),
    ("F3_2_visual_pipeline_vlm",     build_f3_2_visual_vlm),
    ("F3_3_audio_pipeline",          build_f3_3_audio),
    ("F3_4_summarization_hierarchy", build_f3_4_summarization),
]


def _png_dimensions(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        f.seek(16)
        w, h = struct.unpack(">II", f.read(8))
    return w, h


def _flatten_png_to_white(png_path: Path) -> None:
    """Composite a transparent-background PNG onto opaque white. Graphviz
    emits PNGs whose perimeter is alpha=0; viewers with dark themes
    render that as a black border. Flattening to white removes the
    apparent border."""
    img = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(bg, img).convert("RGB")
    combined.save(png_path, "PNG")


def _render_one(name: str, dot: graphviz.Digraph) -> tuple[Path, Path]:
    out_base = IMG_DIR / name
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    for p in (pdf_path, png_path):
        if p.exists():
            p.unlink()
    dot.format = "pdf"
    dot.render(filename=str(out_base), format="pdf", cleanup=True)
    dot.format = "png"
    dot.render(filename=str(out_base), format="png", cleanup=True)
    if not pdf_path.exists() or not png_path.exists():
        raise RuntimeError(f"render failed for {name}")
    _flatten_png_to_white(png_path)
    return pdf_path, png_path


# --- verification ----------------------------------------------------------

# Strings that must NOT appear in any rendered figure's DOT source.
# (Checked against `dot.source` per builder, since OCR'ing the PNG is
# overkill — the DOT source is the ground truth of what graphviz drew.)
FORBIDDEN_SUBSTRINGS = [
    "(48 items)", "(38 items)", "(39 headlines)", "(39 items)",
    "(100 items)", "(1 567 words)", "(59 chunks)",
    "(9 x 59)", "9 x 59", "531 calls", "59 paragraphs per model",
    "Whisper transcription", "Whisper-quality",
    "20-min", "20 min",
    "JSON response", "raw JSON",
]


def _check_no_opaque_black_border(png_path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Walks all 4 sides of the PNG; fails if any
    opaque (alpha > 0) near-black pixel is found on the perimeter."""
    img = Image.open(png_path).convert("RGB")  # we flattened to RGB already
    w, h = img.size
    px = img.load()
    # We composited onto white, so a real border would show as opaque
    # dark pixels here. Sample the entire perimeter.
    for x in range(w):
        for y in (0, h - 1):
            r, g, b = px[x, y]
            if max(r, g, b) < 40:
                return False, f"opaque dark pixel at ({x},{y}) = {(r,g,b)}"
    for y in range(h):
        for x in (0, w - 1):
            r, g, b = px[x, y]
            if max(r, g, b) < 40:
                return False, f"opaque dark pixel at ({x},{y}) = {(r,g,b)}"
    # And corners specifically, for clarity in the report.
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        r, g, b = px[x, y]
        if (r, g, b) != (255, 255, 255):
            return False, f"corner ({x},{y}) is {(r,g,b)} (expected white)"
    return True, "perimeter is opaque white"


def main() -> int:
    if shutil.which("dot") is None:
        sys.stderr.write(
            "ERROR: the Graphviz `dot` binary is not on PATH.\n"
            "       On Windows: download the installer from\n"
            "       https://graphviz.org/download/  (pick the 64-bit EXE)\n"
            "       and tick \"Add Graphviz to the system PATH\" during setup.\n"
            "       Then restart your shell and re-run this script.\n"
        )
        return 2

    dpi = int(DPI)
    # When embedded with \includegraphics[width=16cm] these are the actual
    # rendered dimensions on the thesis page.
    target_page_w_cm = 16.0
    max_page_h_cm = 12.0

    rendered = []
    forbidden_hits: list[tuple[str, str]] = []
    for name, builder in FIGURES:
        dot = builder()
        # Scan DOT source for forbidden substrings before rendering.
        src = dot.source
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in src:
                forbidden_hits.append((name, needle))
        pdf, png = _render_one(name, dot)
        w, h = _png_dimensions(png)
        rendered.append((name, pdf, png, w, h))

    print(f"Rendered {len(rendered)} figures to {IMG_DIR} at {dpi} DPI.\n")
    print(f"Native PNG dimensions and projected page-embed size when\n"
          f"included at \\includegraphics[width={target_page_w_cm:.0f}cm]:\n")
    print(f"  {'name':<32}  {'native (cm)':<14}  {'at 16 cm width':<14}  fits page?")
    print(f"  {'-' * 32}  {'-' * 14}  {'-' * 14}  {'-' * 10}")
    for name, pdf, png, w, h in rendered:
        in_w, in_h = w / dpi, h / dpi
        cm_w, cm_h = in_w * 2.54, in_h * 2.54
        scaled_h_cm = target_page_w_cm * (cm_h / cm_w)
        fits = "YES" if scaled_h_cm <= max_page_h_cm else \
               f"too tall by {scaled_h_cm - max_page_h_cm:.1f} cm"
        native = f"{cm_w:.1f} x {cm_h:.1f}"
        scaled = f"16.0 x {scaled_h_cm:.1f}"
        print(f"  {name:<32}  {native:<14}  {scaled:<14}  {fits}")
    print()
    for name, pdf, png, w, h in rendered:
        in_w, in_h = w / dpi, h / dpi
        print(f"  {name}")
        print(f"    PDF : {pdf.name}  ({pdf.stat().st_size:,} B)")
        print(f"    PNG : {png.name}  {w}x{h} px  "
              f"({in_w:.2f} x {in_h:.2f} in)")

    # --- Verification checklist ---
    print()
    print("Verification checklist:")
    print()

    # (1) opaque-white perimeter — black-border check
    all_borders_ok = True
    border_reasons = []
    for name, _pdf, png, _w, _h in rendered:
        ok, reason = _check_no_opaque_black_border(png)
        if not ok:
            all_borders_ok = False
            border_reasons.append((name, reason))
    if all_borders_ok:
        print("  [PASS] No black outer border on any figure.")
        print("         (Perimeter and all 4 corners are opaque white "
              "on all 4 PNGs.)")
    else:
        print("  [FAIL] Black border detected on:")
        for n, r in border_reasons:
            print(f"           - {n}: {r}")

    # (2) Forbidden substrings (item counts, implementation specifics)
    if not forbidden_hits:
        print("  [PASS] No forbidden substrings in any DOT source.")
        print("         (No item counts, no 'Whisper'/'20-min'/'JSON' in "
              "labels, no '531 calls'/'9 x 59'/'59 paragraphs per model'.)")
    else:
        print("  [FAIL] Forbidden substrings found in DOT source:")
        for n, needle in forbidden_hits:
            print(f"           - {n}: {needle!r}")

    # (3) F3.4 has a single linear main flow (no visible backward edges).
    # We scan only the *visible* edges in the DOT source — invisible
    # layout-ordering edges (`style=invis`) are layout scaffolding and
    # don't render, so they shouldn't count as backward arrows.
    f3_4_dot = build_f3_4_summarization().source
    visible_edges = [
        ln for ln in f3_4_dot.splitlines()
        if "->" in ln and "style=invis" not in ln
    ]
    backward_patterns = [
        ("l1_out -> l1",   "visible backward edge from summaries into L1"),
        ("summaries -> l1", "visible backward edge from summaries into L1"),
        ("summary -> l1",  "visible backward edge from summary into L1"),
        ("eval -> l1",     "visible backward edge from eval into L1"),
        ("eval -> l2",     "visible backward edge from eval into L2"),
        ("report -> ",     "visible outgoing edge from report (terminal node)"),
    ]
    # Also: variance should connect to ONE of L1 or L2 (per spec), not both.
    variance_targets = sum(
        1 for ln in visible_edges
        if ln.strip().startswith("variance ->")
    )
    f3_4_problems = [
        desc for pat, desc in backward_patterns
        if any(pat in ln for ln in visible_edges)
    ]
    if variance_targets > 1:
        f3_4_problems.append(
            f"variance note has {variance_targets} outgoing edges "
            "(spec says exactly one)")
    if not f3_4_problems:
        print("  [PASS] F3.4 has a single linear main flow.")
        print("         (No backward dashed arrows, no re-feeding of "
              "summaries into Level 1.)")
    else:
        print("  [FAIL] F3.4 still contains:")
        for p in f3_4_problems:
            print(f"           - {p}")

    # (4) Fits within 16 cm x 12 cm bounding box when scaled to 16 cm wide.
    all_fit = True
    for name, _pdf, _png, w, h in rendered:
        cm_w, cm_h = (w / dpi) * 2.54, (h / dpi) * 2.54
        scaled_h = 16.0 * (cm_h / cm_w)
        if scaled_h > 12.0:
            all_fit = False
            print(f"  [FAIL] {name}: at 16 cm wide, height would be "
                  f"{scaled_h:.1f} cm (>12 cm).")
    if all_fit:
        print("  [PASS] All four figures fit within 16 x 12 cm when "
              "embedded at width=16cm.")

    print()
    ok = all_borders_ok and not forbidden_hits and not f3_4_problems and all_fit
    if ok:
        print("All checks passed.")
        return 0
    print("Some checks failed — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
