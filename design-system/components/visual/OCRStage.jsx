import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Real cropped panorama segment + REAL EasyOCR bboxes when
 * `bboxesUrl` is provided and resolves. Otherwise falls back to a
 * synthetic bbox layout so the animation still plays.
 *
 * The bboxes JSON is produced by site/scripts/instrument_slice_A.py.
 */
const FALLBACK_TOKENS = [
  "GAZA", "HEALTH", "MINISTRY:", "716", "PALESTINIANS", "KILLED",
  "IN", "ISRAELI", "ATTACKS", "SINCE", "CEASEFIRE", "STARTED",
];

export default function OCRStage({ segmentSrc, bboxesUrl }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [visibleCount, setVisibleCount] = useState(0);
  const [bboxes, setBboxes] = useState(null);     // {x, y, w, h, text}[] — real, in original-slice px
  const [segSize, setSegSize] = useState(null);   // [w, h] of original slice
  const [isReal, setIsReal] = useState(false);

  // Load real bboxes if available
  useEffect(() => {
    if (!bboxesUrl) return;
    fetch(bboxesUrl).then(r => r.ok ? r.json() : null).then(d => {
      if (d?.words?.length) {
        setBboxes(d.words);
        setSegSize(d.segment_size_px);
        setIsReal(true);
      }
    }).catch(() => {});
  }, [bboxesUrl]);

  // Fallback synthetic boxes (used only if no real data)
  const synthetic = FALLBACK_TOKENS.map((t, i) => ({
    text: t,
    leftPct: (i * 7.5) + 2,
    widthPct: t.length * 1.4 + 1,
  }));

  // Drive the reveal: cap synthetic at its length, real at its length
  const totalReveal = isReal ? bboxes.length : synthetic.length;
  useEffect(() => {
    if (!inView) return;
    setVisibleCount(0);
    let i = 0;
    // Real bboxes can be 50+ words → speed up the stagger so the
    // animation finishes in ~3 s either way.
    const stepMs = isReal ? Math.max(40, Math.floor(2200 / totalReveal)) : 220;
    const id = setInterval(() => {
      i += 1;
      setVisibleCount(i);
      if (i >= totalReveal) clearInterval(id);
    }, stepMs);
    return () => clearInterval(id);
  }, [inView, isReal, totalReveal]);

  // Convert real bboxes (in original-slice px) to % of the on-screen segment.
  // We still display the same WebP segment image — its rendered width is
  // 100% of the container, so we scale x/w by segSize[0].
  const realOverlays = isReal && segSize
    ? bboxes.slice(0, visibleCount).map((b, i) => ({
        key: i,
        leftPct: (b.x / segSize[0]) * 100,
        widthPct: (b.w / segSize[0]) * 100,
        topPct:  (b.y / segSize[1]) * 100,
        heightPct: (b.h / segSize[1]) * 100,
        text: b.text,
      }))
    : null;

  const tokens = isReal
    ? bboxes.slice(0, visibleCount).map(b => b.text)
    : synthetic.slice(0, visibleCount).map(s => s.text);

  return (
    <StagePanel
      number="4"
      title="Optical character recognition"
      blurb={
        <p>
          Each panorama is sliced into overlapping 3000-px windows and fed to
          {" "}<strong>EasyOCR</strong> for word-level recognition.
          <strong> Tesseract</strong> runs a second pass dedicated to
          finding the thin <code className="mono">" - "</code> delimiter that
          separates one headline from the next — a glyph EasyOCR consistently
          skips. This hybrid strategy delivers the highest end-to-end F1 in
          the engine comparison.
          {isReal && (
            <> Boxes shown here are the <strong>actual EasyOCR detections</strong>{" "}
              on this 3000-px segment.
            </>
          )}
        </p>
      }
      accent="visual"
      data={{
        primary_engine: "EasyOCR (CRAFT + CRNN, English)",
        delimiter_engine: "Tesseract LSTM PSM 6",
        slice_width_px: 3000,
        slice_stride_px: 1500,
        bbox_source: isReal ? "instrument_slice_A.py" : "synthetic",
        word_count: isReal ? bboxes.length : null,
      }}
    >
      <div ref={ref} className="space-y-4">
        <div className="relative w-full">
          {segmentSrc && (
            <img
              src={segmentSrc}
              alt="Panorama segment"
              className={`w-full ${isReal ? "h-16 sm:h-20" : "h-12"} object-cover ring-1 ring-border rounded-sm`}
            />
          )}
          <div className="absolute inset-0">
            {realOverlays
              ? realOverlays.map(o => (
                  <motion.div
                    key={o.key}
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.16 }}
                    className="absolute ring-1 ring-visual rounded-[2px] bg-visual/10"
                    style={{
                      left: `${o.leftPct}%`,
                      width: `${o.widthPct}%`,
                      top: `${o.topPct}%`,
                      height: `${o.heightPct}%`,
                    }}
                  />
                ))
              : synthetic.slice(0, visibleCount).map((b, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.18 }}
                    className="absolute top-1.5 bottom-1.5 ring-1 ring-visual rounded-[2px] bg-visual/10"
                    style={{ left: `${b.leftPct}%`, width: `${b.widthPct}%` }}
                  />
                ))}
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-text-muted">
          <span>Token stream <span className="mono ml-2">{visibleCount} / {totalReveal}</span></span>
          {isReal && (
            <span className="text-[10px] uppercase tracking-wide text-success">live data</span>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5 min-h-16">
          {tokens.map((t, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="px-2 py-1 rounded-md bg-visual/10 text-visual mono text-xs"
            >
              {t}
            </motion.span>
          ))}
        </div>
      </div>
    </StagePanel>
  );
}
