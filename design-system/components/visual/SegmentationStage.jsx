import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

const STREAM = [
  ["GAZA", "HEALTH", "MINISTRY:", "716", "PALESTINIANS", "KILLED",
   "IN", "ISRAELI", "ATTACKS", "SINCE", "CEASEFIRE"],
  ["-"],
  ["LEBANON'S", "HEALTH", "MINISTRY:", "FOUR", "PEOPLE", "KILLED",
   "AND", "39", "INJURED", "IN", "RAID"],
  ["-"],
  ["ARTEMIS", "II", "ASTRONAUTS", "GLIMPSE", "MOON'S", "GRAND", "CANYON"],
];

export default function SegmentationStage() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [stepIdx, setStepIdx] = useState(0);   // index into flattened tokens
  const [headlines, setHeadlines] = useState([[]]);

  // Build a flat list of (token, isDash) so we can step one at a time
  const flat = STREAM.flat();

  useEffect(() => {
    if (!inView) return;
    let i = 0;
    const id = setInterval(() => {
      if (i >= flat.length) { clearInterval(id); return; }
      const tok = flat[i];
      setStepIdx(i + 1);
      setHeadlines((prev) => {
        const next = prev.map((h) => [...h]);
        if (tok === "-") {
          next.push([]); // start a new headline
        } else {
          next[next.length - 1].push(tok);
        }
        return next;
      });
      i += 1;
    }, 200);
    return () => clearInterval(id);
  }, [inView]);

  return (
    <StagePanel
      number="5"
      title="Segmentation into headlines"
      blurb={
        <p>
          The continuous OCR token stream is split into individual news items
          at every <code className="mono">" - "</code> delimiter.  Fuzzy
          deduplication then collapses near-identical re-readings of the same
          headline (the ticker repeats every few minutes), keeping the
          highest-quality version of each.
        </p>
      }
      accent="visual"
      data={{
        delimiter: " - ",
        dedup_threshold: 60,
        partial_threshold: 80,
      }}
    >
      <div ref={ref} className="grid grid-cols-12 gap-4">
        {/* token stream (left) */}
        <div className="col-span-5">
          <div className="text-xs text-text-muted mb-2">Stream</div>
          <div className="flex flex-wrap gap-1 max-h-44 overflow-hidden">
            {flat.slice(0, stepIdx).map((t, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: t === "-" ? 0.4 : 1, y: 0 }}
                transition={{ duration: 0.15 }}
                className={`mono text-[11px] px-1.5 py-0.5 rounded
                  ${t === "-" ? "bg-warn/30 text-warn" : "bg-surface-2 text-text-muted"}`}
              >
                {t}
              </motion.span>
            ))}
          </div>
        </div>

        {/* arrow */}
        <div className="col-span-1 flex items-center justify-center text-text-subtle">
          →
        </div>

        {/* headline cards (right) */}
        <div className="col-span-6 space-y-2">
          <div className="text-xs text-text-muted mb-2">
            Headlines <span className="mono">{headlines.filter(h=>h.length).length}</span>
          </div>
          <AnimatePresence>
            {headlines.filter((h) => h.length).map((h, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 6, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                className="card p-2.5 text-xs leading-snug"
              >
                {h.join(" ")}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </StagePanel>
  );
}
