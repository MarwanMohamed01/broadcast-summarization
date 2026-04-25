import { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

const BEFORE = "ISRAEL AIR STRIKE ON SOUTHERN LEBAN0N KILS SEVEN PEPPLE INCLUDING FOUR-YEAR-OLD GIRL";
const AFTER  = "ISRAELI AIR STRIKE ON SOUTHERN LEBANON KILLS SEVEN PEOPLE, INCLUDING FOUR-YEAR-OLD GIRL";

/** Naive char-diff for visualisation only — not the real LCS algo. */
function diffChars(a, b) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (a[i] === b[j]) { out.push({ ch: b[j], type: "same" }); i++; j++; }
    else if (b[j] && (!a[i] || a[i + 1] === b[j])) { out.push({ ch: b[j], type: "ins" }); j++; if (a[i] !== b[j]) i++; }
    else { out.push({ ch: b[j] || "", type: "ins" }); j++; i++; }
  }
  return out;
}

export default function LLMCleanStage() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [showAfter, setShowAfter] = useState(false);

  useEffect(() => {
    if (!inView) return;
    const id = setTimeout(() => setShowAfter(true), 900);
    return () => clearTimeout(id);
  }, [inView]);

  const diff = diffChars(BEFORE, AFTER);

  return (
    <StagePanel
      number="6"
      title="LLM cleaning pass"
      blurb={
        <p>
          A second pass sends batches of headlines to <strong>Gemini 2.5
          Flash</strong> with a tightly-scoped prompt: split merged items,
          complete truncated ones using cross-headline context, and fix
          obvious OCR typos.  This single pass lifts F1 from
          {" "}<strong>84.8% → 88.1%</strong> on the 14-hour validation set.
        </p>
      }
      accent="visual"
      data={{
        cleaner_model: "gemini-2.5-flash",
        batch_size: 15,
        scope: "split / complete / fix typos — never invent",
      }}
    >
      <div ref={ref} className="space-y-3">
        <div className="text-xs text-text-muted">Before · raw OCR</div>
        <motion.div
          animate={{ opacity: showAfter ? 0.45 : 1 }}
          transition={{ duration: 0.4 }}
          className="card p-3 mono text-xs leading-relaxed"
        >
          {BEFORE}
        </motion.div>

        <div className="text-xs text-text-muted mt-4">After · LLM-cleaned</div>
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={inView ? { opacity: showAfter ? 1 : 0, y: showAfter ? 0 : 6 } : {}}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="card p-3 mono text-xs leading-relaxed"
        >
          {diff.map((d, i) => (
            <span
              key={i}
              className={d.type === "ins" ? "bg-success/30 text-success rounded-sm" : ""}
            >{d.ch}</span>
          ))}
        </motion.div>

        <div className="flex gap-3 pt-2 text-xs text-text-muted">
          <span className="pill">F1 84.8% → 88.1%</span>
          <span className="pill">CER 23.4% → 13.9%</span>
        </div>
      </div>
    </StagePanel>
  );
}
