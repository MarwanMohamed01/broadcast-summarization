import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Per-model ROUGE-L / BERTScore F1 bar chart.
 *
 * Real-data mode: pass `scoresUrl` (the eval_scores.json built by
 *   site/scripts/build_assets.py) plus a `variant` key — one of:
 *     "visual_27min" | "visual_14h_cleaned" | "asr_final"
 *
 * Fallback mode: if scoresUrl is omitted or fails, the inline FALLBACK
 *   below renders so the chart still tells a story.
 */
const FALLBACK = [
  { model: "Llama 3.3 70B (Groq)",  rougeL: 0.41, bertscore: 0.87 },
  { model: "Gemini 2.5 Flash",      rougeL: 0.39, bertscore: 0.86 },
  { model: "Qwen3 32B",             rougeL: 0.38, bertscore: 0.85 },
  { model: "Llama 4 Scout",         rougeL: 0.36, bertscore: 0.85 },
  { model: "Cohere Command-R",      rougeL: 0.34, bertscore: 0.84 },
  { model: "Llama 3.1 8B (Groq)",   rougeL: 0.32, bertscore: 0.83 },
  { model: "HF Llama 3 8B",         rougeL: 0.30, bertscore: 0.82 },
  { model: "Llama 3.2 3B (Ollama)", rougeL: 0.27, bertscore: 0.81 },
  { model: "Llama 3.1 8B (Ollama)", rougeL: 0.26, bertscore: 0.81 },
];

export default function EvalStage({
  scoresUrl,
  variant = "visual_14h_cleaned",
  scores,                  // optional: directly-supplied rows
  accent = "visual",
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [sortKey, setSortKey] = useState("rougeL");
  const [rows, setRows] = useState(scores ?? null);
  const [isReal, setIsReal] = useState(Boolean(scores));

  useEffect(() => {
    if (rows) return;
    if (!scoresUrl) {
      setRows(FALLBACK);
      return;
    }
    fetch(scoresUrl)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        const variantRows = data?.[variant];
        if (Array.isArray(variantRows) && variantRows.length) {
          setRows(variantRows);
          setIsReal(true);
        } else {
          setRows(FALLBACK);
        }
      })
      .catch(() => setRows(FALLBACK));
  }, [scoresUrl, variant]);

  const data = rows ?? FALLBACK;
  const sorted = [...data].sort((a, b) => (b[sortKey] ?? 0) - (a[sortKey] ?? 0));
  const accentBg = accent === "audio" ? "bg-audio" : "bg-visual";

  return (
    <StagePanel
      number={accent === "audio" ? "5" : "8"}
      title="ROUGE & BERTScore evaluation"
      blurb={
        <p>
          Each summary is scored against a human-written reference using
          {" "}<strong>ROUGE-L</strong> (lexical overlap) and
          {" "}<strong>BERTScore F1</strong> (semantic similarity). The two
          metrics agree on the leaders and disagree at the tail — flip the
          sort to see which models trade lexical fidelity for semantic recall.
          {!isReal && (
            <> <em className="text-text-subtle">(Showing representative
            numbers — instrument the eval pipeline for real ones.)</em></>
          )}
        </p>
      }
      accent={accent}
      data={data}
    >
      <div ref={ref}>
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xs text-text-muted">Sort by</span>
          <button
            onClick={() => setSortKey("rougeL")}
            className={`text-xs px-2.5 py-1 rounded-md border transition-colors
              ${sortKey === "rougeL" ? "bg-surface-2 border-border text-text" : "border-transparent text-text-muted"}`}
          >ROUGE-L</button>
          <button
            onClick={() => setSortKey("bertscore")}
            className={`text-xs px-2.5 py-1 rounded-md border transition-colors
              ${sortKey === "bertscore" ? "bg-surface-2 border-border text-text" : "border-transparent text-text-muted"}`}
          >BERTScore</button>
          {isReal && (
            <span className="ml-auto text-[10px] uppercase tracking-wide text-success">
              live data
            </span>
          )}
        </div>

        <div className="space-y-2">
          {sorted.map((row, i) => (
            <motion.div
              key={row.model}
              layout
              transition={{ type: "spring", damping: 18, stiffness: 200 }}
              className="grid grid-cols-12 gap-3 items-center text-xs"
            >
              <span className="col-span-4 truncate text-text-muted">{row.model}</span>
              <div className="col-span-5 flex items-center gap-2">
                <div className="flex-1 h-2 rounded-full bg-surface-2 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={inView ? { width: `${(row[sortKey] ?? 0) * 100}%` } : {}}
                    transition={{ delay: 0.1 + i * 0.05, duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
                    className={`h-full ${accentBg}`}
                  />
                </div>
                <span className="mono w-10 text-right text-text">
                  {((row[sortKey] ?? 0) * 100).toFixed(1)}
                </span>
              </div>
              <span className="col-span-3 text-right text-text-subtle mono">
                R {(row.rougeL ?? 0).toFixed(2)} · B {(row.bertscore ?? 0).toFixed(2)}
              </span>
            </motion.div>
          ))}
        </div>
      </div>
    </StagePanel>
  );
}
