import { useEffect, useState } from "react";
import { motion } from "framer-motion";

/** Loads validation_report.json and renders the headline metrics. */
export default function ResultsHero({ reportUrl }) {
  const [report, setReport] = useState(null);

  useEffect(() => {
    fetch(reportUrl).then(r => r.json()).then(setReport).catch(() => {});
  }, [reportUrl]);

  if (!report) {
    return <div className="text-text-subtle text-sm">loading metrics…</div>;
  }

  // Pull combined-slice numbers for raw + cleaned at threshold 70
  const findRow = (stage, slice) =>
    report.results?.find(r => r.stage === stage && r.slice === slice);

  const raw      = findRow("raw_v6", "combined");
  const cleaned  = findRow("llm_cleaned", "combined");

  const cells = [
    {
      label: "Raw v6",
      f1:   raw?.thresholds?.["70"]?.f1,
      prec: raw?.thresholds?.["70"]?.precision,
      rec:  raw?.thresholds?.["70"]?.recall,
      cer:  raw?.avg_cer_at_70,
      hl:   false,
    },
    {
      label: "LLM-cleaned",
      f1:   cleaned?.thresholds?.["70"]?.f1,
      prec: cleaned?.thresholds?.["70"]?.precision,
      rec:  cleaned?.thresholds?.["70"]?.recall,
      cer:  cleaned?.avg_cer_at_70,
      hl:   true,
    },
  ];

  return (
    <div className="grid sm:grid-cols-2 gap-6">
      {cells.map((c, i) => (
        <motion.div
          key={c.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 + i * 0.1, duration: 0.5 }}
          className={`card p-6 ${c.hl ? "ring-2 ring-visual" : ""}`}
        >
          <div className="text-xs uppercase tracking-wide text-text-subtle mb-3">
            {c.label}
          </div>
          <div className="text-5xl font-semibold tracking-tight mb-2">
            {((c.f1 ?? 0) * 100).toFixed(1)}<span className="text-2xl text-text-muted">% F1</span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm mt-4">
            <Stat label="Precision" value={c.prec} suffix="%" />
            <Stat label="Recall"    value={c.rec}  suffix="%" />
            <Stat label="CER"       value={c.cer}  suffix="%" />
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function Stat({ label, value, suffix }) {
  return (
    <div>
      <div className="text-text-subtle text-[11px] uppercase tracking-wide">{label}</div>
      <div className="mono text-base">
        {value == null ? "—" : `${(value * 100).toFixed(1)}${suffix ?? ""}`}
      </div>
    </div>
  );
}
