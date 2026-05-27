import { useEffect, useState } from "react";
import { motion } from "framer-motion";

/**
 * Renders the three-way OCR vs Gemini vs Mistral comparison.
 *
 * Loads `vlm_opensource_evaluation.json` which already bundles the
 * Gemini canonical row and the OCR baseline alongside the open-source
 * VLM aggregates. No external coordination required.
 *
 * Props:
 *   reportUrl  – defaults to /data/vlm/vlm_opensource_evaluation.json
 *   compact    – if true, drops the per-slice rows and shows only the
 *                three highlight cards (OCR / Gemini / Mistral)
 */
export default function VLMComparison({
  reportUrl = "/data/vlm/vlm_opensource_evaluation.json",
  compact = false,
}) {
  const [report, setReport] = useState(null);

  useEffect(() => {
    fetch(reportUrl).then(r => r.json()).then(setReport).catch(() => {});
  }, [reportUrl]);

  if (!report) {
    return <div className="text-text-subtle text-sm">loading VLM metrics…</div>;
  }

  const ocr     = report.ocr_baseline;
  const gemini  = report.gemini_canonical;
  const aggs    = report.aggregates ?? {};
  const perRun  = report.per_run ?? [];

  // Mistral roll-up = mean across the two GT slices in aggregates.
  const gtKeys = Object.keys(aggs);
  const mean = (arr) =>
    arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const meanField = (f) =>
    mean(gtKeys.map(k => aggs[k]?.[f]).filter(v => v != null && !Number.isNaN(v)));
  const meanRecall =
    mean(perRun.map(r => r.thresholds?.["70"]?.recall).filter(v => v != null));
  const meanPrec =
    mean(perRun.map(r => r.thresholds?.["70"]?.precision).filter(v => v != null));
  const meanWall =
    mean(perRun.map(r => r.wall_seconds).filter(v => v != null));

  const cards = [
    {
      label: "OCR pipeline",
      sub: "v6 + LLM-clean",
      accent: "visual",
      f1: ocr?.f1_at_70,
      recall: ocr?.recall_at_70,
      precision: ocr?.precision_at_70,
      cost: "$0.05",
      time: "~12 min",
      hl: true,
    },
    {
      label: "Gemini 2.5 Flash",
      sub: "closed, paid",
      accent: "vlm",
      f1: gemini?.thresholds?.["70"]?.f1,
      recall: gemini?.thresholds?.["70"]?.recall,
      precision: gemini?.thresholds?.["70"]?.precision,
      // Real Google billing across all VLM experiments was €8.28. The
      // per-run `cost_usd` (~$0.87) is an unverified token estimate that
      // under-reports actual spend, so we show the authoritative billed
      // total here instead.
      cost: "€8.28",
      time: gemini ? `${Math.round((gemini.wall_seconds ?? 0) / 60)} min` : "—",
      hl: false,
    },
    {
      label: "Ministral 3 14B",
      sub: "open-weights, free",
      accent: "vlm",
      f1: meanField("f1_mean"),
      recall: meanRecall,
      precision: meanPrec,
      cost: "$0.00",
      time: meanWall ? `${Math.round(meanWall / 60)} min` : "—",
      hl: true,
    },
  ];

  return (
    <div className="space-y-6">
      <div className={`grid gap-6 ${compact ? "sm:grid-cols-3" : "lg:grid-cols-3"}`}>
        {cards.map((c, i) => (
          <motion.div
            key={c.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 + i * 0.08, duration: 0.45 }}
            className={`card p-6 ${c.hl ? "ring-2 ring-visual" : ""}`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs uppercase tracking-wide text-text-subtle">
                {c.label}
              </div>
              <span className="pill">{c.sub}</span>
            </div>
            <div className="text-5xl font-semibold tracking-tight mb-1">
              {fmtPct(c.f1)}
              <span className="text-2xl text-text-muted ml-1">F1</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm mt-4">
              <Stat label="Recall"    value={c.recall} suffix="%" />
              <Stat label="Precision" value={c.precision} suffix="%" />
              <Stat label="Cost"      raw={c.cost} />
              <Stat label="Time"      raw={c.time} />
            </div>
          </motion.div>
        ))}
      </div>

      {!compact && (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface-2/50">
                <Th>Model</Th>
                <Th>GT slice</Th>
                <Th>Extracted</Th>
                <Th>F1 @70</Th>
                <Th>Recall</Th>
                <Th>Precision</Th>
                <Th>CER</Th>
                <Th>Halluc</Th>
              </tr>
            </thead>
            <tbody>
              {perRun.map((r, i) => (
                <tr key={i}
                    className="border-b border-border last:border-0 hover:bg-surface-2/40">
                  <Td mono>{report.model}</Td>
                  <Td>{r.gt_slice}</Td>
                  <Td mono>{r.extracted_count}</Td>
                  <Td mono>{fmtPct(r.thresholds?.["70"]?.f1)}</Td>
                  <Td mono>{fmtPct(r.thresholds?.["70"]?.recall)}</Td>
                  <Td mono>{fmtPct(r.thresholds?.["70"]?.precision)}</Td>
                  <Td mono>{fmtPct(r.avg_cer_at_70)}</Td>
                  <Td mono>{fmtPct(r.hallucination_rate_at_60)}</Td>
                </tr>
              ))}
              {gemini && (
                <tr className="border-b border-border last:border-0">
                  <Td mono>gemini-2.5-flash</Td>
                  <Td>{gemini.gt_slice}</Td>
                  <Td mono>{gemini.extracted_count}</Td>
                  <Td mono>{fmtPct(gemini.thresholds?.["70"]?.f1)}</Td>
                  <Td mono>{fmtPct(gemini.thresholds?.["70"]?.recall)}</Td>
                  <Td mono>{fmtPct(gemini.thresholds?.["70"]?.precision)}</Td>
                  <Td mono>{fmtPct(gemini.avg_cer_at_70)}</Td>
                  <Td mono>{fmtPct(gemini.hallucination_rate_at_60)}</Td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, suffix, raw }) {
  return (
    <div>
      <div className="text-text-subtle text-[11px] uppercase tracking-wide">
        {label}
      </div>
      <div className="mono text-base">
        {raw ?? (value == null ? "—" : `${(value * 100).toFixed(1)}${suffix ?? ""}`)}
      </div>
    </div>
  );
}

function Th({ children }) {
  return (
    <th className="text-left font-medium text-text-subtle text-[11px] uppercase tracking-wide px-4 py-2.5">
      {children}
    </th>
  );
}

function Td({ children, mono }) {
  return (
    <td className={`px-4 py-2.5 ${mono ? "font-mono text-xs" : ""}`}>
      {children}
    </td>
  );
}

function fmtPct(v) {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
