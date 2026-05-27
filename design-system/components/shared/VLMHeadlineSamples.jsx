import { useEffect, useState } from "react";

/**
 * Side-by-side sample of headlines from each VLM at the head of the
 * stitched panorama. Pulls the small `headlines_<model>_sample.json`
 * files emitted by build_assets.py so the page weight stays under
 * ~30 KB even though Mistral emitted 3,214 headlines in total.
 */
export default function VLMHeadlineSamples({
  geminiUrl  = "/data/vlm/headlines_gemini_sample.json",
  mistralUrl = "/data/vlm/headlines_mistral_sample.json",
  limit = 20,
}) {
  const [gemini,  setGemini]  = useState(null);
  const [mistral, setMistral] = useState(null);

  useEffect(() => {
    fetch(geminiUrl).then(r => r.json()).then(setGemini).catch(() => {});
    fetch(mistralUrl).then(r => r.json()).then(setMistral).catch(() => {});
  }, [geminiUrl, mistralUrl]);

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <Column
        title="Gemini 2.5 Flash"
        subtitle="closed-source · paid"
        data={gemini}
        limit={limit}
        accent="text-text"
      />
      <Column
        title="Ministral 3 14B"
        subtitle="open-weights · free"
        data={mistral}
        limit={limit}
        accent="text-visual"
        highlight
      />
    </div>
  );
}

function Column({ title, subtitle, data, limit, accent, highlight }) {
  if (!data) {
    return (
      <div className="card p-5 text-text-subtle text-sm">loading…</div>
    );
  }
  const items = (data.headlines ?? []).slice(0, limit);
  return (
    <div className={`card p-5 ${highlight ? "ring-2 ring-visual" : ""}`}>
      <div className="flex items-end justify-between mb-3">
        <div>
          <div className={`text-sm font-semibold ${accent}`}>{title}</div>
          <div className="text-xs text-text-subtle">{subtitle}</div>
        </div>
        <div className="text-xs text-text-subtle mono">
          {data.total_extracted} extracted · showing first {Math.min(limit, items.length)}
        </div>
      </div>
      <ol className="space-y-1.5 text-xs leading-snug">
        {items.map((h, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-text-subtle mono w-6 text-right shrink-0">
              {i + 1}.
            </span>
            <span className="font-mono">{h.text}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
