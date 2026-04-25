import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const TABS = [
  { key: "headlines",  label: "Headlines"  },
  { key: "transcript", label: "Transcript" },
  { key: "summaries",  label: "Summaries"  },
  { key: "raw",        label: "Raw OCR"    },
];

export default function Explorer({ urls }) {
  const [tab, setTab] = useState("headlines");
  const [query, setQuery] = useState("");

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-6">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`text-sm px-3 py-1.5 rounded-md border transition-colors
              ${tab === t.key ? "bg-text text-bg border-text" : "border-border hover:bg-surface-2"}`}
          >{t.label}</button>
        ))}
        <div className="flex-1" />
        <input
          type="search"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Filter…"
          className="text-sm px-3 py-1.5 rounded-md border border-border bg-surface w-48"
        />
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          {tab === "headlines"  && <Headlines  url={urls.headlines}  q={query} />}
          {tab === "transcript" && <Transcript url={urls.transcript} q={query} />}
          {tab === "summaries"  && <Summaries  url={urls.summaries}  q={query} />}
          {tab === "raw"        && <RawText    url={urls.raw}        q={query} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

function useJson(url) {
  const [d, setD] = useState(null);
  useEffect(() => {
    if (!url) return;
    fetch(url).then(r => r.json()).then(setD).catch(() => {});
  }, [url]);
  return d;
}

function useText(url) {
  const [d, setD] = useState("");
  useEffect(() => {
    if (!url) return;
    fetch(url).then(r => r.text()).then(setD).catch(() => {});
  }, [url]);
  return d;
}

function Headlines({ url, q }) {
  const data = useJson(url) ?? [];
  const filt = data.filter(it => !q || (it.text || "").toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="grid sm:grid-cols-2 gap-2">
      {filt.map((it, i) => (
        <div key={i} className="card p-3 text-sm">
          <div className="text-[10px] mono text-text-subtle mb-1">#{it.id ?? i}</div>
          {it.text}
        </div>
      ))}
      {!filt.length && <div className="text-text-subtle text-sm">no matches</div>}
    </div>
  );
}

function Transcript({ url, q }) {
  const data = useJson(url);
  const segs = data?.segments ?? data ?? [];
  const filt = segs.filter(s => !q || (s.text || "").toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="card divide-y divide-border max-h-[600px] overflow-auto">
      {filt.slice(0, 500).map((s, i) => (
        <div key={i} className="px-3 py-2 text-sm flex gap-3">
          <span className="mono text-[10px] text-text-subtle w-20 shrink-0 pt-0.5">
            {s.start?.toFixed?.(1)}–{s.end?.toFixed?.(1)}
          </span>
          <span className="leading-snug">{s.text}</span>
        </div>
      ))}
      {filt.length > 500 && (
        <div className="p-3 text-text-subtle text-xs">+{filt.length - 500} more (refine search)</div>
      )}
    </div>
  );
}

function Summaries({ url, q }) {
  const data = useJson(url);
  const items = data ? Object.entries(data) : [];
  const filt = items.filter(([k, v]) => !q ||
    k.toLowerCase().includes(q.toLowerCase()) ||
    JSON.stringify(v).toLowerCase().includes(q.toLowerCase())
  );
  return (
    <div className="space-y-3">
      {filt.map(([k, v]) => (
        <details key={k} className="card p-4 text-sm">
          <summary className="cursor-pointer font-medium">{k}</summary>
          <pre className="mt-3 mono text-xs whitespace-pre-wrap leading-relaxed">
            {typeof v === "string" ? v : JSON.stringify(v, null, 2)}
          </pre>
        </details>
      ))}
    </div>
  );
}

function RawText({ url, q }) {
  const data = useText(url);
  const lines = data.split("\n").filter(l => !q || l.toLowerCase().includes(q.toLowerCase()));
  return (
    <pre className="card p-4 max-h-[600px] overflow-auto mono text-xs leading-relaxed">
      {lines.join("\n")}
    </pre>
  );
}
