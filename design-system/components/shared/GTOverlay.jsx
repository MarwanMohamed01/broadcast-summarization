import { useEffect, useState } from "react";
import { motion } from "framer-motion";

/**
 * Side-by-side ground-truth vs extracted headlines, with a fuzzy-match
 * highlight per pair. Pure visualization — uses naive Jaro-Winkler in the
 * browser to flag the matched extracted item without needing a server.
 */
function similarity(a, b) {
  if (!a || !b) return 0;
  const A = a.toUpperCase(), B = b.toUpperCase();
  // Quick character-overlap heuristic (NOT real ROUGE — just for the UI cue)
  const setA = new Set(A.split(/\s+/));
  const setB = new Set(B.split(/\s+/));
  const inter = [...setA].filter(x => setB.has(x)).length;
  return inter / Math.max(setA.size, setB.size);
}

export default function GTOverlay({ gtUrl, extractedUrl, sliceLabel = "Slice A" }) {
  const [gt, setGt] = useState([]);
  const [ext, setExt] = useState([]);

  useEffect(() => {
    fetch(gtUrl).then(r => r.text()).then(t => {
      setGt(t.split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("#")));
    }).catch(() => {});
  }, [gtUrl]);

  useEffect(() => {
    fetch(extractedUrl).then(r => r.json()).then(arr => {
      setExt((arr ?? []).map(it => it.text).filter(Boolean));
    }).catch(() => {});
  }, [extractedUrl]);

  const matches = gt.map(g => {
    let best = { i: -1, sim: 0 };
    ext.forEach((e, i) => {
      const s = similarity(g, e);
      if (s > best.sim) best = { i, sim: s };
    });
    return best;
  });

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">{sliceLabel} · ground truth ↔ extracted</h3>
        <div className="text-xs text-text-muted">
          {gt.length} GT · {ext.length} extracted
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 max-h-[600px] overflow-auto">
        <div className="space-y-1.5">
          <div className="text-xs uppercase tracking-wide text-text-subtle">Ground truth</div>
          {gt.map((g, i) => (
            <motion.div
              key={i}
              whileHover={{ x: 2 }}
              className="text-xs px-2 py-1.5 rounded bg-surface-2 border border-border"
            >
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${
                  matches[i].sim > 0.5 ? "bg-success" :
                  matches[i].sim > 0.3 ? "bg-warn" : "bg-error"
                }`}></span>
                <span className="line-clamp-2">{g}</span>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="space-y-1.5">
          <div className="text-xs uppercase tracking-wide text-text-subtle">Extracted</div>
          {ext.map((e, i) => {
            const isMatched = matches.some(m => m.i === i);
            return (
              <motion.div
                key={i}
                whileHover={{ x: 2 }}
                className={`text-xs px-2 py-1.5 rounded border ${
                  isMatched
                    ? "bg-success/10 border-success/30"
                    : "bg-surface-2 border-border"
                }`}
              >
                <span className="line-clamp-2">{e}</span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
