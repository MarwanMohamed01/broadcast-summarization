import { useState, useEffect, useRef } from "react";
import { motion, useInView } from "framer-motion";
import { RotateCw, ChevronDown } from "lucide-react";

/**
 * Container for one pipeline stage. Reused across both sites.
 *
 * Props:
 *   number     – ordinal label like "1" or "5/8"
 *   title      – short title
 *   blurb      – longer paragraph (string or node)
 *   accent     – "visual" | "audio" — controls accent color
 *   children   – the illustration (right column on desktop, below on mobile)
 *   data       – optional object/string surfaced in the "Show data" disclosure
 *   replayKey  – number used to reset the illustration; bumped by ReplayButton
 *   onReplay   – called with new key
 */
export default function StagePanel({
  number,
  title,
  blurb,
  accent = "visual",
  children,
  data,
  showReplay = true,
}) {
  const [replayKey, setReplayKey] = useState(0);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px -15% 0px" });

  const accentClass = accent === "audio" ? "text-audio" : "text-visual";
  const accentSoftBg = accent === "audio" ? "bg-audio/10" : "bg-visual/10";

  return (
    <section
      ref={ref}
      className="stage-panel border-t border-border first:border-t-0"
    >
      <div className="max-w-content mx-auto px-6 py-20 lg:py-28 grid lg:grid-cols-12 gap-10 items-start">
        {/* left: text */}
        <motion.div
          className="lg:col-span-5"
          initial={{ opacity: 0, y: 12 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="flex items-center gap-3 mb-4">
            <span
              className={`inline-flex w-7 h-7 rounded-md ${accentSoftBg} ${accentClass}
                          items-center justify-center font-mono text-xs font-bold`}
            >
              {number}
            </span>
            <span className={`pill ${accent === "audio" ? "pill-audio" : "pill-visual"}`}>
              {accent === "audio" ? "Audio path" : "Visual path"}
            </span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-semibold tracking-tight text-text mb-4">
            {title}
          </h2>
          <div className="prose-thesis">{blurb}</div>

          <div className="flex items-center gap-3 mt-6">
            {showReplay && (
              <button
                type="button"
                onClick={() => setReplayKey((k) => k + 1)}
                className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5
                           rounded-md border border-border hover:bg-surface-2 transition-colors"
              >
                <RotateCw size={14} /> Replay
              </button>
            )}
            {data !== undefined && (
              <button
                type="button"
                onClick={() => setOpen((o) => !o)}
                aria-expanded={open}
                className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5
                           rounded-md border border-border hover:bg-surface-2 transition-colors"
              >
                <ChevronDown size={14}
                  style={{
                    transform: open ? "rotate(180deg)" : "rotate(0deg)",
                    transition: "transform 200ms",
                  }} />
                {open ? "Hide data" : "Show data"}
              </button>
            )}
          </div>

          {open && data !== undefined && (
            <pre className="mt-4 text-xs leading-relaxed font-mono p-4 rounded-md
                            bg-surface-2 border border-border max-h-72 overflow-auto">
              {typeof data === "string" ? data : JSON.stringify(data, null, 2)}
            </pre>
          )}
        </motion.div>

        {/* right: illustration */}
        <div className="lg:col-span-7" key={replayKey}>
          <div className="card p-4 sm:p-6 overflow-hidden">{children}</div>
        </div>
      </div>
    </section>
  );
}
