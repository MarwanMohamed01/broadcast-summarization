import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Visualises the L1 -> L2 collapse: 59 small chunk cards send particles
 * to a final L2 summary card, per LLM. Shown for one model column for
 * clarity (the others are alluded to as a stack behind).
 */
export default function HierarchicalSummaryStage({ totalChunks = 59 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const cardCount = 18; // representative grid

  return (
    <StagePanel
      number="3.5"
      title="Hierarchical summarisation: L1 → L2"
      blurb={
        <p>
          A single LLM call cannot fit the {totalChunks}-chunk transcript.
          The pipeline runs <strong>two passes</strong> per model:
          <strong> Level&nbsp;1</strong> summarises each chunk into one short
          paragraph; <strong>Level&nbsp;2</strong> concatenates that model's
          59 paragraphs and summarises again into the day's top stories.
          Total per-model API calls: <strong>{totalChunks} + 1</strong>.
        </p>
      }
      accent="audio"
      data={{ level_1_calls_per_model: totalChunks, level_2_calls_per_model: 1, total_models: 9 }}
    >
      <div ref={ref} className="relative h-72">
        {/* L1 cards on the left */}
        <div className="absolute left-0 top-0 bottom-0 w-1/2 grid grid-cols-6 gap-1 content-start">
          {Array.from({ length: cardCount }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.7 }}
              animate={inView ? {
                opacity: 1, scale: 1,
                x: inView ? 0 : 0,
              } : {}}
              transition={{ delay: 0.05 + i * 0.025 }}
              className="h-7 rounded-sm bg-audio/15 ring-1 ring-audio/30"
            />
          ))}
        </div>

        {/* particle trails sweeping right */}
        {inView && Array.from({ length: 12 }).map((_, i) => (
          <motion.div
            key={`p${i}`}
            initial={{ left: "20%", top: `${20 + (i * 7) % 60}%`, opacity: 0 }}
            animate={{ left: "78%", top: "50%", opacity: [0, 1, 0] }}
            transition={{ delay: 1.2 + i * 0.12, duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
            className="absolute w-1.5 h-1.5 rounded-full bg-audio shadow-[0_0_8px_var(--color-audio)]"
          />
        ))}

        {/* L2 final card on the right */}
        <motion.div
          initial={{ opacity: 0, scale: 0.85 }}
          animate={inView ? { opacity: 1, scale: 1 } : {}}
          transition={{ delay: 2.6, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="absolute right-0 top-1/2 -translate-y-1/2 w-1/2 max-w-xs"
        >
          <div className="card p-3">
            <div className="text-[10px] uppercase tracking-wide text-text-subtle mb-1">Level 2 summary</div>
            <p className="text-xs leading-snug">
              The day's coverage focused on escalating Israel–Iran exchanges
              with strikes on energy facilities, continued operations in Gaza
              and southern Lebanon, and Trump-administration responses…
            </p>
          </div>
          {/* stacked echo behind to imply 9 models */}
          <div className="absolute -z-10 inset-0 translate-y-2 translate-x-2 card opacity-40"></div>
          <div className="absolute -z-20 inset-0 translate-y-4 translate-x-4 card opacity-20"></div>
        </motion.div>
      </div>
    </StagePanel>
  );
}
