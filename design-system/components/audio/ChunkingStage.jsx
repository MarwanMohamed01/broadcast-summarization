import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * 59 transcript chunks visualized as a 12-card sample with a "+47 more" tile.
 */
export default function ChunkingStage({ totalChunks = 59, chunkMinutes = 15 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const sample = Array.from({ length: 12 }, (_, i) =>
    Math.round((i / 11) * (totalChunks - 1))
  );

  return (
    <StagePanel
      number="3"
      title={`${chunkMinutes}-minute chunking`}
      blurb={
        <p>
          The transcript is split at <strong>{chunkMinutes}-minute</strong>
          {" "}boundaries based on segment timestamps.  Each chunk holds
          ~2,000–2,600 words — small enough to fit comfortably in any of the
          nine LLMs' context windows, with room left for the system prompt.
          The 14-hour broadcast becomes <strong>{totalChunks}</strong> chunks.
        </p>
      }
      accent="audio"
      data={{ chunk_minutes: chunkMinutes, total_chunks: totalChunks, target_words_per_chunk: "2,000–2,600" }}
    >
      <div ref={ref}>
        <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
          {sample.map((idx, i) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.85, y: 8 }}
              animate={inView ? { opacity: 1, scale: 1, y: 0 } : {}}
              transition={{ delay: 0.1 + i * 0.04, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="card aspect-square p-2 flex flex-col justify-between"
            >
              <span className="text-[9px] uppercase tracking-wide text-text-subtle">chunk</span>
              <span className="mono text-sm">{String(idx).padStart(3, "0")}</span>
            </motion.div>
          ))}
          <motion.div
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ delay: 0.1 + sample.length * 0.04 }}
            className="aspect-square p-2 grid place-items-center rounded-lg
                       border border-dashed border-border text-xs text-text-subtle"
          >
            +{totalChunks - sample.length} more
          </motion.div>
        </div>
      </div>
    </StagePanel>
  );
}
