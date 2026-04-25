import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Pulls the real per-frame Δx distribution from
 * site/public/data/instrumented/slice_A/scroll_deltas.json when present
 * and uses the median for the on-screen label. Falls back to a
 * representative 12 px otherwise.
 */
export default function ScrollDetectionStage({ deltasUrl, fallbackDeltaPx = 12 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (!deltasUrl) return;
    fetch(deltasUrl).then(r => r.ok ? r.json() : null).then(setStats).catch(() => {});
  }, [deltasUrl]);

  const deltaPx = Math.round(stats?.median_px ?? fallbackDeltaPx);
  const sampleCount = stats?.count;

  return (
    <StagePanel
      number="2"
      title="Scroll detection"
      blurb={
        <p>
          Ticker text scrolls leftward across the bottom of the screen.
          Sampled frames are <strong>cross-correlated</strong> against the
          next frame to recover the per-frame pixel offset. Knowing how
          far the ticker scrolled lets the next stage stitch frames into a
          single un-scrolled panorama without duplicating or skipping
          characters.
          {sampleCount != null && (
            <> On Slice A the median offset is{" "}
              <strong>Δx ≈ {deltaPx} px</strong> across{" "}
              <strong>{sampleCount.toLocaleString()}</strong> sampled frames.
            </>
          )}
        </p>
      }
      accent="visual"
      data={
        stats ?? {
          method: "OpenCV template matching, middle 50% of ticker",
          delta_px_displayed: deltaPx,
          source: "fallback (instrumented data not built yet)",
        }
      }
    >
      <div ref={ref} className="relative h-44 grid place-items-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.4 }}
          className="absolute top-4 left-4 right-12 h-12 rounded-sm bg-surface-2 ring-1 ring-border
                     flex items-center pl-4 mono text-xs text-text-muted overflow-hidden"
        >
          <span className="opacity-60">frame N · </span>
          <span className="ml-2 whitespace-nowrap">
            ISRAELI AIR STRIKE ON SOUTHERN LEBANON KILLS SEVEN PEOPLE
          </span>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, x: 0 }}
          animate={inView ? { opacity: 0.85, x: -deltaPx } : {}}
          transition={{ delay: 0.5, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="absolute top-12 left-4 right-12 h-12 rounded-sm bg-visual/10 ring-1 ring-visual/40
                     flex items-center pl-4 mono text-xs text-text-muted overflow-hidden"
        >
          <span className="opacity-60">frame N+1 · </span>
          <span className="ml-2 whitespace-nowrap">
            ISRAELI AIR STRIKE ON SOUTHERN LEBANON KILLS SEVEN PEOPLE
          </span>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ delay: 1.1, duration: 0.3 }}
          className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3"
        >
          <svg width="100" height="20" viewBox="0 0 100 20">
            <line x1="10" y1="10" x2="90" y2="10" stroke="currentColor"
              className="text-visual" strokeWidth="2"/>
            <polygon points="6,10 14,5 14,15" fill="currentColor" className="text-visual"/>
          </svg>
          <span className="pill pill-visual mono">Δx = {deltaPx} px</span>
        </motion.div>
      </div>
    </StagePanel>
  );
}
