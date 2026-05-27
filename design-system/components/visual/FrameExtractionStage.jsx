import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * If `statsUrl` is provided and resolves, real numbers from
 * site/public/data/instrumented/slice_A/frame_stats.json (and as a
 * fallback site/public/data/stats/pipeline_stats.json) drive the
 * counts. Otherwise the props/defaults are used.
 */
export default function FrameExtractionStage({
  statsUrl,
  pipelineStatsUrl,
  sampleRate: sampleRateProp = 5,
  totalFrames: totalFramesProp = 30,
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });

  const [sampleRate, setSampleRate] = useState(sampleRateProp);
  const [sourceFrames, setSourceFrames] = useState(null);   // real source-frame count if known
  const [keptFrames, setKeptFrames] = useState(null);       // real kept-frame count if known

  useEffect(() => {
    let cancelled = false;
    async function load() {
      // Prefer instrumented Slice A stats; fall back to whole-pipeline stats
      try {
        const r = await fetch(statsUrl);
        if (r.ok) {
          const d = await r.json();
          if (!cancelled) {
            setSampleRate(d.frame_sample_rate ?? sampleRate);
            setSourceFrames(d.source_frames ?? null);
            setKeptFrames(d.kept_frames ?? null);
            return;
          }
        }
      } catch {}
      try {
        const r = await fetch(pipelineStatsUrl);
        if (r.ok) {
          const d = await r.json();
          if (!cancelled) {
            const src = d.video?.total_frames;
            const sr = d.config?.frame_sample_rate ?? sampleRate;
            setSampleRate(sr);
            setSourceFrames(src ?? null);
            setKeptFrames(src ? Math.ceil(src / sr) : null);
          }
        }
      } catch {}
    }
    load();
    return () => { cancelled = true; };
  }, [statsUrl, pipelineStatsUrl]);

  // 30-frame visualization (illustrative regardless of underlying numbers)
  const tileCount = totalFramesProp;
  const tiles = Array.from({ length: tileCount }, (_, i) => i + 1);
  const sampledTiles = Math.ceil(tileCount / sampleRate);

  return (
    <StagePanel
      number="1"
      title="Frame extraction"
      blurb={
        <p>
          The video runs at 30 fps — far too dense to OCR every frame. The
          pipeline samples every <strong>{sampleRate}th frame</strong>
          {" "}({Math.round(30 / sampleRate)} fps), capturing each ticker
          scroll step with no redundant work.
          {sourceFrames != null && keptFrames != null && (
            <> On Slice A this means{" "}
              <strong>{keptFrames.toLocaleString()}</strong> sampled frames
              out of <strong>{sourceFrames.toLocaleString()}</strong> source
              frames.
            </>
          )}
        </p>
      }
      accent="visual"
      data={{
        frame_sample_rate: sampleRate,
        source_frames: sourceFrames,
        kept_frames: keptFrames,
        illustration_tiles: tileCount,
      }}
    >
      <div ref={ref} className="space-y-4">
        <div className="text-xs text-text-muted mono">
          {sourceFrames != null && keptFrames != null
            ? `${keptFrames.toLocaleString()} / ${sourceFrames.toLocaleString()} frames kept`
            : `${sampledTiles} / ${tileCount} frames kept (illustrative)`}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {tiles.map((n, i) => {
            const sampled = (i % sampleRate) === 0;
            return (
              <motion.div
                key={n}
                initial={{ opacity: 0, y: 8 }}
                animate={inView ? {
                  opacity: sampled ? 1 : 0.35,
                  y: sampled ? -4 : 0,
                  scale: sampled ? 1.05 : 1,
                } : {}}
                transition={{ delay: i * 0.04, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                className={`relative h-12 w-7 rounded-sm ring-1 ${
                  sampled ? "bg-visual/20 ring-visual" : "bg-surface-2 ring-border"
                }`}
              >
                <span className="absolute inset-x-0 bottom-0 text-[9px] text-center mono opacity-60">
                  {n}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </StagePanel>
  );
}
