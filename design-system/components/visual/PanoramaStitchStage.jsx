import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Pre-cropped strips from a real panorama slide together and snap into
 * one long strip. `segmentSrcs` is an array of WebP URLs from build_assets.py.
 */
export default function PanoramaStitchStage({ segmentSrcs = [], totalWidthPx = 142165 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });

  // Show first 6 segments, slide them in
  const segs = segmentSrcs.slice(0, 6);

  return (
    <StagePanel
      number="3"
      title="Panorama stitching"
      blurb={
        <p>
          Knowing the per-frame scroll offset, the pipeline takes a vertical
          slice from each kept frame and pastes them edge-to-edge.  The result
          is a single uninterrupted image of the entire ticker —
          {" "}<strong>{totalWidthPx.toLocaleString()} px wide</strong> for the
          30-minute slice.  Every character that appeared on screen is now
          present exactly once in this panorama.
        </p>
      }
      accent="visual"
      data={{ panorama_width_px: totalWidthPx, segments_visualised: segs.length }}
    >
      <div ref={ref} className="space-y-4">
        <div className="flex items-center gap-1 overflow-hidden">
          {segs.map((src, i) => (
            <motion.img
              key={i}
              src={src}
              alt=""
              initial={{ opacity: 0, x: -40 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ delay: i * 0.18, duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
              className="h-10 w-1/6 object-cover ring-1 ring-border rounded-sm"
            />
          ))}
        </div>

        <div className="space-y-1.5">
          <div className="text-xs text-text-muted mono flex justify-between">
            <span>Stitched width</span>
            <motion.span
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 1 } : {}}
              transition={{ delay: segs.length * 0.18 + 0.3 }}
              className="text-visual"
            >
              {totalWidthPx.toLocaleString()} px
            </motion.span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={inView ? { width: "100%" } : {}}
              transition={{ delay: 0.2, duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
              className="h-full bg-visual"
            />
          </div>
        </div>
      </div>
    </StagePanel>
  );
}
