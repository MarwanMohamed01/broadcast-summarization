import { useEffect, useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

export default function SourceVideoStage({ videoSrc, label = "AlJazeera 14h sample" }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: false, margin: "-15% 0px" });
  const videoRef = useRef(null);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (inView) v.play().catch(() => {});
    else v.pause();
  }, [inView]);

  return (
    <StagePanel
      number="0"
      title="The source: 14 hours of broadcast"
      blurb={
        <p>
          The pipeline takes a continuous AlJazeera English broadcast as input.
          Two parallel processes run on the same source — <strong>vision</strong>
          {" "}reads the scrolling ticker bar, <strong>audio</strong> transcribes
          the spoken news. This 30-second slice is what the rest of the page
          will trace through both pipelines.
        </p>
      }
      accent="visual"
    >
      <div ref={ref} className="relative">
        <video
          ref={videoRef}
          src={videoSrc}
          muted
          loop
          playsInline
          className="w-full rounded-md ring-1 ring-border"
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ delay: 0.3 }}
          className="mt-3 flex items-center justify-between text-xs text-text-muted"
        >
          <span className="mono">{label} · 08:30:00 → 08:30:30</span>
          <span className="mono">480p · 30 fps · 30 s</span>
        </motion.div>
      </div>
    </StagePanel>
  );
}
