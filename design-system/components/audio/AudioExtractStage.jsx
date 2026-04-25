import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Visualises the moment audio is peeled off the video container.
 * No real audio plays — purely visual metaphor of bars detaching and
 * re-arranging into a horizontal waveform.
 */
export default function AudioExtractStage() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });

  // Synthetic bar heights — looks like an audio strip
  const bars = Array.from({ length: 36 }, (_, i) =>
    20 + Math.abs(Math.sin(i * 0.7)) * 35 + (i % 3) * 5
  );

  return (
    <StagePanel
      number="1"
      title="Audio extraction"
      blurb={
        <p>
          The MP4 container holds both video and audio.  PyAV is used to
          decode the audio track and resample it to the format
          {" "}<strong>Whisper</strong> expects — 16 kHz mono PCM.  No
          ffmpeg CLI is required, which keeps the dependency surface small
          and the build predictable on Windows.
        </p>
      }
      accent="audio"
      data={{ codec: "PCM s16le", sample_rate_hz: 16000, channels: "mono" }}
    >
      <div ref={ref} className="space-y-6">
        <div className="aspect-video bg-surface-2 rounded-md ring-1 ring-border grid place-items-center text-xs text-text-subtle">
          Video container (MP4)
        </div>
        <div className="flex items-end justify-center gap-1 h-20">
          {bars.map((h, i) => (
            <motion.div
              key={i}
              initial={{ height: 4, opacity: 0.4 }}
              animate={inView ? { height: h, opacity: 1 } : {}}
              transition={{ delay: 0.2 + i * 0.02, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="w-1.5 rounded-full bg-audio"
            />
          ))}
        </div>
        <div className="text-xs text-text-muted text-center mono">
          16 kHz · mono · 30 s
        </div>
      </div>
    </StagePanel>
  );
}
