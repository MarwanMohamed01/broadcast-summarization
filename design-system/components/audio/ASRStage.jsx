import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

/**
 * Real waveform peaks (decimated by build_assets.py) + a moving playhead +
 * synced transcript segment.  No real audio playback — keeps the visit
 * silent and avoids licensing entanglements with the source broadcast.
 *
 * Props:
 *   peaksUrl       — URL of /data/audio/peaks.json
 *   transcriptUrl  — URL of /data/transcript/transcript_slice_A.json
 */
export default function ASRStage({ peaksUrl, transcriptUrl }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const [peaks, setPeaks] = useState(null);
  const [segments, setSegments] = useState([]);
  const [t, setT] = useState(0);   // playhead seconds

  useEffect(() => {
    if (!peaksUrl) return;
    fetch(peaksUrl).then(r => r.ok ? r.json() : null).then(setPeaks).catch(() => {});
  }, [peaksUrl]);

  useEffect(() => {
    if (!transcriptUrl) return;
    fetch(transcriptUrl).then(r => r.ok ? r.json() : null).then((data) => {
      if (Array.isArray(data?.segments)) setSegments(data.segments);
      else if (Array.isArray(data)) setSegments(data);
    }).catch(() => {});
  }, [transcriptUrl]);

  // Auto-advance the playhead while in view
  useEffect(() => {
    if (!inView || !peaks) return;
    const dur = peaks.duration_sec || 30;
    let raf, start = performance.now();
    const tick = (now) => {
      const elapsed = (now - start) / 1000;
      setT(elapsed % dur);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, peaks]);

  const dur = peaks?.duration_sec ?? 30;
  const playheadPct = (t / dur) * 100;

  // Segment at the playhead (slice_A transcript starts at 30600s — adjust)
  const baseT = segments[0]?.start ?? 0;
  const localT = baseT + t;
  const current = segments.find((s) => s.start <= localT && localT < s.end);

  return (
    <StagePanel
      number="2"
      title="Whisper ASR transcription"
      blurb={
        <p>
          The 16 kHz mono audio is fed to <strong>faster-whisper</strong>
          {" "}(small int8 model, English).  The 14-hour file is too large to
          load at once — the transcriber streams 20-minute chunks from disk
          and saves intermediate transcripts so a crash doesn't mean
          re-running for hours.  Output: 13,517 timestamped segments at
          {" "}<strong>3.07× realtime</strong> on CPU.
        </p>
      }
      accent="audio"
      data={{
        model: "faster-whisper small int8 English",
        chunking: "20-minute internal stream chunks",
        segments_total: 13517,
        speed: "3.07x realtime CPU",
      }}
    >
      <div ref={ref} className="space-y-4">
        {/* waveform */}
        <div className="relative h-24 bg-surface-2 rounded-md ring-1 ring-border overflow-hidden">
          {peaks ? (
            <svg className="w-full h-full" viewBox={`0 0 ${peaks.peaks.length} 100`} preserveAspectRatio="none">
              {peaks.peaks.map(([min, max], i) => (
                <line
                  key={i}
                  x1={i} x2={i}
                  y1={50 + min * 50}
                  y2={50 + max * 50}
                  stroke="currentColor"
                  className="text-audio"
                  strokeOpacity={0.6}
                  strokeWidth={1}
                />
              ))}
            </svg>
          ) : (
            <div className="absolute inset-0 grid place-items-center text-xs text-text-subtle">
              loading waveform…
            </div>
          )}
          {/* playhead */}
          <div
            className="absolute top-0 bottom-0 w-px bg-audio"
            style={{ left: `${playheadPct}%`, boxShadow: "0 0 6px var(--color-audio)" }}
          />
        </div>

        {/* transcript window */}
        <motion.div
          key={current?.id ?? "empty"}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18 }}
          className="card p-3 min-h-[64px] text-sm leading-relaxed"
        >
          {current ? (
            <>
              <div className="text-[10px] mono text-text-subtle mb-1">
                {current.start.toFixed(1)} – {current.end.toFixed(1)} s
              </div>
              {current.text}
            </>
          ) : (
            <span className="text-text-subtle text-xs">…silence / interlude…</span>
          )}
        </motion.div>
      </div>
    </StagePanel>
  );
}
