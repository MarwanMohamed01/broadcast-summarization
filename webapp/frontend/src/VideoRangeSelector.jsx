import { useEffect, useRef, useState } from "react";
import Slider from "rc-slider";
import "rc-slider/assets/index.css";
import { Play } from "lucide-react";

function fmt(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  return `${m}:${String(ss).padStart(2, "0")}`;
}

export default function VideoRangeSelector({
  src, duration, start, end, maxSegment = 30 * 60, onChange,
}) {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(start || 0);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => setCurrentTime(v.currentTime);
    v.addEventListener("timeupdate", onTime);
    return () => v.removeEventListener("timeupdate", onTime);
  }, []);

  function handleSliderChange(values) {
    const [s, e] = values;
    let cs = s, ce = e;
    if (e - s > maxSegment) {
      if (Math.abs(s - start) > Math.abs(e - end)) cs = e - maxSegment;
      else ce = s + maxSegment;
    }
    onChange(cs, ce);
    if (videoRef.current) {
      const movedStart = Math.abs(cs - start) > Math.abs(ce - end);
      videoRef.current.currentTime = movedStart ? cs : ce;
    }
  }

  function jumpTo(sec) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = sec;
    v.play().catch(() => {});
  }

  const segMin = ((end - start) / 60).toFixed(1);
  const overLimit = end - start > maxSegment;

  return (
    <div className="space-y-3">
      <video
        ref={videoRef} src={src} controls
        className="w-full max-h-[360px] rounded-md bg-black ring-1 ring-border"
      />

      <div className="card p-4">
        <div className="flex justify-between text-[11px] text-text-muted mono mb-2">
          <span>0:00</span>
          <span>playhead: <strong className="text-text">{fmt(currentTime)}</strong></span>
          <span>{fmt(duration)}</span>
        </div>

        <div className="px-2.5">
          <Slider
            range
            min={0}
            max={Math.max(1, Math.floor(duration))}
            value={[Math.floor(start), Math.floor(end)]}
            onChange={handleSliderChange}
            step={1}
            allowCross={false}
            pushable={1}
            styles={{
              track: { background: "var(--color-visual)", height: 6 },
              rail:  { background: "var(--color-border)", height: 6 },
              handle: {
                background: "var(--color-visual)",
                border: "2px solid var(--color-bg)",
                boxShadow: "0 0 0 2px var(--color-visual)",
                width: 16, height: 16, marginTop: -5, opacity: 1,
              },
            }}
          />
        </div>

        <div className="flex justify-between items-center mt-4 text-sm flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <span className="text-text-muted">Start</span>
            <strong className="mono">{fmt(start)}</strong>
            <button onClick={() => jumpTo(start)} className="btn-secondary !px-2 !py-1 text-[11px]">
              <Play size={10} />
            </button>
          </div>
          <div className={`font-semibold ${overLimit ? "text-error" : "text-text"}`}>
            {segMin} min{overLimit && ` (max ${maxSegment / 60} min)`}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-text-muted">End</span>
            <strong className="mono">{fmt(end)}</strong>
            <button onClick={() => jumpTo(end)} className="btn-secondary !px-2 !py-1 text-[11px]">
              <Play size={10} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
