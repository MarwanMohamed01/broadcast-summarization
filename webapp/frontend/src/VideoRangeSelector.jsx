import { useEffect, useRef, useState } from "react";
import Slider from "rc-slider";
import "rc-slider/assets/index.css";

/**
 * Video player + draggable range picker.
 *
 * Props:
 *   src        — video URL
 *   duration   — total duration in seconds (from backend metadata)
 *   start      — current selected start (seconds)
 *   end        — current selected end (seconds)
 *   maxSegment — maximum allowed segment length in seconds (default 1800 = 30 min)
 *   onChange   — (start, end) => void
 */
export default function VideoRangeSelector({
  src,
  duration,
  start,
  end,
  maxSegment = 30 * 60,
  onChange,
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
    // Clamp segment length to maxSegment
    let clampedStart = s;
    let clampedEnd = e;
    if (e - s > maxSegment) {
      // Keep whichever handle the user was dragging closer to its previous value
      if (Math.abs(s - start) > Math.abs(e - end)) {
        clampedStart = e - maxSegment;
      } else {
        clampedEnd = s + maxSegment;
      }
    }
    onChange(clampedStart, clampedEnd);
    if (videoRef.current) {
      // Jump playback to whichever handle just moved
      const movedStart = Math.abs(clampedStart - start) > Math.abs(clampedEnd - end);
      videoRef.current.currentTime = movedStart ? clampedStart : clampedEnd;
    }
  }

  function jumpTo(sec) {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      videoRef.current.play().catch(() => {});
    }
  }

  function fmt(s) {
    s = Math.floor(s);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
    return `${m}:${String(ss).padStart(2, "0")}`;
  }

  const segMin = ((end - start) / 60).toFixed(1);
  const overLimit = end - start > maxSegment;

  return (
    <div style={{ marginBottom: 12 }}>
      <video
        ref={videoRef}
        src={src}
        controls
        style={{
          width: "100%",
          maxHeight: 360,
          background: "#000",
          borderRadius: 6,
        }}
      />

      <div
        style={{
          background: "#f9fafb",
          border: "1px solid #e5e7eb",
          borderRadius: 6,
          padding: "12px 16px",
          marginTop: 8,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.85rem",
            color: "#6b7280",
            marginBottom: 6,
          }}
        >
          <span>0:00</span>
          <span>
            playhead: <strong>{fmt(currentTime)}</strong>
          </span>
          <span>{fmt(duration)}</span>
        </div>

        <div style={{ padding: "0 10px" }}>
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
              track: { background: "#3b82f6", height: 6 },
              rail: { background: "#d1d5db", height: 6 },
              handle: {
                background: "#3b82f6",
                border: "2px solid #fff",
                boxShadow: "0 0 0 2px #3b82f6",
                width: 16,
                height: 16,
                marginTop: -5,
                opacity: 1,
              },
            }}
          />
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 12,
            fontSize: "0.9rem",
          }}
        >
          <div>
            <strong>Start:</strong> {fmt(start)}
            <button
              onClick={() => jumpTo(start)}
              style={{
                marginLeft: 6,
                padding: "2px 8px",
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              ▶ preview
            </button>
          </div>
          <div
            style={{
              color: overLimit ? "#dc2626" : "#374151",
              fontWeight: 600,
            }}
          >
            Segment: {segMin} min
            {overLimit && ` (max ${maxSegment / 60} min)`}
          </div>
          <div>
            <strong>End:</strong> {fmt(end)}
            <button
              onClick={() => jumpTo(end)}
              style={{
                marginLeft: 6,
                padding: "2px 8px",
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              ▶ preview
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
