import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Play, FileVideo, Loader2, CheckCircle2, XCircle, Download,
  Clock, Cpu, Sun, Moon,
} from "lucide-react";
import { api } from "./api";
import VideoRangeSelector from "./VideoRangeSelector";

function fmtSec(s) {
  if (!s && s !== 0) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = Math.round(s % 60);
  if (h) return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  return `${m}:${String(ss).padStart(2, "0")}`;
}

const STATUS_STYLE = {
  queued:    { bg: "bg-surface-2",      fg: "text-text-muted", Icon: Clock        },
  running:   { bg: "bg-visual/15",      fg: "text-visual",     Icon: Loader2      },
  done:      { bg: "bg-success/15",     fg: "text-success",    Icon: CheckCircle2 },
  failed:    { bg: "bg-error/15",       fg: "text-error",      Icon: XCircle      },
  cancelled: { bg: "bg-warn/15",        fg: "text-warn",       Icon: XCircle      },
};

function StatusPill({ status }) {
  const s = STATUS_STYLE[status] || STATUS_STYLE.queued;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium ${s.bg} ${s.fg}`}>
      <s.Icon size={12} className={status === "running" ? "animate-spin" : ""} />
      {status}
    </span>
  );
}

function ProgressBar({ value }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="bg-surface-2 rounded-full h-2 overflow-hidden">
      <motion.div
        className="bg-visual h-full"
        initial={false}
        animate={{ width: `${pct}%` }}
        transition={{ type: "spring", damping: 18, stiffness: 200 }}
      />
    </div>
  );
}

function SummaryCard({ summary }) {
  const ok = summary.status === "success";
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`card p-4 ${ok ? "" : "ring-1 ring-error/30"}`}
    >
      <div className="flex justify-between items-start mb-2">
        <strong className="text-text">{summary.display_name}</strong>
        {ok && (
          <span className="text-[11px] text-text-muted mono">
            {summary.latency_seconds?.toFixed(1)}s · {summary.output_tokens}t
          </span>
        )}
      </div>
      {ok ? (
        <p className="text-sm text-text-muted leading-relaxed whitespace-pre-wrap m-0">
          {summary.summary}
        </p>
      ) : (
        <p className="text-sm text-error m-0">FAILED: {summary.error}</p>
      )}
    </motion.div>
  );
}

function JobHistoryItem({ job, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded-md text-xs border transition-colors ${
        active ? "border-visual bg-visual/10" : "border-transparent hover:bg-surface-2"
      }`}
    >
      <div className="flex justify-between items-center mb-1">
        <code className="text-text-muted text-[11px]">{job.job_id.slice(0, 8)}</code>
        <StatusPill status={job.status} />
      </div>
      <div className="text-text-subtle text-[11px]">
        {job.task} · {Math.round((job.end_sec - job.start_sec) / 60)} min
      </div>
    </button>
  );
}

function ThemeToggle() {
  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={() => {
        const cur = document.documentElement.dataset.theme;
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("theme", next);
      }}
      className="w-9 h-9 grid place-items-center rounded-md hover:bg-surface-2 transition-colors"
    >
      <Sun size={16} className="dark:hidden" />
      <Moon size={16} className="hidden dark:inline-block" />
    </button>
  );
}

export default function App() {
  // Restore stored theme
  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const sysDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = stored || (sysDark ? "dark" : "light");
  }, []);

  const [video, setVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [task, setTask] = useState("ticker");
  const [startSec, setStartSec] = useState(0);
  const [endSec, setEndSec] = useState(600);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    api.models()
      .then((m) => { setModels(m); setSelectedModels(m.map((x) => x.display_name)); })
      .catch((e) => setError(`Backend unreachable: ${e.message}`));
  }, []);

  useEffect(() => {
    const load = () => fetch("http://localhost:8000/api/jobs")
      .then((r) => r.json()).then(setHistory).catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const id = setInterval(async () => {
      try {
        const updated = await api.jobStatus(job.job_id);
        setJob(updated);
        if (updated.status === "done") {
          const res = await api.jobResult(job.job_id);
          setResult(res);
        }
      } catch (e) { setError(e.message); clearInterval(id); }
    }, 2000);
    return () => clearInterval(id);
  }, [job]);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true); setError(null);
    try {
      const info = await api.uploadVideo(file);
      setVideo(info);
      setEndSec(Math.min(info.duration_seconds, 600));
    } catch (e) { setError(e.message); }
    finally { setUploading(false); }
  }

  async function handleSubmit() {
    setError(null); setResult(null); setJob(null);
    try {
      const j = await api.submitJob({
        video_id: video.video_id,
        task,
        start_sec: Number(startSec),
        end_sec: Number(endSec),
        models: selectedModels.length === models.length ? null : selectedModels,
      });
      setJob(j);
    } catch (e) { setError(e.message); }
  }

  function toggleModel(name) {
    setSelectedModels((prev) =>
      prev.includes(name) ? prev.filter((p) => p !== name) : [...prev, name],
    );
  }

  async function loadHistoryJob(jobId) {
    setError(null);
    try {
      const j = await api.jobStatus(jobId);
      setJob(j);
      setResult(j.status === "done" ? await api.jobResult(jobId) : null);
    } catch (e) { setError(e.message); }
  }

  function downloadResult() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `result_${job?.job_id || "job"}.json`;
    a.click(); URL.revokeObjectURL(url);
  }

  const segMin = ((endSec - startSec) / 60).toFixed(1);
  const segValid = endSec > startSec && endSec - startSec <= 30 * 60;

  return (
    <div className="min-h-screen bg-bg text-text">
      {/* ── Header ─────────────────────────── */}
      <header className="sticky top-0 z-40 bg-bg/80 backdrop-blur border-b border-border">
        <div className="max-w-content mx-auto px-6 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="inline-flex w-7 h-7 rounded-md bg-visual/10 ring-1 ring-visual/40
                             items-center justify-center text-visual font-mono text-xs font-bold">TV</span>
            <div>
              <div className="font-semibold text-sm tracking-tight">Multimodal Extraction</div>
              <div className="text-[11px] text-text-subtle">interactive workbench</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <a href="/" className="hidden sm:inline pill">Defense site →</a>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="max-w-content mx-auto px-6 py-8 grid lg:grid-cols-12 gap-6">
        {/* ── Sidebar: history ──────────────── */}
        <aside className="lg:col-span-3 lg:sticky lg:top-24 self-start">
          <h2 className="text-xs uppercase tracking-wide text-text-subtle mb-3">
            Jobs <span className="mono ml-1">({history.length})</span>
          </h2>
          <div className="card p-2 max-h-[80vh] overflow-y-auto space-y-1">
            {history.length === 0 && (
              <p className="text-xs text-text-subtle px-2 py-3">No jobs yet — upload a video to start.</p>
            )}
            {history.slice().reverse().map((j) => (
              <JobHistoryItem key={j.job_id} job={j}
                active={job?.job_id === j.job_id}
                onClick={() => loadHistoryJob(j.job_id)} />
            ))}
          </div>
        </aside>

        {/* ── Main column ───────────────────── */}
        <main className="lg:col-span-9 space-y-6">
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="card border-error/40 bg-error/5 text-error p-3 text-sm flex items-center gap-2"
              >
                <XCircle size={16} /> {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Step 1 — Upload */}
          <section className="card p-6">
            <div className="flex items-center gap-3 mb-4">
              <span className="inline-flex w-7 h-7 rounded-md bg-visual/10 text-visual mono text-xs font-bold items-center justify-center">1</span>
              <h2 className="text-lg font-semibold">Upload a video</h2>
            </div>

            <label className={`flex items-center justify-center gap-3 border-2 border-dashed border-border
                              rounded-lg py-10 cursor-pointer hover:bg-surface-2 transition-colors
                              ${uploading ? "opacity-60 cursor-wait" : ""}`}>
              <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} className="sr-only" />
              {uploading ? (
                <><Loader2 size={20} className="animate-spin text-visual"/> Uploading…</>
              ) : (
                <><Upload size={20} className="text-text-subtle"/> <span>Drop an MP4 or click to choose</span></>
              )}
            </label>

            {video && (
              <motion.div
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className="mt-4 flex items-center gap-3 px-4 py-3 rounded-md bg-surface-2 text-sm"
              >
                <FileVideo size={18} className="text-success shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{video.filename}</div>
                  <div className="text-xs text-text-muted mono">
                    {fmtSec(video.duration_seconds)} · {video.width}×{video.height} @ {video.fps}fps · {video.size_mb} MB
                  </div>
                </div>
              </motion.div>
            )}
          </section>

          {/* Step 2 — Configure */}
          <AnimatePresence>
            {video && (
              <motion.section
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                className="card p-6"
              >
                <div className="flex items-center gap-3 mb-5">
                  <span className="inline-flex w-7 h-7 rounded-md bg-visual/10 text-visual mono text-xs font-bold items-center justify-center">2</span>
                  <h2 className="text-lg font-semibold">Configure the run</h2>
                </div>

                {/* task selector */}
                <div className="mb-5">
                  <div className="text-xs uppercase tracking-wide text-text-subtle mb-2">Pipeline</div>
                  <div className="grid grid-cols-3 gap-2 max-w-md">
                    {[
                      ["ticker", "Visual",  "visual"],
                      ["audio",  "Audio",   "audio"],
                      ["both",   "Both",    "visual"],
                    ].map(([k, label, accent]) => (
                      <button
                        key={k}
                        onClick={() => setTask(k)}
                        className={`px-3 py-2 rounded-md border text-sm font-medium transition-colors ${
                          task === k
                            ? `border-${accent} bg-${accent}/10 text-${accent}`
                            : "border-border hover:bg-surface-2"
                        }`}
                      >{label}</button>
                    ))}
                  </div>
                </div>

                <VideoRangeSelector
                  src={api.videoUrl(video.video_id)}
                  duration={video.duration_seconds}
                  start={startSec} end={endSec}
                  maxSegment={30 * 60}
                  onChange={(s, e) => { setStartSec(s); setEndSec(e); }}
                />

                {/* model toggles */}
                <div className="mt-5">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs uppercase tracking-wide text-text-subtle">
                      Models <span className="mono">({selectedModels.length}/{models.length})</span>
                    </div>
                    <button
                      onClick={() => setSelectedModels(
                        selectedModels.length === models.length ? [] : models.map(m => m.display_name)
                      )}
                      className="text-[11px] text-text-muted hover:text-text"
                    >
                      {selectedModels.length === models.length ? "deselect all" : "select all"}
                    </button>
                  </div>
                  <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {models.map((m) => {
                      const on = selectedModels.includes(m.display_name);
                      return (
                        <label key={m.display_name}
                          className={`flex items-center gap-2 px-3 py-2 rounded-md border text-xs cursor-pointer transition-colors ${
                            on ? "border-visual bg-visual/10" : "border-border hover:bg-surface-2"
                          }`}>
                          <input type="checkbox" checked={on} onChange={() => toggleModel(m.display_name)} className="sr-only" />
                          <Cpu size={12} className={on ? "text-visual" : "text-text-subtle"} />
                          <span className="truncate">{m.display_name}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>

                <div className="flex items-center justify-between mt-6">
                  <div className="text-xs text-text-muted">
                    {segMin} min segment · {selectedModels.length} model{selectedModels.length === 1 ? "" : "s"}
                  </div>
                  <button onClick={handleSubmit}
                    disabled={!segValid || selectedModels.length === 0}
                    className="btn-primary"
                  >
                    <Play size={14} /> Run pipeline
                  </button>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Step 3 — Progress */}
          <AnimatePresence>
            {job && (
              <motion.section
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                className="card p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <span className="inline-flex w-7 h-7 rounded-md bg-visual/10 text-visual mono text-xs font-bold items-center justify-center">3</span>
                  <h2 className="text-lg font-semibold">Progress</h2>
                </div>
                <div className="flex items-center gap-3 mb-3">
                  <code className="text-[11px] mono text-text-muted">{job.job_id}</code>
                  <StatusPill status={job.status} />
                  <span className="text-xs text-text-muted ml-auto mono">
                    {Math.round((job.progress || 0) * 100)}%
                  </span>
                </div>
                <ProgressBar value={job.progress} />
                <div className="mt-3 text-xs text-text-muted">
                  <strong className="text-text">{job.stage}</strong>
                  {job.message ? ` — ${job.message}` : ""}
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Step 4 — Results */}
          <AnimatePresence>
            {result && (
              <motion.section
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                className="card p-6"
              >
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-3">
                    <span className="inline-flex w-7 h-7 rounded-md bg-success/15 text-success mono text-xs font-bold items-center justify-center">4</span>
                    <h2 className="text-lg font-semibold">Results</h2>
                  </div>
                  <button onClick={downloadResult} className="btn-secondary">
                    <Download size={14} /> JSON
                  </button>
                </div>

                {result.task === "ticker" && (
                  <ResultTicker result={result} />
                )}
                {result.task === "audio" && (
                  <ResultAudio result={result} />
                )}
                {result.task === "both" && (
                  <>
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-text-subtle mb-3">Visual path</h3>
                    <ResultTicker result={result.ticker || {}} />
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-text-subtle mt-8 mb-3">Audio path</h3>
                    <ResultAudio result={result.audio || {}} />
                  </>
                )}
              </motion.section>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

function ResultTicker({ result }) {
  return (
    <>
      <div className="mb-5">
        <h3 className="text-sm font-semibold mb-2">
          Headlines <span className="text-text-subtle mono">({result.cleaned_items?.length || 0})</span>
        </h3>
        <div className="grid sm:grid-cols-2 gap-2">
          {(result.cleaned_items || []).map((it, i) => (
            <div key={i} className="bg-surface-2 rounded-md px-3 py-2 text-xs">
              <span className="text-text-subtle mono mr-2">#{i + 1}</span>{it}
            </div>
          ))}
        </div>
      </div>
      <h3 className="text-sm font-semibold mb-3">9-LLM summaries</h3>
      <div className="grid lg:grid-cols-2 gap-3">
        {(result.summaries || []).map((s, i) => <SummaryCard key={i} summary={s} />)}
      </div>
    </>
  );
}

function ResultAudio({ result }) {
  return (
    <>
      <div className="mb-5">
        <p className="text-sm text-text-muted">
          Transcript: <strong className="text-text">{result.transcript_segments}</strong> segments,{" "}
          {result.transcript_length_chars?.toLocaleString()} chars
        </p>
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-visual">Transcript preview</summary>
          <pre className="bg-surface-2 p-3 text-xs mono whitespace-pre-wrap max-h-72 overflow-auto rounded mt-2">
            {result.transcript_preview}
          </pre>
        </details>
      </div>
      <h3 className="text-sm font-semibold mb-3">9-LLM summaries</h3>
      <div className="grid lg:grid-cols-2 gap-3">
        {(result.summaries || []).map((s, i) => <SummaryCard key={i} summary={s} />)}
      </div>
    </>
  );
}
