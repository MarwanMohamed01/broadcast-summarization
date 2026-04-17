import { useEffect, useState } from "react";
import { api } from "./api";
import VideoRangeSelector from "./VideoRangeSelector";
import "./App.css";

function fmtSec(s) {
  if (!s && s !== 0) return "—";
  const m = Math.floor(s / 60);
  const ss = Math.round(s - m * 60);
  return `${m}:${String(ss).padStart(2, "0")}`;
}

function StatusPill({ status }) {
  const cls = {
    queued: "bg-gray-400",
    running: "bg-blue-500",
    done: "bg-green-600",
    failed: "bg-red-600",
    cancelled: "bg-yellow-600",
  }[status] || "bg-gray-500";
  return (
    <span className={`${cls} text-white rounded-full px-3 py-0.5 text-xs font-semibold`}>
      {status}
    </span>
  );
}

function ProgressBar({ value }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="bg-gray-200 rounded-md h-3 overflow-hidden">
      <div
        className="bg-blue-500 h-full transition-all duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function SummaryCard({ summary }) {
  const ok = summary.status === "success";
  return (
    <div
      className={`border rounded-lg p-4 mb-3 ${
        ok ? "border-gray-200 bg-gray-50" : "border-red-200 bg-red-50"
      }`}
    >
      <div className="flex justify-between items-start mb-2">
        <strong className="text-gray-900">{summary.display_name}</strong>
        <span className="text-xs text-gray-500">
          {summary.latency_seconds?.toFixed(1)}s · {summary.output_tokens} tokens
        </span>
      </div>
      {ok ? (
        <p className="text-sm text-gray-800 leading-relaxed m-0 whitespace-pre-wrap">
          {summary.summary}
        </p>
      ) : (
        <p className="text-sm text-red-700 m-0">FAILED: {summary.error}</p>
      )}
    </div>
  );
}

function JobHistoryItem({ job, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-2 mb-1 rounded text-xs border ${
        active
          ? "border-blue-500 bg-blue-50"
          : "border-transparent hover:bg-gray-50"
      }`}
    >
      <div className="flex justify-between items-center mb-1">
        <code className="text-gray-600">{job.job_id.slice(0, 8)}</code>
        <StatusPill status={job.status} />
      </div>
      <div className="text-gray-500">
        {job.task} · {Math.round((job.end_sec - job.start_sec) / 60)} min
      </div>
    </button>
  );
}

export default function App() {
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
    api
      .models()
      .then((m) => {
        setModels(m);
        setSelectedModels(m.map((x) => x.display_name));
      })
      .catch((e) => setError(`Backend unreachable: ${e.message}`));
  }, []);

  // Refresh job history periodically
  useEffect(() => {
    const load = () => fetch("http://localhost:8000/api/jobs")
      .then((r) => r.json())
      .then(setHistory)
      .catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  // Poll active job
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
      } catch (e) {
        setError(e.message);
        clearInterval(id);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [job]);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const info = await api.uploadVideo(file);
      setVideo(info);
      setEndSec(Math.min(info.duration_seconds, 600));
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSubmit() {
    setError(null);
    setResult(null);
    setJob(null);
    try {
      const j = await api.submitJob({
        video_id: video.video_id,
        task,
        start_sec: Number(startSec),
        end_sec: Number(endSec),
        models:
          selectedModels.length === models.length ? null : selectedModels,
      });
      setJob(j);
    } catch (e) {
      setError(e.message);
    }
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
      if (j.status === "done") {
        const res = await api.jobResult(jobId);
        setResult(res);
      } else {
        setResult(null);
      }
    } catch (e) {
      setError(e.message);
    }
  }

  function downloadResult() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `result_${job?.job_id || "job"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const segMin = ((endSec - startSec) / 60).toFixed(1);
  const segValid = endSec > startSec && endSec - startSec <= 30 * 60;

  return (
    <div className="min-h-screen">
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="max-w-7xl mx-auto flex items-baseline justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 m-0">
              TV News Summarization
            </h1>
            <p className="text-sm text-gray-500 m-0">
              Upload a video, pick a range, get 9 LLM summaries
            </p>
          </div>
          <span className="text-xs text-gray-400">MVP · localhost</span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 flex gap-6">
        {/* ── History sidebar ── */}
        <aside className="w-64 shrink-0">
          <h2 className="text-sm font-semibold text-gray-700 mb-2">
            Jobs ({history.length})
          </h2>
          <div className="bg-white border border-gray-200 rounded-lg p-2 max-h-[80vh] overflow-y-auto">
            {history.length === 0 && (
              <p className="text-xs text-gray-400 m-2">No jobs yet</p>
            )}
            {history
              .slice()
              .reverse()
              .map((j) => (
                <JobHistoryItem
                  key={j.job_id}
                  job={j}
                  active={job?.job_id === j.job_id}
                  onClick={() => loadHistoryJob(j.job_id)}
                />
              ))}
          </div>
        </aside>

        {/* ── Main content ── */}
        <main className="flex-1 min-w-0">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 rounded-md p-3 mb-4 text-sm">
              {error}
            </div>
          )}

          {/* Step 1: Upload */}
          <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-5 mb-5">
            <h2 className="text-lg font-semibold mb-3 text-gray-900">
              1. Upload video
            </h2>
            <input
              type="file"
              accept="video/*"
              onChange={handleUpload}
              disabled={uploading}
              className="text-sm"
            />
            {uploading && (
              <span className="ml-2 text-sm text-gray-500">Uploading…</span>
            )}
            {video && (
              <div className="mt-3 text-sm text-gray-700">
                <span className="text-green-600 font-semibold">✓ </span>
                <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">
                  {video.video_id}
                </code>{" "}
                — {video.filename}
                <br />
                Duration: <strong>{fmtSec(video.duration_seconds)}</strong> ·{" "}
                {video.width}×{video.height} @ {video.fps}fps · {video.size_mb}MB
              </div>
            )}
          </section>

          {/* Step 2: Configure */}
          {video && (
            <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-5 mb-5">
              <h2 className="text-lg font-semibold mb-3 text-gray-900">
                2. Configure job
              </h2>

              <div className="mb-4">
                <div className="text-sm font-semibold text-gray-700 mb-1">
                  Task:
                </div>
                {["ticker", "audio", "both"].map((t) => (
                  <label
                    key={t}
                    className={`inline-flex items-center mr-4 px-3 py-1 rounded cursor-pointer border ${
                      task === t
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-300 hover:bg-gray-50"
                    }`}
                  >
                    <input
                      type="radio"
                      name="task"
                      value={t}
                      checked={task === t}
                      onChange={() => setTask(t)}
                      className="mr-2"
                    />
                    {t}
                  </label>
                ))}
              </div>

              <VideoRangeSelector
                src={api.videoUrl(video.video_id)}
                duration={video.duration_seconds}
                start={startSec}
                end={endSec}
                maxSegment={30 * 60}
                onChange={(s, e) => {
                  setStartSec(s);
                  setEndSec(e);
                }}
              />

              <div className="mt-4">
                <div className="text-sm font-semibold text-gray-700 mb-1">
                  LLMs ({selectedModels.length}/{models.length}):
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-sm">
                  {models.map((m) => (
                    <label key={m.display_name} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(m.display_name)}
                        onChange={() => toggleModel(m.display_name)}
                        className="mr-2"
                      />
                      {m.display_name}
                    </label>
                  ))}
                </div>
              </div>

              <button
                onClick={handleSubmit}
                disabled={!segValid || selectedModels.length === 0}
                className={`mt-5 px-5 py-2 rounded font-semibold text-white ${
                  segValid && selectedModels.length > 0
                    ? "bg-blue-500 hover:bg-blue-600 cursor-pointer"
                    : "bg-gray-300 cursor-not-allowed"
                }`}
              >
                ▶ Summarize
              </button>
              <span className="ml-3 text-xs text-gray-500">
                {segMin} min segment · {selectedModels.length} LLM
                {selectedModels.length === 1 ? "" : "s"}
              </span>
            </section>
          )}

          {/* Step 3: Progress */}
          {job && (
            <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-5 mb-5">
              <h2 className="text-lg font-semibold mb-3 text-gray-900">
                3. Job progress
              </h2>
              <div className="flex items-center gap-3 mb-2">
                <code className="text-xs bg-gray-100 px-2 py-0.5 rounded">
                  {job.job_id}
                </code>
                <StatusPill status={job.status} />
                <span className="text-xs text-gray-500">
                  {Math.round((job.progress || 0) * 100)}%
                </span>
              </div>
              <ProgressBar value={job.progress} />
              <div className="mt-2 text-xs text-gray-600">
                <strong className="text-gray-800">{job.stage}</strong>
                {job.message ? ` — ${job.message}` : ""}
              </div>
            </section>
          )}

          {/* Step 4: Results */}
          {result && (
            <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-5 mb-5">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-lg font-semibold text-gray-900 m-0">
                  4. Results
                </h2>
                <button
                  onClick={downloadResult}
                  className="text-xs border border-gray-300 hover:bg-gray-50 rounded px-3 py-1"
                >
                  ⬇ Download JSON
                </button>
              </div>

              {result.task === "ticker" && (
                <>
                  <h3 className="text-base font-semibold mt-0 mb-2">
                    Headlines ({result.cleaned_items?.length || 0})
                  </h3>
                  <ol className="text-sm text-gray-700 pl-5 mb-4">
                    {(result.cleaned_items || []).map((it, i) => (
                      <li key={i}>{it}</li>
                    ))}
                  </ol>
                  <h3 className="text-base font-semibold mt-4 mb-2">
                    Summaries
                  </h3>
                  {(result.summaries || []).map((s, i) => (
                    <SummaryCard key={i} summary={s} />
                  ))}
                </>
              )}

              {result.task === "audio" && (
                <>
                  <p className="text-sm text-gray-700">
                    Transcript:{" "}
                    <strong>{result.transcript_segments}</strong> segments,{" "}
                    {result.transcript_length_chars?.toLocaleString()} chars
                  </p>
                  <details className="mb-4">
                    <summary className="cursor-pointer text-sm text-blue-600">
                      Transcript preview
                    </summary>
                    <pre className="bg-gray-50 p-3 text-xs whitespace-pre-wrap max-h-72 overflow-auto rounded mt-2">
                      {result.transcript_preview}
                    </pre>
                  </details>
                  <h3 className="text-base font-semibold mt-4 mb-2">
                    Summaries
                  </h3>
                  {(result.summaries || []).map((s, i) => (
                    <SummaryCard key={i} summary={s} />
                  ))}
                </>
              )}

              {result.task === "both" && (
                <>
                  <h3 className="text-base font-semibold mt-0 mb-2">Ticker</h3>
                  <ol className="text-sm text-gray-700 pl-5 mb-4">
                    {(result.ticker?.cleaned_items || []).map((it, i) => (
                      <li key={i}>{it}</li>
                    ))}
                  </ol>
                  {(result.ticker?.summaries || []).map((s, i) => (
                    <SummaryCard key={`t${i}`} summary={s} />
                  ))}
                  <h3 className="text-base font-semibold mt-6 mb-2">Audio</h3>
                  {(result.audio?.summaries || []).map((s, i) => (
                    <SummaryCard key={`a${i}`} summary={s} />
                  ))}
                </>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
