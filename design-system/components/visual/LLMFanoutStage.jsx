import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import StagePanel from "../shared/StagePanel.jsx";

const MODELS = [
  { id: "groq-70b",  label: "Llama 3.3 70B",   provider: "Groq" },
  { id: "groq-8b",   label: "Llama 3.1 8B",    provider: "Groq" },
  { id: "groq-qwen", label: "Qwen3 32B",       provider: "Groq" },
  { id: "groq-l4",   label: "Llama 4 Scout",   provider: "Groq" },
  { id: "ollama-3b", label: "Llama 3.2 3B",    provider: "Ollama" },
  { id: "ollama-8b", label: "Llama 3.1 8B",    provider: "Ollama" },
  { id: "gemini",    label: "Gemini 2.5 Flash",provider: "Google" },
  { id: "hf-llama",  label: "Llama 3 8B",      provider: "HuggingFace" },
  { id: "cohere",    label: "Command-R",       provider: "Cohere" },
];

export default function LLMFanoutStage({ accent = "visual" }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-15% 0px" });
  const accentBg = accent === "audio" ? "bg-audio" : "bg-visual";
  const accentText = accent === "audio" ? "text-audio" : "text-visual";

  return (
    <StagePanel
      number={accent === "audio" ? "4" : "7"}
      title="9-LLM summarization fan-out"
      blurb={
        <p>
          The cleaned headlines (or, on the audio path, the chunked transcript)
          are sent to <strong>nine</strong> different language models in
          parallel using one shared prompt.  Each model produces an independent
          summary so the evaluation can rank them on the same input rather
          than the same family of model.
        </p>
      }
      accent={accent}
      data={{ models: MODELS.map(m => `${m.provider} · ${m.label}`) }}
    >
      <div ref={ref} className="relative h-72">
        {/* central input */}
        <div className="absolute inset-0 grid place-items-center">
          <div className="card px-4 py-2 text-xs mono">38 headlines</div>
        </div>

        {/* radial model nodes */}
        {MODELS.map((m, i) => {
          const angle = (i / MODELS.length) * Math.PI * 2 - Math.PI / 2;
          const r = 130;
          const x = Math.cos(angle) * r;
          const y = Math.sin(angle) * r;
          return (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, x: 0, y: 0, scale: 0.6 }}
              animate={inView ? { opacity: 1, x, y, scale: 1 } : {}}
              transition={{ delay: 0.2 + i * 0.05, duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
            >
              <div className="card px-2.5 py-1.5 flex flex-col items-center gap-0.5 min-w-[110px]">
                <span className="text-[9px] uppercase tracking-wide text-text-subtle">{m.provider}</span>
                <span className="text-xs font-medium">{m.label}</span>
              </div>
            </motion.div>
          );
        })}

        {/* connecting lines */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="-180 -150 360 300">
          {MODELS.map((m, i) => {
            const angle = (i / MODELS.length) * Math.PI * 2 - Math.PI / 2;
            const r = 110;
            const x = Math.cos(angle) * r;
            const y = Math.sin(angle) * r;
            return (
              <motion.line
                key={m.id}
                x1={0} y1={0} x2={x} y2={y}
                stroke="currentColor"
                className={accentText}
                strokeOpacity={0.25}
                strokeWidth={1}
                initial={{ pathLength: 0 }}
                animate={inView ? { pathLength: 1 } : {}}
                transition={{ delay: 0.1 + i * 0.05, duration: 0.4 }}
              />
            );
          })}
        </svg>
      </div>
    </StagePanel>
  );
}
