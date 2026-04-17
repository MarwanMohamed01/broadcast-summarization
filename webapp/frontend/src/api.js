// Thin API client for the FastAPI backend.
// Default base URL points at the dev server on localhost:8000.

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  models: () => request("/api/models"),

  uploadVideo: async (file) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/api/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  videoInfo: (id) => request(`/api/videos/${id}/info`),
  videoUrl: (id) => `${BASE}/api/videos/${id}`,

  submitJob: (body) =>
    request("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  jobStatus: (id) => request(`/api/jobs/${id}`),
  jobResult: (id) => request(`/api/jobs/${id}/result`),
};
