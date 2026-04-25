# webapp/frontend/ — Interactive workbench

The hands-on side of the project: upload a video, pick a time range, choose
which of the 9 LLMs to invoke, and watch the FastAPI backend run the real
OCR / ASR / summarisation pipeline. Companion to the read-only
[defense site](../../site/) — both share visual identity via
[`design-system/`](../../design-system/).

## Stack

- **React 18 + Vite 5** — same setup as the original MVP
- **Tailwind 3** — wired to `../../design-system/tokens.css` (no duplicated colors)
- **Framer Motion 11** — micro-interactions on uploads, progress, results
- **Lucide React** — icons
- **`@fontsource/{inter, jetbrains-mono}`** — fonts ship offline
- **rc-slider** — already used by the existing `VideoRangeSelector`

The backend (`webapp/backend/`, FastAPI) is **unchanged** — this redesign is
UI-only. All API endpoints, request bodies, and response shapes stay the
same.

## Develop

```bash
cd webapp/frontend
npm install
npm run dev                  # http://localhost:5173

# Backend (in another terminal):
cd webapp/backend
uvicorn main:app --reload    # http://localhost:8000
```

Set `VITE_API_BASE` to point at a different backend if needed:
```bash
VITE_API_BASE=http://my-server:8000 npm run dev
```

## Production build

```bash
npm run build
npm run preview              # local preview of built assets
```

The `dist/` folder is a static bundle — serve from any web server, or hand
off to the same FastAPI process via static-file middleware (existing pattern).

## What changed in the redesign

| Before | After |
|---|---|
| Plain inline-style cards | Token-driven `card` / `pill` / `btn-primary` classes |
| One mega `App.jsx` with hard-coded blue/gray utilities | Same single file, but consumes design-system tokens |
| `VideoRangeSelector` with 200 lines of inline `style={{...}}` | Same component, Tailwind + tokens |
| Hard-coded light theme | Light + dark mode (toggle in nav, persists to `localStorage`) |
| `<header>` says "TV News Summarization · MVP · localhost" | Branded as part of the thesis (links to defense site) |
| Static status pills | Animated status pills with Lucide icons |
| `bg-blue-500` everywhere | `bg-visual` semantic accents (matches defense site) |
| Status drops into the page abruptly | Framer Motion enter/exit transitions for every step |
| Drag-and-drop visually identical to a file picker | Big dashed drop zone with hover state |

Behaviourally identical — same backend, same workflow, same shortcuts.

## Folder layout

```
webapp/frontend/
├── package.json
├── vite.config.js              (allows imports from ../../design-system/)
├── tailwind.config.js          (pulls in design-system tokens + content glob)
├── postcss.config.js
├── index.html
├── src/
│   ├── main.jsx
│   ├── App.jsx                 (redesigned)
│   ├── App.css                 (rc-slider focus-ring patches only)
│   ├── index.css               (tokens + base utility classes)
│   ├── api.js                  (UNCHANGED — same endpoints, same shapes)
│   └── VideoRangeSelector.jsx  (redesigned — Tailwind, tokens, Lucide)
└── public/
```

## Backend contract (unchanged)

The frontend talks to FastAPI on `localhost:8000` via `src/api.js`:

- `GET  /health`
- `GET  /api/models`
- `POST /api/upload`            (multipart `file`)
- `GET  /api/videos/:id`        (raw video stream for the player)
- `POST /api/jobs`              (create — `video_id`, `task`, `start_sec`, `end_sec`, `models`)
- `GET  /api/jobs`              (list)
- `GET  /api/jobs/:id`          (status poll)
- `GET  /api/jobs/:id/result`   (final payload when status == done)
