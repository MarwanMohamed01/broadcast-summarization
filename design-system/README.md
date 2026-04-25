# design-system/ — shared visual identity

Single source of truth for the tokens, components, and styles used by both
[`site/`](../site/) (the read-only defense site) and
[`webapp/frontend/`](../webapp/frontend/) (the interactive workbench).

## Layout

```
design-system/
├── tokens.css                  CSS custom properties (colors, type, motion, spacing)
└── components/
    ├── shared/
    │   ├── StagePanel.jsx      Wrapper for any pipeline stage (text + illustration)
    │   ├── ResultsHero.jsx     Headline P/R/F1/CER cards on /results
    │   ├── GTOverlay.jsx       Side-by-side GT-vs-extracted comparison
    │   └── Explorer.jsx        Tabbed data browser (headlines / transcript / etc.)
    ├── visual/
    │   ├── SourceVideoStage.jsx
    │   ├── FrameExtractionStage.jsx
    │   ├── ScrollDetectionStage.jsx
    │   ├── PanoramaStitchStage.jsx
    │   ├── OCRStage.jsx
    │   ├── SegmentationStage.jsx
    │   ├── LLMCleanStage.jsx
    │   ├── LLMFanoutStage.jsx       (also used by audio path with accent="audio")
    │   └── EvalStage.jsx            (also used by audio path with accent="audio")
    └── audio/
        ├── AudioExtractStage.jsx
        ├── ASRStage.jsx
        ├── ChunkingStage.jsx
        └── HierarchicalSummaryStage.jsx
```

## Consuming this from another project

There is **no build step** for this folder — components are imported by
relative path. Both consumer projects already pull React + Framer Motion +
Lucide React from their own `node_modules`.

In `site/` (Astro):
```jsx
import StagePanel from "../../../design-system/components/shared/StagePanel.jsx";
```

In `webapp/frontend/` (Vite):
```jsx
import StagePanel from "../../../design-system/components/shared/StagePanel.jsx";
```

Both projects' Tailwind configs include the design-system folder in their
`content` glob so utility classes get extracted.

Both projects' Vite configs allow filesystem access above their root via
`server.fs.allow: ['..', '../..']` so the imports resolve in dev.

## Tokens

Edit `tokens.css`. Both sites pick up the change instantly (no build needed
for the tokens alone, since both projects `@import` the CSS file).

The token system covers:
- Colors (light + dark mode, semantic + path accents)
- Type (Inter sans + JetBrains Mono)
- Motion (fast/medium/slow + ease curves)
- Radii, shadows, spacing
- Skip-to-content / focus rings / `prefers-reduced-motion` short-circuit

## Conventions

- Every illustration component takes `accent: "visual" | "audio"` (default
  `"visual"`) so the same component can be reused in both pipelines.
- Every stage component is wrapped in `<StagePanel>` and respects its
  scroll-into-view trigger and replay key.
- No component fetches at the module top level — all `fetch()` calls are
  inside `useEffect` so build-time SSR doesn't hit the network.
