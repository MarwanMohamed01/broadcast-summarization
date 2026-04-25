import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NM = path.resolve(__dirname, 'node_modules');

// Shared components under ../design-system/ have no node_modules of their
// own. Astro's React integration handles 'react' / 'react-dom' resolution
// for us; we only need to redirect the third-party packages those
// components import (framer-motion, lucide-react) to *this* project's
// node_modules. Aliasing react itself loads its CJS jsx-runtime as ESM
// and crashes — don't.
export default defineConfig({
  output: 'static',
  integrations: [
    react(),
    tailwind({ applyBaseStyles: false }),
  ],
  vite: {
    server: {
      fs: { allow: ['..'] },
    },
    resolve: {
      alias: {
        'framer-motion': path.join(NM, 'framer-motion'),
        'lucide-react':  path.join(NM, 'lucide-react'),
      },
    },
    optimizeDeps: {
      include: ['framer-motion', 'lucide-react'],
    },
  },
});
