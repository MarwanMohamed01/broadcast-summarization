import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NM = path.resolve(__dirname, 'node_modules');

// Shared components under ../../design-system/ have no node_modules of
// their own. The React Vite plugin handles 'react' / 'react-dom' for us;
// we only redirect the third-party packages those components import to
// *this* project's node_modules. Aliasing react itself would load its CJS
// jsx-runtime as ESM and crash — don't.
export default defineConfig({
  plugins: [react()],
  server: {
    fs: { allow: ['..', '../..'] },
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
});
