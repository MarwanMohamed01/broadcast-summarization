/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
    "../../design-system/components/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg:           'var(--color-bg)',
        surface:      'var(--color-surface)',
        'surface-2':  'var(--color-surface-2)',
        border:       'var(--color-border)',
        text:         'var(--color-text)',
        'text-muted': 'var(--color-text-muted)',
        'text-subtle':'var(--color-text-subtle)',
        visual:       'var(--color-visual)',
        audio:        'var(--color-audio)',
        success:      'var(--color-success)',
        warn:         'var(--color-warn)',
        error:        'var(--color-error)',
      },
      fontFamily: {
        sans: 'var(--font-sans)',
        mono: 'var(--font-mono)',
      },
      maxWidth: {
        content: 'var(--max-content)',
      },
      borderRadius: {
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
        xl: 'var(--radius-xl)',
      },
      boxShadow: {
        card:     'var(--shadow-card)',
        elevated: 'var(--shadow-elevated)',
      },
    },
  },
  plugins: [],
};
