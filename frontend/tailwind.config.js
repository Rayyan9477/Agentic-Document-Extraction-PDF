/** @type {import('tailwindcss').Config} */
// V3 Phase 8 — design tokens with semantic aliases, dark-mode support,
// and motion / elevation / focus tokens. Documented in
// frontend/src/lib/design.md. The semantic palette (canvas / surface /
// text-primary / accent-brand / etc.) maps to CSS custom properties
// defined in frontend/src/app/globals.css for both :root and .dark.
module.exports = {
  // Phase 8: dark mode is `class`-driven so ThemeProvider can flip it.
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // ─── Semantic tokens (Phase 8) ────────────────────────────
        // Map to CSS custom properties so dark mode flips entire UI
        // without touching Tailwind classes.
        canvas: 'rgb(var(--bg-canvas-rgb) / <alpha-value>)',
        surface: {
          DEFAULT: 'rgb(var(--bg-surface-rgb) / <alpha-value>)',
          raised: 'rgb(var(--bg-surface-raised-rgb) / <alpha-value>)',
          overlay: 'rgb(var(--bg-overlay-rgb) / <alpha-value>)',
          // Legacy ramp kept for compat with pre-Phase-8 components.
          50: '#fafafa',
          100: '#f4f4f5',
          200: '#e4e4e7',
          300: '#d4d4d8',
          400: '#a1a1aa',
          500: '#71717a',
          600: '#52525b',
          700: '#3f3f46',
          800: '#27272a',
          900: '#18181b',
        },
        // Semantic text aliases — every component should prefer these.
        'text-primary': 'rgb(var(--text-primary-rgb) / <alpha-value>)',
        'text-secondary': 'rgb(var(--text-secondary-rgb) / <alpha-value>)',
        'text-muted': 'rgb(var(--text-muted-rgb) / <alpha-value>)',
        // Semantic borders.
        'border-default': 'rgb(var(--border-default-rgb) / <alpha-value>)',
        'border-strong': 'rgb(var(--border-strong-rgb) / <alpha-value>)',
        // Semantic accents — flip-safe across light/dark.
        'accent-brand': 'rgb(var(--accent-brand-rgb) / <alpha-value>)',
        'accent-brand-soft': 'rgb(var(--accent-brand-soft-rgb) / <alpha-value>)',
        'accent-success': 'rgb(var(--accent-success-rgb) / <alpha-value>)',
        'accent-warning': 'rgb(var(--accent-warning-rgb) / <alpha-value>)',
        'accent-danger': 'rgb(var(--accent-danger-rgb) / <alpha-value>)',
        'accent-info': 'rgb(var(--accent-info-rgb) / <alpha-value>)',

        // ─── Legacy ramps (kept; new code prefers semantic above) ──
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        accent: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        success: {
          light: '#dcfce7',
          DEFAULT: '#22c55e',
          dark: '#15803d',
        },
        warning: {
          light: '#fef3c7',
          DEFAULT: '#f59e0b',
          dark: '#b45309',
        },
        error: {
          light: '#fee2e2',
          DEFAULT: '#ef4444',
          dark: '#b91c1c',
        },
        info: {
          light: '#dbeafe',
          DEFAULT: '#3b82f6',
          dark: '#1d4ed8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      // Phase 8 — typography scale (see design.md).
      fontSize: {
        'display': ['2rem', { lineHeight: '2.5rem', fontWeight: '600' }],
        'h1': ['1.5rem', { lineHeight: '2rem', fontWeight: '600' }],
        'h2': ['1.25rem', { lineHeight: '1.75rem', fontWeight: '600' }],
        'h3': ['1rem', { lineHeight: '1.5rem', fontWeight: '500' }],
        'body': ['0.875rem', { lineHeight: '1.25rem' }],
        'small': ['0.75rem', { lineHeight: '1rem' }],
      },
      // Phase 8 — elevation tokens. `elev-modal` is the existing
      // modal shadow; `elev-{0,1,2,3}` are the documented stops.
      boxShadow: {
        'elev-0': 'none',
        'elev-1': '0 1px 2px 0 rgb(0 0 0 / 0.06)',
        'elev-2':
          '0 2px 15px -3px rgb(0 0 0 / 0.07), 0 10px 20px -2px rgb(0 0 0 / 0.04)',
        'elev-3': '0 16px 40px -8px rgb(0 0 0 / 0.10)',
        'elev-modal': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
        // Legacy aliases.
        'soft': '0 2px 15px -3px rgb(0 0 0 / 0.07), 0 10px 20px -2px rgb(0 0 0 / 0.04)',
        'glow': '0 0 20px rgb(59 130 246 / 0.3)',
        'inner-soft': 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
      },
      transitionDuration: {
        // Phase 8 — motion scale.
        'instant': '0ms',
        'fast': '120ms',
        'base': '200ms',
        'slow': '320ms',
        '400': '400ms',
      },
      transitionTimingFunction: {
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      animation: {
        'fade-in': 'fadeIn 200ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-up': 'slideUp 200ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-down': 'slideDown 200ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-in-right': 'slideInRight 200ms cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 2s linear infinite',
        'bounce-subtle': 'bounceSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        bounceSubtle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'mesh-gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      },
      borderRadius: {
        '4xl': '2rem',
      },
    },
  },
  plugins: [],
};
