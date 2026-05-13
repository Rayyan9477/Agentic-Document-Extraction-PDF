'use client';

/**
 * V3 Phase 8 — Theme toggle.
 *
 * Three-state cycle: light → dark → system → light.
 * Mounted in the header next to the user menu.
 *
 * Renders the icon for the *current resolved* theme so users can
 * quickly tell light vs dark; a tooltip shows the underlying
 * choice (including 'system').
 */

import { useState } from 'react';
import { Moon, Sun, Monitor } from 'lucide-react';
import { useTheme, type ThemeChoice } from '@/components/ThemeProvider';

const NEXT_CHOICE: Record<ThemeChoice, ThemeChoice> = {
  light: 'dark',
  dark: 'system',
  system: 'light',
};

const LABEL: Record<ThemeChoice, string> = {
  light: 'Light theme',
  dark: 'Dark theme',
  system: 'System theme',
};

export function ThemeToggle() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const [hover, setHover] = useState(false);

  const Icon =
    theme === 'system' ? Monitor : resolvedTheme === 'dark' ? Moon : Sun;

  return (
    <button
      type="button"
      onClick={() => setTheme(NEXT_CHOICE[theme])}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      aria-label={`Toggle theme — currently ${LABEL[theme]}`}
      title={LABEL[theme]}
      className="
        relative inline-flex h-9 w-9 items-center justify-center rounded-lg
        text-text-secondary hover:text-text-primary hover:bg-surface
        focus-ring transition-colors duration-fast
      "
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      {hover && (
        <span
          role="tooltip"
          className="
            pointer-events-none absolute top-full mt-2 right-0
            whitespace-nowrap rounded-md bg-surface-raised border border-default
            px-2 py-1 text-small text-text-primary shadow-elev-3
            animate-fade-in
          "
        >
          {LABEL[theme]}
        </span>
      )}
    </button>
  );
}
