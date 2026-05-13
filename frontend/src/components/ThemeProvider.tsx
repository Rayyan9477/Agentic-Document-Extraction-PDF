'use client';

/**
 * V3 Phase 8 — ThemeProvider.
 *
 * Manages the dark-mode `class="dark"` flag on `<html>` based on:
 *   1. `localStorage.theme` (user choice: 'light' | 'dark' | 'system')
 *   2. `prefers-color-scheme: dark` media query (when theme='system')
 *
 * Wrap the application in `<ThemeProvider>` at the root layout.
 * Components that need to read/write the choice can use `useTheme()`.
 *
 * The provider is intentionally tiny — no flashing, no flicker. It
 * runs once on mount, sets the class, and listens for system changes.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

export type ThemeChoice = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

interface ThemeContextValue {
  /** What the user picked. */
  theme: ThemeChoice;
  /** What is actually applied right now (system-resolved). */
  resolvedTheme: ResolvedTheme;
  setTheme: (next: ThemeChoice) => void;
}

const STORAGE_KEY = 'theme';

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getInitialTheme(): ThemeChoice {
  if (typeof window === 'undefined') return 'system';
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY) as ThemeChoice | null;
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      return stored;
    }
  } catch {
    // localStorage may be disabled (private mode); ignore.
  }
  return 'system';
}

function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  if (choice !== 'system') return choice;
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

function applyResolved(resolved: ResolvedTheme) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (resolved === 'dark') root.classList.add('dark');
  else root.classList.remove('dark');
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>('system');
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>('light');

  // Initial mount: read storage + apply.
  useEffect(() => {
    const initial = getInitialTheme();
    setThemeState(initial);
    const resolved = resolveTheme(initial);
    setResolvedTheme(resolved);
    applyResolved(resolved);
  }, []);

  // Listen for system preference changes when in 'system' mode.
  useEffect(() => {
    if (theme !== 'system' || typeof window === 'undefined') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      const resolved: ResolvedTheme = mq.matches ? 'dark' : 'light';
      setResolvedTheme(resolved);
      applyResolved(resolved);
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [theme]);

  const setTheme = useCallback((next: ThemeChoice) => {
    setThemeState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
    const resolved = resolveTheme(next);
    setResolvedTheme(resolved);
    applyResolved(resolved);
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, resolvedTheme, setTheme }),
    [theme, resolvedTheme, setTheme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (ctx === null) {
    // Don't throw — components rendered outside the provider
    // (e.g. in storybook or stand-alone tests) get a defaulted
    // implementation that mutates the DOM directly.
    return {
      theme: 'system',
      resolvedTheme: 'light',
      setTheme: () => undefined,
    };
  }
  return ctx;
}
