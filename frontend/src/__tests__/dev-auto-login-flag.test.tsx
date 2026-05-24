/**
 * R1.3 regression — DEV_AUTO_LOGIN must be hard-off in production.
 *
 * Before R1.3 the flag was hardcoded to ``true`` (a literal const
 * with no env / NODE_ENV gate, so every ``next build`` shipped an auth
 * bypass into production. The fix gates it on:
 *   1. ``process.env.NODE_ENV !== 'production'`` (hard fail-closed)
 *   2. ``process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN === 'true'`` (opt-in)
 *
 * The matching build-time guard lives in ``next.config.js`` — that
 * one is impossible to unit-test here because vitest doesn't load
 * ``next.config.js``. Coverage for that guard lives in a manual run:
 *
 *     NODE_ENV=production NEXT_PUBLIC_DEV_AUTO_LOGIN=true \
 *         npm run build   # must FAIL
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const ORIGINAL_NODE_ENV = process.env.NODE_ENV;
const ORIGINAL_DEV_FLAG = process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN;

function setEnv(nodeEnv: string | undefined, devFlag: string | undefined): void {
  if (nodeEnv === undefined) {
    delete (process.env as Record<string, string | undefined>).NODE_ENV;
  } else {
    process.env.NODE_ENV = nodeEnv;
  }
  if (devFlag === undefined) {
    delete process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN;
  } else {
    process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN = devFlag;
  }
}

beforeEach(() => {
  // Reset module cache so re-importing ProtectedRoute picks up the
  // new env vars (the flag is computed at module load time).
  vi.resetModules();
});

afterEach(() => {
  setEnv(ORIGINAL_NODE_ENV, ORIGINAL_DEV_FLAG);
  vi.resetModules();
});

/**
 * Helper: import ProtectedRoute fresh, then probe whether the
 * DEV_AUTO_LOGIN literal resolved truthy by checking whether the
 * component renders its children unconditionally (the only effect of
 * the flag inside the component).
 *
 * We don't import the constant directly because it isn't exported.
 * Instead we render the component with auth state set to "not
 * authenticated" — if DEV_AUTO_LOGIN is true the component renders
 * children; if false it shows the redirect loader.
 */
async function probeDevAutoLoginEffect(): Promise<'children' | 'loader'> {
  // Mock the auth + router modules BEFORE importing ProtectedRoute.
  vi.doMock('@/hooks/useAuth', () => ({
    useAuth: () => ({ isAuthenticated: false, isLoading: false }),
  }));
  vi.doMock('next/navigation', () => ({
    useRouter: () => ({ replace: vi.fn() }),
  }));
  vi.doMock('@/components/ui', () => ({
    PageLoader: ({ text }: { text: string }) => (
      <div data-testid="page-loader">{text}</div>
    ),
  }));

  const React = (await import('react')).default;
  const { render, screen } = await import('@testing-library/react');
  const ProtectedRouteModule = await import(
    '@/components/auth/ProtectedRoute'
  );
  const ProtectedRoute = ProtectedRouteModule.default;

  render(
    React.createElement(
      ProtectedRoute,
      null,
      React.createElement(
        'div',
        { 'data-testid': 'protected-content' },
        'secret',
      ),
    ),
  );

  if (screen.queryByTestId('protected-content')) {
    return 'children';
  }
  return 'loader';
}

describe('DEV_AUTO_LOGIN env gating', () => {
  it('production + flag=true → bypass MUST be OFF (loader rendered)', async () => {
    setEnv('production', 'true');
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('loader');
  });

  it('production + flag=false → bypass OFF (loader rendered)', async () => {
    setEnv('production', 'false');
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('loader');
  });

  it('production + flag unset → bypass OFF (loader rendered)', async () => {
    setEnv('production', undefined);
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('loader');
  });

  it('development + flag=true → bypass ON (children rendered)', async () => {
    setEnv('development', 'true');
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('children');
  });

  it('development + flag=false → bypass OFF (loader rendered)', async () => {
    setEnv('development', 'false');
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('loader');
  });

  it('development + flag unset → bypass OFF by default (loader rendered)', async () => {
    setEnv('development', undefined);
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('loader');
  });

  it('test env + flag=true → bypass ON (children rendered)', async () => {
    // ``test`` is the vitest default, also acts as non-production.
    setEnv('test', 'true');
    const effect = await probeDevAutoLoginEffect();
    expect(effect).toBe('children');
  });
});
