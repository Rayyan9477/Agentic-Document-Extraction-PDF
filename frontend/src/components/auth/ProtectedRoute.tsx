'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { PageLoader } from '@/components/ui';

// DEV MODE: Skip auth check entirely.
//
// R1.3 (P0) fix: this was previously a hardcoded ``true`` literal with
// no env / NODE_ENV gate, so every ``next build`` shipped an auth
// bypass into production. The flag now:
//   1. Is FORCED off whenever ``NODE_ENV === 'production'`` regardless
//      of what the operator set.
//   2. Otherwise reads ``NEXT_PUBLIC_DEV_AUTO_LOGIN``, defaulting to
//      OFF in dev too — operators must opt in by setting it to
//      ``"true"`` in ``.env.local``.
//
// A matching build-time guard in ``next.config.js`` refuses to produce
// a production build at all when the env var is set to ``"true"`` AND
// ``NODE_ENV === 'production'``, so an operator can't silently re-enable
// the bypass by typo.
const DEV_AUTO_LOGIN =
  process.env.NODE_ENV !== 'production' &&
  process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN === 'true';

interface ProtectedRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  /** Optional message shown during redirect */
  redirectMessage?: string;
}

/**
 * Protected route wrapper that handles authentication state.
 *
 * SECURITY: Shows loading state during auth check AND during redirect
 * to prevent blank screen flash that could expose page structure.
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  redirectTo = '/login',
  redirectMessage = 'Redirecting to login...',
}) => {
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();
  const [isRedirecting, setIsRedirecting] = useState(false);

  // Handle redirect when not authenticated. Hook is declared
  // unconditionally so React's hook-order invariant holds even when
  // ``DEV_AUTO_LOGIN`` short-circuits below — the effect simply no-ops
  // in dev mode.
  useEffect(() => {
    if (DEV_AUTO_LOGIN) return;
    if (!isLoading && !isAuthenticated && !isRedirecting) {
      setIsRedirecting(true);
      // Use replace to prevent back button returning to protected page
      router.replace(redirectTo);
    }
  }, [isAuthenticated, isLoading, isRedirecting, router, redirectTo]);

  // DEV MODE: Skip all auth checks
  if (DEV_AUTO_LOGIN) {
    return <>{children}</>;
  }

  // Show loading while checking auth
  if (isLoading) {
    return <PageLoader text="Checking authentication..." />;
  }

  // Show loading while redirecting (prevents blank screen)
  if (!isAuthenticated || isRedirecting) {
    return <PageLoader text={redirectMessage} />;
  }

  // User is authenticated - render children
  return <>{children}</>;
};

export default ProtectedRoute;
