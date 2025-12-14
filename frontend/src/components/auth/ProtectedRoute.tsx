'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { PageLoader } from '@/components/ui';

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

  // Handle redirect when not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated && !isRedirecting) {
      setIsRedirecting(true);
      // Use replace to prevent back button returning to protected page
      router.replace(redirectTo);
    }
  }, [isAuthenticated, isLoading, isRedirecting, router, redirectTo]);

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
