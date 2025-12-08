'use client';

import React from 'react';
import { useRequireAuth } from '@/hooks/useAuth';
import { PageLoader } from '@/components/ui';

interface ProtectedRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  redirectTo = '/login',
}) => {
  const { isAuthenticated, isLoading } = useRequireAuth(redirectTo);

  if (isLoading) {
    return <PageLoader text="Checking authentication..." />;
  }

  if (!isAuthenticated) {
    return null; // Will redirect via the hook
  }

  return <>{children}</>;
};

export default ProtectedRoute;
