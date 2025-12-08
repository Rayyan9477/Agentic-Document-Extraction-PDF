'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/store/authStore';
import { authApi, getAccessToken } from '@/lib/api';
import type { User } from '@/types/api';

export function useAuth() {
  const { user, isAuthenticated, isLoading, setUser, setLoading, logout } = useAuthStore();

  // Fetch current user on mount if token exists
  const { data, isLoading: queryLoading, error } = useQuery<User>({
    queryKey: ['auth', 'currentUser'],
    queryFn: () => authApi.getCurrentUser(),
    enabled: !!getAccessToken() && !user,
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  useEffect(() => {
    if (data) {
      setUser(data);
    } else if (error) {
      logout();
    } else if (!getAccessToken()) {
      setLoading(false);
    }
  }, [data, error, setUser, logout, setLoading]);

  const handleLogout = async () => {
    try {
      await authApi.logout();
    } catch {
      // Ignore errors during logout
    } finally {
      logout();
    }
  };

  return {
    user,
    isAuthenticated,
    isLoading: isLoading || queryLoading,
    logout: handleLogout,
  };
}

// Hook for protected routes
export function useRequireAuth(redirectTo: string = '/login') {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push(redirectTo);
    }
  }, [isAuthenticated, isLoading, router, redirectTo]);

  return { isAuthenticated, isLoading };
}

// Hook to redirect if already authenticated
export function useRedirectIfAuthenticated(redirectTo: string = '/dashboard') {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push(redirectTo);
    }
  }, [isAuthenticated, isLoading, router, redirectTo]);

  return { isAuthenticated, isLoading };
}
