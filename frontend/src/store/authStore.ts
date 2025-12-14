import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from '@/types/api';

// Token storage keys (must match api.ts)
const ACCESS_TOKEN_KEY = 'pdf_access_token';
const REFRESH_TOKEN_KEY = 'pdf_refresh_token';

/**
 * Clear all auth tokens from localStorage.
 *
 * SECURITY: This is called synchronously with store state clearing
 * to prevent race conditions where tokens exist but user state is null.
 */
const clearAllAuthTokens = (): void => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }
};

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      isLoading: true,

      setUser: (user) =>
        set({
          user,
          isAuthenticated: !!user,
          isLoading: false,
        }),

      setLoading: (loading) =>
        set({
          isLoading: loading,
        }),

      logout: () => {
        // SECURITY FIX: Clear tokens FIRST (synchronously) to prevent
        // race condition where user state is null but tokens still exist.
        // This ensures auth state and token state are always consistent.
        clearAllAuthTokens();

        // Then clear store state
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
      // SECURITY: Also clear persisted state on logout by using onRehydrate
      onRehydrateStorage: () => (state) => {
        // On rehydrate, verify tokens exist if user is authenticated
        if (state?.isAuthenticated) {
          if (typeof window !== 'undefined') {
            const hasAccessToken = !!localStorage.getItem(ACCESS_TOKEN_KEY);
            const hasRefreshToken = !!localStorage.getItem(REFRESH_TOKEN_KEY);

            // If state says authenticated but no tokens, force logout
            if (!hasAccessToken && !hasRefreshToken) {
              state.logout();
            }
          }
        }
      },
    }
  )
);
