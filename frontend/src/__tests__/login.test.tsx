/**
 * Login Page Component Tests
 *
 * Tests cover:
 * - Form rendering
 * - Form validation
 * - Error handling
 * - Loading states
 * - Successful login flow
 * - Navigation after login
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import toast from 'react-hot-toast';
import LoginPage from '@/app/login/page';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';

// Mock the API
vi.mock('@/lib/api', () => ({
  authApi: {
    login: vi.fn(),
    getCurrentUser: vi.fn(),
  },
}));

// Mock the router
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => '/login',
}));

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    useAuthStore.setState({ user: null, isAuthenticated: false });
  });

  describe('Rendering', () => {
    it('renders login form with all required fields', () => {
      render(<LoginPage />);

      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    it('renders logo and branding', () => {
      render(<LoginPage />);

      expect(screen.getByText(/PDF Document Extraction/i)).toBeInTheDocument();
      expect(screen.getByText(/Sign in to your account/i)).toBeInTheDocument();
    });

    it('renders link to signup page', () => {
      render(<LoginPage />);

      const signupLink = screen.getByText(/Sign up/i);
      expect(signupLink).toBeInTheDocument();
      expect(signupLink.closest('a')).toHaveAttribute('href', '/signup');
    });

    it('renders forgot password link', () => {
      render(<LoginPage />);

      const forgotLink = screen.getByText(/Forgot password/i);
      expect(forgotLink).toBeInTheDocument();
    });

    it('renders remember me checkbox', () => {
      render(<LoginPage />);

      expect(screen.getByText(/Remember me/i)).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('shows error when submitting empty username', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/username is required/i)).toBeInTheDocument();
      });
    });

    it('shows error when submitting empty password', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      await user.type(usernameInput, 'testuser');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/password is required/i)).toBeInTheDocument();
      });
    });

    it('shows error when password is too short', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, '12345'); // Less than 6 characters

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(
          screen.getByText(/password must be at least 6 characters/i)
        ).toBeInTheDocument();
      });
    });

    it('clears errors when user starts typing', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/username is required/i)).toBeInTheDocument();
      });

      const usernameInput = screen.getByLabelText(/username/i);
      await user.type(usernameInput, 't');

      await waitFor(() => {
        expect(screen.queryByText(/username is required/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Successful Login', () => {
    it('calls login API with correct credentials', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);
      const mockGetUser = vi.mocked(authApi.getCurrentUser);

      mockLogin.mockResolvedValueOnce({
        access_token: 'mock-access-token',
        refresh_token: 'mock-refresh-token',
        token_type: 'bearer',
        expires_in: 1800,
      });

      mockGetUser.mockResolvedValueOnce({
        user_id: 'user-123',
        username: 'testuser',
        email: 'test@example.com',
        roles: ['viewer'],
        permissions: ['document:read'],
      });

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          username: 'testuser',
          password: 'password123',
        });
      });
    });

    it('shows loading state during login', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);

      // Create a promise that we can control
      let resolveLogin: any;
      const loginPromise = new Promise((resolve) => {
        resolveLogin = resolve;
      });

      mockLogin.mockReturnValueOnce(loginPromise as any);

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      // Button should be disabled during loading
      await waitFor(() => {
        expect(submitButton).toBeDisabled();
      });

      // Resolve the login
      resolveLogin({
        access_token: 'token',
        refresh_token: 'refresh',
        token_type: 'bearer',
        expires_in: 1800,
      });
    });

    it('redirects to dashboard after successful login', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);
      const mockGetUser = vi.mocked(authApi.getCurrentUser);

      mockLogin.mockResolvedValueOnce({
        access_token: 'mock-token',
        refresh_token: 'mock-refresh',
        token_type: 'bearer',
        expires_in: 1800,
      });

      mockGetUser.mockResolvedValueOnce({
        user_id: 'user-123',
        username: 'testuser',
        email: 'test@example.com',
        roles: ['viewer'],
        permissions: ['document:read'],
      });

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/dashboard');
      });
    });

    it('shows success toast after login', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);
      const mockGetUser = vi.mocked(authApi.getCurrentUser);

      mockLogin.mockResolvedValueOnce({
        access_token: 'token',
        refresh_token: 'refresh',
        token_type: 'bearer',
        expires_in: 1800,
      });

      mockGetUser.mockResolvedValueOnce({
        user_id: 'user-123',
        username: 'testuser',
        email: 'test@example.com',
        roles: ['viewer'],
        permissions: ['document:read'],
      });

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Welcome back!');
      });
    });

    it('updates auth store with user data', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);
      const mockGetUser = vi.mocked(authApi.getCurrentUser);

      const mockUserData = {
        user_id: 'user-123',
        username: 'testuser',
        email: 'test@example.com',
        roles: ['viewer'],
        permissions: ['document:read'],
      };

      mockLogin.mockResolvedValueOnce({
        access_token: 'token',
        refresh_token: 'refresh',
        token_type: 'bearer',
        expires_in: 1800,
      });

      mockGetUser.mockResolvedValueOnce(mockUserData);

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        const state = useAuthStore.getState();
        expect(state.user).toEqual(mockUserData);
        expect(state.isAuthenticated).toBe(true);
      });
    });
  });

  describe('Error Handling', () => {
    it('shows error toast on invalid credentials', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);

      mockLogin.mockRejectedValueOnce(new Error('Invalid username or password'));

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'wronguser');
      await user.type(passwordInput, 'wrongpass');
      await user.click(submitButton);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Invalid username or password');
      });
    });

    it('shows generic error on API failure', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);

      mockLogin.mockRejectedValueOnce(new Error());

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Invalid credentials');
      });
    });

    it('re-enables form after error', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);

      mockLogin.mockRejectedValueOnce(new Error('Login failed'));

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });
    });

    it('handles network errors gracefully', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);

      mockLogin.mockRejectedValueOnce(new Error('Network error'));

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalled();
      });
    });
  });

  describe('Form Interaction', () => {
    it('allows typing in username field', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i) as HTMLInputElement;
      await user.type(usernameInput, 'myusername');

      expect(usernameInput.value).toBe('myusername');
    });

    it('allows typing in password field', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);

      const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement;
      await user.type(passwordInput, 'mypassword');

      expect(passwordInput.value).toBe('mypassword');
    });

    it('password field has type password', () => {
      render(<LoginPage />);

      const passwordInput = screen.getByLabelText(/password/i);
      expect(passwordInput).toHaveAttribute('type', 'password');
    });

    it('can submit form by pressing Enter', async () => {
      const user = userEvent.setup();
      const mockLogin = vi.mocked(authApi.login);
      const mockGetUser = vi.mocked(authApi.getCurrentUser);

      mockLogin.mockResolvedValueOnce({
        access_token: 'token',
        refresh_token: 'refresh',
        token_type: 'bearer',
        expires_in: 1800,
      });

      mockGetUser.mockResolvedValueOnce({
        user_id: 'user-123',
        username: 'testuser',
        email: 'test@example.com',
        roles: ['viewer'],
        permissions: ['document:read'],
      });

      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);

      await user.type(usernameInput, 'testuser');
      await user.type(passwordInput, 'password123{Enter}');

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalled();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper form labels', () => {
      render(<LoginPage />);

      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });

    it('has proper autocomplete attributes', () => {
      render(<LoginPage />);

      const usernameInput = screen.getByLabelText(/username/i);
      const passwordInput = screen.getByLabelText(/password/i);

      expect(usernameInput).toHaveAttribute('autocomplete', 'username');
      expect(passwordInput).toHaveAttribute('autocomplete', 'current-password');
    });

    it('submit button has descriptive text', () => {
      render(<LoginPage />);

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      expect(submitButton).toBeInTheDocument();
    });
  });
});
