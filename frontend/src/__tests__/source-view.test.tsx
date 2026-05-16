import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AuthenticatedImage } from '@/components/document/AuthenticatedImage';

beforeEach(() => {
  global.fetch = vi.fn();
  // @ts-expect-error - jsdom doesn't ship URL.createObjectURL
  global.URL.createObjectURL = vi.fn(() => 'blob:mock');
  // @ts-expect-error - same
  global.URL.revokeObjectURL = vi.fn();
  localStorage.clear();
});

describe('AuthenticatedImage', () => {
  it('renders loading placeholder before fetch resolves', () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockReturnValue(
      new Promise(() => {}),
    );
    render(<AuthenticatedImage src="/api/page" alt="page 1" />);
    expect(screen.getByRole('img', { name: /loading/i })).toBeInTheDocument();
  });

  it('renders img after blob fetch succeeds', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob()),
    });
    render(<AuthenticatedImage src="/api/page" alt="page 1" />);
    const img = await screen.findByRole('img', { name: 'page 1' });
    expect(img).toHaveAttribute('src', 'blob:mock');
  });

  it('renders error state on HTTP failure', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 401,
    });
    render(<AuthenticatedImage src="/api/page" alt="page 1" />);
    expect(
      await screen.findByRole('img', { name: /failed to load/i }),
    ).toBeInTheDocument();
  });

  it('sends Authorization header with bearer token', async () => {
    localStorage.setItem('access_token', 'test-token');
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob()),
    });
    render(<AuthenticatedImage src="/api/page" alt="page 1" />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const init = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(init.headers.Authorization).toBe('Bearer test-token');
  });

  it('omits Authorization header when no token', async () => {
    localStorage.removeItem('access_token');
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob()),
    });
    render(<AuthenticatedImage src="/api/page" alt="page 1" />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const init = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1];
    expect(init.headers.Authorization).toBeUndefined();
  });
});
