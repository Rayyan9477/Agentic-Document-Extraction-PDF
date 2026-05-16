'use client';

/**
 * V3 Phase 8.5 — Authenticated <img> replacement.
 *
 * <img src=...> tags cannot send Authorization: Bearer headers, so they
 * 401 on protected page-image endpoints (GET /api/v1/documents/{id}/pages/{n}).
 * This component fetches the bytes with the bearer token, builds a blob
 * URL, and renders <img src={blobUrl}>. Releases the blob URL on unmount
 * or src change.
 */

import { useEffect, useRef, useState } from 'react';
import { getAccessToken } from '@/lib/api';

interface AuthenticatedImageProps {
  src: string;
  alt: string;
  className?: string;
  onLoad?: (event: { currentTarget: HTMLImageElement }) => void;
  onError?: () => void;
}

export function AuthenticatedImage({
  src,
  alt,
  className,
  onLoad,
  onError,
}: AuthenticatedImageProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const currentUrlRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(false);
    setBlobUrl(null);

    const token = getAccessToken();
    const headers: Record<string, string> = { Accept: 'image/*' };
    if (token) headers.Authorization = `Bearer ${token}`;

    fetch(src, { headers })
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        if (currentUrlRef.current) URL.revokeObjectURL(currentUrlRef.current);
        currentUrlRef.current = url;
        setBlobUrl(url);
      })
      .catch(() => {
        if (cancelled) return;
        setError(true);
        onError?.();
      });

    return () => {
      cancelled = true;
    };
  }, [src, onError]);

  useEffect(() => {
    return () => {
      if (currentUrlRef.current) {
        URL.revokeObjectURL(currentUrlRef.current);
        currentUrlRef.current = null;
      }
    };
  }, []);

  if (error) {
    return (
      <div
        className={className}
        role="img"
        aria-label={`Failed to load: ${alt}`}
      />
    );
  }

  if (!blobUrl) {
    return <div className={className} role="img" aria-label={`Loading: ${alt}`} />;
  }

  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={blobUrl}
      alt={alt}
      className={className}
      onLoad={(e) => onLoad?.({ currentTarget: e.currentTarget })}
    />
  );
}
