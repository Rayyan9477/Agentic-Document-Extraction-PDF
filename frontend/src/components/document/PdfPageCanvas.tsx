'use client';

/**
 * V3 Phase 8 — PDF page canvas (PNG mode, the default renderer).
 *
 * Loads the rendered page image from the existing Phase 4 endpoint
 * `GET /api/v1/documents/{id}/pages/{n}` and stacks a BboxOverlay
 * on top. Cheap (~0KB extra bundle), fast (backend already
 * rasterised), and works without `react-pdf` / `pdfjs-dist`.
 *
 * For the high-fidelity opt-in (PDF text layer, in-PDF search),
 * see `PdfPageCanvasNative.tsx` which is loaded via
 * `next/dynamic({ ssr: false })`.
 */

import { useEffect, useState } from 'react';
import { Skeleton } from '@/components/ui';
import { pageImageUrl } from '@/lib/api/provenance';
import type { FieldProvenance } from '@/lib/api/provenance';
import { BboxOverlay, bboxesForPage } from './BboxOverlay';
import { AuthenticatedImage } from './AuthenticatedImage';

interface PdfPageCanvasProps {
  processingId: string;
  pageNumber: number;
  fields: Record<string, FieldProvenance>;
  activeFieldName: string | null;
  onSelectField: (fieldName: string) => void;
}

export function PdfPageCanvas({
  processingId,
  pageNumber,
  fields,
  activeFieldName,
  onSelectField,
}: PdfPageCanvasProps) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Reset state when page changes.
  useEffect(() => {
    setNaturalSize(null);
    setError(null);
  }, [pageNumber, processingId]);

  const url = pageImageUrl(processingId, pageNumber);

  const items = bboxesForPage(fields, pageNumber);

  // Display dimensions = scaled natural with a max-width cap.
  // Aspect ratio preserved.
  const maxWidth = 720;
  let displayW = naturalSize?.w ?? maxWidth;
  let displayH = naturalSize?.h ?? maxWidth * 1.4;
  if (naturalSize) {
    const scale = Math.min(1, maxWidth / naturalSize.w);
    displayW = naturalSize.w * scale;
    displayH = naturalSize.h * scale;
  }

  return (
    <div
      className="relative inline-block rounded-lg overflow-hidden bg-canvas border border-default shadow-elev-2"
      style={{ width: displayW, height: displayH }}
    >
      {!naturalSize && !error && (
        <Skeleton className="absolute inset-0" />
      )}
      <AuthenticatedImage
        src={url}
        alt={`Page ${pageNumber} of document`}
        onLoad={(e) => {
          const img = e.currentTarget;
          setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        }}
        onError={() => setError('Failed to load page image')}
        className="block w-full h-full object-contain"
      />
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-canvas text-accent-danger text-body">
          {error}
        </div>
      )}
      {naturalSize && (
        <BboxOverlay
          items={items}
          activeFieldName={activeFieldName}
          onSelect={onSelectField}
          width={displayW}
          height={displayH}
        />
      )}
    </div>
  );
}
