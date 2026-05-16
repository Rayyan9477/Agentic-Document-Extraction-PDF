'use client';

/**
 * V3 Phase 8 — PDF page canvas (NATIVE PDF mode, opt-in).
 *
 * High-fidelity renderer using `react-pdf` for PDF text layer +
 * Ctrl+F search + native selection. Loaded via `next/dynamic({
 * ssr: false })` so the ~150KB gzip cost is paid only when the user
 * opts in via the render-mode switch.
 *
 * If `react-pdf` is not installed (or the backend doesn't expose
 * `/pdf`), the component renders a fallback message telling the
 * user to switch back to PNG mode.
 */

import { useEffect, useRef, useState } from 'react';
import { Skeleton } from '@/components/ui';
import { pdfDownloadUrl } from '@/lib/api/provenance';
import type { FieldProvenance } from '@/lib/api/provenance';
import { BboxOverlay, bboxesForPage } from './BboxOverlay';

interface PdfPageCanvasNativeProps {
  processingId: string;
  pageNumber: number;
  fields: Record<string, FieldProvenance>;
  activeFieldName: string | null;
  onSelectField: (fieldName: string) => void;
}

// Lazy + optional import. ``react-pdf`` is a soft dependency —
// installations that don't include the [pdf-viewer] extra fall
// back to a graceful error message inside the canvas. We use
// ``any`` for the module shape so the project type-checks even
// when ``react-pdf`` is not installed.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ReactPdfModule = any;

export default function PdfPageCanvasNative({
  processingId,
  pageNumber,
  fields,
  activeFieldName,
  onSelectField,
}: PdfPageCanvasNativeProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [rpdf, setRpdf] = useState<ReactPdfModule | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // @ts-expect-error -- react-pdf is an opt-in soft dependency;
        // installations without the [pdf-viewer] extra fall through
        // to the importError path below.
        const mod = await import('react-pdf');
        // Worker pinned to a Next.js public asset; air-gap safe.
        mod.pdfjs.GlobalWorkerOptions.workerSrc = new URL(
          'pdfjs-dist/build/pdf.worker.min.mjs',
          import.meta.url,
        ).toString();
        if (!cancelled) setRpdf(mod);
      } catch {
        if (!cancelled) {
          setImportError(
            'PDF mode is unavailable in this build. Install the ' +
              '[pdf-viewer] extra or switch to Image mode.',
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (importError) {
    return (
      <div className="rounded-lg border border-default bg-canvas p-6 text-body text-text-secondary">
        {importError}
      </div>
    );
  }

  if (!rpdf) {
    return <Skeleton className="w-[720px] h-[1000px] rounded-lg" />;
  }

  const { Document, Page } = rpdf;
  const items = bboxesForPage(fields, pageNumber);

  return (
    <div
      ref={containerRef}
      className="relative inline-block rounded-lg overflow-hidden bg-canvas border border-default shadow-elev-2"
    >
      <Document
        file={pdfDownloadUrl(processingId)}
        loading={<Skeleton className="w-[720px] h-[1000px]" />}
        error={
          <div className="p-6 text-body text-accent-danger">
            Failed to load PDF. Switch back to Image mode.
          </div>
        }
      >
        <Page
          pageNumber={pageNumber}
          width={720}
          onLoadSuccess={(p: { width: number; height: number }) =>
            setSize({ w: p.width, h: p.height })
          }
          renderTextLayer
          renderAnnotationLayer={false}
        />
      </Document>
      {size && (
        <BboxOverlay
          items={items}
          activeFieldName={activeFieldName}
          onSelect={onSelectField}
          width={size.w}
          height={size.h}
        />
      )}
    </div>
  );
}
