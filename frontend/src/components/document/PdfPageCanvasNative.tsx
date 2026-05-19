'use client';

/**
 * Veridoc — PDF page canvas (NATIVE PDF mode, opt-in).
 *
 * Originally this component lazy-loaded ``react-pdf`` for a text-layer
 * + Ctrl-F search experience. The current build ships without ``react-pdf``
 * in dependencies because:
 *
 * 1. The default Source View renderer (``PdfPageCanvas`` PNG mode) is
 *    sufficient for the demo flow and for the air-gapped on-prem
 *    deployment story.
 * 2. ``react-pdf`` + ``pdfjs-dist`` pull in a ~150 KB worker that
 *    Next.js's Terser cannot minify cleanly, and the dependency
 *    pulls in build-time pain for an opt-in feature.
 *
 * This stub keeps the file present so the existing dynamic import in
 * ``SourceViewTab.tsx`` continues to compile; at render time it shows
 * an explanatory message and steers the user back to PNG mode. To
 * activate the native renderer, an operator can:
 *
 * * ``npm install react-pdf pdfjs-dist`` (re-add the optional
 *   dependency)
 * * Replace this file's body with the original react-pdf
 *   implementation (kept in git history)
 */

import { Skeleton as _Skeleton } from '@/components/ui';
import type { FieldProvenance } from '@/lib/api/provenance';

interface PdfPageCanvasNativeProps {
  processingId: string;
  pageNumber: number;
  fields: Record<string, FieldProvenance>;
  activeFieldName: string | null;
  onSelectField: (fieldName: string) => void;
}

// Re-export to keep the skeleton type visible to downstream tooling.
export const Skeleton = _Skeleton;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function PdfPageCanvasNative(_props: PdfPageCanvasNativeProps) {
  return (
    <div
      role="status"
      className="rounded-lg border border-default bg-canvas p-6 text-body text-text-secondary max-w-[720px]"
    >
      <p className="font-semibold text-text-primary mb-2">
        PDF mode is unavailable in this build
      </p>
      <p>
        The native PDF renderer (with text-layer + Ctrl-F search) is an
        opt-in feature that requires the <code>react-pdf</code>{' '}
        dependency. The default <strong>Image</strong> mode renders the
        same bounding-box overlay and click-to-source workflow without
        the extra bundle weight.
      </p>
      <p className="mt-2 text-text-muted">
        To enable PDF mode, install <code>react-pdf</code> and{' '}
        <code>pdfjs-dist</code>, then replace{' '}
        <code>PdfPageCanvasNative.tsx</code> with the
        git-history version of the native renderer.
      </p>
    </div>
  );
}
