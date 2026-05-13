'use client';

/**
 * V3 Phase 8 — Source View tab.
 *
 * The headline frontend deliverable for V3. Shows the rendered
 * document page side-by-side with the extracted-field list,
 * with two-way bbox highlighting and a per-field provenance
 * timeline.
 *
 * Layout: 60/40 split.
 *   Left  — page canvas + page navigator + render-mode switch
 *   Right — field list + active-field provenance timeline drawer
 *
 * The canvas renderer is dual-mode:
 *   * 'png' (default) — lightweight PNG mode, ~0KB extra bundle
 *   * 'pdf' (opt-in)  — react-pdf with text layer + search
 *
 * The mode preference persists in localStorage.sourceViewRenderMode.
 */

import { useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { useQuery } from '@tanstack/react-query';
import {
  ChevronLeft,
  ChevronRight,
  FileImage,
  FileText,
  Info,
} from 'lucide-react';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  EmptyState,
  Skeleton,
} from '@/components/ui';
import {
  ApiError,
  fetchProvenance,
  isProvenanceEmpty,
  type FieldProvenance,
} from '@/lib/api/provenance';
import { BRANDING } from '@/lib/branding';
import { PdfPageCanvas } from './PdfPageCanvas';
import { ProvenanceTimeline } from './ProvenanceTimeline';
import {
  ProvenanceProvider,
  useProvenance,
} from './ProvenanceContext';

/**
 * Native PDF renderer is dynamically imported so the ~150KB
 * react-pdf bundle is only paid when the user opts in.
 */
const PdfPageCanvasNative = dynamic(
  () => import('./PdfPageCanvasNative'),
  {
    ssr: false,
    loading: () => <Skeleton className="w-[720px] h-[1000px] rounded-lg" />,
  },
);

type RenderMode = 'png' | 'pdf';

const STORAGE_KEY = 'sourceViewRenderMode';

interface SourceViewTabProps {
  processingId: string;
}

export function SourceViewTab({ processingId }: SourceViewTabProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['provenance', processingId],
    queryFn: () => fetchProvenance(processingId),
    retry: (count, err) => {
      // Don't retry 404 — provenance is genuinely unavailable.
      if (err instanceof ApiError && err.status === 404) return false;
      return count < 2;
    },
  });

  if (isLoading) {
    return <Skeleton className="w-full h-[800px] rounded-lg" />;
  }

  if (error instanceof ApiError && error.status === 404) {
    return <SourceViewEmptyState />;
  }

  if (error || isProvenanceEmpty(data)) {
    return <SourceViewEmptyState />;
  }

  return (
    <ProvenanceProvider provenance={data!}>
      <SourceViewLayout processingId={processingId} />
    </ProvenanceProvider>
  );
}

function SourceViewEmptyState() {
  return (
    <EmptyState
      icon={<Info className="w-8 h-8" />}
      title="Source view unavailable"
      description={
        `Source view requires the V3 dual-VLM extraction engine ` +
        `with provenance enabled. Documents extracted under the ` +
        `legacy single-VLM mode don't carry per-field source links. ` +
        `See the documentation for migration steps.`
      }
      action={{
        label: 'Learn more',
        onClick: () => {
          if (typeof window !== 'undefined') {
            window.open(BRANDING.docsUrl, '_blank', 'noopener,noreferrer');
          }
        },
      }}
    />
  );
}

function SourceViewLayout({ processingId }: { processingId: string }) {
  const { provenance, currentPage, setCurrentPage } = useProvenance();
  const [renderMode, setRenderMode] = useState<RenderMode>('png');

  // Hydrate render mode from localStorage on first mount.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY) as RenderMode | null;
      if (stored === 'png' || stored === 'pdf') setRenderMode(stored);
    } catch {
      // ignore
    }
  }, []);

  const switchMode = (next: RenderMode) => {
    setRenderMode(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  };

  // Determine total pages from the provenance data: max bbox.page.
  const totalPages = useMemo(() => {
    if (!provenance) return 1;
    let max = 1;
    for (const p of Object.values(provenance.fields)) {
      if (p.page && p.page > max) max = p.page;
      if (p.bbox?.page && p.bbox.page > max) max = p.bbox.page;
    }
    return max;
  }, [provenance]);

  if (!provenance) return null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
      {/* Left pane (60%): canvas + page navigator + mode switch */}
      <div className="lg:col-span-3 space-y-3">
        <SourceViewToolbar
          renderMode={renderMode}
          onSwitchMode={switchMode}
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={setCurrentPage}
        />
        <div className="flex justify-center">
          <SourceViewCanvas
            processingId={processingId}
            pageNumber={currentPage}
            renderMode={renderMode}
          />
        </div>
      </div>

      {/* Right pane (40%): field list + active timeline */}
      <div className="lg:col-span-2">
        <SourceViewFieldList />
      </div>
    </div>
  );
}

function SourceViewToolbar({
  renderMode,
  onSwitchMode,
  currentPage,
  totalPages,
  onPageChange,
}: {
  renderMode: RenderMode;
  onSwitchMode: (m: RenderMode) => void;
  currentPage: number;
  totalPages: number;
  onPageChange: (p: number) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 flex-wrap">
      <div
        role="group"
        aria-label="Page navigation"
        className="flex items-center gap-2"
      >
        <Button
          variant="ghost"
          size="icon"
          aria-label="Previous page"
          disabled={currentPage <= 1}
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        >
          <ChevronLeft className="w-4 h-4" aria-hidden="true" />
        </Button>
        <span className="text-body text-text-secondary tabular-nums">
          Page {currentPage} of {totalPages}
        </span>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Next page"
          disabled={currentPage >= totalPages}
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        >
          <ChevronRight className="w-4 h-4" aria-hidden="true" />
        </Button>
      </div>

      <div
        role="group"
        aria-label="Render mode"
        className="inline-flex rounded-lg border border-default overflow-hidden text-small"
      >
        <button
          type="button"
          // eslint-disable-next-line jsx-a11y/aria-proptypes -- ternary returns valid "true"/"false" literals; linter can't statically resolve
          aria-pressed={renderMode === 'png' ? 'true' : 'false'}
          onClick={() => onSwitchMode('png')}
          className={
            renderMode === 'png'
              ? 'flex items-center gap-1.5 px-3 py-1.5 bg-accent-brand-soft text-accent-brand'
              : 'flex items-center gap-1.5 px-3 py-1.5 text-text-secondary hover:bg-canvas'
          }
        >
          <FileImage className="w-3.5 h-3.5" aria-hidden="true" />
          Image
        </button>
        <button
          type="button"
          // eslint-disable-next-line jsx-a11y/aria-proptypes -- ternary returns valid "true"/"false" literals; linter can't statically resolve
          aria-pressed={renderMode === 'pdf' ? 'true' : 'false'}
          onClick={() => onSwitchMode('pdf')}
          className={
            renderMode === 'pdf'
              ? 'flex items-center gap-1.5 px-3 py-1.5 bg-accent-brand-soft text-accent-brand border-l border-default'
              : 'flex items-center gap-1.5 px-3 py-1.5 text-text-secondary hover:bg-canvas border-l border-default'
          }
        >
          <FileText className="w-3.5 h-3.5" aria-hidden="true" />
          PDF
        </button>
      </div>
    </div>
  );
}

function SourceViewCanvas({
  processingId,
  pageNumber,
  renderMode,
}: {
  processingId: string;
  pageNumber: number;
  renderMode: RenderMode;
}) {
  const { provenance, activeFieldName, setActiveFieldName } = useProvenance();
  if (!provenance) return null;

  const fields = provenance.fields;

  if (renderMode === 'pdf') {
    return (
      <PdfPageCanvasNative
        processingId={processingId}
        pageNumber={pageNumber}
        fields={fields}
        activeFieldName={activeFieldName}
        onSelectField={setActiveFieldName}
      />
    );
  }

  return (
    <PdfPageCanvas
      processingId={processingId}
      pageNumber={pageNumber}
      fields={fields}
      activeFieldName={activeFieldName}
      onSelectField={setActiveFieldName}
    />
  );
}

function SourceViewFieldList() {
  const { provenance, activeFieldName, setActiveFieldName, activeField } =
    useProvenance();
  if (!provenance) return null;

  const fieldEntries = Object.entries(provenance.fields).sort(
    ([a], [b]) => a.localeCompare(b),
  );

  return (
    <div className="space-y-3">
      <Card variant="outlined" padding="none">
        <CardHeader className="p-4">
          <h2 className="text-h3 text-text-primary">
            Fields ({fieldEntries.length})
          </h2>
        </CardHeader>
        <CardContent>
          <ul className="divide-y divide-border-default">
            {fieldEntries.map(([name, field]) => (
              <FieldRow
                key={name}
                name={name}
                field={field}
                active={name === activeFieldName}
                onClick={() => setActiveFieldName(name)}
              />
            ))}
          </ul>
        </CardContent>
      </Card>

      {activeField && (
        <Card variant="elevated">
          <CardHeader>
            <h2 className="text-h3 text-text-primary">Provenance timeline</h2>
            <p className="text-small text-text-secondary font-mono mt-0.5">
              {activeField.field_name}
            </p>
          </CardHeader>
          <CardContent>
            <ProvenanceTimeline field={activeField} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function FieldRow({
  name,
  field,
  active,
  onClick,
}: {
  name: string;
  field: FieldProvenance;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        // eslint-disable-next-line jsx-a11y/aria-proptypes -- ternary returns valid "true"/"false" literals
        aria-pressed={active ? 'true' : 'false'}
        className={
          active
            ? 'w-full text-left px-4 py-3 bg-accent-brand-soft transition-colors duration-fast'
            : 'w-full text-left px-4 py-3 hover:bg-canvas transition-colors duration-fast'
        }
      >
        <div className="flex items-center justify-between gap-3">
          <span className="font-mono text-body text-text-primary truncate">
            {name}
          </span>
          <Badge
            size="sm"
            variant={field.confidence >= 0.9 ? 'success' : field.confidence >= 0.7 ? 'warning' : 'default'}
          >
            {(field.confidence * 100).toFixed(0)}%
          </Badge>
        </div>
        <div className="mt-1 flex items-center gap-2 text-small text-text-muted">
          <span>Page {field.page}</span>
          {field.vlm_model_id && (
            <>
              <span aria-hidden="true">·</span>
              <span className="truncate">{field.vlm_model_id}</span>
            </>
          )}
        </div>
      </button>
    </li>
  );
}
