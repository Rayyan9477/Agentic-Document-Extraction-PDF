'use client';

/**
 * V3 Phase 8 — Bbox overlay.
 *
 * Absolutely-positioned SVG layer that draws one `<rect>` per
 * field bbox over a rendered PDF page (PNG canvas or react-pdf).
 * The active bbox gets accent-brand stroke + 0.15 opacity fill;
 * inactive bboxes get text-muted stroke and transparent fill.
 *
 * Clicking a rect sets the active field via the parent's callback.
 * Keyboard-focusable for screen-reader / keyboard parity.
 */

import { useMemo } from 'react';
import type { FieldProvenance, NormalisedBbox } from '@/lib/api/provenance';

export interface BboxItem {
  fieldName: string;
  bbox: NormalisedBbox;
}

interface BboxOverlayProps {
  /** All bboxes on the *current* page only. */
  items: BboxItem[];
  /** Currently active field name, or null. */
  activeFieldName: string | null;
  /** Click handler from parent (sets active field). */
  onSelect: (fieldName: string) => void;
  /** Canvas display size in pixels (the overlay sizes to match). */
  width: number;
  height: number;
}

export function BboxOverlay({
  items,
  activeFieldName,
  onSelect,
  width,
  height,
}: BboxOverlayProps) {
  // Convert normalised coords to display pixels.
  const rects = useMemo(
    () =>
      items.map(({ fieldName, bbox }) => {
        const x = bbox.x * width;
        const y = bbox.y * height;
        const w = bbox.width * width;
        const h = bbox.height * height;
        const active = fieldName === activeFieldName;
        return { fieldName, x, y, w, h, active };
      }),
    [items, activeFieldName, width, height],
  );

  return (
    <svg
      width={width}
      height={height}
      className="absolute inset-0 pointer-events-none"
      role="presentation"
    >
      {rects.map(({ fieldName, x, y, w, h, active }) => (
        <g key={fieldName}>
          <rect
            x={x}
            y={y}
            width={w}
            height={h}
            fill={active ? 'rgb(var(--accent-brand-rgb) / 0.15)' : 'transparent'}
            stroke={
              active
                ? 'rgb(var(--accent-brand-rgb))'
                : 'rgb(var(--text-muted-rgb) / 0.6)'
            }
            strokeWidth={active ? 2 : 1}
            rx={2}
            className="pointer-events-auto cursor-pointer transition-all duration-fast"
            onClick={() => onSelect(fieldName)}
            tabIndex={0}
            role="button"
            aria-label={`Select source region for ${fieldName}`}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect(fieldName);
              }
            }}
          />
          {active && (
            <text
              x={x + 4}
              y={y - 4}
              fontSize={11}
              fill="rgb(var(--accent-brand-rgb))"
              className="pointer-events-none select-none font-mono"
            >
              {fieldName}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

/** Helper: filter the full field map down to bboxes on a given page. */
export function bboxesForPage(
  fields: Record<string, FieldProvenance>,
  page: number,
): BboxItem[] {
  const items: BboxItem[] = [];
  for (const [fieldName, p] of Object.entries(fields)) {
    if (!p.bbox) continue;
    if (p.bbox.page !== page) continue;
    items.push({ fieldName, bbox: p.bbox });
  }
  return items;
}
