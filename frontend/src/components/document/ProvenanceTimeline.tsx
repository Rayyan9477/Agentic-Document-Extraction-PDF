'use client';

/**
 * V3 Phase 8 — Provenance timeline.
 *
 * Vertical step-list showing the per-stage history of a single
 * field: Pass1 → Pass2 → Reconciler → Critic → Validator. Each step
 * shows stage name, agent label, confidence delta from the previous
 * step, and (when available) timestamp.
 *
 * Reads from FieldProvenance — falls back to `extraction_path`
 * when the richer `stages` array isn't populated (the V3 backend
 * may emit either shape depending on the release).
 */

import { CheckCircle2, ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui';
import type { FieldProvenance, ProvenanceStage } from '@/lib/api/provenance';

interface ProvenanceTimelineProps {
  field: FieldProvenance;
}

function stagesFromField(field: FieldProvenance): ProvenanceStage[] {
  if (field.stages && field.stages.length > 0) return field.stages;
  // Fall back to extraction_path with no per-step metadata. We
  // synthesise the final confidence onto the last stage so the UI
  // still shows *something* useful.
  return field.extraction_path.map((stage, i, all) => ({
    stage,
    agent: field.agent_signatures[i],
    confidence: i === all.length - 1 ? field.confidence : undefined,
  }));
}

function deltaBadge(prev: number | undefined, curr: number | undefined) {
  if (prev === undefined || curr === undefined) return null;
  const d = curr - prev;
  if (Math.abs(d) < 0.005) return null;
  const sign = d > 0 ? '+' : '';
  const variant = d > 0 ? 'success' : 'warning';
  return (
    <Badge size="sm" variant={variant as 'success' | 'warning'}>
      Δ {sign}
      {d.toFixed(2)}
    </Badge>
  );
}

export function ProvenanceTimeline({ field }: ProvenanceTimelineProps) {
  const stages = stagesFromField(field);

  if (stages.length === 0) {
    return (
      <p className="text-small text-text-muted">
        No provenance lineage recorded for this field.
      </p>
    );
  }

  return (
    <ol
      aria-label={`Provenance timeline for ${field.field_name}`}
      className="relative space-y-3"
    >
      {/* Spine */}
      <div
        aria-hidden="true"
        className="absolute left-[7px] top-1 bottom-1 w-px bg-border-default"
      />
      {stages.map((stage, i) => {
        const prevConf = i > 0 ? stages[i - 1].confidence : undefined;
        return (
          <li key={`${stage.stage}-${i}`} className="relative flex gap-3 pl-6">
            <CheckCircle2
              aria-hidden="true"
              className="absolute left-0 top-0.5 w-4 h-4 text-accent-brand bg-canvas rounded-full"
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-small text-text-primary">
                  {stage.stage}
                </span>
                {stage.agent && (
                  <>
                    <ArrowRight
                      aria-hidden="true"
                      className="w-3 h-3 text-text-muted"
                    />
                    <span className="text-small text-text-secondary">
                      {stage.agent}
                    </span>
                  </>
                )}
                {stage.confidence !== undefined && (
                  <Badge size="sm" variant="default">
                    {(stage.confidence * 100).toFixed(0)}%
                  </Badge>
                )}
                {deltaBadge(prevConf, stage.confidence)}
              </div>
              {stage.reason && (
                <p className="mt-0.5 text-small text-text-muted">
                  {stage.reason}
                </p>
              )}
              {stage.timestamp && (
                <p className="mt-0.5 text-small text-text-muted font-mono">
                  {stage.timestamp}
                </p>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
