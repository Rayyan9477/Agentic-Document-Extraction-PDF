'use client';

/**
 * V3 Phase 8 — ProvenanceContext.
 *
 * Shared state for the Source View tab: which field is active,
 * what page that field lives on, and the loaded provenance map.
 *
 * Bidirectional sync:
 *   * Click a field in the right pane → setActiveField(name)
 *     → canvas auto-pages and highlights the bbox.
 *   * Click a bbox on the canvas → setActiveField(bbox.field_name)
 *     → field expands in the right pane.
 */

import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type {
  DocumentProvenanceResponse,
  FieldProvenance,
} from '@/lib/api/provenance';

interface ProvenanceContextValue {
  activeFieldName: string | null;
  setActiveFieldName: (name: string | null) => void;
  /** Currently active field's provenance object, if any. */
  activeField: FieldProvenance | null;
  /** Full provenance payload from the API. */
  provenance: DocumentProvenanceResponse | null;
  /** Current page in the canvas (1-indexed). */
  currentPage: number;
  setCurrentPage: (page: number) => void;
}

const ProvenanceContext = createContext<ProvenanceContextValue | null>(null);

export function ProvenanceProvider({
  children,
  provenance,
  defaultPage = 1,
}: {
  children: React.ReactNode;
  provenance: DocumentProvenanceResponse | null;
  defaultPage?: number;
}) {
  const [activeFieldName, setActiveFieldNameState] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(defaultPage);

  const setActiveFieldName = useCallback(
    (name: string | null) => {
      setActiveFieldNameState(name);
      if (name && provenance?.fields[name]?.page) {
        // Auto-page to the field's source page.
        setCurrentPage(provenance.fields[name].page);
      }
    },
    [provenance],
  );

  const activeField = useMemo<FieldProvenance | null>(() => {
    if (!activeFieldName || !provenance) return null;
    return provenance.fields[activeFieldName] ?? null;
  }, [activeFieldName, provenance]);

  const value = useMemo<ProvenanceContextValue>(
    () => ({
      activeFieldName,
      setActiveFieldName,
      activeField,
      provenance,
      currentPage,
      setCurrentPage,
    }),
    [activeFieldName, setActiveFieldName, activeField, provenance, currentPage],
  );

  return (
    <ProvenanceContext.Provider value={value}>
      {children}
    </ProvenanceContext.Provider>
  );
}

export function useProvenance(): ProvenanceContextValue {
  const ctx = useContext(ProvenanceContext);
  if (ctx === null) {
    throw new Error(
      'useProvenance must be used inside <ProvenanceProvider>',
    );
  }
  return ctx;
}
