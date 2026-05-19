'use client';

import React from 'react';
import {
  Shield,
  FileOutput,
  Zap,
  Users,
  User,
  ScanLine,
  PenLine,
  Image as ImageIcon,
  Printer,
  Table as TableIcon,
  ClipboardList,
} from 'lucide-react';
import { Card, CardHeader, CardContent, Select, Input } from '@/components/ui';
import type { SelectOption } from '@/components/ui';
import { MODE_LABELS, type ModeKey } from '@/lib/branding';
import type { ExportFormat, ExtractionMode, ProcessingPriority } from '@/types/api';

// WS-3 modality canonical names — must match the backend's
// src/agents/modality.py constants. Empty modalityOverride array means
// "auto-detect" on the backend.
export type Modality =
  | 'printed'
  | 'handwritten'
  | 'visual'
  | 'fax'
  | 'table'
  | 'form';

export interface UploadOptions {
  schemaName: string;
  exportFormat: ExportFormat;
  priority: ProcessingPriority;
  extractionMode: ExtractionMode;
  maskPhi: boolean;
  outputDir: string;
  // WS-3: empty = auto-detect; else override the analyzer's choice.
  modalityOverride: Modality[];
  // WS-6: null = use server default; true/false = per-request override.
  phiMode: boolean | null;
  // Phase K: top-level mode chip. ``auto`` lets the analyzer detect the
  // profile; ``healthcare`` and ``general`` force a specific profile.
  mode: ModeKey;
}

interface UploadOptionsProps {
  options: UploadOptions;
  onChange: (options: UploadOptions) => void;
  schemas?: Array<{ name: string; description: string }>;
  loading?: boolean;
}

const UploadOptionsComponent: React.FC<UploadOptionsProps> = ({
  options,
  onChange,
  schemas,
}) => {
  const schemaOptions: SelectOption[] = [
    { value: '', label: 'Auto-detect schema' },
    ...(Array.isArray(schemas) ? schemas : []).map((s) => ({
      value: s.name,
      label: s.name,
    })),
  ];

  const formatOptions: SelectOption[] = [
    { value: 'json', label: 'JSON', icon: <FileOutput className="w-4 h-4" /> },
    { value: 'excel', label: 'Excel (.xlsx)', icon: <FileOutput className="w-4 h-4" /> },
    { value: 'markdown', label: 'Markdown', icon: <FileOutput className="w-4 h-4" /> },
    { value: 'both', label: 'JSON + Excel', icon: <FileOutput className="w-4 h-4" /> },
    { value: 'all', label: 'All Formats', icon: <FileOutput className="w-4 h-4" /> },
  ];

  const priorityOptions: SelectOption[] = [
    { value: 'low', label: 'Low Priority', icon: <Zap className="w-4 h-4 text-surface-400" /> },
    { value: 'normal', label: 'Normal Priority', icon: <Zap className="w-4 h-4 text-primary-500" /> },
    { value: 'high', label: 'High Priority', icon: <Zap className="w-4 h-4 text-warning-500" /> },
  ];

  const extractionModeOptions: SelectOption[] = [
    { value: 'multi', label: 'Multi-Record', icon: <Users className="w-4 h-4 text-primary-600" /> },
    { value: 'single', label: 'Single-Record', icon: <User className="w-4 h-4 text-surface-500" /> },
    { value: 'auto', label: 'Auto-Detect', icon: <Users className="w-4 h-4 text-success-500" /> },
  ];

  const updateOption = <K extends keyof UploadOptions>(
    key: K,
    value: UploadOptions[K]
  ) => {
    onChange({ ...options, [key]: value });
  };

  // WS-3: each chip toggles one modality on / off in the override list.
  // Empty list ⇒ auto-detect (the analyzer chooses).
  const MODALITY_CHIPS: Array<{
    value: Modality;
    label: string;
    icon: React.ReactNode;
    hint: string;
  }> = [
    { value: 'printed', label: 'Printed', icon: <Printer className="w-3.5 h-3.5" />,
      hint: 'Standard printed text (default).' },
    { value: 'handwritten', label: 'Handwritten', icon: <PenLine className="w-3.5 h-3.5" />,
      hint: 'Treat values as low-confidence; null on doubt.' },
    { value: 'visual', label: 'Visual', icon: <ImageIcon className="w-3.5 h-3.5" />,
      hint: 'Radiology / ultrasound / photo: do not invent fields.' },
    { value: 'fax', label: 'Fax', icon: <ScanLine className="w-3.5 h-3.5" />,
      hint: 'Otsu binarization + morphology; verify each digit.' },
    { value: 'table', label: 'Table', icon: <TableIcon className="w-3.5 h-3.5" />,
      hint: 'Respect column boundaries; record empty cells as null.' },
    { value: 'form', label: 'Form', icon: <ClipboardList className="w-3.5 h-3.5" />,
      hint: 'Extract by labelled boxes / numbered sections.' },
  ];

  const toggleModality = (m: Modality) => {
    const current = options.modalityOverride ?? [];
    const next = current.includes(m)
      ? current.filter((x) => x !== m)
      : [...current, m];
    updateOption('modalityOverride', next);
  };

  const phiModeOptions: SelectOption[] = [
    { value: 'auto', label: 'Server default (auto)' },
    { value: 'on', label: 'Force PHI redaction' },
    { value: 'off', label: 'Bypass PHI redaction' },
  ];
  const phiModeValue =
    options.phiMode === null || options.phiMode === undefined
      ? 'auto'
      : options.phiMode
      ? 'on'
      : 'off';

  // Phase K — top-level mode chips. Defaults to ``auto`` so existing
  // behaviour is preserved when the user doesn't touch the row.
  const activeMode: ModeKey = options.mode ?? 'auto';

  return (
    <Card variant="elevated" padding="md">
      <CardHeader
        title="Processing Options"
        description="Configure how your documents will be processed"
      />
      <CardContent className="mt-4 space-y-4">
        {/* Phase K: Top-level mode selector (Healthcare / General / Auto). */}
        <div className="space-y-2">
          <div className="flex items-baseline justify-between">
            <label className="block text-sm font-medium text-surface-900">
              Document Mode
            </label>
            <span className="text-xs text-surface-500">
              {MODE_LABELS[activeMode].label}
            </span>
          </div>
          <p className="text-xs text-surface-500">
            {MODE_LABELS[activeMode].description}
          </p>
          <div
            role="radiogroup"
            aria-label="Document mode"
            className="flex flex-wrap gap-2"
          >
            {(Object.keys(MODE_LABELS) as ModeKey[]).map((modeKey) => {
              const cfg = MODE_LABELS[modeKey];
              const active = activeMode === modeKey;
              return (
                <button
                  type="button"
                  key={modeKey}
                  role="radio"
                  aria-checked={active ? 'true' : 'false'}
                  title={cfg.description}
                  onClick={() => updateOption('mode', modeKey)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
                    active
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-surface-50 text-surface-700 border-surface-200 hover:bg-surface-100'
                  }`}
                >
                  <span aria-hidden="true">{cfg.icon}</span>
                  {cfg.short}
                </button>
              );
            })}
          </div>
        </div>

        {/* Extraction Mode */}
        <Select
          label="Extraction Mode"
          options={extractionModeOptions}
          value={options.extractionMode}
          onChange={(value) => updateOption('extractionMode', value as ExtractionMode)}
          hint="Multi-Record separates distinct entities (patients, invoices) per row"
        />

        {/* Schema Selection */}
        <Select
          label="Document Schema"
          options={schemaOptions}
          value={options.schemaName}
          onChange={(value) => updateOption('schemaName', value)}
          placeholder="Select schema or auto-detect"
          hint="Choose a predefined schema or let the system auto-detect"
        />

        {/* Export Format */}
        <Select
          label="Export Format"
          options={formatOptions}
          value={options.exportFormat}
          onChange={(value) => updateOption('exportFormat', value as ExportFormat)}
        />

        {/* Priority */}
        <Select
          label="Processing Priority"
          options={priorityOptions}
          value={options.priority}
          onChange={(value) => updateOption('priority', value as ProcessingPriority)}
        />

        {/* Output Directory */}
        <Input
          label="Output Directory"
          value={options.outputDir}
          onChange={(e) => updateOption('outputDir', e.target.value)}
          placeholder="./output"
          hint="Where to save the exported files"
          leftIcon={<FileOutput className="w-4 h-4" />}
        />

        {/* WS-3: modality override chips (multi-select). Empty = auto-detect. */}
        <div className="space-y-2">
          <div className="flex items-baseline justify-between">
            <label className="block text-sm font-medium text-surface-900">
              Specialised Mode
            </label>
            <span className="text-xs text-surface-500">
              {(options.modalityOverride?.length ?? 0) === 0
                ? 'Auto-detect (recommended)'
                : `${options.modalityOverride.length} selected`}
            </span>
          </div>
          <p className="text-xs text-surface-500">
            Override how the document is preprocessed and prompted. Pick none to let the analyzer choose.
          </p>
          <div className="flex flex-wrap gap-2">
            {MODALITY_CHIPS.map((chip) => {
              const active = (options.modalityOverride ?? []).includes(chip.value);
              return (
                <button
                  type="button"
                  key={chip.value}
                  title={chip.hint}
                  onClick={() => toggleModality(chip.value)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all border ${
                    active
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-surface-50 text-surface-700 border-surface-200 hover:bg-surface-100'
                  }`}
                >
                  {chip.icon}
                  {chip.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* PHI Masking Toggle (export-time defence-in-depth) */}
        <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
              <Shield className="w-5 h-5 text-success-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-surface-900">
                Mask PHI in exports
              </p>
              <p className="text-xs text-surface-500">
                Regex-based redaction of names / SSN / email / phone in JSON / Excel / Markdown.
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={options.maskPhi}
              onChange={(e) => updateOption('maskPhi', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-surface-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        {/* WS-6: PHI mode tri-state (server default / force on / force off) */}
        <Select
          label="PHI Mode (extraction-time)"
          options={phiModeOptions}
          value={phiModeValue}
          onChange={(value) => {
            const next: boolean | null =
              value === 'auto' ? null : value === 'on' ? true : false;
            updateOption('phiMode', next);
          }}
          hint="Routes extracted strings through the openai/privacy-filter token classifier (or regex fallback) before storage."
        />
      </CardContent>
    </Card>
  );
};

export default UploadOptionsComponent;
