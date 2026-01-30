'use client';

import React from 'react';
import { Shield, FileOutput, Zap, Users, User } from 'lucide-react';
import { Card, CardHeader, CardContent, Select, Input } from '@/components/ui';
import type { SelectOption } from '@/components/ui';
import type { ExportFormat, ExtractionMode, ProcessingPriority } from '@/types/api';

interface UploadOptions {
  schemaName: string;
  exportFormat: ExportFormat;
  priority: ProcessingPriority;
  extractionMode: ExtractionMode;
  maskPhi: boolean;
  outputDir: string;
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

  return (
    <Card variant="elevated" padding="md">
      <CardHeader
        title="Processing Options"
        description="Configure how your documents will be processed"
      />
      <CardContent className="mt-4 space-y-4">
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

        {/* PHI Masking Toggle */}
        <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
              <Shield className="w-5 h-5 text-success-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-surface-900">
                HIPAA PHI Masking
              </p>
              <p className="text-xs text-surface-500">
                Mask sensitive health information in exports
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
      </CardContent>
    </Card>
  );
};

export default UploadOptionsComponent;
