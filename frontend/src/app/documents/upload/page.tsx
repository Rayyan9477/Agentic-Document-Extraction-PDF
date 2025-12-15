'use client';

import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { Upload, ArrowRight, FileText, CheckCircle } from 'lucide-react';
import { AppLayout } from '@/components/layout';
import { FileUploader, UploadOptions } from '@/components/documents';
import type { UploadFile } from '@/components/documents';
import { Card, CardHeader, CardContent, Button, StepProgress } from '@/components/ui';
import { documentsApi, schemaApi } from '@/lib/api';
import { generateId } from '@/lib/utils';
import type { ExportFormat, ProcessingPriority } from '@/types/api';

const UPLOAD_STEPS = [
  { label: 'Select Files', description: 'Choose PDF documents' },
  { label: 'Configure', description: 'Set processing options' },
  { label: 'Process', description: 'Upload and process' },
  { label: 'Complete', description: 'View results' },
];

export default function DocumentUploadPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [options, setOptions] = useState({
    schemaName: '',
    exportFormat: 'json' as ExportFormat,
    priority: 'normal' as ProcessingPriority,
    maskPhi: false,
    outputDir: './output',
  });

  // Fetch available schemas
  const { data: schemas = [] } = useQuery({
    queryKey: ['schemas'],
    queryFn: () => schemaApi.list(),
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: UploadFile) => {
      return documentsApi.upload(file.file, {
        schema_name: options.schemaName || undefined,
        export_format: options.exportFormat,
        priority: options.priority,
        mask_phi: options.maskPhi,
      });
    },
    onSuccess: (response, variables) => {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === variables.id
            ? { ...f, status: 'success', progress: 100, taskId: response.task_id }
            : f
        )
      );
      toast.success(`${variables.file.name} uploaded successfully`);
    },
    onError: (error: Error, variables) => {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === variables.id
            ? { ...f, status: 'error', error: error.message }
            : f
        )
      );
      toast.error(`Failed to upload ${variables.file.name}`);
    },
  });

  const handleFilesSelected = useCallback((newFiles: File[]) => {
    const uploadFiles: UploadFile[] = newFiles.map((file) => ({
      id: generateId(),
      file,
      status: 'pending',
      progress: 0,
    }));
    setFiles((prev) => [...prev, ...uploadFiles]);
  }, []);

  const handleFileRemove = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const handleUpload = async () => {
    setCurrentStep(2);

    // Track uploaded count for step progression
    const pendingFiles = files.filter((f) => f.status === 'pending');
    let successCount = files.filter((f) => f.status === 'success').length;
    const totalFiles = files.length;

    for (const file of pendingFiles) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id ? { ...f, status: 'uploading', progress: 30 } : f
        )
      );

      try {
        await uploadMutation.mutateAsync(file);
        successCount++;
      } catch {
        // Error is handled in onError callback
        // Don't count as success
      }
    }

    // Move to completion step only when ALL files have succeeded
    // NOTE: This check uses tracked counts since React state is async
    if (successCount === totalFiles && totalFiles > 0) {
      setCurrentStep(3);
    }
  };

  const canProceedToStep2 = files.length > 0 && files.some((f) => f.status === 'pending');
  const canStartUpload = currentStep >= 1 && files.some((f) => f.status === 'pending');
  const isUploading = files.some((f) => f.status === 'uploading');
  const allComplete = files.length > 0 && files.every((f) => f.status === 'success');

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-2xl font-bold text-surface-900">Upload Documents</h1>
          <p className="text-surface-500 mt-1">
            Upload PDF documents for AI-powered field extraction
          </p>
        </motion.div>

        {/* Progress Steps */}
        <Card variant="elevated" padding="md">
          <StepProgress steps={UPLOAD_STEPS} currentStep={currentStep} />
        </Card>

        {/* Step Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* File Upload Area */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card variant="elevated" padding="md">
                <CardHeader
                  title="Select Documents"
                  description="Upload PDF files for processing"
                  action={
                    files.length > 0 && (
                      <span className="text-sm text-surface-500">
                        {files.filter((f) => f.status === 'pending').length} pending
                      </span>
                    )
                  }
                />
                <CardContent className="mt-4">
                  <FileUploader
                    onFilesSelected={handleFilesSelected}
                    onFileRemove={handleFileRemove}
                    files={files}
                    maxFiles={10}
                    disabled={isUploading || allComplete}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Options Panel */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <UploadOptions
                options={options}
                onChange={setOptions}
                schemas={schemas}
              />
            </motion.div>
          </div>
        </div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex items-center justify-between"
        >
          <Button
            variant="secondary"
            onClick={() => router.back()}
            disabled={isUploading}
          >
            Cancel
          </Button>

          <div className="flex items-center gap-3">
            {currentStep === 0 && (
              <Button
                variant="primary"
                onClick={() => setCurrentStep(1)}
                disabled={!canProceedToStep2}
                rightIcon={<ArrowRight className="w-4 h-4" />}
              >
                Configure Options
              </Button>
            )}

            {currentStep === 1 && (
              <Button
                variant="primary"
                onClick={handleUpload}
                disabled={!canStartUpload}
                loading={isUploading}
                leftIcon={<Upload className="w-4 h-4" />}
              >
                Start Processing
              </Button>
            )}

            {currentStep === 2 && !allComplete && (
              <Button variant="primary" disabled loading>
                Processing...
              </Button>
            )}

            {(currentStep === 3 || allComplete) && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-success-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm font-medium">All files processed!</span>
                </div>
                <Button
                  variant="primary"
                  onClick={() => router.push('/tasks')}
                  rightIcon={<ArrowRight className="w-4 h-4" />}
                >
                  View Tasks
                </Button>
              </div>
            )}
          </div>
        </motion.div>

        {/* Info Box */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-info-50 border border-info-200 rounded-xl p-4"
        >
          <div className="flex gap-3">
            <FileText className="w-5 h-5 text-info-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-info-800">
                Processing Information
              </p>
              <ul className="mt-2 text-sm text-info-700 space-y-1">
                <li>• Documents are processed asynchronously in the background</li>
                <li>• You can track progress in the Task Queue</li>
                <li>• Results will be available once processing completes</li>
                <li>• HIPAA-compliant processing with optional PHI masking</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </AppLayout>
  );
}
