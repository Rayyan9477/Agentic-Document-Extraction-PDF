'use client';

import React, { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft,
  FileText,
  Download,
  RefreshCw,
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Copy,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Loader2,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import {
  Card,
  Button,
  Badge,
  Modal,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Tooltip,
  Skeleton,
} from '@/components/ui';
import { documentsApi, previewApi, exportApi } from '@/lib/api';
import {
  formatDuration,
  formatConfidence,
  getConfidenceLevel,
  getConfidenceColor,
  getStatusText,
  copyToClipboard,
  cn,
} from '@/lib/utils';
import type { FieldResult, ConfidenceLevel } from '@/types/api';

// Field card component for displaying extraction results
interface FieldCardProps {
  name: string;
  result: FieldResult;
  maskPhi: boolean;
}

const FieldCard: React.FC<FieldCardProps> = ({ name, result, maskPhi }) => {
  const [expanded, setExpanded] = useState(false);
  const confidenceLevel = getConfidenceLevel(result.confidence);

  const displayValue = () => {
    if (maskPhi && result.value) {
      // HIPAA-compliant PHI masking
      // SECURITY: Never reveal any characters, length, or format of PHI
      // Use consistent mask to prevent pattern analysis
      return '••••••••';
    }
    if (typeof result.value === 'object') {
      return JSON.stringify(result.value, null, 2);
    }
    return String(result.value ?? 'N/A');
  };

  const getConfidenceIcon = (level: ConfidenceLevel) => {
    switch (level) {
      case 'high':
        return <CheckCircle className="w-4 h-4 text-success-600" />;
      case 'medium':
        return <AlertTriangle className="w-4 h-4 text-warning-600" />;
      case 'low':
        return <XCircle className="w-4 h-4 text-error-600" />;
    }
  };

  return (
    <motion.div
      layout
      className={cn(
        'border rounded-xl p-4 transition-all',
        result.validation_passed ? 'border-surface-200' : 'border-warning-300 bg-warning-50',
        !result.passes_agree && 'ring-2 ring-error-200'
      )}
    >
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-surface-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-surface-400" />
          )}
          <span className="font-medium text-surface-900">{name}</span>
        </div>
        <div className="flex items-center gap-2">
          <Tooltip content={`${formatConfidence(result.confidence)} confidence`}>
            <div className="flex items-center gap-1">
              {getConfidenceIcon(confidenceLevel)}
              <span className={cn('text-sm', getConfidenceColor(confidenceLevel))}>
                {formatConfidence(result.confidence)}
              </span>
            </div>
          </Tooltip>
        </div>
      </div>

      <div className="mt-2 pl-7">
        <div className="text-sm text-surface-700 font-mono bg-surface-50 p-2 rounded">
          {displayValue()}
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 pt-3 border-t border-surface-100 pl-7 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-surface-500">Location</span>
                <span className="text-surface-700">{result.location || 'Not specified'}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-surface-500">Passes Agree</span>
                <Badge variant={result.passes_agree ? 'success' : 'error'}>
                  {result.passes_agree ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-surface-500">Validation</span>
                <Badge variant={result.validation_passed ? 'success' : 'warning'}>
                  {result.validation_passed ? 'Passed' : 'Review Needed'}
                </Badge>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const processingId = params.id as string;

  const [activeTab, setActiveTab] = useState('fields');
  const [maskPhi, setMaskPhi] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  // Fetch document data
  const {
    data: document,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['document', processingId],
    queryFn: () => documentsApi.get(processingId),
    retry: 1,
  });

  // Fetch markdown preview
  const { data: markdownPreview } = useQuery({
    queryKey: ['document', processingId, 'preview', maskPhi],
    queryFn: () => previewApi.markdown(processingId, maskPhi),
    enabled: activeTab === 'preview',
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: () => documentsApi.delete(processingId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      router.push('/documents');
    },
  });

  // Reprocess mutation
  const reprocessMutation = useMutation({
    mutationFn: () => documentsApi.reprocess(processingId),
    onSuccess: () => {
      refetch();
    },
  });

  // Export handlers
  const handleExport = async (format: 'json' | 'excel' | 'markdown' | 'fhir') => {
    try {
      await exportApi.download(processingId, format);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  // Copy handler
  const handleCopy = async (text: string, fieldName: string) => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopiedField(fieldName);
      setTimeout(() => setCopiedField(null), 2000);
    }
  };

  // Human review handler - opens review workflow modal
  const handleStartReview = () => {
    setShowReviewModal(true);
  };

  // Mark review as complete (would typically call an API)
  const handleCompleteReview = async (approved: boolean, notes: string) => {
    // TODO: Implement API call to mark review as complete
    // For now, close modal and show success message
    console.log('Review completed:', { processingId, approved, notes });
    setShowReviewModal(false);
    // Refresh document data to get updated review status
    refetch();
  };

  if (isLoading) {
    return (
      <AppLayout>
        <div className="space-y-6">
          <div className="flex items-center gap-4">
            <Skeleton variant="circular" width="2.5rem" height="2.5rem" />
            <div className="space-y-2">
              <Skeleton width="200px" />
              <Skeleton width="150px" height="0.75rem" />
            </div>
          </div>
          <Skeleton height="200px" />
          <Skeleton height="400px" />
        </div>
      </AppLayout>
    );
  }

  if (error || !document) {
    return (
      <AppLayout>
        <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
          <div className="w-16 h-16 rounded-full bg-error-100 flex items-center justify-center mb-4">
            <XCircle className="w-8 h-8 text-error-600" />
          </div>
          <h2 className="text-xl font-bold text-surface-900 mb-2">Document Not Found</h2>
          <p className="text-surface-500 mb-6">
            The document you&apos;re looking for doesn&apos;t exist or has been deleted.
          </p>
          <Link href="/documents">
            <Button variant="primary">Back to Documents</Button>
          </Link>
        </div>
      </AppLayout>
    );
  }

  const fieldCount = Object.keys(document.field_metadata || {}).length;
  const highConfidenceFields = Object.values(document.field_metadata || {}).filter(
    (f) => f.confidence >= 0.85
  ).length;
  const validationIssues = document.validation?.errors?.length || 0;

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4"
        >
          <div className="flex items-center gap-4">
            <Link href="/documents">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center">
              <FileText className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-surface-900 flex items-center gap-2">
                Document Details
                <Badge variant={document.status === 'completed' ? 'success' : 'default'}>
                  {getStatusText(document.status)}
                </Badge>
              </h1>
              <p className="text-sm text-surface-500 font-mono">{processingId}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Tooltip content={maskPhi ? 'Show PHI' : 'Mask PHI'}>
              <Button
                variant="secondary"
                size="icon"
                onClick={() => setMaskPhi(!maskPhi)}
              >
                {maskPhi ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </Tooltip>
            <Button
              variant="secondary"
              leftIcon={<RefreshCw className="w-4 h-4" />}
              onClick={() => reprocessMutation.mutate()}
              disabled={reprocessMutation.isPending}
            >
              {reprocessMutation.isPending ? 'Reprocessing...' : 'Reprocess'}
            </Button>
            <Button
              variant="secondary"
              leftIcon={<Download className="w-4 h-4" />}
              onClick={() => handleExport('json')}
            >
              Export
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowDeleteModal(true)}
              className="text-error-600 hover:bg-error-50"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </motion.div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card variant="default" padding="md">
            <div className="flex items-center gap-3">
              <div
                className={cn(
                  'w-10 h-10 rounded-lg flex items-center justify-center',
                  getConfidenceColor(document.confidence_level)
                )}
              >
                {document.overall_confidence >= 0.85 ? (
                  <CheckCircle className="w-5 h-5" />
                ) : document.overall_confidence >= 0.6 ? (
                  <AlertTriangle className="w-5 h-5" />
                ) : (
                  <XCircle className="w-5 h-5" />
                )}
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {formatConfidence(document.overall_confidence)}
                </p>
                <p className="text-sm text-surface-500">Overall Confidence</p>
              </div>
            </div>
          </Card>

          <Card variant="default" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                <FileText className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">{fieldCount}</p>
                <p className="text-sm text-surface-500">Fields Extracted</p>
              </div>
            </div>
          </Card>

          <Card variant="default" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-success-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">{highConfidenceFields}</p>
                <p className="text-sm text-surface-500">High Confidence</p>
              </div>
            </div>
          </Card>

          <Card variant="default" padding="md">
            <div className="flex items-center gap-3">
              <div
                className={cn(
                  'w-10 h-10 rounded-lg flex items-center justify-center',
                  validationIssues > 0 ? 'bg-error-100' : 'bg-success-100'
                )}
              >
                {validationIssues > 0 ? (
                  <AlertTriangle className="w-5 h-5 text-error-600" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-success-600" />
                )}
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">{validationIssues}</p>
                <p className="text-sm text-surface-500">Validation Issues</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Human Review Warning */}
        {document.requires_human_review && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-4 p-4 bg-warning-50 border border-warning-200 rounded-xl"
          >
            <div className="w-10 h-10 rounded-full bg-warning-100 flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="w-5 h-5 text-warning-600" />
            </div>
            <div className="flex-1">
              <p className="font-medium text-warning-800">Human Review Required</p>
              <p className="text-sm text-warning-700">
                {document.human_review_reason || 'This document requires manual verification.'}
              </p>
            </div>
            <Button variant="secondary" onClick={handleStartReview}>
              Start Review
            </Button>
          </motion.div>
        )}

        {/* Tabs */}
        <Card variant="elevated" padding="none">
          <Tabs value={activeTab} onChange={setActiveTab}>
            <TabsList className="border-b border-surface-200 px-4">
              <TabsTrigger value="fields">Extracted Fields</TabsTrigger>
              <TabsTrigger value="validation">Validation</TabsTrigger>
              <TabsTrigger value="preview">Preview</TabsTrigger>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
            </TabsList>

            <TabsContent value="fields" className="p-6">
              <div className="space-y-4">
                {Object.entries(document.field_metadata || {}).map(([name, result]) => (
                  <FieldCard key={name} name={name} result={result} maskPhi={maskPhi} />
                ))}
                {Object.keys(document.field_metadata || {}).length === 0 && (
                  <div className="text-center py-12 text-surface-500">
                    No fields extracted from this document.
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="validation" className="p-6">
              {document.validation ? (
                <div className="space-y-6">
                  <div className="flex items-center gap-4">
                    <Badge
                      variant={document.validation.is_valid ? 'success' : 'error'}
                      size="lg"
                    >
                      {document.validation.is_valid ? 'Valid' : 'Invalid'}
                    </Badge>
                  </div>

                  {document.validation.errors && document.validation.errors.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-surface-700 mb-3">Errors</h3>
                      <div className="space-y-2">
                        {document.validation.errors.map((error, i) => (
                          <div
                            key={i}
                            className="flex items-start gap-3 p-3 bg-error-50 border border-error-200 rounded-lg"
                          >
                            <XCircle className="w-5 h-5 text-error-600 flex-shrink-0 mt-0.5" />
                            <span className="text-sm text-error-700">{error}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {document.validation.warnings && document.validation.warnings.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-surface-700 mb-3">Warnings</h3>
                      <div className="space-y-2">
                        {document.validation.warnings.map((warning, i) => (
                          <div
                            key={i}
                            className="flex items-start gap-3 p-3 bg-warning-50 border border-warning-200 rounded-lg"
                          >
                            <AlertTriangle className="w-5 h-5 text-warning-600 flex-shrink-0 mt-0.5" />
                            <span className="text-sm text-warning-700">{warning}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {document.validation.hallucination_flags &&
                    document.validation.hallucination_flags.length > 0 && (
                      <div>
                        <h3 className="text-sm font-medium text-surface-700 mb-3">
                          Hallucination Flags
                        </h3>
                        <div className="space-y-2">
                          {document.validation.hallucination_flags.map((flag, i) => (
                            <div
                              key={i}
                              className="flex items-start gap-3 p-3 bg-error-50 border border-error-200 rounded-lg"
                            >
                              <AlertTriangle className="w-5 h-5 text-error-600 flex-shrink-0 mt-0.5" />
                              <span className="text-sm text-error-700">{flag}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
              ) : (
                <div className="text-center py-12 text-surface-500">
                  No validation data available.
                </div>
              )}
            </TabsContent>

            <TabsContent value="preview" className="p-6">
              {markdownPreview ? (
                <div className="prose prose-sm max-w-none">
                  <pre className="bg-surface-50 p-4 rounded-xl overflow-auto text-sm">
                    {markdownPreview}
                  </pre>
                </div>
              ) : (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-primary-500 animate-spin" />
                </div>
              )}
            </TabsContent>

            <TabsContent value="metadata" className="p-6">
              <div className="space-y-4">
                {document.metadata && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-surface-50 rounded-lg">
                      <p className="text-sm text-surface-500 mb-1">Processing Time</p>
                      <p className="font-medium text-surface-900">
                        {document.metadata.processing_time_ms
                          ? formatDuration(document.metadata.processing_time_ms)
                          : 'N/A'}
                      </p>
                    </div>
                    <div className="p-4 bg-surface-50 rounded-lg">
                      <p className="text-sm text-surface-500 mb-1">VLM Calls</p>
                      <p className="font-medium text-surface-900">
                        {document.metadata.vlm_calls ?? 'N/A'}
                      </p>
                    </div>
                    <div className="p-4 bg-surface-50 rounded-lg">
                      <p className="text-sm text-surface-500 mb-1">Pages Processed</p>
                      <p className="font-medium text-surface-900">
                        {document.metadata.pages_processed ?? 'N/A'}
                      </p>
                    </div>
                    <div className="p-4 bg-surface-50 rounded-lg">
                      <p className="text-sm text-surface-500 mb-1">Retries</p>
                      <p className="font-medium text-surface-900">
                        {document.metadata.retries ?? 0}
                      </p>
                    </div>
                  </div>
                )}

                <div className="mt-6">
                  <h3 className="text-sm font-medium text-surface-700 mb-3">Raw Data</h3>
                  <div className="relative">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2"
                      onClick={() =>
                        handleCopy(JSON.stringify(document.data, null, 2), 'raw-data')
                      }
                    >
                      {copiedField === 'raw-data' ? (
                        <CheckCircle className="w-4 h-4 text-success-600" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </Button>
                    <pre className="bg-surface-50 p-4 rounded-xl overflow-auto text-sm font-mono max-h-96">
                      {JSON.stringify(document.data, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        {/* Delete Confirmation Modal */}
        <Modal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          title="Delete Document"
        >
          <div className="space-y-4">
            <p className="text-surface-600">
              Are you sure you want to delete this document? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowDeleteModal(false)}>
                Cancel
              </Button>
              <Button
                variant="danger"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </Button>
            </div>
          </div>
        </Modal>

        {/* Human Review Modal */}
        <Modal
          isOpen={showReviewModal}
          onClose={() => setShowReviewModal(false)}
          title="Human Review"
        >
          <HumanReviewForm
            document={document}
            onSubmit={handleCompleteReview}
            onCancel={() => setShowReviewModal(false)}
          />
        </Modal>
      </div>
    </AppLayout>
  );
}

// Human Review Form Component
interface HumanReviewFormProps {
  document: any;
  onSubmit: (approved: boolean, notes: string) => void;
  onCancel: () => void;
}

function HumanReviewForm({ document, onSubmit, onCancel }: HumanReviewFormProps) {
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (approved: boolean) => {
    setIsSubmitting(true);
    try {
      await onSubmit(approved, notes);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Review Reason */}
      <div className="p-4 bg-warning-50 border border-warning-200 rounded-lg">
        <p className="text-sm font-medium text-warning-800">Review Required Because:</p>
        <p className="text-sm text-warning-700 mt-1">
          {document?.human_review_reason || 'Low confidence extraction or validation issues detected.'}
        </p>
      </div>

      {/* Key Fields Summary */}
      <div>
        <h4 className="text-sm font-medium text-surface-700 mb-2">Key Fields to Verify:</h4>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {Object.entries(document?.field_metadata || {})
            .filter(([_, result]: [string, any]) => result.confidence < 0.85 || !result.validation_passed)
            .slice(0, 5)
            .map(([name, result]: [string, any]) => (
              <div key={name} className="flex items-center justify-between p-2 bg-surface-50 rounded">
                <span className="text-sm font-medium text-surface-700">{name}</span>
                <span className="text-sm text-surface-600">{String(result.value)}</span>
              </div>
            ))}
        </div>
      </div>

      {/* Review Notes */}
      <div>
        <label htmlFor="review-notes" className="block text-sm font-medium text-surface-700 mb-1">
          Review Notes (optional)
        </label>
        <textarea
          id="review-notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add any notes about your review..."
          className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
          rows={3}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end gap-3 pt-2">
        <Button variant="secondary" onClick={onCancel} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button
          variant="danger"
          onClick={() => handleSubmit(false)}
          disabled={isSubmitting}
        >
          Reject
        </Button>
        <Button
          variant="primary"
          onClick={() => handleSubmit(true)}
          disabled={isSubmitting}
        >
          Approve
        </Button>
      </div>
    </div>
  );
}
