'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  FileText,
  Upload,
  Search,
  Filter,
  Download,
  Eye,
  MoreVertical,
  Clock,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import {
  Card,
  Button,
  Input,
  StatusBadge,
  ConfidenceBadge,
  Dropdown,
  NoDocumentsState,
  Skeleton,
} from '@/components/ui';
import { documentsApi } from '@/lib/api';
import { formatDuration, truncate } from '@/lib/utils';
import type { ProcessResponse } from '@/types/api';

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('');

  const { data: documents, isLoading } = useQuery({
    queryKey: ['documents', 'recent'],
    queryFn: () => documentsApi.listRecent(),
  });

  const filteredDocuments = documents?.filter((doc) =>
    doc.processing_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleExport = async (doc: ProcessResponse, format: 'json' | 'excel' | 'markdown') => {
    try {
      const blob = await documentsApi.export(doc.processing_id, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${doc.processing_id}.${format === 'excel' ? 'xlsx' : format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Documents</h1>
            <p className="text-surface-500 mt-1">
              View and manage processed documents
            </p>
          </div>
          <Link href="/documents/upload">
            <Button variant="primary" leftIcon={<Upload className="w-4 h-4" />}>
              Upload New
            </Button>
          </Link>
        </motion.div>

        {/* Search and Filter */}
        <Card variant="elevated" padding="md">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Input
                placeholder="Search by ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="w-4 h-4" />}
              />
            </div>
            <Button variant="secondary" leftIcon={<Filter className="w-4 h-4" />}>
              Filters
            </Button>
          </div>
        </Card>

        {/* Documents List */}
        <Card variant="elevated" padding="none">
          {isLoading ? (
            <div className="p-6 space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex items-center gap-4 p-4 border rounded-xl">
                  <Skeleton variant="rectangular" width="3rem" height="3rem" />
                  <div className="flex-1 space-y-2">
                    <Skeleton width="60%" />
                    <Skeleton width="40%" height="0.75rem" />
                  </div>
                  <Skeleton width="5rem" />
                </div>
              ))}
            </div>
          ) : filteredDocuments?.length === 0 ? (
            <div className="p-8">
              <NoDocumentsState
                onUpload={() => window.location.href = '/documents/upload'}
              />
            </div>
          ) : (
            <div className="divide-y divide-surface-100">
              {/* Table Header */}
              <div className="hidden md:grid grid-cols-12 gap-4 px-6 py-3 bg-surface-50 text-xs font-medium text-surface-500 uppercase tracking-wider">
                <div className="col-span-4">Document</div>
                <div className="col-span-2">Status</div>
                <div className="col-span-2">Confidence</div>
                <div className="col-span-2">Processed</div>
                <div className="col-span-2 text-right">Actions</div>
              </div>

              {/* Document Rows */}
              {filteredDocuments?.map((doc, index) => (
                <motion.div
                  key={doc.processing_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className="grid grid-cols-1 md:grid-cols-12 gap-4 px-6 py-4 hover:bg-surface-50 transition-colors"
                >
                  {/* Document Info */}
                  <div className="col-span-4 flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-6 h-6 text-primary-600" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-surface-900 truncate">
                        {truncate(doc.processing_id, 20)}
                      </p>
                      <p className="text-xs text-surface-500">
                        {doc.metadata?.fields_extracted || 0} fields extracted
                      </p>
                    </div>
                  </div>

                  {/* Status */}
                  <div className="col-span-2 flex items-center">
                    <StatusBadge status={doc.status} />
                  </div>

                  {/* Confidence */}
                  <div className="col-span-2 flex items-center">
                    <ConfidenceBadge
                      level={doc.confidence_level}
                      value={doc.overall_confidence}
                    />
                  </div>

                  {/* Processed Time */}
                  <div className="col-span-2 flex items-center">
                    <div className="text-sm text-surface-600">
                      {doc.metadata?.processing_time_ms
                        ? formatDuration(doc.metadata.processing_time_ms)
                        : 'N/A'}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="col-span-2 flex items-center justify-end gap-2">
                    <Link href={`/documents/${doc.processing_id}`}>
                      <Button variant="ghost" size="icon">
                        <Eye className="w-4 h-4" />
                      </Button>
                    </Link>
                    <Dropdown
                      align="right"
                      trigger={
                        <Button variant="ghost" size="icon">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      }
                      items={[
                        {
                          label: 'View Details',
                          icon: <Eye className="w-4 h-4" />,
                          onClick: () =>
                            (window.location.href = `/documents/${doc.processing_id}`),
                        },
                        {
                          label: 'Export JSON',
                          icon: <Download className="w-4 h-4" />,
                          onClick: () => handleExport(doc, 'json'),
                        },
                        {
                          label: 'Export Excel',
                          icon: <Download className="w-4 h-4" />,
                          onClick: () => handleExport(doc, 'excel'),
                        },
                        { divider: true, label: '' },
                        {
                          label: 'Reprocess',
                          icon: <Clock className="w-4 h-4" />,
                          onClick: () => {},
                        },
                      ]}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </Card>

        {/* Stats Summary */}
        {documents && documents.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-4"
          >
            <Card variant="default" padding="md" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-success-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {documents.filter((d) => d.status === 'completed').length}
                </p>
                <p className="text-sm text-surface-500">Completed</p>
              </div>
            </Card>
            <Card variant="default" padding="md" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-warning-100 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-warning-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {documents.filter((d) => d.requires_human_review).length}
                </p>
                <p className="text-sm text-surface-500">Need Review</p>
              </div>
            </Card>
            <Card variant="default" padding="md" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                <FileText className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {documents.length}
                </p>
                <p className="text-sm text-surface-500">Total Documents</p>
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </AppLayout>
  );
}
