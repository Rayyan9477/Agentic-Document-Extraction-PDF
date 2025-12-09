'use client';

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  Database,
  Search,
  FileText,
  Grid3X3,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import {
  Card,
  CardHeader,
  CardContent,
  Input,
  Badge,
  Skeleton,
} from '@/components/ui';
import { schemaApi } from '@/lib/api';
import type { SchemaInfo } from '@/types/api';

export default function SchemasPage() {
  const [searchQuery, setSearchQuery] = useState('');

  const { data: schemasData, isLoading, error } = useQuery({
    queryKey: ['schemas'],
    queryFn: () => schemaApi.list(),
  });

  const schemas = schemasData || [];
  const filteredSchemas = schemas.filter((schema) =>
    schema.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    schema.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    schema.document_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Document Schemas</h1>
            <p className="text-surface-500 mt-1">
              Available extraction schemas for different document types
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card variant="outline" padding="md">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                  <Database className="w-5 h-5 text-primary-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-surface-900">{schemas.length}</p>
                  <p className="text-sm text-surface-500">Total Schemas</p>
                </div>
              </div>
            </Card>

            <Card variant="outline" padding="md">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-success-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-surface-900">
                    {schemas.reduce((sum, s) => sum + s.field_count, 0)}
                  </p>
                  <p className="text-sm text-surface-500">Total Fields</p>
                </div>
              </div>
            </Card>

            <Card variant="outline" padding="md">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-info-100 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-info-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-surface-900">
                    {new Set(schemas.map(s => s.document_type)).size}
                  </p>
                  <p className="text-sm text-surface-500">Document Types</p>
                </div>
              </div>
            </Card>
          </div>
        </motion.div>

        {/* Search Bar */}
        <Card variant="outline" padding="none">
          <div className="p-4">
            <Input
              placeholder="Search schemas by name, type, or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-4 h-4" />}
            />
          </div>
        </Card>

        {/* Schemas Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <Card key={i} variant="outline" padding="lg">
                <Skeleton className="h-48" />
              </Card>
            ))}
          </div>
        ) : error ? (
          <Card variant="outline" padding="lg">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-16 h-16 rounded-full bg-danger-100 flex items-center justify-center mb-4">
                <AlertCircle className="w-8 h-8 text-danger-600" />
              </div>
              <h3 className="text-lg font-semibold text-surface-900 mb-2">
                Failed to Load Schemas
              </h3>
              <p className="text-surface-500 max-w-md">
                There was an error loading the schemas. Please try again later.
              </p>
            </div>
          </Card>
        ) : filteredSchemas.length === 0 ? (
          <Card variant="outline" padding="lg">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-16 h-16 rounded-full bg-surface-100 flex items-center justify-center mb-4">
                <Database className="w-8 h-8 text-surface-400" />
              </div>
              <h3 className="text-lg font-semibold text-surface-900 mb-2">
                {searchQuery ? 'No schemas found' : 'No schemas available'}
              </h3>
              <p className="text-surface-500 max-w-md">
                {searchQuery
                  ? 'Try adjusting your search query'
                  : 'No document schemas are currently configured'}
              </p>
            </div>
          </Card>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {filteredSchemas.map((schema) => (
              <SchemaCard key={schema.name} schema={schema} />
            ))}
          </motion.div>
        )}
      </div>
    </AppLayout>
  );
}

interface SchemaCardProps {
  schema: SchemaInfo;
}

function SchemaCard({ schema }: SchemaCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Card variant="elevated" padding="none" className="h-full">
        {/* Header with gradient */}
        <div className="relative h-24 bg-gradient-to-br from-primary-500 to-primary-700 rounded-t-xl">
          <div className="absolute inset-0 bg-grid-white/10" />
          <div className="relative h-full flex items-center justify-center">
            <div className="w-14 h-14 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
              <Database className="w-7 h-7 text-white" />
            </div>
          </div>
        </div>

        <CardContent className="p-6">
          {/* Schema Name */}
          <div className="mb-3">
            <h3 className="text-lg font-bold text-surface-900 mb-1">
              {schema.name}
            </h3>
            <Badge size="sm" variant="default">
              {schema.document_type}
            </Badge>
          </div>

          {/* Description */}
          <p className="text-sm text-surface-600 mb-4 line-clamp-3">
            {schema.description || 'No description available'}
          </p>

          {/* Stats */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-surface-500 flex items-center gap-2">
                <Grid3X3 className="w-4 h-4" />
                Fields
              </span>
              <span className="font-semibold text-surface-900">
                {schema.field_count}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-surface-500 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                Version
              </span>
              <span className="font-semibold text-surface-900">
                {schema.version}
              </span>
            </div>
          </div>

          {/* Status */}
          <div className="mt-4 pt-4 border-t border-surface-100">
            <div className="flex items-center gap-2 text-sm text-success-600">
              <CheckCircle className="w-4 h-4" />
              <span className="font-medium">Active</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
