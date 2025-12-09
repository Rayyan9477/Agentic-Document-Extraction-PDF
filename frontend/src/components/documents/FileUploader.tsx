'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { cn, formatFileSize } from '@/lib/utils';
import { Button, Progress } from '@/components/ui';

export interface UploadFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  taskId?: string;
}

interface FileUploaderProps {
  onFilesSelected: (files: File[]) => void;
  onFileRemove: (fileId: string) => void;
  files: UploadFile[];
  maxFiles?: number;
  maxSize?: number; // in bytes
  disabled?: boolean;
}

const FileUploader: React.FC<FileUploaderProps> = ({
  onFilesSelected,
  onFileRemove,
  files,
  maxFiles = 10,
  maxSize = 50 * 1024 * 1024, // 50MB
  disabled = false,
}) => {
  const [dragError, setDragError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: { file: File; errors: { code: string; message: string }[] }[]) => {
      setDragError(null);

      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles.map((f) => {
          if (f.errors[0]?.code === 'file-too-large') {
            return `${f.file.name} is too large (max ${formatFileSize(maxSize)})`;
          }
          if (f.errors[0]?.code === 'file-invalid-type') {
            return `${f.file.name} is not a PDF file`;
          }
          return `${f.file.name}: ${f.errors[0]?.message}`;
        });
        setDragError(errors.join(', '));
        return;
      }

      if (files.length + acceptedFiles.length > maxFiles) {
        setDragError(`Maximum ${maxFiles} files allowed`);
        return;
      }

      onFilesSelected(acceptedFiles);
    },
    [files.length, maxFiles, maxSize, onFilesSelected]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxSize,
    maxFiles: maxFiles - files.length,
    disabled: disabled || files.length >= maxFiles,
  });

  const getFileStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-success-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-error-500" />;
      default:
        return <FileText className="w-5 h-5 text-surface-400" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          'relative border-2 border-dashed rounded-2xl p-8 transition-all duration-200 cursor-pointer',
          'flex flex-col items-center justify-center text-center',
          isDragActive && !isDragReject
            ? 'border-primary-500 bg-primary-50'
            : isDragReject
            ? 'border-error-500 bg-error-50'
            : 'border-surface-300 bg-surface-50 hover:border-primary-400 hover:bg-primary-50/50',
          (disabled || files.length >= maxFiles) && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input {...getInputProps()} />
        <div
          className={cn(
            'w-16 h-16 rounded-full flex items-center justify-center mb-4',
            isDragActive ? 'bg-primary-100' : 'bg-surface-100'
          )}
        >
          <Upload
            className={cn(
              'w-8 h-8',
              isDragActive ? 'text-primary-600' : 'text-surface-400'
            )}
          />
        </div>
        <p className="text-lg font-medium text-surface-700 mb-1">
          {isDragActive
            ? 'Drop your PDFs here'
            : 'Drag & drop PDF files here'}
        </p>
        <p className="text-sm text-surface-500 mb-4">
          or click to browse from your computer
        </p>
        <div className="flex items-center gap-4 text-xs text-surface-400">
          <span>PDF files only</span>
          <span>•</span>
          <span>Max {formatFileSize(maxSize)} per file</span>
          <span>•</span>
          <span>Up to {maxFiles} files</span>
        </div>
      </div>

      {/* Error Message */}
      <AnimatePresence>
        {dragError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-2 p-3 bg-error-50 border border-error-200 rounded-xl text-sm text-error-700"
          >
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {dragError}
          </motion.div>
        )}
      </AnimatePresence>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-surface-700">
              Selected Files ({files.length}/{maxFiles})
            </p>
          </div>
          <div className="space-y-2">
            <AnimatePresence>
              {files.map((file) => (
                <motion.div
                  key={file.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center gap-3 p-3 bg-white border border-surface-200 rounded-xl"
                >
                  <div className="w-10 h-10 rounded-lg bg-surface-100 flex items-center justify-center flex-shrink-0">
                    {getFileStatusIcon(file.status)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-surface-900 truncate">
                      {file.file.name}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-surface-500">
                        {formatFileSize(file.file.size)}
                      </span>
                      {file.error && (
                        <span className="text-xs text-error-600">{file.error}</span>
                      )}
                      {file.taskId && (
                        <span className="text-xs text-success-600">
                          Task: {file.taskId.slice(0, 8)}...
                        </span>
                      )}
                    </div>
                    {file.status === 'uploading' && (
                      <Progress
                        value={file.progress}
                        size="sm"
                        color="primary"
                        className="mt-2"
                      />
                    )}
                  </div>
                  {(file.status === 'pending' || file.status === 'error') && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onFileRemove(file.id)}
                      className="flex-shrink-0"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
