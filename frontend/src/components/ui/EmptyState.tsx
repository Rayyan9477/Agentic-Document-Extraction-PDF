'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { FileQuestion, Inbox, Search, AlertCircle } from 'lucide-react';
import Button from './Button';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  variant?: 'default' | 'compact' | 'large';
  className?: string;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  secondaryAction,
  variant = 'default',
  className,
}) => {
  const variants = {
    compact: {
      container: 'py-8',
      iconSize: 'w-10 h-10',
      iconWrapper: 'w-16 h-16',
      title: 'text-base',
      description: 'text-sm',
    },
    default: {
      container: 'py-12',
      iconSize: 'w-12 h-12',
      iconWrapper: 'w-20 h-20',
      title: 'text-lg',
      description: 'text-sm',
    },
    large: {
      container: 'py-16',
      iconSize: 'w-16 h-16',
      iconWrapper: 'w-24 h-24',
      title: 'text-xl',
      description: 'text-base',
    },
  };

  const styles = variants[variant];

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center',
        styles.container,
        className
      )}
    >
      <div
        className={cn(
          'flex items-center justify-center rounded-full bg-surface-100 mb-4',
          styles.iconWrapper
        )}
      >
        {icon || (
          <Inbox className={cn('text-surface-400', styles.iconSize)} />
        )}
      </div>
      <h3 className={cn('font-semibold text-surface-900', styles.title)}>
        {title}
      </h3>
      {description && (
        <p className={cn('mt-2 text-surface-500 max-w-sm', styles.description)}>
          {description}
        </p>
      )}
      {(action || secondaryAction) && (
        <div className="flex items-center gap-3 mt-6">
          {secondaryAction && (
            <Button variant="secondary" onClick={secondaryAction.onClick}>
              {secondaryAction.label}
            </Button>
          )}
          {action && (
            <Button variant="primary" onClick={action.onClick}>
              {action.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
};

// Preset Empty States
export const NoDocumentsState: React.FC<{ onUpload?: () => void }> = ({ onUpload }) => (
  <EmptyState
    icon={<FileQuestion className="w-12 h-12 text-surface-400" />}
    title="No documents yet"
    description="Upload your first PDF document to get started with extraction."
    action={onUpload ? { label: 'Upload Document', onClick: onUpload } : undefined}
  />
);

export const NoResultsState: React.FC<{ query?: string; onClear?: () => void }> = ({
  query,
  onClear,
}) => (
  <EmptyState
    icon={<Search className="w-12 h-12 text-surface-400" />}
    title="No results found"
    description={
      query
        ? `No results found for "${query}". Try adjusting your search.`
        : 'Try adjusting your filters or search terms.'
    }
    action={onClear ? { label: 'Clear Search', onClick: onClear } : undefined}
  />
);

export const ErrorState: React.FC<{
  message?: string;
  onRetry?: () => void;
}> = ({ message, onRetry }) => (
  <EmptyState
    icon={<AlertCircle className="w-12 h-12 text-error-500" />}
    title="Something went wrong"
    description={message || 'An unexpected error occurred. Please try again.'}
    action={onRetry ? { label: 'Try Again', onClick: onRetry } : undefined}
  />
);

export default EmptyState;
