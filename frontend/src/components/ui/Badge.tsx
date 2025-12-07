'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import type { TaskStatus, ConfidenceLevel } from '@/types/api';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
  pulse?: boolean;
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', size = 'md', dot = false, pulse = false, children, ...props }, ref) => {
    const variants = {
      default: 'bg-surface-100 text-surface-700 border-surface-200',
      primary: 'bg-primary-100 text-primary-700 border-primary-200',
      success: 'bg-success-100 text-success-700 border-success-200',
      warning: 'bg-warning-100 text-warning-700 border-warning-200',
      error: 'bg-error-100 text-error-700 border-error-200',
      info: 'bg-info-100 text-info-700 border-info-200',
      outline: 'bg-transparent text-surface-600 border-surface-300',
    };

    const sizes = {
      sm: 'text-xs px-2 py-0.5',
      md: 'text-xs px-2.5 py-1',
      lg: 'text-sm px-3 py-1',
    };

    const dotColors = {
      default: 'bg-surface-500',
      primary: 'bg-primary-500',
      success: 'bg-success-500',
      warning: 'bg-warning-500',
      error: 'bg-error-500',
      info: 'bg-info-500',
      outline: 'bg-surface-500',
    };

    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center gap-1.5 font-medium rounded-full border',
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {dot && (
          <span
            className={cn(
              'w-1.5 h-1.5 rounded-full',
              dotColors[variant],
              pulse && 'animate-pulse'
            )}
          />
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';

// Status Badge Component
interface StatusBadgeProps extends Omit<BadgeProps, 'variant'> {
  status: TaskStatus;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, ...props }) => {
  const statusConfig: Record<TaskStatus, { variant: BadgeProps['variant']; label: string; pulse: boolean }> = {
    pending: { variant: 'default', label: 'Pending', pulse: false },
    started: { variant: 'info', label: 'Started', pulse: true },
    processing: { variant: 'primary', label: 'Processing', pulse: true },
    validating: { variant: 'info', label: 'Validating', pulse: true },
    exporting: { variant: 'primary', label: 'Exporting', pulse: true },
    completed: { variant: 'success', label: 'Completed', pulse: false },
    failed: { variant: 'error', label: 'Failed', pulse: false },
    retrying: { variant: 'warning', label: 'Retrying', pulse: true },
    cancelled: { variant: 'default', label: 'Cancelled', pulse: false },
  };

  const config = statusConfig[status];

  return (
    <Badge variant={config.variant} dot pulse={config.pulse} {...props}>
      {config.label}
    </Badge>
  );
};

// Confidence Badge Component
interface ConfidenceBadgeProps extends Omit<BadgeProps, 'variant'> {
  level: ConfidenceLevel;
  value?: number;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ level, value, ...props }) => {
  const levelConfig: Record<ConfidenceLevel, { variant: BadgeProps['variant']; label: string }> = {
    high: { variant: 'success', label: 'High' },
    medium: { variant: 'warning', label: 'Medium' },
    low: { variant: 'error', label: 'Low' },
  };

  const config = levelConfig[level];

  return (
    <Badge variant={config.variant} {...props}>
      {value !== undefined ? `${Math.round(value * 100)}%` : config.label}
    </Badge>
  );
};

export default Badge;
