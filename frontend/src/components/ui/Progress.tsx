'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface ProgressProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  striped?: boolean;
  className?: string;
}

const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  size = 'md',
  color = 'primary',
  showLabel = false,
  label,
  animated = true,
  striped = false,
  className,
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizes = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const colors = {
    primary: 'bg-primary-600',
    success: 'bg-success-600',
    warning: 'bg-warning-600',
    error: 'bg-error-600',
    info: 'bg-info-600',
  };

  return (
    <div className={cn('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-surface-700">
            {label || 'Progress'}
          </span>
          {showLabel && (
            <span className="text-sm text-surface-500">{Math.round(percentage)}%</span>
          )}
        </div>
      )}
      <div
        className={cn(
          'w-full bg-surface-200 rounded-full overflow-hidden',
          sizes[size]
        )}
      >
        <motion.div
          initial={animated ? { width: 0 } : false}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={cn(
            'h-full rounded-full',
            colors[color],
            striped && 'bg-striped'
          )}
        />
      </div>
    </div>
  );
};

// Circular Progress Component
interface CircularProgressProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info';
  showLabel?: boolean;
  className?: string;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 80,
  strokeWidth = 8,
  color = 'primary',
  showLabel = true,
  className,
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  const colors = {
    primary: 'text-primary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    error: 'text-error-600',
    info: 'text-info-600',
  };

  return (
    <div className={cn('relative inline-flex', className)}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-surface-200"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={colors[color]}
        />
      </svg>
      {showLabel && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-semibold text-surface-700">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
    </div>
  );
};

// Step Progress Component
interface StepProgressProps {
  steps: Array<{ label: string; description?: string }>;
  currentStep: number;
  className?: string;
}

export const StepProgress: React.FC<StepProgressProps> = ({
  steps,
  currentStep,
  className,
}) => {
  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center">
        {steps.map((step, index) => (
          <React.Fragment key={index}>
            {/* Step circle */}
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  'w-10 h-10 rounded-full flex items-center justify-center font-medium text-sm transition-all',
                  index < currentStep
                    ? 'bg-primary-600 text-white'
                    : index === currentStep
                    ? 'bg-primary-100 text-primary-700 border-2 border-primary-600'
                    : 'bg-surface-100 text-surface-400'
                )}
              >
                {index < currentStep ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  index + 1
                )}
              </div>
              <span
                className={cn(
                  'mt-2 text-xs font-medium text-center max-w-[80px]',
                  index <= currentStep ? 'text-surface-700' : 'text-surface-400'
                )}
              >
                {step.label}
              </span>
            </div>
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div className="flex-1 h-0.5 mx-2 -mt-6">
                <div
                  className={cn(
                    'h-full transition-all',
                    index < currentStep ? 'bg-primary-600' : 'bg-surface-200'
                  )}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default Progress;
