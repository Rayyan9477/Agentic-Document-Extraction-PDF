'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'white' | 'surface';
  className?: string;
}

const Spinner: React.FC<SpinnerProps> = ({ size = 'md', color = 'primary', className }) => {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-8 h-8 border-3',
    xl: 'w-12 h-12 border-4',
  };

  const colors = {
    primary: 'border-primary-200 border-t-primary-600',
    white: 'border-white/30 border-t-white',
    surface: 'border-surface-200 border-t-surface-600',
  };

  return (
    <div
      className={cn(
        'rounded-full animate-spin',
        sizes[size],
        colors[color],
        className
      )}
    />
  );
};

// Loading Overlay Component
interface LoadingOverlayProps {
  loading: boolean;
  text?: string;
  blur?: boolean;
  children: React.ReactNode;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  text,
  blur = true,
  children,
}) => {
  return (
    <div className="relative">
      {children}
      {loading && (
        <div
          className={cn(
            'absolute inset-0 flex flex-col items-center justify-center bg-white/80 z-10',
            blur && 'backdrop-blur-sm'
          )}
        >
          <Spinner size="lg" />
          {text && <p className="mt-3 text-sm text-surface-600">{text}</p>}
        </div>
      )}
    </div>
  );
};

// Full Page Loader
interface PageLoaderProps {
  text?: string;
}

export const PageLoader: React.FC<PageLoaderProps> = ({ text = 'Loading...' }) => {
  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-surface-50 z-50">
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-primary-200 rounded-full" />
          <div className="absolute top-0 left-0 w-16 h-16 border-4 border-transparent border-t-primary-600 rounded-full animate-spin" />
        </div>
        <p className="text-surface-600 font-medium">{text}</p>
      </div>
    </div>
  );
};

// Skeleton Loader
interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  variant = 'text',
  width,
  height,
}) => {
  const variants = {
    text: 'rounded-lg',
    circular: 'rounded-full',
    rectangular: 'rounded-xl',
  };

  const defaultDimensions = {
    text: { width: '100%', height: '1rem' },
    circular: { width: '2.5rem', height: '2.5rem' },
    rectangular: { width: '100%', height: '5rem' },
  };

  return (
    <div
      className={cn(
        'animate-pulse bg-surface-200',
        variants[variant],
        className
      )}
      style={{
        width: width || defaultDimensions[variant].width,
        height: height || defaultDimensions[variant].height,
      }}
    />
  );
};

export default Spinner;
