'use client';

import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline' | 'link';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled,
      leftIcon,
      rightIcon,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary:
        'bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500 shadow-sm hover:shadow-md active:scale-[0.98]',
      secondary:
        'bg-surface-100 text-surface-900 hover:bg-surface-200 focus:ring-surface-400 border border-surface-200',
      ghost:
        'text-surface-600 hover:bg-surface-100 hover:text-surface-900 focus:ring-surface-400',
      danger:
        'bg-error-600 text-white hover:bg-error-700 focus:ring-error-500 shadow-sm',
      outline:
        'border-2 border-primary-600 text-primary-600 hover:bg-primary-50 focus:ring-primary-500',
      link: 'text-primary-600 hover:text-primary-700 underline-offset-4 hover:underline focus:ring-primary-500',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm rounded-lg gap-1.5',
      md: 'px-4 py-2 text-sm rounded-xl gap-2',
      lg: 'px-6 py-3 text-base rounded-xl gap-2',
      icon: 'p-2 rounded-xl',
    };

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          leftIcon && <span className="flex-shrink-0">{leftIcon}</span>
        )}
        {children}
        {rightIcon && !loading && <span className="flex-shrink-0">{rightIcon}</span>}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
