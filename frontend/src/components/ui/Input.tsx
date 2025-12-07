'use client';

import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, hint, leftIcon, rightIcon, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-surface-700 mb-1.5"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-surface-400">
              {leftIcon}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={cn(
              'w-full px-4 py-2.5 rounded-xl border bg-white text-surface-900',
              'placeholder:text-surface-400',
              'transition-all duration-200',
              'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
              'disabled:bg-surface-50 disabled:text-surface-400 disabled:cursor-not-allowed',
              error
                ? 'border-error-500 focus:ring-error-500'
                : 'border-surface-200 hover:border-surface-300',
              leftIcon && 'pl-10',
              rightIcon && 'pr-10',
              className
            )}
            {...props}
          />
          {rightIcon && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400">
              {rightIcon}
            </div>
          )}
        </div>
        {error && <p className="mt-1.5 text-sm text-error-600">{error}</p>}
        {hint && !error && <p className="mt-1.5 text-sm text-surface-500">{hint}</p>}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;

// Textarea component
export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  hint?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, hint, id, ...props }, ref) => {
    const textareaId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={textareaId}
            className="block text-sm font-medium text-surface-700 mb-1.5"
          >
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={textareaId}
          className={cn(
            'w-full px-4 py-2.5 rounded-xl border bg-white text-surface-900',
            'placeholder:text-surface-400',
            'transition-all duration-200',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'disabled:bg-surface-50 disabled:text-surface-400 disabled:cursor-not-allowed',
            'min-h-[100px] resize-y',
            error
              ? 'border-error-500 focus:ring-error-500'
              : 'border-surface-200 hover:border-surface-300',
            className
          )}
          {...props}
        />
        {error && <p className="mt-1.5 text-sm text-error-600">{error}</p>}
        {hint && !error && <p className="mt-1.5 text-sm text-surface-500">{hint}</p>}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';
