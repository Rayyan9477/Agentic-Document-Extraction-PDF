'use client';

import React, { forwardRef, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
  icon?: React.ReactNode;
}

interface SelectProps {
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  label?: string;
  error?: string;
  hint?: string;
  disabled?: boolean;
  className?: string;
}

const Select = forwardRef<HTMLDivElement, SelectProps>(
  (
    {
      options,
      value,
      onChange,
      placeholder = 'Select an option',
      label,
      error,
      hint,
      disabled = false,
      className,
    },
    ref
  ) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const selectedOption = options.find((opt) => opt.value === value);

    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
          setIsOpen(false);
        }
      };

      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleSelect = (optionValue: string) => {
      onChange?.(optionValue);
      setIsOpen(false);
    };

    return (
      <div ref={ref} className={cn('w-full', className)}>
        {label && (
          <label className="block text-sm font-medium text-surface-700 mb-1.5">
            {label}
          </label>
        )}
        <div ref={containerRef} className="relative">
          <button
            type="button"
            onClick={() => !disabled && setIsOpen(!isOpen)}
            disabled={disabled}
            className={cn(
              'w-full flex items-center justify-between gap-2',
              'px-4 py-2.5 rounded-xl border bg-white',
              'text-left transition-all duration-200',
              'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
              'disabled:bg-surface-50 disabled:text-surface-400 disabled:cursor-not-allowed',
              error
                ? 'border-error-500'
                : isOpen
                ? 'border-primary-500 ring-2 ring-primary-500'
                : 'border-surface-200 hover:border-surface-300'
            )}
          >
            <span
              className={cn(
                'flex items-center gap-2 truncate',
                selectedOption ? 'text-surface-900' : 'text-surface-400'
              )}
            >
              {selectedOption?.icon}
              {selectedOption?.label || placeholder}
            </span>
            <ChevronDown
              className={cn(
                'w-4 h-4 text-surface-400 transition-transform duration-200',
                isOpen && 'rotate-180'
              )}
            />
          </button>

          <AnimatePresence>
            {isOpen && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className={cn(
                  'absolute z-20 w-full mt-2',
                  'bg-white border border-surface-200 rounded-xl shadow-lg',
                  'max-h-60 overflow-auto'
                )}
              >
                <div className="py-1">
                  {options.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => !option.disabled && handleSelect(option.value)}
                      disabled={option.disabled}
                      className={cn(
                        'w-full flex items-center justify-between gap-2 px-4 py-2.5',
                        'text-left transition-colors duration-150',
                        option.disabled
                          ? 'text-surface-400 cursor-not-allowed'
                          : option.value === value
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-surface-700 hover:bg-surface-50'
                      )}
                    >
                      <span className="flex items-center gap-2 truncate">
                        {option.icon}
                        {option.label}
                      </span>
                      {option.value === value && (
                        <Check className="w-4 h-4 text-primary-600 flex-shrink-0" />
                      )}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {error && <p className="mt-1.5 text-sm text-error-600">{error}</p>}
        {hint && !error && <p className="mt-1.5 text-sm text-surface-500">{hint}</p>}
      </div>
    );
  }
);

Select.displayName = 'Select';

export default Select;
