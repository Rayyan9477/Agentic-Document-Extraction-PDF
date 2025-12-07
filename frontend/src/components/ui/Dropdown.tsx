'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface DropdownItem {
  label: string;
  icon?: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  danger?: boolean;
  divider?: boolean;
}

interface DropdownProps {
  trigger: React.ReactNode;
  items: DropdownItem[];
  align?: 'left' | 'right';
  className?: string;
}

const Dropdown: React.FC<DropdownProps> = ({
  trigger,
  items,
  align = 'left',
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={dropdownRef} className={cn('relative inline-block', className)}>
      <div onClick={() => setIsOpen(!isOpen)}>{trigger}</div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={cn(
              'absolute z-50 mt-2 min-w-[180px]',
              'bg-white border border-surface-200 rounded-xl shadow-lg',
              'overflow-hidden',
              align === 'right' ? 'right-0' : 'left-0'
            )}
          >
            <div className="py-1">
              {items.map((item, index) =>
                item.divider ? (
                  <div key={index} className="my-1 border-t border-surface-100" />
                ) : (
                  <button
                    key={index}
                    onClick={() => {
                      item.onClick?.();
                      setIsOpen(false);
                    }}
                    disabled={item.disabled}
                    className={cn(
                      'w-full flex items-center gap-3 px-4 py-2.5',
                      'text-sm text-left transition-colors duration-150',
                      item.disabled
                        ? 'text-surface-400 cursor-not-allowed'
                        : item.danger
                        ? 'text-error-600 hover:bg-error-50'
                        : 'text-surface-700 hover:bg-surface-50'
                    )}
                  >
                    {item.icon && (
                      <span className="flex-shrink-0 w-4 h-4">{item.icon}</span>
                    )}
                    {item.label}
                  </button>
                )
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Dropdown;
