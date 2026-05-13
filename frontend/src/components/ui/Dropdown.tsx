'use client';

import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
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
  // V3 Phase 8 — full keyboard + ARIA support.
  // * trigger gets ``aria-haspopup="menu"`` and ``aria-expanded``
  // * menu container gets ``role="menu"``
  // * each item gets ``role="menuitem"``
  // * ArrowDown/Up navigates, Home/End jumps, Enter/Space activates,
  //   Escape closes (and returns focus to the trigger).
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number>(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  // Indices of non-divider non-disabled items — these are the
  // keyboard-reachable rows.
  const enabledIndices = items.reduce<number[]>((acc, item, idx) => {
    if (!item.divider && !item.disabled) acc.push(idx);
    return acc;
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
    setActiveIndex(-1);
    triggerRef.current?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setActiveIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleTriggerKey = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    if (['ArrowDown', 'Enter', ' '].includes(e.key)) {
      e.preventDefault();
      setIsOpen(true);
      setActiveIndex(enabledIndices[0] ?? -1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setIsOpen(true);
      setActiveIndex(enabledIndices[enabledIndices.length - 1] ?? -1);
    }
  };

  const handleMenuKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === 'Tab') {
      // Tab closes the menu and lets default tab handling proceed.
      setIsOpen(false);
      setActiveIndex(-1);
      return;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const pos = enabledIndices.indexOf(activeIndex);
      const dir = e.key === 'ArrowDown' ? 1 : -1;
      const nextPos = (pos + dir + enabledIndices.length) % enabledIndices.length;
      setActiveIndex(enabledIndices[nextPos]);
      return;
    }
    if (e.key === 'Home') {
      e.preventDefault();
      setActiveIndex(enabledIndices[0] ?? -1);
      return;
    }
    if (e.key === 'End') {
      e.preventDefault();
      setActiveIndex(enabledIndices[enabledIndices.length - 1] ?? -1);
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (activeIndex >= 0) {
        items[activeIndex].onClick?.();
        close();
      }
    }
  };

  // Auto-focus the active item when it changes.
  useEffect(() => {
    if (!isOpen || activeIndex < 0 || !menuRef.current) return;
    const buttons = menuRef.current.querySelectorAll<HTMLButtonElement>(
      'button[role="menuitem"]',
    );
    const target = Array.from(buttons).find(
      (b) => Number(b.dataset.index) === activeIndex,
    );
    target?.focus({ preventScroll: true });
  }, [activeIndex, isOpen]);

  return (
    <div ref={dropdownRef} className={cn('relative inline-block', className)}>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        // Linter prefers explicit "true"/"false" literals for ARIA booleans
        // because some screen readers historically had quirks with React's
        // boolean coercion. Keep them as strings.
        aria-expanded={isOpen ? 'true' : 'false'}
        aria-controls={isOpen ? menuId : undefined}
        onClick={() => {
          setIsOpen((open) => !open);
          setActiveIndex(enabledIndices[0] ?? -1);
        }}
        onKeyDown={handleTriggerKey}
        className="inline-flex items-center"
      >
        {trigger}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            ref={menuRef}
            id={menuId}
            role="menu"
            aria-orientation="vertical"
            onKeyDown={handleMenuKey}
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
            className={cn(
              'absolute z-50 mt-2 min-w-[180px]',
              'bg-surface-raised text-text-primary border border-default rounded-xl shadow-elev-3',
              'overflow-hidden',
              align === 'right' ? 'right-0' : 'left-0',
            )}
          >
            <div className="py-1">
              {items.map((item, index) =>
                item.divider ? (
                  <div
                    key={index}
                    role="separator"
                    className="my-1 border-t border-default"
                  />
                ) : (
                  <button
                    key={index}
                    type="button"
                    // eslint-disable-next-line jsx-a11y/role-has-required-aria-props -- menu parent is the motion.div above; jsx-a11y can't trace through framer-motion polymorphism
                    role="menuitem"
                    data-index={index}
                    tabIndex={-1}
                    onClick={() => {
                      if (item.disabled) return;
                      item.onClick?.();
                      close();
                    }}
                    onMouseEnter={() => setActiveIndex(index)}
                    disabled={item.disabled}
                    aria-disabled={item.disabled ? 'true' : undefined}
                    className={cn(
                      'w-full flex items-center gap-3 px-4 py-2.5',
                      'text-body text-left transition-colors duration-fast',
                      'focus:outline-none focus-visible:bg-canvas',
                      item.disabled
                        ? 'text-text-muted cursor-not-allowed'
                        : item.danger
                          ? 'text-accent-danger hover:bg-accent-danger/10'
                          : 'text-text-primary hover:bg-canvas',
                    )}
                  >
                    {item.icon && (
                      <span className="flex-shrink-0 w-4 h-4" aria-hidden="true">
                        {item.icon}
                      </span>
                    )}
                    {item.label}
                  </button>
                ),
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Dropdown;
