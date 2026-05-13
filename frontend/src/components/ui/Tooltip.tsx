'use client';

import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
  align?: 'start' | 'center' | 'end';
  delay?: number;
  className?: string;
}

const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  side = 'top',
  align = 'center',
  delay = 200,
  className,
}) => {
  // V3 Phase 8 — proper ARIA tooltip semantics:
  // * trigger gets aria-describedby pointing at the bubble's id
  // * bubble gets role="tooltip"
  // * hover/focus parity (was already correct in the legacy file)
  // * Escape key dismisses the tooltip while focused (keyboard a11y)
  const tooltipId = useId();
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const updatePosition = () => {
    if (!triggerRef.current) return;

    const rect = triggerRef.current.getBoundingClientRect();
    const offset = 8;

    let top = 0;
    let left = 0;

    switch (side) {
      case 'top':
        top = rect.top - offset;
        break;
      case 'bottom':
        top = rect.bottom + offset;
        break;
      case 'left':
        left = rect.left - offset;
        break;
      case 'right':
        left = rect.right + offset;
        break;
    }

    if (side === 'top' || side === 'bottom') {
      switch (align) {
        case 'start':
          left = rect.left;
          break;
        case 'center':
          left = rect.left + rect.width / 2;
          break;
        case 'end':
          left = rect.right;
          break;
      }
    } else {
      switch (align) {
        case 'start':
          top = rect.top;
          break;
        case 'center':
          top = rect.top + rect.height / 2;
          break;
        case 'end':
          top = rect.bottom;
          break;
      }
    }

    setPosition({ top, left });
  };

  const showTooltip = () => {
    timeoutRef.current = setTimeout(() => {
      updatePosition();
      setIsVisible(true);
    }, delay);
  };

  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const getTransformOrigin = () => {
    const origins: Record<string, Record<string, string>> = {
      top: { start: 'bottom left', center: 'bottom center', end: 'bottom right' },
      bottom: { start: 'top left', center: 'top center', end: 'top right' },
      left: { start: 'right top', center: 'right center', end: 'right bottom' },
      right: { start: 'left top', center: 'left center', end: 'left bottom' },
    };
    return origins[side][align];
  };

  const getTranslate = () => {
    const translates: Record<string, Record<string, string>> = {
      top: { start: '0, -100%', center: '-50%, -100%', end: '-100%, -100%' },
      bottom: { start: '0, 0', center: '-50%, 0', end: '-100%, 0' },
      left: { start: '-100%, 0', center: '-100%, -50%', end: '-100%, -100%' },
      right: { start: '0, 0', center: '0, -50%', end: '0, -100%' },
    };
    return `translate(${translates[side][align]})`;
  };

  const tooltipContent = (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          id={tooltipId}
          role="tooltip"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
          style={{
            position: 'fixed',
            top: position.top,
            left: position.left,
            transform: getTranslate(),
            transformOrigin: getTransformOrigin(),
            zIndex: 9999,
          }}
          className={cn(
            // V3 Phase 8 — semantic tokens so dark mode flips.
            'px-3 py-1.5 text-small rounded-lg shadow-elev-3',
            'bg-surface-raised text-text-primary border border-default',
            'max-w-xs break-words',
            className,
          )}
        >
          {content}
        </motion.div>
      )}
    </AnimatePresence>
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape' && isVisible) {
      e.preventDefault();
      hideTooltip();
    }
  };

  return (
    <>
      <div
        ref={triggerRef}
        // V3 Phase 8 — link trigger to bubble for screen readers.
        aria-describedby={isVisible ? tooltipId : undefined}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
        onKeyDown={handleKeyDown}
        className="inline-flex"
      >
        {children}
      </div>
      {typeof window !== 'undefined' && createPortal(tooltipContent, document.body)}
    </>
  );
};

export default Tooltip;
