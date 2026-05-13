/**
 * V3 Phase 8 — Focus trap hook.
 *
 * Captures Tab and Shift+Tab inside a container element while
 * `active=true`, returning focus to the previously-focused element
 * on deactivation. Ships zero dependencies.
 *
 * Usage:
 *   const ref = useRef<HTMLDivElement>(null);
 *   useFocusTrap(ref, isOpen);
 *   <div ref={ref}>...</div>
 */

import { useEffect, type RefObject } from 'react';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

export function useFocusTrap(
  containerRef: RefObject<HTMLElement | null>,
  active: boolean,
): void {
  useEffect(() => {
    if (!active) return;
    const container = containerRef.current;
    if (!container) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;

    // Move focus to first focusable inside the container.
    const focusables = Array.from(
      container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
    );
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (first) {
      first.focus({ preventScroll: true });
    } else {
      // Container itself receives focus so screenreaders announce.
      container.setAttribute('tabindex', '-1');
      container.focus({ preventScroll: true });
    }

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key !== 'Tab' || focusables.length === 0) return;
      const activeEl = document.activeElement as HTMLElement | null;
      if (e.shiftKey) {
        if (activeEl === first || !container?.contains(activeEl)) {
          e.preventDefault();
          last?.focus();
        }
      } else {
        if (activeEl === last || !container?.contains(activeEl)) {
          e.preventDefault();
          first?.focus();
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      // Restore focus to where it was before the trap activated.
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus({ preventScroll: true });
      }
    };
  }, [active, containerRef]);
}
