'use client';

import React from 'react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

interface FooterProps {
  className?: string;
}

const Footer: React.FC<FooterProps> = ({ className }) => {
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className={cn(
        'bg-white border-t border-surface-200 py-4 px-6',
        className
      )}
    >
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="text-sm text-surface-500">
          &copy; {currentYear} PDF Document Extraction System. All rights reserved.
        </div>
        <div className="flex items-center gap-6">
          <Link
            href="/privacy"
            className="text-sm text-surface-500 hover:text-surface-700 transition-colors"
          >
            Privacy Policy
          </Link>
          <Link
            href="/terms"
            className="text-sm text-surface-500 hover:text-surface-700 transition-colors"
          >
            Terms of Service
          </Link>
          <Link
            href="/docs"
            className="text-sm text-surface-500 hover:text-surface-700 transition-colors"
          >
            Documentation
          </Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
