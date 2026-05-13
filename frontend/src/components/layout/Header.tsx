'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import {
  FileText,
  Bell,
  User,
  Search,
  Menu,
  Settings,
  LogOut,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button, Dropdown } from '@/components/ui';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BRANDING } from '@/lib/branding';

interface HeaderProps {
  onMenuClick?: () => void;
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  notifications?: number;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick, user, notifications = 0 }) => {
  const pathname = usePathname();

  const getPageTitle = () => {
    const titles: Record<string, string> = {
      '/': 'Dashboard',
      '/dashboard': 'Dashboard',
      '/documents': 'Documents',
      '/documents/upload': 'Upload Document',
      '/tasks': 'Task Queue',
      '/settings': 'Settings',
    };
    return titles[pathname] || BRANDING.productName;
  };

  return (
    <header className="sticky top-0 z-40 bg-surface/80 backdrop-blur-lg border-b border-default">
      <div className="flex items-center justify-between h-16 px-4 lg:px-6">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          {/* Mobile Menu Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            aria-label="Open navigation menu"
            className="lg:hidden"
          >
            <Menu className="h-5 w-5" aria-hidden="true" />
          </Button>

          {/* Logo (Mobile) */}
          <Link
            href="/"
            aria-label={`${BRANDING.productName} home`}
            className="flex items-center gap-2 lg:hidden"
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" aria-hidden="true" />
            </div>
          </Link>

          {/* Page Title */}
          <div className="hidden sm:block">
            <h1 className="text-h2 text-text-primary">
              {getPageTitle()}
            </h1>
          </div>
        </div>

        {/* Center Section - Search */}
        <div className="hidden md:flex flex-1 max-w-md mx-8">
          <div className="relative w-full">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none"
              aria-hidden="true"
            />
            <input
              type="search"
              placeholder="Search documents..."
              aria-label="Search documents"
              className={cn(
                'w-full pl-10 pr-4 py-2 rounded-xl border border-default',
                'bg-canvas text-text-primary placeholder:text-text-muted',
                'focus:outline-none focus:ring-2 focus:ring-accent-brand focus:border-transparent',
                'transition-all duration-base'
              )}
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-1">
          {/* Search (Mobile) */}
          <Button variant="ghost" size="icon" aria-label="Search" className="md:hidden">
            <Search className="h-5 w-5" aria-hidden="true" />
          </Button>

          {/* V3 Phase 8 — Theme toggle (3-state: light/dark/system) */}
          <ThemeToggle />

          {/* Notifications */}
          <div className="relative">
            <Button
              variant="ghost"
              size="icon"
              aria-label={
                notifications > 0
                  ? `${notifications} unread notifications`
                  : 'Notifications'
              }
            >
              <Bell className="h-5 w-5" aria-hidden="true" />
              {notifications > 0 && (
                <span
                  aria-hidden="true"
                  className="absolute top-1 right-1 w-4 h-4 bg-accent-danger text-white text-small rounded-full flex items-center justify-center"
                >
                  {notifications > 9 ? '9+' : notifications}
                </span>
              )}
            </Button>
          </div>

          {/* User Menu */}
          {user ? (
            <Dropdown
              align="right"
              trigger={
                <Button
                  variant="ghost"
                  className="flex items-center gap-2 px-2"
                  aria-label={`Account menu for ${user.name}`}
                >
                  <div className="w-8 h-8 rounded-full bg-accent-brand-soft flex items-center justify-center">
                    {user.avatar ? (
                      <Image
                        src={user.avatar}
                        alt=""
                        width={32}
                        height={32}
                        className="w-8 h-8 rounded-full"
                      />
                    ) : (
                      <User className="w-4 h-4 text-accent-brand" aria-hidden="true" />
                    )}
                  </div>
                  <span className="hidden sm:block text-body text-text-secondary">
                    {user.name}
                  </span>
                  <ChevronDown className="hidden sm:block w-4 h-4 text-text-muted" aria-hidden="true" />
                </Button>
              }
              items={[
                {
                  label: user.email,
                  disabled: true,
                },
                { divider: true, label: '' },
                {
                  label: 'Settings',
                  icon: <Settings className="w-4 h-4" />,
                  onClick: () => {},
                },
                {
                  label: 'Sign out',
                  icon: <LogOut className="w-4 h-4" />,
                  onClick: () => {},
                  danger: true,
                },
              ]}
            />
          ) : (
            <Link href="/login">
              <Button variant="primary" size="sm">
                Sign In
              </Button>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
