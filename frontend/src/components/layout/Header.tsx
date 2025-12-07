'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
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
import { Button, Dropdown, Badge } from '@/components/ui';

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
    return titles[pathname] || 'PDF Extraction';
  };

  return (
    <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-lg border-b border-surface-200">
      <div className="flex items-center justify-between h-16 px-4 lg:px-6">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          {/* Mobile Menu Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            className="lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </Button>

          {/* Logo (Mobile) */}
          <Link href="/" className="flex items-center gap-2 lg:hidden">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
          </Link>

          {/* Page Title */}
          <div className="hidden sm:block">
            <h1 className="text-lg font-semibold text-surface-900">
              {getPageTitle()}
            </h1>
          </div>
        </div>

        {/* Center Section - Search */}
        <div className="hidden md:flex flex-1 max-w-md mx-8">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
            <input
              type="text"
              placeholder="Search documents..."
              className={cn(
                'w-full pl-10 pr-4 py-2 rounded-xl border border-surface-200',
                'bg-surface-50 text-surface-900 placeholder:text-surface-400',
                'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
                'transition-all duration-200'
              )}
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-2">
          {/* Search (Mobile) */}
          <Button variant="ghost" size="icon" className="md:hidden">
            <Search className="h-5 w-5" />
          </Button>

          {/* Notifications */}
          <div className="relative">
            <Button variant="ghost" size="icon">
              <Bell className="h-5 w-5" />
              {notifications > 0 && (
                <span className="absolute top-1 right-1 w-4 h-4 bg-error-500 text-white text-xs rounded-full flex items-center justify-center">
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
                <Button variant="ghost" className="flex items-center gap-2 px-2">
                  <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                    {user.avatar ? (
                      <img
                        src={user.avatar}
                        alt={user.name}
                        className="w-8 h-8 rounded-full"
                      />
                    ) : (
                      <User className="w-4 h-4 text-primary-600" />
                    )}
                  </div>
                  <span className="hidden sm:block text-sm font-medium text-surface-700">
                    {user.name}
                  </span>
                  <ChevronDown className="hidden sm:block w-4 h-4 text-surface-400" />
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
