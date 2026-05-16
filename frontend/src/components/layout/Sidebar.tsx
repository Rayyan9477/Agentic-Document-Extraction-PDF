'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  FileText,
  Upload,
  ListTodo,
  Settings,
  Activity,
  Database,
  Shield,
  HelpCircle,
  ChevronLeft,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button, Badge } from '@/components/ui';
import { BRANDING } from '@/lib/branding';

interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  badge?: string | number;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  {
    label: 'Dashboard',
    href: '/dashboard',
    icon: <LayoutDashboard className="w-5 h-5" />,
  },
  {
    label: 'Documents',
    href: '/documents',
    icon: <FileText className="w-5 h-5" />,
  },
  {
    label: 'Upload',
    href: '/documents/upload',
    icon: <Upload className="w-5 h-5" />,
  },
  {
    label: 'Task Queue',
    href: '/tasks',
    icon: <ListTodo className="w-5 h-5" />,
  },
  {
    label: 'Health',
    href: '/health',
    icon: <Activity className="w-5 h-5" />,
  },
  {
    label: 'Schemas',
    href: '/schemas',
    icon: <Database className="w-5 h-5" />,
  },
];

const bottomNavItems: NavItem[] = [
  {
    label: 'Settings',
    href: '/settings',
    icon: <Settings className="w-5 h-5" />,
  },
  {
    label: 'Security',
    href: '/security',
    icon: <Shield className="w-5 h-5" />,
  },
  {
    label: 'Help',
    href: '/help',
    icon: <HelpCircle className="w-5 h-5" />,
  },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  isCollapsed = false,
  onToggleCollapse,
}) => {
  const pathname = usePathname();

  const NavLink: React.FC<{ item: NavItem }> = ({ item }) => {
    const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

    return (
      <Link
        href={item.href}
        onClick={() => onClose()}
        // V3 Phase 8 — aria-current marks the active route for AT.
        aria-current={isActive ? 'page' : undefined}
        title={isCollapsed ? item.label : undefined}
        className={cn(
          'flex items-center gap-3 px-3 py-2.5 rounded-xl',
          'transition-all duration-base',
          'group relative',
          isActive
            ? 'bg-accent-brand-soft text-accent-brand'
            : 'text-text-secondary hover:bg-surface hover:text-text-primary'
        )}
      >
        <span
          className={cn(
            'flex-shrink-0 transition-colors',
            isActive ? 'text-accent-brand' : 'text-text-muted group-hover:text-text-secondary'
          )}
          aria-hidden="true"
        >
          {item.icon}
        </span>
        {!isCollapsed && (
          <>
            <span className="flex-1 font-medium text-body">{item.label}</span>
            {item.badge && (
              <Badge size="sm" variant={isActive ? 'primary' : 'default'}>
                {item.badge}
              </Badge>
            )}
          </>
        )}
        {isActive && (
          <motion.div
            layoutId="activeNav"
            className="absolute left-0 w-1 h-8 bg-accent-brand rounded-r-full"
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            aria-hidden="true"
          />
        )}
      </Link>
    );
  };

  const sidebarContent = (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-default">
        <Link
          href="/"
          aria-label={`${BRANDING.productName} home`}
          className="flex items-center gap-3"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-elev-1">
            <FileText className="w-6 h-6 text-white" aria-hidden="true" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col">
              <span className="text-h3 font-bold text-text-primary">{BRANDING.productName}</span>
              <span className="text-small text-text-muted">{BRANDING.versionLabel}</span>
            </div>
          )}
        </Link>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close navigation"
          className="lg:hidden"
        >
          <X className="w-5 h-5" aria-hidden="true" />
        </Button>
      </div>

      {/* Navigation */}
      <nav
        aria-label="Primary navigation"
        className="flex-1 overflow-y-auto p-4 space-y-1"
      >
        {navItems.map((item) => (
          <NavLink key={item.href} item={item} />
        ))}
      </nav>

      {/* Bottom Navigation */}
      <nav
        aria-label="Secondary navigation"
        className="p-4 border-t border-default space-y-1"
      >
        {bottomNavItems.map((item) => (
          <NavLink key={item.href} item={item} />
        ))}
      </nav>

      {/* Collapse Toggle (Desktop) */}
      {onToggleCollapse && (
        <div className="hidden lg:block p-4 border-t border-surface-100">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleCollapse}
            className="w-full justify-center"
          >
            <ChevronLeft
              className={cn(
                'w-5 h-5 transition-transform',
                isCollapsed && 'rotate-180'
              )}
            />
          </Button>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isOpen && (
          <motion.aside
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 left-0 w-72 bg-surface dark:bg-surface-raised border-r border-surface-200 z-50 lg:hidden"
          >
            {sidebarContent}
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <aside
        className={cn(
          'hidden lg:flex flex-col bg-surface dark:bg-surface-raised border-r border-surface-200',
          'transition-all duration-300',
          isCollapsed ? 'w-20' : 'w-72'
        )}
      >
        {sidebarContent}
      </aside>
    </>
  );
};

export default Sidebar;
