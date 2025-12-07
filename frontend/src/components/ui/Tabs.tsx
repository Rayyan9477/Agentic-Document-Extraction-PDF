'use client';

import React, { createContext, useContext, useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (value: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

const useTabsContext = () => {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('Tabs components must be used within a Tabs provider');
  }
  return context;
};

interface TabsProps {
  defaultValue?: string;
  value?: string;
  onChange?: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}

const Tabs: React.FC<TabsProps> = ({
  defaultValue = '',
  value,
  onChange,
  children,
  className,
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const activeTab = value !== undefined ? value : internalValue;

  const setActiveTab = (newValue: string) => {
    if (value === undefined) {
      setInternalValue(newValue);
    }
    onChange?.(newValue);
  };

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className={cn('w-full', className)}>{children}</div>
    </TabsContext.Provider>
  );
};

interface TabsListProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'pills' | 'underlined';
}

const TabsList: React.FC<TabsListProps> = ({
  children,
  className,
  variant = 'default',
}) => {
  const variants = {
    default: 'bg-surface-100 p-1 rounded-xl',
    pills: 'gap-2',
    underlined: 'border-b border-surface-200',
  };

  return (
    <div
      className={cn(
        'flex items-center',
        variants[variant],
        className
      )}
      role="tablist"
    >
      {children}
    </div>
  );
};

interface TabsTriggerProps {
  value: string;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  icon?: React.ReactNode;
}

const TabsTrigger: React.FC<TabsTriggerProps> = ({
  value,
  children,
  className,
  disabled = false,
  icon,
}) => {
  const { activeTab, setActiveTab } = useTabsContext();
  const isActive = activeTab === value;

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      disabled={disabled}
      onClick={() => setActiveTab(value)}
      className={cn(
        'relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg',
        'transition-all duration-200',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        isActive
          ? 'text-primary-700'
          : 'text-surface-500 hover:text-surface-700 hover:bg-surface-50',
        className
      )}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
      {isActive && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-0 bg-white rounded-lg shadow-sm -z-10"
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      )}
    </button>
  );
};

interface TabsContentProps {
  value: string;
  children: React.ReactNode;
  className?: string;
}

const TabsContent: React.FC<TabsContentProps> = ({ value, children, className }) => {
  const { activeTab } = useTabsContext();

  if (activeTab !== value) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      transition={{ duration: 0.2 }}
      role="tabpanel"
      className={cn('mt-4', className)}
    >
      {children}
    </motion.div>
  );
};

export { Tabs, TabsList, TabsTrigger, TabsContent };
export default Tabs;
