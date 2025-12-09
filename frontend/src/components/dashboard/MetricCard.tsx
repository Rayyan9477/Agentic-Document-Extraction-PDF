'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card } from '@/components/ui';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label?: string;
  };
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info';
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  color = 'primary',
  loading = false,
}) => {
  const iconBgClasses = {
    primary: 'bg-primary-100 text-primary-600',
    success: 'bg-success-100 text-success-600',
    warning: 'bg-warning-100 text-warning-600',
    error: 'bg-error-100 text-error-600',
    info: 'bg-info-100 text-info-600',
  };

  if (loading) {
    return (
      <Card variant="elevated" padding="md" className="animate-pulse">
        <div className="flex items-start justify-between">
          <div className="space-y-3">
            <div className="h-4 w-24 bg-surface-200 rounded" />
            <div className="h-8 w-32 bg-surface-200 rounded" />
            <div className="h-3 w-20 bg-surface-200 rounded" />
          </div>
          <div className="w-12 h-12 bg-surface-200 rounded-xl" />
        </div>
      </Card>
    );
  }

  const getTrendIcon = () => {
    if (!trend) return null;
    if (trend.value > 0) return <TrendingUp className="w-4 h-4" />;
    if (trend.value < 0) return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getTrendColor = () => {
    if (!trend) return '';
    if (trend.value > 0) return 'text-success-600';
    if (trend.value < 0) return 'text-error-600';
    return 'text-surface-500';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card variant="elevated" padding="md" hover>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-surface-500">{title}</p>
            <p className="text-3xl font-bold text-surface-900">{value}</p>
            <div className="flex items-center gap-2">
              {trend && (
                <span className={cn('flex items-center gap-1 text-sm font-medium', getTrendColor())}>
                  {getTrendIcon()}
                  {Math.abs(trend.value)}%
                  {trend.label && (
                    <span className="text-surface-400 font-normal">{trend.label}</span>
                  )}
                </span>
              )}
              {subtitle && !trend && (
                <span className="text-sm text-surface-500">{subtitle}</span>
              )}
            </div>
          </div>
          {icon && (
            <div className={cn('w-12 h-12 rounded-xl flex items-center justify-center', iconBgClasses[color])}>
              {icon}
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  );
};

export default MetricCard;
