'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, Eye, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { cn, formatRelativeTime } from '@/lib/utils';
import { Card, CardHeader, CardContent, StatusBadge, Skeleton } from '@/components/ui';
import type { RecentActivity } from '@/types/api';

interface RecentActivityListProps {
  activities?: RecentActivity[];
  loading?: boolean;
}

const RecentActivityList: React.FC<RecentActivityListProps> = ({
  activities = [],
  loading = false,
}) => {
  const getActivityIcon = (type: RecentActivity['type']) => {
    const icons = {
      process: <FileText className="w-4 h-4" />,
      export: <Download className="w-4 h-4" />,
      review: <Eye className="w-4 h-4" />,
      error: <AlertCircle className="w-4 h-4" />,
    };
    return icons[type];
  };

  const getActivityColor = (type: RecentActivity['type']) => {
    const colors = {
      process: 'bg-primary-100 text-primary-600',
      export: 'bg-info-100 text-info-600',
      review: 'bg-warning-100 text-warning-600',
      error: 'bg-error-100 text-error-600',
    };
    return colors[type];
  };

  if (loading) {
    return (
      <Card variant="elevated" padding="md">
        <CardHeader title="Recent Activity" />
        <CardContent className="mt-4 space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-start gap-3">
              <Skeleton variant="circular" />
              <div className="flex-1 space-y-2">
                <Skeleton width="70%" height="1rem" />
                <Skeleton width="40%" height="0.75rem" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="elevated" padding="md">
      <CardHeader title="Recent Activity" description="Latest document processing activity" />
      <CardContent className="mt-4">
        {activities.length === 0 ? (
          <div className="text-center py-8 text-surface-500">
            <Clock className="w-8 h-8 mx-auto mb-2 text-surface-400" />
            <p>No recent activity</p>
          </div>
        ) : (
          <div className="space-y-4">
            {activities.map((activity, index) => (
              <motion.div
                key={activity.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-start gap-3 p-3 rounded-xl hover:bg-surface-50 transition-colors"
              >
                <div
                  className={cn(
                    'w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0',
                    getActivityColor(activity.type)
                  )}
                >
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-surface-900 truncate">
                    {activity.description}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    {activity.document_name && (
                      <span className="text-xs text-surface-500 truncate max-w-[150px]">
                        {activity.document_name}
                      </span>
                    )}
                    <span className="text-xs text-surface-400">
                      {formatRelativeTime(activity.timestamp)}
                    </span>
                  </div>
                </div>
                <StatusBadge status={activity.status} size="sm" />
              </motion.div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentActivityList;
