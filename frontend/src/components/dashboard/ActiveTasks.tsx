'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { FileText, ArrowRight, Clock, Loader2 } from 'lucide-react';
import { cn, formatRelativeTime } from '@/lib/utils';
import { Card, CardHeader, CardContent, Button, StatusBadge, Progress, Skeleton } from '@/components/ui';
import type { TaskStatusResponse } from '@/types/api';

interface ActiveTasksProps {
  tasks?: TaskStatusResponse[];
  loading?: boolean;
}

const ActiveTasks: React.FC<ActiveTasksProps> = ({ tasks = [], loading = false }) => {
  if (loading) {
    return (
      <Card variant="elevated" padding="md">
        <CardHeader title="Active Tasks" />
        <CardContent className="mt-4 space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="p-4 border rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton width="60%" height="1rem" />
                <Skeleton width="5rem" height="1.5rem" />
              </div>
              <Skeleton variant="rectangular" height="0.5rem" />
              <Skeleton width="40%" height="0.75rem" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="elevated" padding="md">
      <CardHeader
        title="Active Tasks"
        description={`${tasks.length} task${tasks.length !== 1 ? 's' : ''} in progress`}
        action={
          <Link href="/tasks">
            <Button variant="ghost" size="sm" rightIcon={<ArrowRight className="w-4 h-4" />}>
              View All
            </Button>
          </Link>
        }
      />
      <CardContent className="mt-4">
        {tasks.length === 0 ? (
          <div className="text-center py-8 text-surface-500">
            <Clock className="w-8 h-8 mx-auto mb-2 text-surface-400" />
            <p>No active tasks</p>
            <Link href="/documents/upload">
              <Button variant="primary" size="sm" className="mt-4">
                Upload Document
              </Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {tasks.slice(0, 5).map((task, index) => (
              <motion.div
                key={task.task_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="p-4 border border-surface-200 rounded-xl hover:border-primary-200 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3 min-w-0">
                    <div className="w-10 h-10 rounded-xl bg-primary-100 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-primary-600" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-surface-900 truncate">
                        Task #{task.task_id.slice(0, 8)}...
                      </p>
                      <p className="text-xs text-surface-500">
                        {task.progress?.stage || 'Processing'}
                      </p>
                    </div>
                  </div>
                  <StatusBadge status={task.status as any} size="sm" />
                </div>

                {task.progress && (
                  <div className="mt-3">
                    <Progress
                      value={task.progress.current}
                      max={task.progress.total}
                      size="sm"
                      color="primary"
                    />
                    <p className="mt-1 text-xs text-surface-500">
                      {task.progress.current} of {task.progress.total} completed
                    </p>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ActiveTasks;
