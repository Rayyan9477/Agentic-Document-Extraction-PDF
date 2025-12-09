'use client';

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  RefreshCw,
  X,
  RotateCcw,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Server,
  Database,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import {
  Card,
  CardHeader,
  CardContent,
  Button,
  StatusBadge,
  Progress,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  ConfirmModal,
  Skeleton,
} from '@/components/ui';
import { tasksApi, queueApi } from '@/lib/api';
import { formatRelativeTime, formatDuration } from '@/lib/utils';
import type { TaskStatusResponse } from '@/types/api';

export default function TasksPage() {
  const queryClient = useQueryClient();
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [showCancelModal, setShowCancelModal] = useState(false);

  // Fetch tasks
  const { data: tasksData, isLoading: tasksLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => tasksApi.list({ limit: 50 }),
    refetchInterval: 5000,
  });

  // Fetch queue stats
  const { data: queueStats, isLoading: queueLoading } = useQuery({
    queryKey: ['queue', 'stats'],
    queryFn: () => queueApi.getStats(),
    refetchInterval: 10000,
  });

  // Fetch workers
  const { data: workers, isLoading: workersLoading } = useQuery({
    queryKey: ['queue', 'workers'],
    queryFn: () => queueApi.getWorkers(),
    refetchInterval: 10000,
  });

  // Cancel task mutation
  const cancelMutation = useMutation({
    mutationFn: (taskId: string) => tasksApi.cancel(taskId),
    onSuccess: () => {
      toast.success('Task cancelled');
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      setShowCancelModal(false);
      setSelectedTask(null);
    },
    onError: () => {
      toast.error('Failed to cancel task');
    },
  });

  // Retry task mutation
  const retryMutation = useMutation({
    mutationFn: (taskId: string) => tasksApi.retry(taskId),
    onSuccess: () => {
      toast.success('Task queued for retry');
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
    onError: () => {
      toast.error('Failed to retry task');
    },
  });

  const tasks: TaskStatusResponse[] = tasksData || [];
  const activeTasks = tasks.filter((t: TaskStatusResponse) =>
    ['started', 'processing', 'validating', 'exporting'].includes(t.status)
  );
  const pendingTasks = tasks.filter((t: TaskStatusResponse) => t.status === 'pending');
  const completedTasks = tasks.filter((t: TaskStatusResponse) => t.status === 'completed');
  const failedTasks = tasks.filter((t: TaskStatusResponse) => ['failed', 'cancelled'].includes(t.status));

  const TaskRow: React.FC<{ task: TaskStatusResponse; index: number }> = ({
    task,
    index,
  }) => (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03 }}
      className="p-4 border border-surface-200 rounded-xl hover:border-primary-200 transition-colors"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-primary-100 flex items-center justify-center flex-shrink-0">
            {task.status === 'processing' ? (
              <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
            ) : task.status === 'completed' ? (
              <CheckCircle className="w-5 h-5 text-success-600" />
            ) : task.status === 'failed' ? (
              <AlertCircle className="w-5 h-5 text-error-600" />
            ) : (
              <Clock className="w-5 h-5 text-surface-400" />
            )}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-surface-900">
              Task #{task.task_id.slice(0, 12)}...
            </p>
            <p className="text-xs text-surface-500">
              {task.progress?.stage || task.status}
            </p>
            {task.error && (
              <p className="text-xs text-error-600 mt-1">{task.error}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={task.status} size="sm" />
          {task.status === 'failed' && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => retryMutation.mutate(task.task_id)}
              loading={retryMutation.isPending}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          )}
          {['pending', 'processing'].includes(task.status) && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                setSelectedTask(task.task_id);
                setShowCancelModal(true);
              }}
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      {task.progress && (
        <div className="mt-3">
          <Progress
            value={task.progress.current}
            max={task.progress.total}
            size="sm"
            color={task.status === 'failed' ? 'error' : 'primary'}
          />
          <div className="flex items-center justify-between mt-1">
            <span className="text-xs text-surface-500">
              {task.progress.current} of {task.progress.total}
            </span>
            <span className="text-xs text-surface-500">
              {task.progress.stage}
            </span>
          </div>
        </div>
      )}
    </motion.div>
  );

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Task Queue</h1>
            <p className="text-surface-500 mt-1">
              Monitor and manage processing tasks
            </p>
          </div>
          <Button
            variant="secondary"
            leftIcon={<RefreshCw className="w-4 h-4" />}
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ['tasks'] });
              queryClient.invalidateQueries({ queryKey: ['queue'] });
            }}
          >
            Refresh
          </Button>
        </motion.div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card variant="elevated" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {activeTasks.length}
                </p>
                <p className="text-sm text-surface-500">Active</p>
              </div>
            </div>
          </Card>
          <Card variant="elevated" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-warning-100 flex items-center justify-center">
                <Clock className="w-5 h-5 text-warning-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {pendingTasks.length}
                </p>
                <p className="text-sm text-surface-500">Pending</p>
              </div>
            </div>
          </Card>
          <Card variant="elevated" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-success-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {completedTasks.length}
                </p>
                <p className="text-sm text-surface-500">Completed</p>
              </div>
            </div>
          </Card>
          <Card variant="elevated" padding="md">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-error-100 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-error-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-surface-900">
                  {failedTasks.length}
                </p>
                <p className="text-sm text-surface-500">Failed</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Tasks List */}
          <div className="lg:col-span-2">
            <Card variant="elevated" padding="none">
              <Tabs defaultValue="active">
                <div className="px-4 pt-4">
                  <TabsList>
                    <TabsTrigger value="active">
                      Active ({activeTasks.length})
                    </TabsTrigger>
                    <TabsTrigger value="pending">
                      Pending ({pendingTasks.length})
                    </TabsTrigger>
                    <TabsTrigger value="completed">
                      Completed ({completedTasks.length})
                    </TabsTrigger>
                    <TabsTrigger value="failed">
                      Failed ({failedTasks.length})
                    </TabsTrigger>
                  </TabsList>
                </div>

                <div className="p-4">
                  <TabsContent value="active">
                    {tasksLoading ? (
                      <div className="space-y-3">
                        {[...Array(3)].map((_, i) => (
                          <Skeleton key={i} variant="rectangular" height="5rem" />
                        ))}
                      </div>
                    ) : activeTasks.length === 0 ? (
                      <div className="text-center py-8 text-surface-500">
                        No active tasks
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {activeTasks.map((task: TaskStatusResponse, index: number) => (
                          <TaskRow key={task.task_id} task={task} index={index} />
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="pending">
                    {pendingTasks.length === 0 ? (
                      <div className="text-center py-8 text-surface-500">
                        No pending tasks
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {pendingTasks.map((task: TaskStatusResponse, index: number) => (
                          <TaskRow key={task.task_id} task={task} index={index} />
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="completed">
                    {completedTasks.length === 0 ? (
                      <div className="text-center py-8 text-surface-500">
                        No completed tasks
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {completedTasks.map((task: TaskStatusResponse, index: number) => (
                          <TaskRow key={task.task_id} task={task} index={index} />
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="failed">
                    {failedTasks.length === 0 ? (
                      <div className="text-center py-8 text-surface-500">
                        No failed tasks
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {failedTasks.map((task: TaskStatusResponse, index: number) => (
                          <TaskRow key={task.task_id} task={task} index={index} />
                        ))}
                      </div>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            </Card>
          </div>

          {/* Sidebar - Queue & Workers */}
          <div className="space-y-6">
            {/* Queue Stats */}
            <Card variant="elevated" padding="md">
              <CardHeader title="Queue Status" />
              <CardContent className="mt-4 space-y-3">
                {queueLoading ? (
                  [...Array(2)].map((_, i) => (
                    <Skeleton key={i} variant="rectangular" height="4rem" />
                  ))
                ) : queueStats && queueStats.length > 0 ? (
                  queueStats.map((queue) => (
                    <div
                      key={queue.name}
                      className="p-3 bg-surface-50 rounded-xl"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-surface-700 capitalize">
                          {queue.name}
                        </span>
                        <Database className="w-4 h-4 text-surface-400" />
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div>
                          <p className="text-lg font-bold text-surface-900">
                            {queue.pending}
                          </p>
                          <p className="text-xs text-surface-500">Pending</p>
                        </div>
                        <div>
                          <p className="text-lg font-bold text-primary-600">
                            {queue.active}
                          </p>
                          <p className="text-xs text-surface-500">Active</p>
                        </div>
                        <div>
                          <p className="text-lg font-bold text-surface-600">
                            {queue.reserved}
                          </p>
                          <p className="text-xs text-surface-500">Reserved</p>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-surface-500">
                    No queue data available
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Workers */}
            <Card variant="elevated" padding="md">
              <CardHeader title="Workers" />
              <CardContent className="mt-4 space-y-3">
                {workersLoading ? (
                  [...Array(2)].map((_, i) => (
                    <Skeleton key={i} variant="rectangular" height="4rem" />
                  ))
                ) : workers && workers.length > 0 ? (
                  workers.map((worker) => (
                    <div
                      key={worker.name}
                      className="p-3 bg-surface-50 rounded-xl"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Server className="w-4 h-4 text-surface-400" />
                          <span className="text-sm font-medium text-surface-700">
                            {worker.name}
                          </span>
                        </div>
                        <span
                          className={`text-xs px-2 py-1 rounded-full ${
                            worker.status === 'online'
                              ? 'bg-success-100 text-success-700'
                              : 'bg-error-100 text-error-700'
                          }`}
                        >
                          {worker.status}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-xs">
                        <div>
                          <p className="font-semibold text-surface-900">
                            {worker.active_tasks}
                          </p>
                          <p className="text-surface-500">Active</p>
                        </div>
                        <div>
                          <p className="font-semibold text-success-600">
                            {worker.processed}
                          </p>
                          <p className="text-surface-500">Processed</p>
                        </div>
                        <div>
                          <p className="font-semibold text-error-600">
                            {worker.failed}
                          </p>
                          <p className="text-surface-500">Failed</p>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-surface-500">
                    No workers online
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Cancel Confirmation Modal */}
      <ConfirmModal
        isOpen={showCancelModal}
        onClose={() => {
          setShowCancelModal(false);
          setSelectedTask(null);
        }}
        onConfirm={() => selectedTask && cancelMutation.mutate(selectedTask)}
        title="Cancel Task"
        message="Are you sure you want to cancel this task? This action cannot be undone."
        confirmText="Cancel Task"
        variant="danger"
        loading={cancelMutation.isPending}
      />
    </AppLayout>
  );
}
