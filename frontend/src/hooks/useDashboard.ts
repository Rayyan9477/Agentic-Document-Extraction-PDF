'use client';

import { useQuery } from '@tanstack/react-query';
import { dashboardApi, healthApi, tasksApi, queueApi } from '@/lib/api';
import type { DashboardMetrics, RecentActivity, HealthResponse, TaskStatusResponse, QueueStats } from '@/types/api';

// Dashboard Metrics Hook
export function useDashboardMetrics() {
  return useQuery<DashboardMetrics>({
    queryKey: ['dashboard', 'metrics'],
    queryFn: () => dashboardApi.getMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

// Recent Activity Hook
export function useRecentActivity(limit: number = 10) {
  return useQuery<RecentActivity[]>({
    queryKey: ['dashboard', 'activity', limit],
    queryFn: () => dashboardApi.getActivity(limit),
    refetchInterval: 15000, // Refresh every 15 seconds
  });
}

// Health Status Hook
export function useHealthStatus() {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: () => healthApi.detailed(),
    refetchInterval: 60000, // Refresh every minute
  });
}

// Active Tasks Hook
export function useActiveTasks() {
  return useQuery<TaskStatusResponse[]>({
    queryKey: ['tasks', 'active'],
    queryFn: () => tasksApi.listActive(),
    refetchInterval: 5000, // Refresh every 5 seconds for real-time feel
  });
}

// Queue Stats Hook
export function useQueueStats() {
  return useQuery<QueueStats[]>({
    queryKey: ['queue', 'stats'],
    queryFn: () => queueApi.getStats(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });
}

// Combined Dashboard Data Hook
export function useDashboardData() {
  const metrics = useDashboardMetrics();
  const activity = useRecentActivity();
  const health = useHealthStatus();
  const activeTasks = useActiveTasks();
  const queueStats = useQueueStats();

  return {
    metrics,
    activity,
    health,
    activeTasks,
    queueStats,
    isLoading: metrics.isLoading || activity.isLoading || health.isLoading,
    isError: metrics.isError || activity.isError || health.isError,
  };
}
