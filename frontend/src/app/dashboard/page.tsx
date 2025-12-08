'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  FileText,
  CheckCircle,
  AlertTriangle,
  Clock,
  TrendingUp,
  Users,
  Zap,
  BarChart3,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import { MetricCard, RecentActivityList, SystemStatus, ActiveTasks } from '@/components/dashboard';
import { useDashboardData } from '@/hooks/useDashboard';
import { formatPercentage, formatDuration } from '@/lib/utils';

export default function DashboardPage() {
  const { metrics, activity, health, activeTasks, isLoading } = useDashboardData();

  const metricsData = metrics.data;
  const activityData = activity.data;
  const healthData = health.data;
  const tasksData = activeTasks.data;

  return (
    <AppLayout notifications={3}>
      <div className="space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Dashboard</h1>
            <p className="text-surface-500 mt-1">
              Overview of your document processing system
            </p>
          </div>
        </motion.div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Documents Today"
            value={metricsData?.documents_processed_today || 0}
            subtitle="Processed today"
            icon={<FileText className="w-6 h-6" />}
            color="primary"
            loading={isLoading}
          />
          <MetricCard
            title="Success Rate"
            value={metricsData ? formatPercentage(metricsData.success_rate) : '0%'}
            trend={{ value: 2.5, label: 'vs last week' }}
            icon={<CheckCircle className="w-6 h-6" />}
            color="success"
            loading={isLoading}
          />
          <MetricCard
            title="Avg. Processing Time"
            value={metricsData ? formatDuration(metricsData.average_processing_time) : '0s'}
            subtitle="Per document"
            icon={<Zap className="w-6 h-6" />}
            color="info"
            loading={isLoading}
          />
          <MetricCard
            title="Pending Review"
            value={metricsData?.human_review_pending || 0}
            subtitle="Documents need review"
            icon={<AlertTriangle className="w-6 h-6" />}
            color="warning"
            loading={isLoading}
          />
        </div>

        {/* Secondary Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Weekly Total"
            value={metricsData?.documents_processed_week || 0}
            subtitle="Documents this week"
            icon={<BarChart3 className="w-6 h-6" />}
            color="primary"
            loading={isLoading}
          />
          <MetricCard
            title="Active Tasks"
            value={metricsData?.active_tasks || 0}
            subtitle="Currently processing"
            icon={<Clock className="w-6 h-6" />}
            color="info"
            loading={isLoading}
          />
          <MetricCard
            title="Queue Pending"
            value={metricsData?.pending_tasks || 0}
            subtitle="Waiting in queue"
            icon={<Users className="w-6 h-6" />}
            color="primary"
            loading={isLoading}
          />
          <MetricCard
            title="Failed Today"
            value={metricsData?.failed_tasks_today || 0}
            subtitle="Require attention"
            icon={<AlertTriangle className="w-6 h-6" />}
            color="error"
            loading={isLoading}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Active Tasks - 2 columns */}
          <div className="lg:col-span-2">
            <ActiveTasks tasks={tasksData} loading={activeTasks.isLoading} />
          </div>

          {/* System Status - 1 column */}
          <div>
            <SystemStatus health={healthData} loading={health.isLoading} />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <RecentActivityList activities={activityData} loading={activity.isLoading} />

          {/* Quick Stats Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-primary-600 to-primary-800 rounded-2xl p-6 text-white"
          >
            <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 gap-4">
              <a
                href="/documents/upload"
                className="flex flex-col items-center justify-center p-4 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
              >
                <FileText className="w-8 h-8 mb-2" />
                <span className="text-sm font-medium">Upload Document</span>
              </a>
              <a
                href="/tasks"
                className="flex flex-col items-center justify-center p-4 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
              >
                <Clock className="w-8 h-8 mb-2" />
                <span className="text-sm font-medium">View Queue</span>
              </a>
              <a
                href="/documents"
                className="flex flex-col items-center justify-center p-4 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
              >
                <BarChart3 className="w-8 h-8 mb-2" />
                <span className="text-sm font-medium">Browse Results</span>
              </a>
              <a
                href="/settings"
                className="flex flex-col items-center justify-center p-4 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
              >
                <TrendingUp className="w-8 h-8 mb-2" />
                <span className="text-sm font-medium">Analytics</span>
              </a>
            </div>
          </motion.div>
        </div>
      </div>
    </AppLayout>
  );
}
