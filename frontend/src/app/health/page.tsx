'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  Activity,
  Server,
  Database,
  Cpu,
  HardDrive,
  Wifi,
  CheckCircle,
  AlertTriangle,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import { Card, CardHeader, CardContent, Button, Skeleton } from '@/components/ui';
import { healthApi } from '@/lib/api';
import type { ComponentHealth } from '@/types/api';
import { cn, formatDateTime } from '@/lib/utils';

export default function HealthPage() {
  const { data: health, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['health', 'detailed'],
    queryFn: () => healthApi.detailed(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const getStatusIcon = (status: ComponentHealth['status']) => {
    const icons = {
      healthy: <CheckCircle className="w-5 h-5 text-success-500" />,
      degraded: <AlertTriangle className="w-5 h-5 text-warning-500" />,
      unhealthy: <XCircle className="w-5 h-5 text-error-500" />,
    };
    return icons[status];
  };

  const getStatusBg = (status: ComponentHealth['status']) => {
    const colors = {
      healthy: 'bg-success-50 border-success-200',
      degraded: 'bg-warning-50 border-warning-200',
      unhealthy: 'bg-error-50 border-error-200',
    };
    return colors[status];
  };

  const getComponentIcon = (name: string) => {
    const icons: Record<string, React.ReactNode> = {
      api: <Server className="w-6 h-6" />,
      database: <Database className="w-6 h-6" />,
      redis: <HardDrive className="w-6 h-6" />,
      vlm: <Cpu className="w-6 h-6" />,
      celery: <Activity className="w-6 h-6" />,
      network: <Wifi className="w-6 h-6" />,
    };
    return icons[name.toLowerCase()] || <Server className="w-6 h-6" />;
  };

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
            <h1 className="text-2xl font-bold text-surface-900">System Health</h1>
            <p className="text-surface-500 mt-1">
              Monitor the status of all system components
            </p>
          </div>
          <Button
            variant="secondary"
            leftIcon={<RefreshCw className={cn('w-4 h-4', isFetching && 'animate-spin')} />}
            onClick={() => refetch()}
            disabled={isFetching}
          >
            Refresh
          </Button>
        </motion.div>

        {/* Overall Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card
            variant="elevated"
            padding="lg"
            className={cn(
              'border-2',
              health?.status === 'healthy'
                ? 'border-success-200 bg-success-50'
                : 'border-warning-200 bg-warning-50'
            )}
          >
            <div className="flex items-center gap-6">
              <div
                className={cn(
                  'w-20 h-20 rounded-2xl flex items-center justify-center',
                  health?.status === 'healthy' ? 'bg-success-100' : 'bg-warning-100'
                )}
              >
                {health?.status === 'healthy' ? (
                  <CheckCircle className="w-10 h-10 text-success-600" />
                ) : (
                  <AlertTriangle className="w-10 h-10 text-warning-600" />
                )}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-surface-900">
                  {health?.status === 'healthy'
                    ? 'All Systems Operational'
                    : 'System Degraded'}
                </h2>
                <p className="text-surface-600 mt-1">
                  {health?.timestamp
                    ? `Last checked: ${formatDateTime(health.timestamp)}`
                    : 'Checking system status...'}
                </p>
                {health?.version && (
                  <p className="text-sm text-surface-500 mt-1">
                    Version: {health.version}
                  </p>
                )}
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Components Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoading ? (
            [...Array(6)].map((_, i) => (
              <Card key={i} variant="elevated" padding="md">
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <Skeleton variant="circular" width="3rem" height="3rem" />
                    <div className="flex-1 space-y-2">
                      <Skeleton width="60%" />
                      <Skeleton width="40%" height="0.75rem" />
                    </div>
                  </div>
                  <Skeleton variant="rectangular" height="3rem" />
                </div>
              </Card>
            ))
          ) : health?.components ? (
            Object.entries(health.components).map(([name, component], index) => (
              <motion.div
                key={name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
              >
                <Card
                  variant="elevated"
                  padding="md"
                  className={cn('border', getStatusBg(component.status))}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          'w-12 h-12 rounded-xl flex items-center justify-center',
                          component.status === 'healthy'
                            ? 'bg-success-100 text-success-600'
                            : component.status === 'degraded'
                            ? 'bg-warning-100 text-warning-600'
                            : 'bg-error-100 text-error-600'
                        )}
                      >
                        {getComponentIcon(name)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-surface-900 capitalize">
                          {name.replace(/_/g, ' ')}
                        </h3>
                        <span
                          className={cn(
                            'text-xs font-medium capitalize',
                            component.status === 'healthy'
                              ? 'text-success-600'
                              : component.status === 'degraded'
                              ? 'text-warning-600'
                              : 'text-error-600'
                          )}
                        >
                          {component.status}
                        </span>
                      </div>
                    </div>
                    {getStatusIcon(component.status)}
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-3">
                    {component.latency_ms !== undefined && (
                      <div className="p-2 bg-white rounded-lg">
                        <p className="text-xs text-surface-500">Latency</p>
                        <p className="text-sm font-semibold text-surface-900">
                          {component.latency_ms}ms
                        </p>
                      </div>
                    )}
                    {component.error && (
                      <div className="col-span-2 p-2 bg-error-50 rounded-lg">
                        <p className="text-xs text-error-600 font-medium">Error</p>
                        <p className="text-sm text-error-700">{component.error}</p>
                      </div>
                    )}
                  </div>
                </Card>
              </motion.div>
            ))
          ) : (
            <div className="col-span-full text-center py-8 text-surface-500">
              No component data available
            </div>
          )}
        </div>

        {/* System Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Card variant="default" padding="md">
            <CardHeader title="System Information" />
            <CardContent className="mt-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-surface-500">Version</p>
                  <p className="font-medium text-surface-900">
                    {health?.version || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-surface-500">Environment</p>
                  <p className="font-medium text-surface-900">Production</p>
                </div>
                <div>
                  <p className="text-surface-500">Uptime</p>
                  <p className="font-medium text-surface-900">99.9%</p>
                </div>
                <div>
                  <p className="text-surface-500">Last Deployment</p>
                  <p className="font-medium text-surface-900">2 days ago</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </AppLayout>
  );
}
