'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Database, Server, Cpu, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardHeader, CardContent, Skeleton } from '@/components/ui';
import type { HealthResponse, ComponentHealth } from '@/types/api';

interface SystemStatusProps {
  health?: HealthResponse;
  loading?: boolean;
}

const SystemStatus: React.FC<SystemStatusProps> = ({ health, loading = false }) => {
  const getStatusIcon = (status: ComponentHealth['status']) => {
    const icons = {
      healthy: <CheckCircle className="w-4 h-4 text-success-500" />,
      degraded: <AlertTriangle className="w-4 h-4 text-warning-500" />,
      unhealthy: <XCircle className="w-4 h-4 text-error-500" />,
    };
    return icons[status];
  };

  const getStatusColor = (status: ComponentHealth['status']) => {
    const colors = {
      healthy: 'bg-success-50 border-success-200',
      degraded: 'bg-warning-50 border-warning-200',
      unhealthy: 'bg-error-50 border-error-200',
    };
    return colors[status];
  };

  const getComponentIcon = (name: string) => {
    const icons: Record<string, React.ReactNode> = {
      api: <Server className="w-4 h-4" />,
      database: <Database className="w-4 h-4" />,
      redis: <Database className="w-4 h-4" />,
      vlm: <Cpu className="w-4 h-4" />,
      celery: <Activity className="w-4 h-4" />,
    };
    return icons[name.toLowerCase()] || <Activity className="w-4 h-4" />;
  };

  if (loading) {
    return (
      <Card variant="elevated" padding="md">
        <CardHeader title="System Status" />
        <CardContent className="mt-4 space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="flex items-center justify-between p-3 border rounded-xl">
              <div className="flex items-center gap-3">
                <Skeleton variant="circular" width="2rem" height="2rem" />
                <Skeleton width="6rem" height="1rem" />
              </div>
              <Skeleton width="4rem" height="1rem" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  const overallStatus = health?.status || 'healthy';
  const components = health?.components || {};

  return (
    <Card variant="elevated" padding="md">
      <CardHeader
        title="System Status"
        action={
          <div
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium',
              overallStatus === 'healthy'
                ? 'bg-success-100 text-success-700'
                : 'bg-warning-100 text-warning-700'
            )}
          >
            {overallStatus === 'healthy' ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <AlertTriangle className="w-4 h-4" />
            )}
            {overallStatus === 'healthy' ? 'All Systems Operational' : 'Degraded Performance'}
          </div>
        }
      />
      <CardContent className="mt-4">
        <div className="space-y-3">
          {Object.entries(components).map(([name, component], index) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={cn(
                'flex items-center justify-between p-3 rounded-xl border',
                getStatusColor(component.status)
              )}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center text-surface-600">
                  {getComponentIcon(name)}
                </div>
                <div>
                  <p className="text-sm font-medium text-surface-900 capitalize">
                    {name.replace(/_/g, ' ')}
                  </p>
                  {component.latency_ms !== undefined && (
                    <p className="text-xs text-surface-500">
                      Latency: {component.latency_ms}ms
                    </p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(component.status)}
                <span className="text-sm font-medium capitalize">{component.status}</span>
              </div>
            </motion.div>
          ))}
        </div>
        {health?.timestamp && (
          <p className="mt-4 text-xs text-surface-400 text-center">
            Last updated: {new Date(health.timestamp).toLocaleTimeString()}
          </p>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemStatus;
