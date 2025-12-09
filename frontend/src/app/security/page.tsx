'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  Lock,
  Key,
  UserCheck,
  AlertTriangle,
  CheckCircle,
  Info,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import { Card, CardHeader, CardContent, Badge } from '@/components/ui';

export default function SecurityPage() {
  const securityFeatures = [
    {
      icon: <Lock className="w-5 h-5" />,
      title: 'AES-256 Encryption',
      description: 'HIPAA-compliant data encryption at rest',
      status: 'active',
    },
    {
      icon: <Key className="w-5 h-5" />,
      title: 'JWT Authentication',
      description: 'Secure token-based authentication',
      status: 'active',
    },
    {
      icon: <UserCheck className="w-5 h-5" />,
      title: 'RBAC',
      description: 'Role-based access control with granular permissions',
      status: 'active',
    },
    {
      icon: <Shield className="w-5 h-5" />,
      title: 'Security Headers',
      description: 'OWASP recommended security headers configured',
      status: 'active',
    },
  ];

  const securityMetrics = [
    {
      label: 'Bcrypt Rounds',
      value: '14',
      icon: <Lock className="w-5 h-5 text-primary-600" />,
    },
    {
      label: 'Token Expiry',
      value: '15 min',
      icon: <Key className="w-5 h-5 text-info-600" />,
    },
    {
      label: 'Rate Limiting',
      value: 'Active',
      icon: <Shield className="w-5 h-5 text-success-600" />,
    },
  ];

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Security Overview</h1>
            <p className="text-surface-500 mt-1">
              System security configuration and compliance status
            </p>
          </div>
        </motion.div>

        {/* Security Status */}
        <Card variant="elevated" padding="lg">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-14 h-14 rounded-xl bg-success-100 flex items-center justify-center">
              <CheckCircle className="w-7 h-7 text-success-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-surface-900">System Secure</h2>
              <p className="text-surface-500 text-sm">
                All security checks passed
              </p>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {securityMetrics.map((metric, index) => (
              <div
                key={index}
                className="p-4 bg-surface-50 rounded-xl flex items-center gap-3"
              >
                <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center">
                  {metric.icon}
                </div>
                <div>
                  <p className="text-sm text-surface-500">{metric.label}</p>
                  <p className="text-lg font-bold text-surface-900">{metric.value}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Security Features */}
        <div>
          <h2 className="text-lg font-bold text-surface-900 mb-4">Security Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {securityFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card variant="outline" padding="lg">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center flex-shrink-0">
                      {feature.icon}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-surface-900">
                          {feature.title}
                        </h3>
                        <Badge
                          variant={feature.status === 'active' ? 'success' : 'default'}
                          size="sm"
                        >
                          {feature.status}
                        </Badge>
                      </div>
                      <p className="text-sm text-surface-600">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Compliance */}
        <Card variant="outline" padding="lg">
          <CardHeader
            title="Compliance Status"
            description="Current compliance and security standards"
          />
          <CardContent className="mt-4 space-y-3">
            <div className="flex items-center justify-between p-3 bg-success-50 rounded-lg">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-success-600" />
                <span className="font-medium text-surface-900">OWASP Top 10</span>
              </div>
              <Badge variant="success" size="sm">8/10 Compliant</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-success-50 rounded-lg">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-success-600" />
                <span className="font-medium text-surface-900">HIPAA Security Rule</span>
              </div>
              <Badge variant="success" size="sm">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-warning-50 rounded-lg">
              <div className="flex items-center gap-3">
                <Info className="w-5 h-5 text-warning-600" />
                <span className="font-medium text-surface-900">CSRF Protection</span>
              </div>
              <Badge variant="warning" size="sm">Recommended</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Security Recommendations */}
        <Card variant="outline" padding="lg">
          <CardHeader
            title="Security Recommendations"
            description="Recommended improvements for production"
          />
          <CardContent className="mt-4">
            <div className="space-y-3">
              {[
                'Implement CSRF protection for state-changing requests',
                'Migrate to httpOnly cookies for token storage',
                'Add Redis for persistent token blacklist',
                'Enable MFA for administrator accounts',
              ].map((recommendation, index) => (
                <div
                  key={index}
                  className="flex items-start gap-3 p-3 bg-info-50 rounded-lg"
                >
                  <Info className="w-5 h-5 text-info-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-surface-700">{recommendation}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
