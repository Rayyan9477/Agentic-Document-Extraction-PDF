'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  Settings,
  User,
  Shield,
  Bell,
  Database,
  Palette,
  Server,
  Key,
  Save,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import {
  Card,
  CardHeader,
  CardContent,
  CardFooter,
  Button,
  Input,
  Select,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '@/components/ui';
import type { SelectOption } from '@/components/ui';

export default function SettingsPage() {
  const [saving, setSaving] = useState(false);

  // Profile settings
  const [profile, setProfile] = useState({
    name: 'Admin User',
    email: 'admin@example.com',
    role: 'Administrator',
  });

  // Processing settings
  const [processing, setProcessing] = useState({
    defaultSchema: '',
    defaultExportFormat: 'json',
    defaultPriority: 'normal',
    autoMaskPhi: true,
    maxConcurrentTasks: '5',
    retryAttempts: '3',
  });

  // Notification settings
  const [notifications, setNotifications] = useState({
    emailOnComplete: true,
    emailOnFail: true,
    slackWebhook: '',
    webhookUrl: '',
  });

  const handleSave = async () => {
    setSaving(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setSaving(false);
    toast.success('Settings saved successfully');
  };

  const exportFormatOptions: SelectOption[] = [
    { value: 'json', label: 'JSON' },
    { value: 'excel', label: 'Excel' },
    { value: 'markdown', label: 'Markdown' },
    { value: 'both', label: 'JSON + Excel' },
    { value: 'all', label: 'All Formats' },
  ];

  const priorityOptions: SelectOption[] = [
    { value: 'low', label: 'Low' },
    { value: 'normal', label: 'Normal' },
    { value: 'high', label: 'High' },
  ];

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-2xl font-bold text-surface-900">Settings</h1>
          <p className="text-surface-500 mt-1">
            Configure your document extraction system
          </p>
        </motion.div>

        {/* Settings Tabs */}
        <Card variant="elevated" padding="none">
          <Tabs defaultValue="profile">
            <div className="px-6 pt-6 border-b border-surface-100">
              <TabsList>
                <TabsTrigger value="profile" icon={<User className="w-4 h-4" />}>
                  Profile
                </TabsTrigger>
                <TabsTrigger value="processing" icon={<Server className="w-4 h-4" />}>
                  Processing
                </TabsTrigger>
                <TabsTrigger value="notifications" icon={<Bell className="w-4 h-4" />}>
                  Notifications
                </TabsTrigger>
                <TabsTrigger value="security" icon={<Shield className="w-4 h-4" />}>
                  Security
                </TabsTrigger>
              </TabsList>
            </div>

            <div className="p-6">
              {/* Profile Settings */}
              <TabsContent value="profile">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Profile Information
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Input
                        label="Full Name"
                        value={profile.name}
                        onChange={(e) =>
                          setProfile({ ...profile, name: e.target.value })
                        }
                      />
                      <Input
                        label="Email Address"
                        type="email"
                        value={profile.email}
                        onChange={(e) =>
                          setProfile({ ...profile, email: e.target.value })
                        }
                      />
                      <Input
                        label="Role"
                        value={profile.role}
                        disabled
                        hint="Contact admin to change role"
                      />
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Change Password
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Input
                        label="Current Password"
                        type="password"
                        placeholder="Enter current password"
                      />
                      <div />
                      <Input
                        label="New Password"
                        type="password"
                        placeholder="Enter new password"
                      />
                      <Input
                        label="Confirm Password"
                        type="password"
                        placeholder="Confirm new password"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Processing Settings */}
              <TabsContent value="processing">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Default Processing Options
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Select
                        label="Default Export Format"
                        options={exportFormatOptions}
                        value={processing.defaultExportFormat}
                        onChange={(v) =>
                          setProcessing({ ...processing, defaultExportFormat: v })
                        }
                      />
                      <Select
                        label="Default Priority"
                        options={priorityOptions}
                        value={processing.defaultPriority}
                        onChange={(v) =>
                          setProcessing({ ...processing, defaultPriority: v })
                        }
                      />
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      System Settings
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Input
                        label="Max Concurrent Tasks"
                        type="number"
                        value={processing.maxConcurrentTasks}
                        onChange={(e) =>
                          setProcessing({
                            ...processing,
                            maxConcurrentTasks: e.target.value,
                          })
                        }
                        hint="Maximum number of concurrent processing tasks"
                      />
                      <Input
                        label="Retry Attempts"
                        type="number"
                        value={processing.retryAttempts}
                        onChange={(e) =>
                          setProcessing({
                            ...processing,
                            retryAttempts: e.target.value,
                          })
                        }
                        hint="Number of retry attempts on failure"
                      />
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Privacy & Compliance
                    </h3>
                    <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-success-100 flex items-center justify-center">
                          <Shield className="w-5 h-5 text-success-600" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-surface-900">
                            Auto-mask PHI by default
                          </p>
                          <p className="text-xs text-surface-500">
                            Automatically mask sensitive health information
                          </p>
                        </div>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={processing.autoMaskPhi}
                          onChange={(e) =>
                            setProcessing({
                              ...processing,
                              autoMaskPhi: e.target.checked,
                            })
                          }
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-surface-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600" />
                      </label>
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Notification Settings */}
              <TabsContent value="notifications">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Email Notifications
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
                        <div>
                          <p className="text-sm font-medium text-surface-900">
                            Email on task completion
                          </p>
                          <p className="text-xs text-surface-500">
                            Receive email when processing completes
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={notifications.emailOnComplete}
                            onChange={(e) =>
                              setNotifications({
                                ...notifications,
                                emailOnComplete: e.target.checked,
                              })
                            }
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-surface-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600" />
                        </label>
                      </div>
                      <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
                        <div>
                          <p className="text-sm font-medium text-surface-900">
                            Email on task failure
                          </p>
                          <p className="text-xs text-surface-500">
                            Receive email when processing fails
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={notifications.emailOnFail}
                            onChange={(e) =>
                              setNotifications({
                                ...notifications,
                                emailOnFail: e.target.checked,
                              })
                            }
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-surface-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-surface-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600" />
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Integrations
                    </h3>
                    <div className="space-y-4">
                      <Input
                        label="Slack Webhook URL"
                        value={notifications.slackWebhook}
                        onChange={(e) =>
                          setNotifications({
                            ...notifications,
                            slackWebhook: e.target.value,
                          })
                        }
                        placeholder="https://hooks.slack.com/services/..."
                        hint="Send notifications to Slack channel"
                      />
                      <Input
                        label="Custom Webhook URL"
                        value={notifications.webhookUrl}
                        onChange={(e) =>
                          setNotifications({
                            ...notifications,
                            webhookUrl: e.target.value,
                          })
                        }
                        placeholder="https://your-server.com/webhook"
                        hint="POST notifications to custom endpoint"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Security Settings */}
              <TabsContent value="security">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      API Keys
                    </h3>
                    <div className="p-4 bg-surface-50 rounded-xl">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <Key className="w-5 h-5 text-surface-600" />
                          <div>
                            <p className="text-sm font-medium text-surface-900">
                              API Key
                            </p>
                            <p className="text-xs text-surface-500">
                              Used for programmatic access
                            </p>
                          </div>
                        </div>
                        <Button variant="secondary" size="sm">
                          Regenerate
                        </Button>
                      </div>
                      <div className="flex items-center gap-2">
                        <Input
                          value="sk-••••••••••••••••••••••••••••••••"
                          disabled
                          className="font-mono text-sm"
                        />
                        <Button variant="secondary" size="sm">
                          Copy
                        </Button>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Session Management
                    </h3>
                    <div className="flex items-center justify-between p-4 bg-surface-50 rounded-xl">
                      <div>
                        <p className="text-sm font-medium text-surface-900">
                          Session Timeout
                        </p>
                        <p className="text-xs text-surface-500">
                          Automatically log out after inactivity
                        </p>
                      </div>
                      <Select
                        options={[
                          { value: '15', label: '15 minutes' },
                          { value: '30', label: '30 minutes' },
                          { value: '60', label: '1 hour' },
                          { value: '120', label: '2 hours' },
                          { value: '0', label: 'Never' },
                        ]}
                        value="30"
                        onChange={() => {}}
                        className="w-40"
                      />
                    </div>
                  </div>

                  <div className="pt-6 border-t border-surface-100">
                    <h3 className="text-lg font-medium text-surface-900 mb-4">
                      Danger Zone
                    </h3>
                    <div className="p-4 border border-error-200 bg-error-50 rounded-xl">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-error-900">
                            Delete Account
                          </p>
                          <p className="text-xs text-error-700">
                            Permanently delete your account and all data
                          </p>
                        </div>
                        <Button variant="danger" size="sm">
                          Delete Account
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </div>

            {/* Save Button */}
            <div className="px-6 py-4 border-t border-surface-100 bg-surface-50 rounded-b-2xl">
              <div className="flex items-center justify-end gap-3">
                <Button variant="secondary">Cancel</Button>
                <Button
                  variant="primary"
                  onClick={handleSave}
                  loading={saving}
                  leftIcon={<Save className="w-4 h-4" />}
                >
                  Save Changes
                </Button>
              </div>
            </div>
          </Tabs>
        </Card>
      </div>
    </AppLayout>
  );
}
