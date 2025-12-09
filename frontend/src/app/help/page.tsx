'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  HelpCircle,
  BookOpen,
  MessageCircle,
  FileText,
  ExternalLink,
  Mail,
  Github,
  AlertCircle,
} from 'lucide-react';
import { AppLayout } from '@/components/layout';
import { Card, CardHeader, CardContent } from '@/components/ui';

export default function HelpPage() {
  const resources = [
    {
      icon: <BookOpen className="w-5 h-5" />,
      title: 'Documentation',
      description: 'Comprehensive guides and API reference',
      link: '/docs',
      external: false,
    },
    {
      icon: <FileText className="w-5 h-5" />,
      title: 'API Reference',
      description: 'Detailed API endpoint documentation',
      link: '/docs',
      external: false,
    },
    {
      icon: <Github className="w-5 h-5" />,
      title: 'GitHub Repository',
      description: 'Source code and issue tracking',
      link: 'https://github.com/yourusername/pdf-extraction',
      external: true,
    },
    {
      icon: <MessageCircle className="w-5 h-5" />,
      title: 'Community Forum',
      description: 'Get help from the community',
      link: '#',
      external: true,
    },
  ];

  const faqs = [
    {
      question: 'How do I upload and process a document?',
      answer:
        'Navigate to the Upload page, select your PDF files, configure processing options (schema, format, priority), and click Upload. Documents will be processed asynchronously.',
    },
    {
      question: 'What document types are supported?',
      answer:
        'The system supports CMS-1500, UB-04, Superbill, and EOB documents. You can view all available schemas on the Schemas page.',
    },
    {
      question: 'How do I export processed data?',
      answer:
        'On the Documents page, click the export button for any processed document and choose your preferred format (JSON, Excel, or Markdown).',
    },
    {
      question: 'What is PHI masking?',
      answer:
        'PHI (Protected Health Information) masking replaces sensitive healthcare data with redacted values to comply with HIPAA regulations.',
    },
  ];

  const contactOptions = [
    {
      icon: <Mail className="w-5 h-5" />,
      title: 'Email Support',
      value: 'support@example.com',
      action: 'mailto:support@example.com',
    },
    {
      icon: <Github className="w-5 h-5" />,
      title: 'Report an Issue',
      value: 'GitHub Issues',
      action: 'https://github.com/yourusername/pdf-extraction/issues',
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
            <h1 className="text-2xl font-bold text-surface-900">Help & Support</h1>
            <p className="text-surface-500 mt-1">
              Documentation, guides, and support resources
            </p>
          </div>
        </motion.div>

        {/* Quick Links */}
        <div>
          <h2 className="text-lg font-bold text-surface-900 mb-4">Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {resources.map((resource, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card variant="outlined" padding="lg" className="h-full">
                  <a
                    href={resource.link}
                    target={resource.external ? '_blank' : undefined}
                    rel={resource.external ? 'noopener noreferrer' : undefined}
                    className="flex items-start gap-4 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center flex-shrink-0 group-hover:bg-primary-200 transition-colors">
                      {resource.icon}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-surface-900 group-hover:text-primary-600 transition-colors">
                          {resource.title}
                        </h3>
                        {resource.external && (
                          <ExternalLink className="w-4 h-4 text-surface-400" />
                        )}
                      </div>
                      <p className="text-sm text-surface-600">
                        {resource.description}
                      </p>
                    </div>
                  </a>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* FAQs */}
        <div>
          <h2 className="text-lg font-bold text-surface-900 mb-4">
            Frequently Asked Questions
          </h2>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + index * 0.1 }}
              >
                <Card variant="outlined" padding="lg">
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-info-100 flex items-center justify-center flex-shrink-0">
                      <HelpCircle className="w-4 h-4 text-info-600" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-surface-900 mb-2">
                        {faq.question}
                      </h3>
                      <p className="text-sm text-surface-600">{faq.answer}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Getting Started */}
        <Card variant="elevated" padding="lg">
          <CardHeader
            title="Getting Started"
            description="Quick steps to begin using the system"
          />
          <CardContent className="mt-4">
            <ol className="space-y-4">
              {[
                {
                  title: 'Upload Documents',
                  description: 'Go to Upload page and select your PDF files',
                },
                {
                  title: 'Configure Options',
                  description: 'Choose schema, export format, and processing priority',
                },
                {
                  title: 'Process & Monitor',
                  description: 'Track processing status on the Tasks page',
                },
                {
                  title: 'Export Results',
                  description: 'Download extracted data in your preferred format',
                },
              ].map((step, index) => (
                <li key={index} className="flex gap-4">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 text-primary-700 font-bold text-sm flex-shrink-0">
                    {index + 1}
                  </div>
                  <div>
                    <h4 className="font-semibold text-surface-900">{step.title}</h4>
                    <p className="text-sm text-surface-600">{step.description}</p>
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>

        {/* Contact Support */}
        <Card variant="outlined" padding="lg">
          <CardHeader
            title="Contact Support"
            description="Need additional help? Reach out to our team"
          />
          <CardContent className="mt-4 space-y-3">
            {contactOptions.map((option, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 bg-surface-50 rounded-xl"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                    {option.icon}
                  </div>
                  <div>
                    <p className="font-medium text-surface-900">{option.title}</p>
                    <p className="text-sm text-surface-500">{option.value}</p>
                  </div>
                </div>
                <a
                  href={option.action}
                  target={option.action.startsWith('http') ? '_blank' : undefined}
                  rel={
                    option.action.startsWith('http')
                      ? 'noopener noreferrer'
                      : undefined
                  }
                  aria-label={`Open ${option.title}`}
                  className="inline-flex items-center justify-center px-3 py-1.5 text-sm font-medium rounded-lg border-2 border-primary-600 text-primary-600 hover:bg-primary-50 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* System Info */}
        <Card variant="outlined" padding="lg">
          <div className="flex items-start gap-4">
            <AlertCircle className="w-5 h-5 text-info-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-surface-900 mb-1">System Version</h3>
              <p className="text-sm text-surface-600">
                PDF Document Extraction System v1.0.0
              </p>
              <p className="text-sm text-surface-500 mt-1">
                Backend API: v1.0.0 | Frontend: v1.0.0
              </p>
            </div>
          </div>
        </Card>
      </div>
    </AppLayout>
  );
}
