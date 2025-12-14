'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from '@/components/ui';

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Fallback component to render on error */
  fallback?: ReactNode;
  /** Called when an error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Whether to show error details (for development) */
  showDetails?: boolean;
  /** Custom reset function */
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary Component
 *
 * Catches JavaScript errors in child component tree and displays
 * a fallback UI instead of crashing the whole app.
 *
 * Usage:
 * ```tsx
 * <ErrorBoundary>
 *   <ComponentThatMightThrow />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });

    // Log error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // In production, you might want to report to an error tracking service
    // Example: Sentry.captureException(error, { extra: errorInfo });
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    this.props.onReset?.();
  };

  handleRefresh = (): void => {
    window.location.reload();
  };

  handleGoHome = (): void => {
    window.location.href = '/';
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo } = this.state;
      const showDetails = this.props.showDetails ?? process.env.NODE_ENV === 'development';

      return (
        <div className="min-h-[400px] flex items-center justify-center p-8">
          <div className="max-w-lg w-full">
            <div className="text-center">
              {/* Error Icon */}
              <div className="mx-auto w-16 h-16 rounded-full bg-error-100 flex items-center justify-center mb-6">
                <AlertTriangle className="w-8 h-8 text-error-600" />
              </div>

              {/* Error Message */}
              <h2 className="text-xl font-semibold text-surface-900 mb-2">
                Something went wrong
              </h2>
              <p className="text-surface-600 mb-6">
                An unexpected error occurred. Please try refreshing the page or go back to the home page.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap justify-center gap-3 mb-6">
                <Button
                  variant="primary"
                  onClick={this.handleReset}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Try Again
                </Button>
                <Button
                  variant="secondary"
                  onClick={this.handleRefresh}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Refresh Page
                </Button>
                <Button
                  variant="ghost"
                  onClick={this.handleGoHome}
                  leftIcon={<Home className="w-4 h-4" />}
                >
                  Go Home
                </Button>
              </div>

              {/* Error Details (Development Only) */}
              {showDetails && error && (
                <div className="mt-6 text-left">
                  <button
                    className="flex items-center gap-2 text-sm text-surface-500 hover:text-surface-700 mb-2"
                    onClick={() => {
                      const details = document.getElementById('error-details');
                      if (details) {
                        details.classList.toggle('hidden');
                      }
                    }}
                  >
                    <Bug className="w-4 h-4" />
                    Show Error Details
                  </button>
                  <div
                    id="error-details"
                    className="hidden p-4 bg-surface-100 rounded-lg overflow-auto max-h-64"
                  >
                    <p className="text-sm font-medium text-error-700 mb-2">
                      {error.name}: {error.message}
                    </p>
                    {errorInfo?.componentStack && (
                      <pre className="text-xs text-surface-600 whitespace-pre-wrap">
                        {errorInfo.componentStack}
                      </pre>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Async Error Boundary for handling async component errors
 *
 * Wraps React.Suspense with ErrorBoundary
 */
interface AsyncBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  loadingFallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

export function AsyncBoundary({
  children,
  fallback,
  loadingFallback,
  onError,
}: AsyncBoundaryProps): JSX.Element {
  return (
    <ErrorBoundary fallback={fallback} onError={onError}>
      <React.Suspense fallback={loadingFallback || <LoadingFallback />}>
        {children}
      </React.Suspense>
    </ErrorBoundary>
  );
}

/**
 * Default loading fallback component
 */
function LoadingFallback(): JSX.Element {
  return (
    <div className="flex items-center justify-center min-h-[200px]">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
    </div>
  );
}

/**
 * Page-level error boundary with full-page styling
 */
interface PageErrorBoundaryProps {
  children: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

export function PageErrorBoundary({
  children,
  onError,
}: PageErrorBoundaryProps): JSX.Element {
  return (
    <ErrorBoundary
      onError={onError}
      fallback={
        <div className="min-h-screen bg-surface-50 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-white rounded-2xl shadow-lg p-8">
            <div className="text-center">
              <div className="mx-auto w-20 h-20 rounded-full bg-error-100 flex items-center justify-center mb-6">
                <AlertTriangle className="w-10 h-10 text-error-600" />
              </div>
              <h1 className="text-2xl font-bold text-surface-900 mb-3">
                Oops! Something went wrong
              </h1>
              <p className="text-surface-600 mb-8">
                We encountered an unexpected error. Our team has been notified and is working on a fix.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button
                  variant="primary"
                  onClick={() => window.location.reload()}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Refresh Page
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => (window.location.href = '/')}
                  leftIcon={<Home className="w-4 h-4" />}
                >
                  Return Home
                </Button>
              </div>
            </div>
          </div>
        </div>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
