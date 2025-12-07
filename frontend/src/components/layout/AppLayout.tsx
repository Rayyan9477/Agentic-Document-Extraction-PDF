'use client';

import React, { useState } from 'react';
import { cn } from '@/lib/utils';
import Header from './Header';
import Sidebar from './Sidebar';
import Footer from './Footer';

interface AppLayoutProps {
  children: React.ReactNode;
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  notifications?: number;
  showFooter?: boolean;
}

const AppLayout: React.FC<AppLayoutProps> = ({
  children,
  user,
  notifications = 0,
  showFooter = true,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-surface-50 flex">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {/* Header */}
        <Header
          onMenuClick={() => setSidebarOpen(true)}
          user={user}
          notifications={notifications}
        />

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <div className="container mx-auto p-4 lg:p-6">
            {children}
          </div>
        </main>

        {/* Footer */}
        {showFooter && <Footer />}
      </div>
    </div>
  );
};

export default AppLayout;
