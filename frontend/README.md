# PDF Document Extraction - Frontend

A modern, professional React/Next.js frontend for the PDF Document Extraction System.

## Features

- **Dashboard**: Real-time metrics, system health, and activity overview
- **Document Upload**: Drag-and-drop PDF upload with progress tracking
- **Document Management**: Browse, search, and export processed documents
- **Task Queue**: Monitor and manage processing tasks in real-time
- **Settings**: Configure processing options, notifications, and security
- **Authentication**: Secure login with JWT tokens

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **React Query** - Server state management
- **Zustand** - Client state management
- **Framer Motion** - Animations
- **Lucide Icons** - Modern icon library

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Start development server
npm run dev
```

### Environment Variables

Create a `.env.local` file with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── dashboard/          # Dashboard page
│   │   ├── documents/          # Document pages
│   │   ├── tasks/              # Task queue page
│   │   ├── settings/           # Settings page
│   │   ├── health/             # System health page
│   │   └── login/              # Authentication
│   ├── components/
│   │   ├── ui/                 # Reusable UI components
│   │   ├── layout/             # Layout components
│   │   ├── dashboard/          # Dashboard-specific components
│   │   └── documents/          # Document-specific components
│   ├── lib/                    # Utility functions and API client
│   ├── hooks/                  # Custom React hooks
│   ├── store/                  # Zustand state stores
│   └── types/                  # TypeScript type definitions
├── public/                     # Static assets
└── tailwind.config.js          # Tailwind configuration
```

## API Integration

The frontend connects to the backend API through:
- Axios HTTP client with interceptors
- JWT token management
- Automatic token refresh
- Error handling and retry logic

## Styling

Custom design system built on Tailwind CSS:
- Primary color palette (blue)
- Status colors (success, warning, error, info)
- Custom animations
- Responsive breakpoints
- Dark mode support (coming soon)

## Components

### UI Components
- Button, Card, Modal, Input, Select
- Badge, Progress, Tabs, Tooltip
- Empty states, Loading states
- Dropdown, Spinner, Skeleton

### Layout Components
- AppLayout, Header, Sidebar, Footer

### Feature Components
- FileUploader, MetricCard, SystemStatus
- ActiveTasks, RecentActivityList

## License

Proprietary - All rights reserved
