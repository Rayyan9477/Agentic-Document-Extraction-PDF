# PDF Document Extraction System - Complete Routes & Functionality

**Date:** 2025-12-09
**Version:** 1.0.0

---

## Table of Contents

1. [Frontend Pages & Routes](#frontend-pages--routes)
2. [Backend API Endpoints](#backend-api-endpoints)
3. [Core Functionalities](#core-functionalities)
4. [System Architecture](#system-architecture)
5. [Authentication & Security](#authentication--security)
6. [Data Flow](#data-flow)

---

## Frontend Pages & Routes

### Public Pages (No Auth Required)

#### 1. Login Page
- **Route:** `/login`
- **File:** [frontend/src/app/login/page.tsx](frontend/src/app/login/page.tsx)
- **Purpose:** User authentication
- **Features:**
  - Username/email + password login
  - JWT token generation
  - Remember me option
  - Link to signup page
  - Error handling for invalid credentials

#### 2. Signup Page
- **Route:** `/signup`
- **File:** [frontend/src/app/signup/page.tsx](frontend/src/app/signup/page.tsx)
- **Purpose:** New user registration
- **Features:**
  - Username, email, password fields
  - Password confirmation
  - Strong password validation (12+ chars, complexity rules)
  - Real-time form validation
  - Terms of service agreement
  - Account creation with auto-role assignment

### Protected Pages (Auth Required)

#### 3. Dashboard (Home)
- **Route:** `/dashboard` or `/`
- **File:** [frontend/src/app/dashboard/page.tsx](frontend/src/app/dashboard/page.tsx)
- **Purpose:** Main overview dashboard
- **Features:**
  - System metrics (documents processed, success rate, avg processing time)
  - Recent activity feed
  - Active task monitoring
  - System health status
  - Quick action buttons (upload, view documents)
  - Real-time statistics

#### 4. Documents List
- **Route:** `/documents`
- **File:** [frontend/src/app/documents/page.tsx](frontend/src/app/documents/page.tsx)
- **Purpose:** View all processed documents
- **Features:**
  - Document list with search functionality
  - Filters (status, date range, document type)
  - Document cards with metadata
  - Export buttons (JSON, Excel, Markdown)
  - Status badges (processing, completed, failed)
  - Confidence scores
  - Pagination

#### 5. Document Detail
- **Route:** `/documents/[id]`
- **File:** [frontend/src/app/documents/[id]/page.tsx](frontend/src/app/documents/[id]/page.tsx)
- **Purpose:** View single document details
- **Features:**
  - Document metadata display
  - Extracted data preview
  - Field-level confidence scores
  - Export options
  - Reprocess option
  - Delete option

#### 6. Document Upload
- **Route:** `/documents/upload`
- **File:** [frontend/src/app/documents/upload/page.tsx](frontend/src/app/documents/upload/page.tsx)
- **Purpose:** Upload and process new documents
- **Features:**
  - Drag-and-drop file uploader
  - Multi-file selection
  - File validation (PDF only)
  - Processing options configuration:
    - Schema selection (auto-detect or manual)
    - Export format (JSON, Excel, Markdown, Both, All)
    - Processing priority (Low, Normal, High)
    - PHI masking toggle
    - Output directory
  - Upload progress tracking
  - Step-by-step wizard (Select → Configure → Process → Complete)
  - Real-time upload status

#### 7. Task Queue
- **Route:** `/tasks`
- **File:** [frontend/src/app/tasks/page.tsx](frontend/src/app/tasks/page.tsx)
- **Purpose:** Monitor async processing tasks
- **Features:**
  - Active tasks list with status
  - Task filtering (all, pending, processing, completed, failed)
  - Task details (ID, status, progress, start time, duration)
  - Cancel task button
  - Retry failed task button
  - Queue statistics
  - Worker status monitoring
  - Real-time task updates

#### 8. Schemas
- **Route:** `/schemas`
- **File:** [frontend/src/app/schemas/page.tsx](frontend/src/app/schemas/page.tsx)
- **Purpose:** View available extraction schemas
- **Features:**
  - Schema cards (CMS-1500, UB-04, Superbill, EOB)
  - Schema statistics (field count, version, document type)
  - Search functionality
  - Schema descriptions
  - Active status indicators
  - Total schemas/fields/types metrics

#### 9. Health Status
- **Route:** `/health`
- **File:** [frontend/src/app/health/page.tsx](frontend/src/app/health/page.tsx)
- **Purpose:** System health monitoring
- **Features:**
  - Overall system health status
  - Component health checks (API, Database, Cache, Queue)
  - Uptime tracking
  - Memory usage
  - CPU usage
  - Disk space
  - Response time metrics
  - Historical health data

#### 10. Settings
- **Route:** `/settings`
- **File:** [frontend/src/app/settings/page.tsx](frontend/src/app/settings/page.tsx)
- **Purpose:** User and system settings
- **Features:**
  - User profile management
  - Password change
  - Notification preferences
  - API key management
  - System preferences
  - Theme selection (light/dark)

#### 11. Security
- **Route:** `/security`
- **File:** [frontend/src/app/security/page.tsx](frontend/src/app/security/page.tsx)
- **Purpose:** Security overview and compliance
- **Features:**
  - Security metrics (Bcrypt rounds, token expiry, rate limits)
  - Active security features list
  - Compliance status (OWASP, HIPAA)
  - Security recommendations
  - Audit log access

#### 12. Help & Support
- **Route:** `/help`
- **File:** [frontend/src/app/help/page.tsx](frontend/src/app/help/page.tsx)
- **Purpose:** Documentation and support resources
- **Features:**
  - Documentation links
  - FAQs
  - Getting started guide
  - Contact support options
  - System version info
  - Video tutorials (placeholder)

---

## Backend API Endpoints

**Base URL:** `http://localhost:8000/api/v1`

### Authentication Endpoints (`/api/v1/auth`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/signup` | Register new user | No |
| POST | `/auth/login` | User login, returns JWT tokens | No |
| POST | `/auth/logout` | Logout, revoke tokens | Yes |
| POST | `/auth/refresh` | Refresh access token | No (refresh token) |
| GET | `/auth/me` | Get current user info | Yes |

**Key Features:**
- JWT-based authentication
- Access token (15 min expiry) + Refresh token (7 days)
- Rate limiting (3 signup/min, 5 login/min)
- Password validation (12+ chars, complexity)
- Account enumeration protection
- Token blacklist for logout

### Document Endpoints (`/api/v1/documents`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/documents/upload` | Upload and process document | Yes |
| POST | `/documents/batch` | Batch upload multiple documents | Yes |
| GET | `/documents` | List recent documents | Yes |
| POST | `/documents/search` | Search documents | Yes |
| POST | `/documents/{id}/export/{format}` | Export document data | Yes |
| POST | `/documents/{id}/reprocess` | Reprocess document | Yes |

**Supported Features:**
- File upload with progress tracking
- Async processing with task queue
- Multiple export formats (JSON, Excel, Markdown, FHIR)
- PHI masking for HIPAA compliance
- Document classification
- Confidence scoring
- Validation and anti-hallucination

### Task Endpoints (`/api/v1/tasks`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/tasks` | List all tasks | Yes |
| GET | `/tasks/active` | List active tasks only | Yes |
| DELETE | `/tasks/{id}` | Cancel task | Yes |
| GET | `/tasks/{id}` | Get task status | Yes |
| GET | `/tasks/{id}/result` | Get task result | Yes |
| POST | `/tasks/{id}/retry` | Retry failed task | Yes |
| POST | `/tasks/{id}/cancel` | Cancel running task | Yes |

**Task States:**
- `pending` - Queued, waiting to start
- `processing` - Currently being processed
- `completed` - Successfully completed
- `failed` - Processing failed
- `cancelled` - Cancelled by user

### Schema Endpoints (`/api/v1/schemas`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/schemas` | List all schemas | Yes |
| GET | `/schemas/{name}` | Get schema details | Yes |
| GET | `/schemas/{name}/fields` | Get schema fields | Yes |
| POST | `/schemas/detect` | Auto-detect document schema | Yes |

**Available Schemas:**
1. **CMS-1500** (74 fields) - Healthcare professional claim form
2. **UB-04** (182 fields) - Institutional provider billing
3. **Superbill** (50 fields) - Itemized service form
4. **EOB** (68 fields) - Explanation of Benefits

### Dashboard Endpoints (`/api/v1/dashboard`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/dashboard/metrics` | Get dashboard metrics | Yes |
| GET | `/dashboard/activity` | Get recent activity | Yes |
| GET | `/dashboard/stats` | Get statistics | Yes |

**Metrics Provided:**
- Total documents processed
- Success rate percentage
- Average processing time
- Documents by status
- Recent activity timeline

### Queue Endpoints (`/api/v1/queue`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/queue/stats` | Get queue statistics | Yes |
| GET | `/queue/workers` | Get worker status | Yes |
| POST | `/queue/{name}/purge` | Purge queue | Yes (Admin) |

**Queue Names:**
- `default` - Standard priority tasks
- `high_priority` - High priority tasks
- `low_priority` - Low priority tasks

### Health Endpoints (`/api/v1/health`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Basic health check | No |
| GET | `/health/detailed` | Detailed health status | Yes |
| GET | `/health/liveness` | Kubernetes liveness probe | No |
| GET | `/health/readiness` | Kubernetes readiness probe | No |
| GET | `/health/database` | Database health | Yes |
| GET | `/health/cache` | Cache health | Yes |
| GET | `/health/queue` | Queue health | Yes |
| GET | `/health/storage` | Storage health | Yes |

---

## Core Functionalities

### 1. Document Processing Pipeline

**Flow:**
```
Upload → Validation → Classification → Extraction → Validation → Export
```

**Steps:**
1. **Upload:** User uploads PDF via UI or API
2. **Validation:** File type, size, format validation
3. **Classification:** Auto-detect document type (CMS-1500, UB-04, etc.)
4. **Extraction:** AI-powered data extraction with GPT-4
5. **Validation:** Multi-layer anti-hallucination validation
6. **Confidence Scoring:** Per-field confidence calculation
7. **Export:** Generate JSON, Excel, Markdown, or FHIR output

**Technologies:**
- **AI Model:** GPT-4 (Claude can be configured)
- **PDF Processing:** PyMuPDF (fitz), pdfplumber
- **OCR:** Tesseract OCR for scanned documents
- **Validation:** Custom validation rules + confidence scoring

### 2. Authentication & Authorization (RBAC)

**Roles:**
- **Admin:** Full system access, user management
- **Editor:** Create, read, update documents
- **Viewer:** Read-only access
- **Guest:** Limited access to public resources

**Permissions:**
- `documents:read` - View documents
- `documents:create` - Upload documents
- `documents:update` - Modify documents
- `documents:delete` - Delete documents
- `schemas:read` - View schemas
- `users:manage` - User management (Admin only)
- `system:configure` - System configuration (Admin only)

**Implementation:**
- JWT tokens with role/permission claims
- Middleware-based permission checks
- Token blacklist for logout
- Refresh token rotation

### 3. Task Queue System

**Technology:** Celery with Redis/RabbitMQ

**Features:**
- Async document processing
- Priority-based queuing (low, normal, high)
- Task retry with exponential backoff
- Task cancellation
- Worker monitoring
- Dead letter queue for failures

**Queues:**
- `high_priority` - SLA-critical documents
- `default` - Standard processing
- `low_priority` - Batch jobs

### 4. Export Formats

#### JSON Export
```json
{
  "document_type": "cms_1500",
  "extracted_data": {
    "patient_name": "John Doe",
    "date_of_service": "2024-01-15",
    ...
  },
  "confidence_scores": {
    "patient_name": 0.98,
    "date_of_service": 0.95,
    ...
  },
  "metadata": {...}
}
```

#### Excel Export
- Structured spreadsheet with tabs
- Extracted data in main sheet
- Confidence scores in separate sheet
- Metadata sheet

#### Markdown Export
- Human-readable format
- Formatted tables
- Includes confidence indicators
- Easy to review and share

#### FHIR Export (Healthcare)
- HL7 FHIR R4 compliant
- Patient, Claim, Coverage resources
- Interoperable with EMR systems

### 5. PHI Masking (HIPAA Compliance)

**Masked Fields:**
- Patient names → `[REDACTED]`
- SSN → `XXX-XX-XXXX`
- Dates of birth → `XX/XX/XXXX`
- Addresses → `[ADDRESS REDACTED]`
- Phone numbers → `XXX-XXX-XXXX`

**Implementation:**
- Regex-based detection
- Named entity recognition (NER)
- Configurable masking rules
- Audit trail for PHI access

### 6. Validation & Anti-Hallucination

**Multi-Layer Validation:**

1. **Schema Validation:** Field presence, data types
2. **Format Validation:** Date formats, SSN patterns, codes
3. **Cross-Field Validation:** Logical consistency checks
4. **Confidence Thresholds:** Flag low-confidence extractions
5. **Human-in-Loop:** Manual review for uncertain fields

**Confidence Scoring:**
- `0.95-1.0` - High confidence (green)
- `0.80-0.94` - Medium confidence (yellow)
- `0.0-0.79` - Low confidence (red, requires review)

### 7. Monitoring & Alerting

**Metrics Collected:**
- Request rate and latency
- Error rates by endpoint
- Queue lengths and processing times
- Document processing success rates
- System resource usage (CPU, memory, disk)

**Alert Channels:**
- Email notifications
- Slack integration
- PagerDuty for critical alerts
- Webhook callbacks

**Alert Rules:**
- High error rate (>5% in 5 min)
- Queue backup (>100 tasks waiting)
- Processing failures (>10 failures in 1 hour)
- System resource exhaustion (>90% usage)

### 8. Security Features

#### Encryption
- **At Rest:** AES-256 encryption for stored data
- **In Transit:** TLS 1.3 for all API communication
- **Database:** Encrypted database fields for sensitive data

#### Security Headers
- Content-Security-Policy (CSP)
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Strict-Transport-Security (HSTS)
- X-XSS-Protection

#### Rate Limiting
- Per-endpoint limits
- Per-user limits
- IP-based rate limiting
- Exponential backoff for repeated violations

#### Audit Logging
- All API requests logged
- User actions tracked
- PHI access logged (HIPAA requirement)
- Tamper-evident log storage
- Log retention: 7 years (compliance)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  Next.js 14 + React 18 + TypeScript + Tailwind CSS         │
│  (http://localhost:3000)                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTPS/REST API
                        │ JWT Authentication
┌───────────────────────▼─────────────────────────────────────┐
│                      API Gateway                             │
│           FastAPI + Uvicorn (Python 3.12)                   │
│                (http://localhost:8000)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Middleware Stack:                                    │  │
│  │  - CORS                                               │  │
│  │  - Security Headers                                   │  │
│  │  - Rate Limiting                                      │  │
│  │  - Authentication (JWT)                               │  │
│  │  - Request Logging                                    │  │
│  │  - Metrics Collection                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└───────┬──────────────┬──────────────┬──────────────┬────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│   Auth    │  │  Document    │  │  Queue   │  │  Cache   │
│  Service  │  │  Processing  │  │ (Celery) │  │ (Redis)  │
│   RBAC    │  │    Engine    │  │          │  │          │
└───────────┘  └──────────────┘  └──────────┘  └──────────┘
        │              │              │              │
        │              ▼              │              │
        │      ┌──────────────┐      │              │
        │      │  AI Models   │      │              │
        │      │   GPT-4      │      │              │
        │      │   Claude     │      │              │
        │      └──────────────┘      │              │
        │                             │              │
        ▼                             ▼              ▼
┌───────────────────────────────────────────────────────────┐
│                     Data Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PostgreSQL  │  │    Redis     │  │  File Store  │  │
│  │   Database   │  │    Cache     │  │   (S3/Local) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- **Framework:** Next.js 14 (React 18, TypeScript)
- **Styling:** Tailwind CSS 3.4
- **State Management:** TanStack Query (React Query)
- **Form Handling:** Custom hooks with validation
- **Animation:** Framer Motion
- **Icons:** Lucide React
- **HTTP Client:** Fetch API with custom wrapper

**Backend:**
- **Framework:** FastAPI 0.110+
- **Language:** Python 3.12
- **ASGI Server:** Uvicorn
- **Task Queue:** Celery + Redis/RabbitMQ
- **Database ORM:** SQLAlchemy 2.0
- **Migrations:** Alembic
- **Validation:** Pydantic 2.0
- **Authentication:** JWT (python-jose)
- **Password Hashing:** Bcrypt (14 rounds)

**AI/ML:**
- **Primary Model:** OpenAI GPT-4
- **Alternative:** Anthropic Claude
- **PDF Processing:** PyMuPDF, pdfplumber
- **OCR:** Tesseract
- **NLP:** spaCy (for NER)

**Infrastructure:**
- **Database:** PostgreSQL 15+
- **Cache:** Redis 7+
- **Message Broker:** Redis or RabbitMQ
- **Storage:** Local filesystem or AWS S3
- **Monitoring:** Prometheus + Grafana
- **Logging:** Structlog + ELK Stack

---

## Authentication & Security

### Authentication Flow

```
1. User Signup
   ↓
   POST /api/v1/auth/signup
   ↓
   - Validate password (12+ chars, complexity)
   - Hash password (bcrypt, 14 rounds)
   - Create user record
   - Assign default role (viewer)
   ↓
   Return success message

2. User Login
   ↓
   POST /api/v1/auth/login
   ↓
   - Validate credentials
   - Check rate limit (5 attempts/min)
   - Generate JWT tokens (access + refresh)
   - Return tokens
   ↓
   Frontend stores tokens (localStorage)

3. Authenticated Request
   ↓
   Request with Authorization: Bearer <token>
   ↓
   - Extract token from header
   - Verify token signature
   - Check token expiry
   - Check token blacklist
   - Extract user/role/permissions
   - Authorize action
   ↓
   Process request

4. Token Refresh
   ↓
   POST /api/v1/auth/refresh
   Body: { refresh_token: "..." }
   ↓
   - Verify refresh token
   - Generate new access token
   - Rotate refresh token
   ↓
   Return new tokens

5. Logout
   ↓
   POST /api/v1/auth/logout
   ↓
   - Add tokens to blacklist
   - Clear frontend storage
   ↓
   Return success
```

### Security Best Practices Implemented

✅ **Authentication:**
- JWT with short-lived access tokens (15 min)
- Refresh token rotation
- Token blacklist on logout
- Bcrypt with 14 rounds (OWASP 2024)
- Rate limiting on auth endpoints

✅ **Authorization:**
- Role-Based Access Control (RBAC)
- Permission-based resource access
- Middleware enforcement

✅ **Data Protection:**
- AES-256 encryption at rest
- TLS 1.3 in transit
- PHI masking for HIPAA compliance
- Secure password storage

✅ **API Security:**
- CORS configured (strict origins)
- Security headers (CSP, HSTS, etc.)
- Input validation (Pydantic schemas)
- SQL injection prevention (ORM)
- XSS protection

✅ **Compliance:**
- HIPAA Security Rule compliant
- OWASP Top 10 (8/10 addressed)
- GDPR data protection
- Audit logging (7-year retention)

---

## Data Flow

### Document Upload & Processing Flow

```
1. User Uploads PDF
   ↓
   Frontend: /documents/upload
   ↓
   POST /api/v1/documents/upload
   - File validation (type, size, format)
   - Generate processing_id (UUID)
   - Store file temporarily
   ↓
   Queue async task (Celery)
   ↓
   Return task_id to frontend

2. Async Processing (Celery Worker)
   ↓
   - Extract text from PDF (PyMuPDF)
   - If scanned: OCR processing (Tesseract)
   ↓
   - Document classification (GPT-4)
   - Select appropriate schema
   ↓
   - Field extraction (GPT-4 with schema)
   - Generate structured data
   ↓
   - Validation layer
     - Schema validation
     - Format validation
     - Cross-field validation
   ↓
   - Confidence scoring
   - Flag low-confidence fields
   ↓
   - Apply PHI masking (if enabled)
   ↓
   - Generate exports (JSON, Excel, etc.)
   - Store results in database
   ↓
   Update task status: COMPLETED

3. Frontend Polling
   ↓
   GET /api/v1/tasks/{task_id}
   ↓
   Check status every 2 seconds
   ↓
   When status = COMPLETED:
   ↓
   GET /api/v1/documents/{processing_id}
   ↓
   Display results to user

4. Export Download
   ↓
   User clicks export button
   ↓
   POST /api/v1/documents/{id}/export/json
   ↓
   Generate export file
   ↓
   Return file blob
   ↓
   Frontend triggers download
```

---

## Summary

### Frontend Pages: 13 Total
- 2 Public (Login, Signup)
- 11 Protected (Dashboard, Documents, Upload, Tasks, Schemas, Health, Settings, Security, Help, Document Detail, Home)

### Backend Endpoints: 40+ Total
- Authentication: 5 endpoints
- Documents: 6 endpoints
- Tasks: 7 endpoints
- Schemas: 4 endpoints
- Dashboard: 3 endpoints
- Queue: 3 endpoints
- Health: 8 endpoints

### Core Technologies:
- **Frontend:** Next.js 14 + TypeScript + Tailwind CSS
- **Backend:** FastAPI + Python 3.12
- **Database:** PostgreSQL
- **Cache:** Redis
- **Queue:** Celery
- **AI:** GPT-4 / Claude

### Security Features:
- JWT Authentication
- RBAC Authorization
- AES-256 Encryption
- HIPAA Compliant PHI Masking
- Rate Limiting
- Audit Logging

### Processing Capabilities:
- 4 Document Schemas (CMS-1500, UB-04, Superbill, EOB)
- 4 Export Formats (JSON, Excel, Markdown, FHIR)
- Async Processing with Priority Queues
- Multi-layer Validation
- Confidence Scoring

---

**Last Updated:** 2025-12-09
**System Status:** ✅ OPERATIONAL
