// API Types for PDF Document Extraction System

// Enums
export type TaskStatus =
  | 'pending'
  | 'started'
  | 'processing'
  | 'validating'
  | 'exporting'
  | 'completed'
  | 'failed'
  | 'retrying'
  | 'cancelled';

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export type ExportFormat = 'json' | 'excel' | 'markdown' | 'both' | 'all';

export type ProcessingPriority = 'low' | 'normal' | 'high';

export type PreviewStyle = 'simple' | 'detailed' | 'summary' | 'technical';

// Field Result
export interface FieldResult {
  value: unknown;
  confidence: number;
  confidence_level: ConfidenceLevel;
  location?: string;
  passes_agree: boolean;
  validation_passed: boolean;
}

// Validation Result
export interface ValidationResult {
  is_valid: boolean;
  field_validations: Record<string, unknown>;
  cross_field_validations: unknown[];
  hallucination_flags: string[];
  warnings: string[];
  errors: string[];
}

// Processing Metadata
export interface ProcessingMetadata {
  processing_time_ms: number;
  vlm_calls: number;
  retries: number;
  pages_processed: number;
  fields_extracted: number;
  validation_checks: number;
}

// Process Request
export interface ProcessRequest {
  pdf_path: string;
  schema_name?: string;
  export_format?: ExportFormat;
  output_dir?: string;
  mask_phi?: boolean;
  priority?: ProcessingPriority;
  async_processing?: boolean;
  callback_url?: string;
}

// Process Response
export interface ProcessResponse {
  processing_id: string;
  status: TaskStatus;
  data: Record<string, unknown>;
  field_metadata: Record<string, FieldResult>;
  validation?: ValidationResult;
  metadata?: ProcessingMetadata;
  overall_confidence: number;
  confidence_level: ConfidenceLevel;
  requires_human_review: boolean;
  human_review_reason?: string;
  output_path?: string;
  errors: string[];
  warnings: string[];
}

// Async Process Response
export interface AsyncProcessResponse {
  task_id: string;
  status: TaskStatus;
  message: string;
  status_url: string;
}

// Batch Request
export interface BatchProcessRequest {
  pdf_paths: string[];
  schema_name?: string;
  export_format?: ExportFormat;
  output_dir: string;
  mask_phi?: boolean;
  stop_on_error?: boolean;
  async_processing?: boolean;
}

// Batch Item Result
export interface BatchItemResult {
  pdf_path: string;
  processing_id: string;
  status: TaskStatus;
  output_path?: string;
  error?: string;
}

// Batch Response
export interface BatchProcessResponse {
  batch_id: string;
  status: TaskStatus;
  total_documents: number;
  successful: number;
  failed: number;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  results: BatchItemResult[];
}

// Task Status Response
export interface TaskStatusResponse {
  task_id: string;
  status: string;
  ready: boolean;
  successful?: boolean;
  progress?: {
    current: number;
    total: number;
    stage: string;
  };
  result?: ProcessResponse | BatchProcessResponse;
  error?: string;
}

// Preview Request
export interface PreviewRequest {
  processing_id: string;
  style?: PreviewStyle;
  include_confidence?: boolean;
  include_validation?: boolean;
  mask_phi?: boolean;
}

// Preview Response
export interface PreviewResponse {
  processing_id: string;
  format: string;
  content: string;
  generated_at: string;
}

// Health Response
export interface HealthResponse {
  status: 'healthy' | 'degraded';
  version: string;
  timestamp: string;
  components: Record<string, ComponentHealth>;
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms?: number;
  error?: string;
  [key: string]: unknown;
}

// Schema Info
export interface SchemaInfo {
  name: string;
  description: string;
  document_type: string;
  field_count: number;
  version: string;
}

export interface SchemaField {
  name: string;
  type: string;
  required: boolean;
  description?: string;
  validation_rules?: string[];
}

// Worker Status
export interface WorkerStatus {
  name: string;
  status: 'online' | 'offline';
  active_tasks: number;
  processed: number;
  failed: number;
}

// Queue Stats
export interface QueueStats {
  name: string;
  pending: number;
  active: number;
  reserved: number;
  scheduled: number;
}

// Error Response
export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

// Auth Types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  roles: string[];
  permissions: string[];
}

// Dashboard Metrics
export interface DashboardMetrics {
  documents_processed_today: number;
  documents_processed_week: number;
  success_rate: number;
  average_processing_time: number;
  active_tasks: number;
  pending_tasks: number;
  failed_tasks_today: number;
  human_review_pending: number;
}

// Recent Activity
export interface RecentActivity {
  id: string;
  type: 'process' | 'export' | 'review' | 'error';
  description: string;
  timestamp: string;
  status: TaskStatus;
  document_name?: string;
}
