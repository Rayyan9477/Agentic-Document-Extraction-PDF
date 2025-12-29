# =============================================================================
# Local Agentic Medical Document Extraction System
# Multi-stage Dockerfile for optimized production builds
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build artifacts
# -----------------------------------------------------------------------------
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.12-slim as runtime

# Labels for container metadata
LABEL maintainer="Document Extraction Team" \
      version="2.0.0" \
      description="HIPAA-compliant document extraction system"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    APP_PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/temp /app/output && \
    chown -R appuser:appgroup /app/data /app/logs /app/temp /app/output

# Switch to non-root user
USER appuser

# Expose port
EXPOSE ${APP_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/health || exit 1

# Default command - run the API server
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
