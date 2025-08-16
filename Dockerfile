# Multi-stage Docker build for DeepSeek RAG Application
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black isort flake8 mypy bandit safety

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to appuser
USER appuser

# Create necessary directories
RUN mkdir -p logs cache temp uploads

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Development command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Install production-only dependencies if any
# RUN pip install gunicorn uvicorn

# Copy only necessary files for production
COPY app.py .
COPY core.py .
COPY config.py .
COPY validation.py .
COPY security.py .
COPY caching.py .
COPY monitoring.py .
COPY resilience.py .
COPY async_processing.py .
COPY logging_config.py .
COPY exceptions.py .
COPY utils.py .

# Create application directories
RUN mkdir -p logs cache temp uploads && \
    chown -R appuser:appuser /app

# Switch to appuser
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Production command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ tests/
COPY pytest.ini .

# Run tests during build
RUN python -m pytest tests/ -m "not integration and not performance" --tb=short

# Final production image
FROM production as final

# Add labels for metadata
LABEL maintainer="DeepSeek RAG Team" \
      version="1.0.0" \
      description="DeepSeek RAG Application for document Q&A" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/DeepseekOllamaRag"

# Final setup
USER appuser