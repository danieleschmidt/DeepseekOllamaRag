# API Documentation

## Overview

This document provides comprehensive API documentation for the DeepSeek RAG application modules.

## Core Modules

### `core.py` - Main RAG Pipeline

#### Classes

##### `DocumentProcessor`
Handles document loading and processing with optimization.

**Methods:**
- `load_document(file_path: str) -> List[Document]`: Load document from file path
- `split_documents(documents: List[Document]) -> List[Document]`: Split documents into optimized chunks
- `_hybrid_chunking(documents: List[Document]) -> List[Document]`: Hybrid chunking strategy
- `_generate_content_hash(contents: List[str]) -> str`: Generate hash for document contents

##### `EmbeddingManager`
Handle embedding generation and management with caching and optimization.

**Methods:**
- `create_vector_store(documents: List[Document]) -> FAISS`: Create optimized FAISS vector store
- `get_retriever(vector_store: FAISS)`: Get retriever from vector store
- `_create_vector_store_batched(documents: List[Document]) -> FAISS`: Create vector store using batch processing

##### `LLMManager`
Handle LLM operations and chain creation with resilience.

**Methods:**
- `create_qa_chain(retriever) -> RetrievalQA`: Create QA chain with retriever
- `ask_question(qa_chain: RetrievalQA, question: str, doc_hash: str) -> Dict[str, Any]`: Ask question with caching and resilience

##### `RAGPipeline`
Main RAG pipeline orchestrator.

**Methods:**
- `process_document(file_path: str) -> Tuple[FAISS, RetrievalQA, str]`: Process document and return components
- `ask_question(qa_chain: RetrievalQA, question: str, doc_hash: str) -> Dict[str, Any]`: Ask question using QA chain

### `config.py` - Configuration Management

#### Classes

##### `ModelConfig`
Configuration for AI models.

**Attributes:**
- `llm_model: str`: LLM model name
- `embedding_model: str`: Embedding model name
- `embedding_device: str`: Device for embeddings
- `ollama_base_url: str`: Ollama service URL
- `max_tokens: int`: Maximum tokens
- `temperature: float`: Model temperature

##### `VectorStoreConfig`
Configuration for vector storage.

**Attributes:**
- `similarity_search_k: int`: Number of similarity search results
- `search_type: str`: Type of search
- `chunk_size: int`: Document chunk size
- `chunk_overlap: int`: Overlap between chunks

##### `AppConfig`
Main application configuration.

**Attributes:**
- `debug: bool`: Debug mode
- `upload_dir: str`: Upload directory
- `temp_dir: str`: Temporary directory
- `max_file_size_mb: int`: Maximum file size
- `allowed_file_types: list`: Allowed file types
- `session_timeout_minutes: int`: Session timeout

##### `Config`
Main configuration class that loads from environment variables.

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert configuration to dictionary

### `validation.py` - Input Validation

#### Classes

##### `ValidationResult`
Result of validation operation.

**Attributes:**
- `is_valid: bool`: Whether validation passed
- `errors: List[str]`: List of validation errors
- `warnings: List[str]`: List of validation warnings
- `metadata: Optional[Dict[str, Any]]`: Additional metadata

##### `FileValidator`
Comprehensive file validation.

**Methods:**
- `validate_file(file_data: bytes, filename: str) -> ValidationResult`: Comprehensive file validation
- `_validate_filename(filename: str) -> List[str]`: Validate filename security
- `_validate_file_size(file_data: bytes) -> List[str]`: Validate file size
- `_validate_file_type(file_data: bytes, filename: str) -> Tuple[List[str], Dict[str, Any]]`: Validate file type
- `_validate_file_content(file_data: bytes) -> List[str]`: Validate file content
- `_validate_security(file_data: bytes, filename: str) -> Tuple[List[str], List[str]]`: Security validation

##### `TextValidator`
Text input validation and sanitization.

**Methods:**
- `validate_question(question: str) -> ValidationResult`: Validate user question input
- `_sanitize_text(text: str) -> str`: Sanitize text input
- `_validate_text_content(text: str) -> Tuple[List[str], List[str]]`: Validate text content

##### `ConfigValidator`
Configuration validation.

**Methods:**
- `validate_config() -> ValidationResult`: Validate application configuration

### `security.py` - Security Management

#### Classes

##### `SecurityManager`
Centralized security management.

**Methods:**
- `generate_session_token(user_id: str) -> str`: Generate secure session token
- `validate_session_token(token: str) -> bool`: Validate session token
- `create_secure_temp_file(suffix: str, content: bytes) -> str`: Create secure temporary file
- `secure_delete_file(file_path: str)`: Securely delete file with overwriting

##### `RateLimiter`
Rate limiting for API endpoints.

**Methods:**
- `is_allowed(identifier: str) -> Tuple[bool, Optional[int]]`: Check if request is allowed

##### `InputSanitizer`
Sanitize and validate inputs for security.

**Methods:**
- `sanitize_filename(filename: str) -> str`: Sanitize filename for security
- `sanitize_text_input(text: str, max_length: int) -> str`: Sanitize text input
- `validate_file_hash(file_content: bytes, expected_hash: str) -> str`: Calculate and validate file hash

##### `SecurityAuditor`
Security auditing and logging.

**Methods:**
- `log_security_event(event_type: str, details: Dict[str, Any], severity: str)`: Log security event
- `get_security_summary(hours: int) -> Dict[str, Any]`: Get security summary

### `caching.py` - Caching System

#### Classes

##### `MemoryCache`
In-memory cache with LRU eviction and TTL support.

**Methods:**
- `get(key: str) -> Optional[Any]`: Get value from cache
- `set(key: str, value: Any, ttl: Optional[int], tags: Optional[List[str]])`: Set value in cache
- `delete(key: str) -> bool`: Delete key from cache
- `clear()`: Clear all cache entries
- `delete_by_tags(tags: List[str])`: Delete entries with specified tags
- `cleanup_expired()`: Remove expired entries
- `get_stats() -> Dict[str, Any]`: Get cache statistics

##### `DiskCache`
Persistent disk-based cache.

**Methods:**
- `get(key: str) -> Optional[Any]`: Get value from disk cache
- `set(key: str, value: Any, ttl: Optional[int], tags: Optional[List[str]])`: Set value in disk cache
- `delete(key: str) -> bool`: Delete key from disk cache
- `clear()`: Clear all cache entries

##### `SmartCache`
Intelligent cache that combines memory and disk caching.

**Methods:**
- `get(key: str) -> Optional[Any]`: Get value from cache
- `set(key: str, value: Any, ttl: Optional[int], memory_only: bool, tags: Optional[List[str]])`: Set value in cache
- `delete(key: str) -> bool`: Delete key from both caches
- `clear()`: Clear both caches
- `get_stats() -> Dict[str, Any]`: Get combined cache statistics

#### Decorators

##### `@cached`
Decorator to cache function results.

**Parameters:**
- `cache_key_func: Optional[Callable]`: Function to generate cache key
- `ttl: int`: Time to live in seconds
- `cache_instance: Optional[SmartCache]`: Cache instance to use

### `monitoring.py` - Health Checks and Monitoring

#### Classes

##### `HealthStatus`
Health status information.

**Attributes:**
- `component: str`: Component name
- `status: str`: Status (healthy, unhealthy, degraded, unknown)
- `timestamp: datetime`: When status was checked
- `response_time_ms: Optional[float]`: Response time
- `details: Optional[Dict[str, Any]]`: Additional details
- `error_message: Optional[str]`: Error message if unhealthy

##### `SystemMetrics`
System performance metrics.

**Attributes:**
- `timestamp: datetime`: When metrics were collected
- `cpu_percent: float`: CPU usage percentage
- `memory_percent: float`: Memory usage percentage
- `memory_used_mb: float`: Used memory in MB
- `memory_available_mb: float`: Available memory in MB
- `disk_usage_percent: float`: Disk usage percentage
- `disk_free_gb: float`: Free disk space in GB

##### `HealthChecker`
Comprehensive health checking system.

**Methods:**
- `check_all() -> Dict[str, HealthStatus]`: Run all health checks
- `get_overall_status() -> str`: Get overall system health status

##### `PerformanceMonitor`
Performance monitoring and metrics collection.

**Methods:**
- `start_monitoring(interval_seconds: int)`: Start continuous monitoring
- `stop_monitoring()`: Stop continuous monitoring
- `collect_metrics() -> SystemMetrics`: Collect current system metrics
- `get_metrics_summary(hours: int) -> Dict[str, Any]`: Get metrics summary
- `export_metrics(file_path: str)`: Export metrics to JSON file

### `resilience.py` - Retry Mechanisms and Circuit Breakers

#### Classes

##### `CircuitBreaker`
Circuit breaker implementation for service protection.

**Methods:**
- `call(func: Callable, *args, **kwargs) -> Any`: Execute function with circuit breaker protection
- `get_state() -> Dict[str, Any]`: Get current circuit breaker state

##### `RetryHandler`
Advanced retry mechanism with exponential backoff.

**Methods:**
- `retry(func: Callable, *args, **kwargs) -> Any`: Execute function with retry logic

##### `ResilienceManager`
Centralized resilience management.

**Methods:**
- `get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig]) -> CircuitBreaker`: Get or create circuit breaker
- `get_retry_handler(name: str, config: Optional[RetryConfig]) -> RetryHandler`: Get or create retry handler
- `get_status() -> Dict[str, Any]`: Get status of all resilience components

#### Decorators

##### `@with_retry`
Decorator to add retry logic to functions.

##### `@with_circuit_breaker`
Decorator to add circuit breaker protection to functions.

##### `@with_resilience`
Decorator to add both retry and circuit breaker protection.

### `async_processing.py` - Asynchronous Processing

#### Classes

##### `AsyncTask`
Asynchronous task representation.

**Attributes:**
- `id: str`: Task ID
- `name: str`: Task name
- `func: Callable`: Function to execute
- `args: tuple`: Function arguments
- `kwargs: dict`: Function keyword arguments
- `status: TaskStatus`: Task status
- `result: Any`: Task result
- `progress: float`: Task progress (0.0 to 1.0)

##### `TaskQueue`
Thread-safe task queue with priority support.

**Methods:**
- `add_task(task: AsyncTask, priority: int) -> str`: Add task to queue
- `get_task(timeout: Optional[float]) -> Optional[AsyncTask]`: Get next task from queue
- `get_task_status(task_id: str) -> Optional[AsyncTask]`: Get task status by ID
- `cancel_task(task_id: str) -> bool`: Cancel pending task

##### `AsyncWorker`
Worker thread for processing async tasks.

**Methods:**
- `start()`: Start the worker thread
- `stop(timeout: float)`: Stop the worker thread
- `get_worker_stats() -> Dict[str, Any]`: Get worker statistics

##### `AsyncTaskManager`
Main async task management system.

**Methods:**
- `start()`: Start all workers
- `stop(timeout: float)`: Stop all workers
- `submit_task(func: Callable, *args, **kwargs) -> str`: Submit task for async execution
- `get_task_status(task_id: str) -> Optional[AsyncTask]`: Get task status
- `wait_for_task(task_id: str, timeout: Optional[float]) -> Any`: Wait for task completion
- `cancel_task(task_id: str) -> bool`: Cancel task
- `get_stats() -> Dict[str, Any]`: Get system statistics

##### `ProgressTracker`
Track progress of long-running tasks.

**Methods:**
- `update(step_name: str, increment: int)`: Update progress
- `get_progress() -> Dict[str, Any]`: Get current progress

#### Decorators

##### `@async_task`
Decorator to make function async-processable.

##### `@with_progress_tracking`
Decorator to add progress tracking to functions.

## Global Instances

- `global_cache`: Global SmartCache instance
- `health_checker`: Global HealthChecker instance
- `performance_monitor`: Global PerformanceMonitor instance
- `security_manager`: Global SecurityManager instance
- `rate_limiter`: Global RateLimiter instance
- `resilience_manager`: Global ResilienceManager instance
- `global_task_manager`: Global AsyncTaskManager instance

## Utility Functions

### Cache Key Generators
- `cache_embedding_key(text: str, model: str) -> str`: Generate cache key for embeddings
- `cache_qa_key(question: str, doc_hash: str, model: str) -> str`: Generate cache key for Q&A results

### Security Functions
- `secure_file_processing(file_content: bytes, filename: str) -> Tuple[str, str]`: Securely process uploaded file
- `check_security_health() -> Dict[str, Any]`: Check overall security health
- `cleanup_security_resources()`: Clean up security resources

### Monitoring Functions
- `get_system_status() -> Dict[str, Any]`: Get comprehensive system status
- `start_background_monitoring()`: Start background monitoring services
- `stop_background_monitoring()`: Stop background monitoring services

### Async Processing Functions
- `start_async_processing()`: Start global async processing
- `stop_async_processing(timeout: float)`: Stop global async processing
- `get_async_stats() -> Dict[str, Any]`: Get async processing statistics

## Error Handling

All modules use custom exceptions defined in `exceptions.py`:

- `DeepSeekRAGException`: Base exception
- `DocumentProcessingError`: Document processing failures
- `EmbeddingGenerationError`: Embedding generation failures
- `VectorStoreError`: Vector store operation failures
- `LLMError`: LLM operation failures
- `ValidationError`: Input validation failures
- `SecurityException`: Security-related errors
- `CircuitBreakerOpenError`: Circuit breaker is open
- `RetryExhaustedError`: All retry attempts exhausted

## Performance Considerations

- All time-sensitive operations are decorated with `@log_execution_time`
- Critical services use `@with_resilience` decorators
- Caching is implemented at multiple levels (memory, disk, and application)
- Async processing is used for long-running operations
- Health checks and monitoring provide real-time performance insights