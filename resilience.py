"""Retry mechanisms and circuit breakers for DeepseekOllamaRag application."""

import time
import random
from functools import wraps
from typing import Callable, Any, Dict, Optional, Type, Union, List
from datetime import datetime, timedelta
from enum import Enum
import threading
from dataclasses import dataclass

from logging_config import global_logger as logger
from exceptions import DeepSeekRAGException, OllamaConnectionError, LLMError


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OllamaConnectionError)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitored_exceptions: tuple = (ConnectionError, TimeoutError, OllamaConnectionError, LLMError)


class CircuitBreaker:
    """Circuit breaker implementation for service protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN: {func.__name__}")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN for {func.__name__}. "
                        f"Next attempt in {self._time_until_reset():.1f} seconds"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.monitored_exceptions as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_seconds
    
    def _time_until_reset(self) -> float:
        """Calculate time until circuit reset attempt."""
        if self.last_failure_time is None:
            return 0
        
        time_since_failure = time.time() - self.last_failure_time
        return max(0, self.config.timeout_seconds - time_since_failure)
    
    def _on_success(self):
        """Handle successful function execution."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker transitioned to CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed function execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker transitioned to OPEN from HALF_OPEN")
            elif (self.state == CircuitState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "time_until_reset": self._time_until_reset() if self.state == CircuitState.OPEN else 0
            }


class RetryHandler:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.2f}s: {str(e)}"
                )
                time.sleep(delay)
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func.__name__}: {str(e)}")
                raise
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            jitter = random.uniform(0, 0.1) * delay
            delay += jitter
        
        return delay


class ResilienceManager:
    """Centralized resilience management."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.default_retry_config = RetryConfig()
        self.default_circuit_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if name not in self.circuit_breakers:
            circuit_config = config or self.default_circuit_config
            self.circuit_breakers[name] = CircuitBreaker(circuit_config)
        return self.circuit_breakers[name]
    
    def get_retry_handler(self, name: str, config: Optional[RetryConfig] = None) -> RetryHandler:
        """Get or create retry handler for service."""
        if name not in self.retry_handlers:
            retry_config = config or self.default_retry_config
            self.retry_handlers[name] = RetryHandler(retry_config)
        return self.retry_handlers[name]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all resilience components."""
        return {
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            "retry_handlers": list(self.retry_handlers.keys()),
            "timestamp": datetime.now().isoformat()
        }


# Decorators for easy use
def with_retry(config: Optional[RetryConfig] = None, service_name: str = "default"):
    """Decorator to add retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = resilience_manager.get_retry_handler(service_name, config)
            return retry_handler.retry(func, *args, **kwargs)
        return wrapper
    return decorator


def with_circuit_breaker(config: Optional[CircuitBreakerConfig] = None, service_name: str = "default"):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            circuit_breaker = resilience_manager.get_circuit_breaker(service_name, config)
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_resilience(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    service_name: str = "default"
):
    """Decorator to add both retry and circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            circuit_breaker = resilience_manager.get_circuit_breaker(service_name, circuit_config)
            retry_handler = resilience_manager.get_retry_handler(service_name, retry_config)
            
            def resilient_func():
                return circuit_breaker.call(func, *args, **kwargs)
            
            return retry_handler.retry(resilient_func)
        return wrapper
    return decorator


# Custom exceptions
class CircuitBreakerOpenError(DeepSeekRAGException):
    """Raised when circuit breaker is open."""
    pass


class RetryExhaustedError(DeepSeekRAGException):
    """Raised when all retry attempts are exhausted."""
    pass


# Global resilience manager
resilience_manager = ResilienceManager()


# Predefined configurations for different services
OLLAMA_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError, OllamaConnectionError)
)

OLLAMA_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=30,
    monitored_exceptions=(ConnectionError, TimeoutError, OllamaConnectionError)
)

EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=1.5,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError)
)

EMBEDDING_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=3,
    timeout_seconds=60,
    monitored_exceptions=(ConnectionError, TimeoutError)
)


def configure_resilience_for_services():
    """Configure resilience for specific services."""
    # Configure Ollama service
    resilience_manager.get_circuit_breaker("ollama", OLLAMA_CIRCUIT_CONFIG)
    resilience_manager.get_retry_handler("ollama", OLLAMA_RETRY_CONFIG)
    
    # Configure embedding service
    resilience_manager.get_circuit_breaker("embedding", EMBEDDING_CIRCUIT_CONFIG)
    resilience_manager.get_retry_handler("embedding", EMBEDDING_RETRY_CONFIG)
    
    logger.info("Resilience configurations applied for all services")


# Initialize service configurations
configure_resilience_for_services()