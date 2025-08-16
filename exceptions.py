"""Custom exceptions for DeepseekOllamaRag application."""

from typing import Optional, Any


class DeepSeekRAGException(Exception):
    """Base exception for DeepSeek RAG application."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(DeepSeekRAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingGenerationError(DeepSeekRAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(DeepSeekRAGException):
    """Raised when vector store operations fail."""
    pass


class LLMError(DeepSeekRAGException):
    """Raised when LLM operations fail."""
    pass


class ConfigurationError(DeepSeekRAGException):
    """Raised when configuration is invalid."""
    pass


class ValidationError(DeepSeekRAGException):
    """Raised when input validation fails."""
    pass


class FileProcessingError(DeepSeekRAGException):
    """Raised when file processing fails."""
    pass


class OllamaConnectionError(LLMError):
    """Raised when connection to Ollama fails."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when specified model is not found."""
    pass