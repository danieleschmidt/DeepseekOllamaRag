"""
Comprehensive validation utilities for the RAG system.
"""

import os
import re
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from config import config
from exceptions import ValidationError
from utils import setup_logging

logger = setup_logging()


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class FileValidator:
    """Validates file uploads and properties."""
    
    def __init__(self):
        self.allowed_extensions = {'.pdf'}
        self.max_file_size = config.app.max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.dangerous_patterns = [
            r'\.exe$', r'\.bat$', r'\.cmd$', r'\.com$',
            r'\.scr$', r'\.vbs$', r'\.js$', r'\.jar$'
        ]
    
    def validate_file(self, file_data: bytes, filename: str) -> ValidationResult:
        """Validate uploaded file."""
        errors = []
        warnings = []
        metadata = {}
        
        # File size validation
        file_size = len(file_data)
        metadata['size_bytes'] = file_size
        
        if file_size == 0:
            errors.append("File is empty")
        elif file_size > self.max_file_size:
            errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
        
        # Filename validation
        if not filename:
            errors.append("Filename is required")
        else:
            filename_validation = self._validate_filename(filename)
            errors.extend(filename_validation['errors'])
            warnings.extend(filename_validation['warnings'])
            metadata.update(filename_validation['metadata'])
        
        # File extension validation
        if filename:
            ext = Path(filename).suffix.lower()
            metadata['extension'] = ext
            
            if ext not in self.allowed_extensions:
                errors.append(f"File type '{ext}' is not allowed. Allowed types: {list(self.allowed_extensions)}")
        
        # Content validation
        if file_data:
            content_validation = self._validate_file_content(file_data, filename)
            errors.extend(content_validation['errors'])
            warnings.extend(content_validation['warnings'])
            metadata.update(content_validation['metadata'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _validate_filename(self, filename: str) -> Dict[str, Any]:
        """Validate filename for security and compliance."""
        errors = []
        warnings = []
        metadata = {}
        
        # Length check
        if len(filename) > 255:
            errors.append("Filename is too long (max 255 characters)")
        
        # Character validation
        invalid_chars = set('<>:"|?*')
        found_invalid = [char for char in filename if char in invalid_chars]
        if found_invalid:
            errors.append(f"Filename contains invalid characters: {found_invalid}")
        
        # Path traversal check
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            errors.append("Filename contains path traversal patterns")
        
        # Dangerous pattern check
        for pattern in self.dangerous_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                errors.append(f"Filename matches dangerous pattern: {pattern}")
        
        # Reserved names (Windows)
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 
                         'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 
                         'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 
                         'LPT7', 'LPT8', 'LPT9'}
        
        base_name = Path(filename).stem.upper()
        if base_name in reserved_names:
            errors.append("Filename uses reserved system name")
        
        metadata['filename_length'] = len(filename)
        metadata['base_name'] = Path(filename).stem
        metadata['extension'] = Path(filename).suffix
        
        return {
            'errors': errors,
            'warnings': warnings,
            'metadata': metadata
        }
    
    def _validate_file_content(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate file content."""
        errors = []
        warnings = []
        metadata = {}
        
        # MIME type detection
        mime_type, _ = mimetypes.guess_type(filename)
        metadata['detected_mime_type'] = mime_type
        
        # Basic PDF validation
        if filename.lower().endswith('.pdf'):
            if not file_data.startswith(b'%PDF-'):
                errors.append("File does not appear to be a valid PDF (missing PDF header)")
            else:
                metadata['pdf_version'] = self._extract_pdf_version(file_data)
        
        # Check for embedded content
        suspicious_patterns = [
            b'<script', b'javascript:', b'vbscript:', 
            b'data:text/html', b'data:application/javascript'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in file_data:
                warnings.append(f"File contains potentially suspicious content: {pattern.decode('utf-8', errors='ignore')}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'metadata': metadata
        }
    
    def _extract_pdf_version(self, file_data: bytes) -> Optional[str]:
        """Extract PDF version from header."""
        try:
            header = file_data[:100]  # First 100 bytes should contain version
            version_match = re.search(rb'%PDF-(\d+\.\d+)', header)
            if version_match:
                return version_match.group(1).decode('ascii')
        except Exception:
            pass
        return None


class TextValidator:
    """Validates text input and content."""
    
    def __init__(self):
        self.max_length = 10000
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval() calls
            r'document\.',  # DOM manipulation
        ]
    
    def validate_text(self, text: str) -> ValidationResult:
        """Validate text input."""
        errors = []
        warnings = []
        metadata = {}
        
        if not text:
            return ValidationResult(True, [], [], {})
        
        # Length validation
        text_length = len(text)
        metadata['length'] = text_length
        
        if text_length > self.max_length:
            errors.append(f"Text is too long ({text_length} characters, max {self.max_length})")
        
        # Content validation
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                warnings.append(f"Text contains potentially suspicious pattern: {pattern}")
        
        # Check for control characters
        control_chars = [char for char in text if ord(char) < 32 and char not in '\n\r\t']
        if control_chars:
            warnings.append(f"Text contains control characters: {len(control_chars)} found")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text for safe use."""
        if not text:
            return ""
        
        # Remove control characters (except common whitespace)
        sanitized = ''.join(
            char for char in text 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # Limit length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]
        
        return sanitized.strip()


class ConfigValidator:
    """Validates configuration settings."""
    
    def validate_config(self) -> ValidationResult:
        """Validate current configuration."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Model configuration validation
            if not config.model.llm_model:
                errors.append("LLM model not specified")
            
            if not config.model.embedding_model:
                errors.append("Embedding model not specified")
            
            if config.model.temperature < 0 or config.model.temperature > 1:
                errors.append("Temperature must be between 0 and 1")
            
            # Vector store configuration
            if config.vector_store.similarity_search_k <= 0:
                errors.append("similarity_search_k must be positive")
            
            if config.vector_store.chunk_size <= 0:
                errors.append("chunk_size must be positive")
            
            if config.vector_store.chunk_overlap < 0:
                errors.append("chunk_overlap cannot be negative")
            
            if config.vector_store.chunk_overlap >= config.vector_store.chunk_size:
                warnings.append("chunk_overlap is >= chunk_size, may cause issues")
            
            # App configuration
            if config.app.max_file_size_mb <= 0:
                errors.append("max_file_size_mb must be positive")
            
            if config.app.session_timeout_minutes <= 0:
                errors.append("session_timeout_minutes must be positive")
            
            # Directory validation
            upload_dir = Path(config.app.upload_dir)
            temp_dir = Path(config.app.temp_dir)
            
            if not upload_dir.exists():
                warnings.append(f"Upload directory does not exist: {upload_dir}")
            
            if not temp_dir.exists():
                warnings.append(f"Temp directory does not exist: {temp_dir}")
            
            metadata['config_valid'] = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating configuration: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )


# Global validator instances
file_validator = FileValidator()
text_validator = TextValidator()
config_validator = ConfigValidator()


def validate_file_upload(file_data: bytes, filename: str) -> ValidationResult:
    """Validate file upload with comprehensive checks."""
    return file_validator.validate_file(file_data, filename)


def validate_user_input(text: str) -> ValidationResult:
    """Validate user text input."""
    return text_validator.validate_text(text)


def validate_system_config() -> ValidationResult:
    """Validate system configuration."""
    return config_validator.validate_config()


def sanitize_user_input(text: str) -> str:
    """Sanitize user text input."""
    return text_validator.sanitize_text(text)