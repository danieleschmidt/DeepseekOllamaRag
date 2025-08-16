"""Input validation and sanitization for DeepseekOllamaRag application."""

import re
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import config
from exceptions import ValidationError
from logging_config import global_logger as logger


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Optional[Dict[str, Any]] = None


class FileValidator:
    """Comprehensive file validation."""
    
    # Dangerous file extensions and patterns
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
        '.app', '.deb', '.pkg', '.dmg', '.bin', '.run', '.msi', '.ps1'
    }
    
    # PDF magic bytes
    PDF_MAGIC_BYTES = [
        b'%PDF-',
        b'\x25\x50\x44\x46\x2d'
    ]
    
    def __init__(self):
        self.max_filename_length = 255
        self.max_path_length = 4096
    
    def validate_file(self, file_data: bytes, filename: str) -> ValidationResult:
        """Comprehensive file validation."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic filename validation
            filename_errors = self._validate_filename(filename)
            errors.extend(filename_errors)
            
            # File size validation
            size_errors = self._validate_file_size(file_data)
            errors.extend(size_errors)
            
            # File type validation
            type_errors, type_metadata = self._validate_file_type(file_data, filename)
            errors.extend(type_errors)
            metadata.update(type_metadata)
            
            # Content validation
            content_errors = self._validate_file_content(file_data)
            errors.extend(content_errors)
            
            # Security validation
            security_errors, security_warnings = self._validate_security(file_data, filename)
            errors.extend(security_errors)
            warnings.extend(security_warnings)
            
            # Generate file metadata
            metadata.update(self._generate_file_metadata(file_data, filename))
            
        except Exception as e:
            logger.error(f"Error during file validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _validate_filename(self, filename: str) -> List[str]:
        """Validate filename security and format."""
        errors = []
        
        if not filename:
            errors.append("Filename cannot be empty")
            return errors
        
        # Length check
        if len(filename) > self.max_filename_length:
            errors.append(f"Filename too long (max {self.max_filename_length} characters)")
        
        # Character validation
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in filename for char in dangerous_chars):
            errors.append("Filename contains dangerous characters")
        
        # Path traversal check
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            errors.append("Filename contains path traversal patterns")
        
        # Reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }\n        \n        base_name = Path(filename).stem.upper()\n        if base_name in reserved_names:\n            errors.append(\"Filename uses reserved system name\")\n        \n        return errors\n    \n    def _validate_file_size(self, file_data: bytes) -> List[str]:\n        \"\"\"Validate file size.\"\"\"\n        errors = []\n        \n        size_bytes = len(file_data)\n        size_mb = size_bytes / (1024 * 1024)\n        \n        if size_bytes == 0:\n            errors.append(\"File is empty\")\n        \n        if size_mb > config.app.max_file_size_mb:\n            errors.append(\n                f\"File size ({size_mb:.1f}MB) exceeds limit of {config.app.max_file_size_mb}MB\"\n            )\n        \n        return errors\n    \n    def _validate_file_type(self, file_data: bytes, filename: str) -> Tuple[List[str], Dict[str, Any]]:\n        \"\"\"Validate file type and extension.\"\"\"\n        errors = []\n        metadata = {}\n        \n        # Get file extension\n        file_extension = Path(filename).suffix.lower()\n        metadata['extension'] = file_extension\n        \n        # Check against allowed types\n        if file_extension.lstrip('.') not in config.app.allowed_file_types:\n            errors.append(\n                f\"File type '{file_extension}' not allowed. \"\n                f\"Allowed types: {', '.join(config.app.allowed_file_types)}\"\n            )\n        \n        # Check for dangerous extensions\n        if file_extension in self.DANGEROUS_EXTENSIONS:\n            errors.append(f\"Dangerous file type detected: {file_extension}\")\n        \n        # MIME type detection\n        mime_type, _ = mimetypes.guess_type(filename)\n        metadata['mime_type'] = mime_type\n        \n        return errors, metadata\n    \n    def _validate_file_content(self, file_data: bytes) -> List[str]:\n        \"\"\"Validate file content and magic bytes.\"\"\"\n        errors = []\n        \n        if not file_data:\n            errors.append(\"File content is empty\")\n            return errors\n        \n        # Check PDF magic bytes\n        is_pdf = any(file_data.startswith(magic) for magic in self.PDF_MAGIC_BYTES)\n        \n        if not is_pdf:\n            errors.append(\"File does not appear to be a valid PDF\")\n        \n        # Check for embedded scripts or suspicious content\n        suspicious_patterns = [\n            b'<script',\n            b'javascript:',\n            b'/JS ',\n            b'/JavaScript',\n            b'/Launch',\n        ]\n        \n        for pattern in suspicious_patterns:\n            if pattern in file_data:\n                errors.append(\"Suspicious content detected in file\")\n                break\n        \n        return errors\n    \n    def _validate_security(self, file_data: bytes, filename: str) -> Tuple[List[str], List[str]]:\n        \"\"\"Security validation checks.\"\"\"\n        errors = []\n        warnings = []\n        \n        # Check file size for potential zip bombs\n        if len(file_data) > 100 * 1024 * 1024:  # 100MB\n            warnings.append(\"Large file detected - potential security risk\")\n        \n        # Check for unusual file patterns\n        if b'\\x00' * 1000 in file_data:  # Large blocks of null bytes\n            warnings.append(\"Unusual file pattern detected\")\n        \n        # Check filename for encoding issues\n        try:\n            filename.encode('utf-8')\n        except UnicodeEncodeError:\n            errors.append(\"Filename contains invalid Unicode characters\")\n        \n        return errors, warnings\n    \n    def _generate_file_metadata(self, file_data: bytes, filename: str) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive file metadata.\"\"\"\n        return {\n            'size_bytes': len(file_data),\n            'size_mb': len(file_data) / (1024 * 1024),\n            'sha256': hashlib.sha256(file_data).hexdigest(),\n            'md5': hashlib.md5(file_data).hexdigest(),\n            'filename': filename,\n            'validated_at': datetime.now().isoformat()\n        }\n\n\nclass TextValidator:\n    \"\"\"Text input validation and sanitization.\"\"\"\n    \n    def __init__(self):\n        self.max_question_length = 2000\n        self.min_question_length = 3\n    \n    def validate_question(self, question: str) -> ValidationResult:\n        \"\"\"Validate user question input.\"\"\"\n        errors = []\n        warnings = []\n        metadata = {}\n        \n        if not question or not question.strip():\n            errors.append(\"Question cannot be empty\")\n            return ValidationResult(False, errors, warnings, metadata)\n        \n        # Sanitize the question\n        sanitized_question = self._sanitize_text(question)\n        metadata['original_length'] = len(question)\n        metadata['sanitized_length'] = len(sanitized_question)\n        metadata['sanitized_question'] = sanitized_question\n        \n        # Length validation\n        if len(sanitized_question) < self.min_question_length:\n            errors.append(f\"Question too short (minimum {self.min_question_length} characters)\")\n        \n        if len(sanitized_question) > self.max_question_length:\n            errors.append(f\"Question too long (maximum {self.max_question_length} characters)\")\n        \n        # Content validation\n        content_errors, content_warnings = self._validate_text_content(sanitized_question)\n        errors.extend(content_errors)\n        warnings.extend(content_warnings)\n        \n        return ValidationResult(\n            is_valid=len(errors) == 0,\n            errors=errors,\n            warnings=warnings,\n            metadata=metadata\n        )\n    \n    def _sanitize_text(self, text: str) -> str:\n        \"\"\"Sanitize text input.\"\"\"\n        if not text:\n            return \"\"\n        \n        # Remove control characters except newlines and tabs\n        sanitized = re.sub(r'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]', '', text)\n        \n        # Normalize whitespace\n        sanitized = re.sub(r'\\s+', ' ', sanitized).strip()\n        \n        # Remove potential injection patterns\n        dangerous_patterns = [\n            r'<script[^>]*>.*?</script>',\n            r'javascript:',\n            r'data:',\n            r'vbscript:',\n        ]\n        \n        for pattern in dangerous_patterns:\n            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)\n        \n        return sanitized\n    \n    def _validate_text_content(self, text: str) -> Tuple[List[str], List[str]]:\n        \"\"\"Validate text content for suspicious patterns.\"\"\"\n        errors = []\n        warnings = []\n        \n        # Check for repeated characters (potential attack)\n        if re.search(r'(.)\\1{100,}', text):\n            warnings.append(\"Text contains excessive repeated characters\")\n        \n        # Check for potential prompt injection\n        injection_patterns = [\n            r'ignore.{0,20}previous.{0,20}instructions',\n            r'system.{0,10}prompt',\n            r'act.{0,10}as.{0,10}if',\n            r'pretend.{0,10}you.{0,10}are',\n        ]\n        \n        for pattern in injection_patterns:\n            if re.search(pattern, text, re.IGNORECASE):\n                warnings.append(\"Potential prompt injection detected\")\n                break\n        \n        return errors, warnings\n\n\nclass ConfigValidator:\n    \"\"\"Configuration validation.\"\"\"\n    \n    @staticmethod\n    def validate_config() -> ValidationResult:\n        \"\"\"Validate application configuration.\"\"\"\n        errors = []\n        warnings = []\n        metadata = {}\n        \n        # Model configuration\n        if not config.model.llm_model:\n            errors.append(\"LLM model not specified\")\n        \n        if not config.model.embedding_model:\n            errors.append(\"Embedding model not specified\")\n        \n        # Vector store configuration\n        if config.vector_store.similarity_search_k <= 0:\n            errors.append(\"Invalid similarity search k value\")\n        \n        if config.vector_store.chunk_size <= 0:\n            errors.append(\"Invalid chunk size\")\n        \n        # App configuration\n        if config.app.max_file_size_mb <= 0:\n            errors.append(\"Invalid max file size\")\n        \n        if not config.app.allowed_file_types:\n            errors.append(\"No allowed file types specified\")\n        \n        # Directory checks\n        required_dirs = [config.app.upload_dir, config.app.temp_dir]\n        for dir_path in required_dirs:\n            if not os.path.exists(dir_path):\n                try:\n                    os.makedirs(dir_path, exist_ok=True)\n                    metadata[f\"created_dir_{dir_path}\"] = True\n                except Exception as e:\n                    errors.append(f\"Cannot create directory {dir_path}: {str(e)}\")\n        \n        return ValidationResult(\n            is_valid=len(errors) == 0,\n            errors=errors,\n            warnings=warnings,\n            metadata=metadata\n        )\n\n\n# Global validator instances\nfile_validator = FileValidator()\ntext_validator = TextValidator()\nconfig_validator = ConfigValidator()"}, {"old_string": "        base_name = Path(filename).stem.upper()\n        if base_name in reserved_names:\n            errors.append(\"Filename uses reserved system name\")", "new_string": "        base_name = Path(filename).stem.upper()\n        if base_name in reserved_names:\n            errors.append(\"Filename uses reserved system name\")"}]