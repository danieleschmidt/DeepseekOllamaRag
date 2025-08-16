"""Security measures and utilities for DeepseekOllamaRag application."""

import os
import hashlib
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
import time

from config import config
from logging_config import global_logger as logger
from exceptions import ValidationError


class SecurityManager:
    """Centralized security management."""
    
    def __init__(self):
        self.session_tokens = {}
        self.rate_limiters = {}
        self.blocked_ips = set()
        self.temp_files = set()
        self.cleanup_thread = None
        self.cleanup_running = False
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                self._cleanup_temp_files()
                self._cleanup_expired_sessions()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in security cleanup: {str(e)}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files older than 1 hour."""
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        for file_path in list(self.temp_files):
            try:
                if os.path.exists(file_path):
                    if os.path.getctime(file_path) < cutoff_time:
                        os.remove(file_path)
                        self.temp_files.discard(file_path)
                        logger.debug(f"Cleaned up temp file: {file_path}")
                else:
                    self.temp_files.discard(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")
    
    def _cleanup_expired_sessions(self):
        """Clean up expired session tokens."""
        current_time = datetime.now()
        expired_tokens = []
        
        for token, data in self.session_tokens.items():
            if current_time > data['expires_at']:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.session_tokens[token]
    
    def generate_session_token(self, user_id: str = "anonymous") -> str:
        """Generate secure session token."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=config.app.session_timeout_minutes)
        
        self.session_tokens[token] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'last_used': datetime.now()
        }
        
        return token
    
    def validate_session_token(self, token: str) -> bool:
        """Validate session token."""
        if token not in self.session_tokens:
            return False
        
        session_data = self.session_tokens[token]
        if datetime.now() > session_data['expires_at']:
            del self.session_tokens[token]
            return False
        
        # Update last used
        session_data['last_used'] = datetime.now()
        return True
    
    def create_secure_temp_file(self, suffix: str = '.tmp', content: bytes = None) -> str:
        """Create secure temporary file."""
        # Create temp file with secure permissions
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            dir=config.app.temp_dir,
            prefix='secure_'
        )
        
        try:
            # Set secure permissions (owner read/write only)
            os.chmod(temp_path, 0o600)
            
            if content:
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)
            else:
                os.close(fd)
            
            # Track for cleanup
            self.temp_files.add(temp_path)
            
            logger.debug(f"Created secure temp file: {temp_path}")
            return temp_path
            
        except Exception as e:
            try:
                os.close(fd)
                os.remove(temp_path)
            except:
                pass
            raise SecurityException(f"Failed to create secure temp file: {str(e)}")
    
    def secure_delete_file(self, file_path: str):
        """Securely delete file with overwriting."""
        try:
            if not os.path.exists(file_path):
                return
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite with random data
            with open(file_path, 'r+b') as f:
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
            
            # Remove file
            os.remove(file_path)
            self.temp_files.discard(file_path)
            
            logger.debug(f"Securely deleted file: {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to securely delete {file_path}: {str(e)}")


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self.blocked_until = {}
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed for identifier."""
        current_time = time.time()
        
        # Check if currently blocked
        if identifier in self.blocked_until:
            if current_time < self.blocked_until[identifier]:
                return False, int(self.blocked_until[identifier] - current_time)
            else:
                del self.blocked_until[identifier]
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            # Block for time window
            self.blocked_until[identifier] = current_time + self.time_window
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False, self.time_window
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True, None


class InputSanitizer:
    """Sanitize and validate inputs for security."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security."""
        if not filename:
            raise ValidationError("Empty filename")
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*\\/'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)\n            filename = name[:255-len(ext)] + ext\n        \n        # Ensure not empty after sanitization\n        if not filename.strip():\n            raise ValidationError(\"Invalid filename after sanitization\")\n        \n        return filename.strip()\n    \n    @staticmethod\n    def sanitize_text_input(text: str, max_length: int = 2000) -> str:\n        \"\"\"Sanitize text input.\"\"\"\n        if not text:\n            return \"\"\n        \n        # Remove control characters except newlines and tabs\n        sanitized = ''.join(\n            char for char in text\n            if ord(char) >= 32 or char in '\\n\\t'\n        )\n        \n        # Limit length\n        if len(sanitized) > max_length:\n            sanitized = sanitized[:max_length]\n        \n        return sanitized.strip()\n    \n    @staticmethod\n    def validate_file_hash(file_content: bytes, expected_hash: str = None) -> str:\n        \"\"\"Calculate and optionally validate file hash.\"\"\"\n        calculated_hash = hashlib.sha256(file_content).hexdigest()\n        \n        if expected_hash and calculated_hash != expected_hash:\n            raise ValidationError(\"File hash mismatch - potential tampering\")\n        \n        return calculated_hash\n\n\nclass SecurityAuditor:\n    \"\"\"Security auditing and logging.\"\"\"\n    \n    def __init__(self):\n        self.audit_log = []\n        self.max_audit_entries = 1000\n    \n    def log_security_event(self, event_type: str, details: Dict[str, Any], \n                          severity: str = \"info\"):\n        \"\"\"Log security event.\"\"\"\n        event = {\n            'timestamp': datetime.now().isoformat(),\n            'event_type': event_type,\n            'severity': severity,\n            'details': details\n        }\n        \n        self.audit_log.append(event)\n        \n        # Keep only recent entries\n        if len(self.audit_log) > self.max_audit_entries:\n            self.audit_log = self.audit_log[-self.max_audit_entries:]\n        \n        # Log to main logger\n        log_msg = f\"Security Event [{event_type}]: {details}\"\n        if severity == \"critical\":\n            logger.critical(log_msg)\n        elif severity == \"error\":\n            logger.error(log_msg)\n        elif severity == \"warning\":\n            logger.warning(log_msg)\n        else:\n            logger.info(log_msg)\n    \n    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:\n        \"\"\"Get security summary for last N hours.\"\"\"\n        cutoff_time = datetime.now() - timedelta(hours=hours)\n        \n        recent_events = [\n            event for event in self.audit_log\n            if datetime.fromisoformat(event['timestamp']) > cutoff_time\n        ]\n        \n        # Count by type and severity\n        event_counts = {}\n        severity_counts = {}\n        \n        for event in recent_events:\n            event_type = event['event_type']\n            severity = event['severity']\n            \n            event_counts[event_type] = event_counts.get(event_type, 0) + 1\n            severity_counts[severity] = severity_counts.get(severity, 0) + 1\n        \n        return {\n            'period_hours': hours,\n            'total_events': len(recent_events),\n            'event_types': event_counts,\n            'severity_levels': severity_counts,\n            'recent_events': recent_events[-10:]  # Last 10 events\n        }\n\n\nclass SecurityException(Exception):\n    \"\"\"Security-related exception.\"\"\"\n    pass\n\n\n# Global security instances\nsecurity_manager = SecurityManager()\nrate_limiter = RateLimiter(max_requests=30, time_window=60)  # 30 requests per minute\ninput_sanitizer = InputSanitizer()\nsecurity_auditor = SecurityAuditor()\n\n\ndef check_security_health() -> Dict[str, Any]:\n    \"\"\"Check overall security health.\"\"\"\n    return {\n        'active_sessions': len(security_manager.session_tokens),\n        'temp_files_tracked': len(security_manager.temp_files),\n        'rate_limited_ips': len(rate_limiter.blocked_until),\n        'audit_entries': len(security_auditor.audit_log),\n        'cleanup_thread_active': security_manager.cleanup_running,\n        'timestamp': datetime.now().isoformat()\n    }\n\n\ndef secure_file_processing(file_content: bytes, filename: str) -> Tuple[str, str]:\n    \"\"\"Securely process uploaded file.\"\"\"\n    try:\n        # Sanitize filename\n        safe_filename = input_sanitizer.sanitize_filename(filename)\n        \n        # Calculate file hash for integrity\n        file_hash = input_sanitizer.validate_file_hash(file_content)\n        \n        # Create secure temporary file\n        temp_path = security_manager.create_secure_temp_file(\n            suffix=Path(safe_filename).suffix,\n            content=file_content\n        )\n        \n        # Log security event\n        security_auditor.log_security_event(\n            'file_upload',\n            {\n                'filename': safe_filename,\n                'size_bytes': len(file_content),\n                'file_hash': file_hash,\n                'temp_path': temp_path\n            }\n        )\n        \n        return temp_path, file_hash\n        \n    except Exception as e:\n        security_auditor.log_security_event(\n            'file_upload_failed',\n            {\n                'filename': filename,\n                'error': str(e)\n            },\n            severity='error'\n        )\n        raise\n\n\ndef cleanup_security_resources():\n    \"\"\"Clean up security resources.\"\"\"\n    try:\n        security_manager.cleanup_running = False\n        if security_manager.cleanup_thread:\n            security_manager.cleanup_thread.join(timeout=5)\n        \n        # Clean up remaining temp files\n        for temp_file in list(security_manager.temp_files):\n            security_manager.secure_delete_file(temp_file)\n        \n        logger.info(\"Security resources cleaned up\")\n        \n    except Exception as e:\n        logger.error(f\"Error cleaning up security resources: {str(e)}\")"}, {"old_string": "            filename = name[:255-len(ext)] + ext", "new_string": "            filename = name[:255-len(ext)] + ext"}]