"""
Security utilities and validation for the RAG system.
"""

import os
import re
import hashlib
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set

from config import config
from utils import setup_logging
from exceptions import ValidationError

logger = setup_logging()


class SecurityManager:
    """Manages security aspects of the application."""
    
    def __init__(self):
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        self.temp_files: Set[str] = set()
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_running = False
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.cleanup_running:
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                self._cleanup_expired_sessions()
                self._cleanup_temp_files()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        timeout = timedelta(minutes=config.app.session_timeout_minutes)
        
        expired_sessions = []
        for token, data in self.session_tokens.items():
            if now - data['created'] > timeout:
                expired_sessions.append(token)
        
        for token in expired_sessions:
            del self.session_tokens[token]
            logger.debug(f"Cleaned up expired session: {token[:8]}...")
    
    def _cleanup_temp_files(self):
        """Clean up old temporary files."""
        files_to_remove = []
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    stat = os.stat(temp_file)
                    # Remove files older than 1 hour
                    if time.time() - stat.st_mtime > 3600:
                        files_to_remove.append(temp_file)
                else:
                    # File doesn't exist, remove from tracking
                    files_to_remove.append(temp_file)
            except Exception as e:
                logger.warning(f"Error checking temp file {temp_file}: {e}")
                files_to_remove.append(temp_file)
        
        for temp_file in files_to_remove:
            self.secure_delete_file(temp_file)
    
    def create_secure_temp_file(self, suffix: str = "", content: bytes = None) -> str:
        """Create a secure temporary file."""
        try:
            fd, path = tempfile.mkstemp(suffix=suffix, dir=config.app.temp_dir)
            
            if content:
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)
            else:
                os.close(fd)
            
            # Set restrictive permissions
            os.chmod(path, 0o600)
            
            self.temp_files.add(path)
            logger.debug(f"Created secure temp file: {path}")
            
            return path
            
        except Exception as e:
            logger.error(f"Error creating secure temp file: {str(e)}")
            raise ValidationError(f"Could not create secure temporary file: {str(e)}")
    
    def secure_delete_file(self, file_path: str):
        """Securely delete a file."""
        try:
            if os.path.exists(file_path):
                # Simple secure deletion (overwrite with zeros)
                with open(file_path, 'r+b') as f:
                    length = f.seek(0, 2)  # Seek to end
                    f.seek(0)
                    f.write(b'\x00' * length)
                    f.flush()
                    os.fsync(f.fileno())
                
                os.remove(file_path)
            
            self.temp_files.discard(file_path)
            logger.debug(f"Securely deleted file: {file_path}")
            
        except Exception as e:
            logger.warning(f"Error securely deleting file {file_path}: {str(e)}")


class RateLimiter:
    """Simple rate limiter for request throttling."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}
        self.blocked_until: Dict[str, float] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        # Check if currently blocked
        if identifier in self.blocked_until:
            if now < self.blocked_until[identifier]:
                return False
            else:
                del self.blocked_until[identifier]
        
        # Initialize request history if needed
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        cutoff_time = now - self.time_window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            # Block for the remaining window time
            self.blocked_until[identifier] = now + self.time_window
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Record this request
        self.requests[identifier].append(now)
        return True


class InputSanitizer:
    """Sanitizes and validates input data."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe use."""
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        # Remove directory traversal attempts
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Ensure not empty after sanitization
        if not filename.strip():
            raise ValidationError("Invalid filename after sanitization")
        
        return filename.strip()
    
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 2000) -> str:
        """Sanitize text input."""
        if not text:
            return ""
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\n\t'
        )
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_hash(file_content: bytes, expected_hash: str = None) -> str:
        """Calculate and optionally validate file hash."""
        calculated_hash = hashlib.sha256(file_content).hexdigest()
        
        if expected_hash and calculated_hash != expected_hash:
            raise ValidationError("File hash mismatch - potential tampering")
        
        return calculated_hash


class SecurityAuditor:
    """Security auditing and logging."""
    
    def __init__(self):
        self.audit_log = []
        self.max_audit_entries = 1000
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = "info"):
        """Log security event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        self.audit_log.append(event)
        
        # Keep only recent entries
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]
        
        # Log to main logger
        log_msg = f"Security Event [{event_type}]: {details}"
        if severity == "critical":
            logger.critical(log_msg)
        elif severity == "error":
            logger.error(log_msg)
        elif severity == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)


# Global security instances
security_manager = SecurityManager()
rate_limiter = RateLimiter(max_requests=30, time_window=60)
input_sanitizer = InputSanitizer()
security_auditor = SecurityAuditor()


def check_security_health() -> Dict[str, Any]:
    """Check overall security health."""
    return {
        'active_sessions': len(security_manager.session_tokens),
        'temp_files_tracked': len(security_manager.temp_files),
        'rate_limited_ips': len(rate_limiter.blocked_until),
        'audit_entries': len(security_auditor.audit_log),
        'cleanup_thread_active': security_manager.cleanup_running,
        'timestamp': datetime.now().isoformat()
    }


def secure_file_processing(file_content: bytes, filename: str) -> tuple:
    """Securely process uploaded file."""
    try:
        # Sanitize filename
        safe_filename = input_sanitizer.sanitize_filename(filename)
        
        # Calculate file hash for integrity
        file_hash = input_sanitizer.validate_file_hash(file_content)
        
        # Create secure temporary file
        temp_path = security_manager.create_secure_temp_file(
            suffix=Path(safe_filename).suffix,
            content=file_content
        )
        
        # Log security event
        security_auditor.log_security_event(
            'file_upload',
            {
                'filename': safe_filename,
                'size_bytes': len(file_content),
                'file_hash': file_hash,
                'temp_path': temp_path
            }
        )
        
        return temp_path, file_hash
        
    except Exception as e:
        security_auditor.log_security_event(
            'file_upload_failed',
            {
                'filename': filename,
                'error': str(e)
            },
            severity='error'
        )
        raise


def cleanup_security_resources():
    """Clean up security resources."""
    try:
        security_manager.cleanup_running = False
        if security_manager.cleanup_thread:
            security_manager.cleanup_thread.join(timeout=5)
        
        # Clean up remaining temp files
        for temp_file in list(security_manager.temp_files):
            security_manager.secure_delete_file(temp_file)
        
        logger.info("Security resources cleaned up")
        
    except Exception as e:
        logger.error(f"Error cleaning up security resources: {str(e)}")