"""Security tests for DeepseekOllamaRag application."""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, Mock, mock_open
from datetime import datetime, timedelta

from security import (
    SecurityManager, RateLimiter, InputSanitizer, SecurityAuditor,
    SecurityException, secure_file_processing
)


class TestSecurityManager:
    """Test security manager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = SecurityManager()
    
    def teardown_method(self):
        """Cleanup after tests."""
        self.manager.cleanup_running = False
    
    def test_generate_session_token(self):
        """Test session token generation."""
        token = self.manager.generate_session_token("test_user")
        
        assert isinstance(token, str)
        assert len(token) > 20  # Should be reasonably long
        assert token in self.manager.session_tokens
        
        session_data = self.manager.session_tokens[token]
        assert session_data["user_id"] == "test_user"
        assert isinstance(session_data["created_at"], datetime)
        assert isinstance(session_data["expires_at"], datetime)
    
    def test_validate_session_token_valid(self):
        """Test validation of valid session token."""
        token = self.manager.generate_session_token()
        assert self.manager.validate_session_token(token)
    
    def test_validate_session_token_invalid(self):
        """Test validation of invalid session token."""
        assert not self.manager.validate_session_token("invalid_token")
    
    def test_validate_session_token_expired(self):
        """Test validation of expired session token."""
        token = self.manager.generate_session_token()
        
        # Manually expire the token
        session_data = self.manager.session_tokens[token]
        session_data["expires_at"] = datetime.now() - timedelta(minutes=1)
        
        assert not self.manager.validate_session_token(token)
        assert token not in self.manager.session_tokens  # Should be removed
    
    def test_validate_session_token_updates_last_used(self):
        """Test that validation updates last_used timestamp."""
        token = self.manager.generate_session_token()
        original_time = self.manager.session_tokens[token]["last_used"]
        
        time.sleep(0.1)  # Small delay
        self.manager.validate_session_token(token)
        
        new_time = self.manager.session_tokens[token]["last_used"]
        assert new_time > original_time
    
    @patch('tempfile.mkstemp')
    @patch('os.chmod')
    def test_create_secure_temp_file(self, mock_chmod, mock_mkstemp):
        """Test secure temporary file creation."""
        mock_fd = 123
        mock_path = "/tmp/secure_test.tmp"
        mock_mkstemp.return_value = (mock_fd, mock_path)
        
        with patch('os.fdopen', mock_open()) as mock_fdopen, \
             patch('os.close') as mock_close:
            
            result_path = self.manager.create_secure_temp_file(content=b"test data")
            
            assert result_path == mock_path
            mock_chmod.assert_called_once_with(mock_path, 0o600)  # Owner read/write only
            assert mock_path in self.manager.temp_files
    
    @patch('tempfile.mkstemp')
    def test_create_secure_temp_file_failure(self, mock_mkstemp):
        """Test secure temp file creation failure handling."""
        mock_mkstemp.side_effect = OSError("Permission denied")
        
        with pytest.raises(SecurityException):
            self.manager.create_secure_temp_file()
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.remove')
    def test_secure_delete_file(self, mock_remove, mock_open_func, mock_getsize, mock_exists):
        """Test secure file deletion with overwriting."""
        mock_exists.return_value = True
        mock_getsize.return_value = 100
        
        test_path = "/tmp/test_file.tmp"
        self.manager.secure_delete_file(test_path)
        
        mock_open_func.assert_called_once()
        mock_remove.assert_called_once_with(test_path)
    
    @patch('os.path.exists')
    def test_secure_delete_file_not_exists(self, mock_exists):
        """Test secure deletion of non-existent file."""
        mock_exists.return_value = False
        
        # Should not raise exception
        self.manager.secure_delete_file("/nonexistent/file.tmp")


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.limiter = RateLimiter(max_requests=3, time_window=1)  # 3 requests per second
    
    def test_rate_limiter_allows_initial_requests(self):
        """Test that initial requests are allowed."""
        for i in range(3):
            allowed, wait_time = self.limiter.is_allowed("test_user")
            assert allowed
            assert wait_time is None
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test that excess requests are blocked."""
        # Make 3 allowed requests
        for i in range(3):
            self.limiter.is_allowed("test_user")
        
        # 4th request should be blocked
        allowed, wait_time = self.limiter.is_allowed("test_user")
        assert not allowed
        assert wait_time > 0
    
    def test_rate_limiter_different_users(self):
        """Test that different users have separate limits."""
        # User 1 makes 3 requests
        for i in range(3):
            allowed, _ = self.limiter.is_allowed("user1")
            assert allowed
        
        # User 2 should still be allowed
        allowed, _ = self.limiter.is_allowed("user2")
        assert allowed
    
    def test_rate_limiter_time_window_reset(self):
        """Test that rate limit resets after time window."""
        # Make 3 requests
        for i in range(3):
            self.limiter.is_allowed("test_user")
        
        # Should be blocked
        allowed, _ = self.limiter.is_allowed("test_user")
        assert not allowed
        
        # Wait for time window to pass
        time.sleep(1.1)
        
        # Should be allowed again
        allowed, _ = self.limiter.is_allowed("test_user")
        assert allowed


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = InputSanitizer.sanitize_filename("test_file.pdf")
        assert result == "test_file.pdf"
    
    def test_sanitize_filename_dangerous_chars(self):
        """Test sanitization of dangerous characters."""
        dangerous_name = "test<>:\"|?*\\/.pdf"
        result = InputSanitizer.sanitize_filename(dangerous_name)
        
        # Dangerous characters should be replaced with underscores
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result
    
    def test_sanitize_filename_path_components(self):
        """Test removal of path components."""
        path_name = "/path/to/file.pdf"
        result = InputSanitizer.sanitize_filename(path_name)
        assert result == "file.pdf"
    
    def test_sanitize_filename_control_chars(self):
        """Test removal of control characters."""
        control_name = "test\\x00\\x01file.pdf"
        result = InputSanitizer.sanitize_filename(control_name)
        assert "\\x00" not in result
        assert "\\x01" not in result
    
    def test_sanitize_filename_too_long(self):
        """Test truncation of overly long filenames."""
        long_name = "a" * 300 + ".pdf"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".pdf")  # Extension should be preserved
    
    def test_sanitize_filename_empty_after_sanitization(self):
        """Test handling of filename that becomes empty after sanitization."""
        with pytest.raises(ValidationError):
            InputSanitizer.sanitize_filename("<<<>>>")
    
    def test_sanitize_text_input_basic(self):
        """Test basic text sanitization."""
        result = InputSanitizer.sanitize_text_input("Hello world!")
        assert result == "Hello world!"
    
    def test_sanitize_text_input_control_chars(self):
        """Test removal of control characters from text."""
        dirty_text = "Hello\\x00\\x01world"
        result = InputSanitizer.sanitize_text_input(dirty_text)
        assert "\\x00" not in result
        assert "\\x01" not in result
        assert "Helloworld" in result
    
    def test_sanitize_text_input_length_limit(self):
        """Test length limiting in text sanitization."""
        long_text = "a" * 3000
        result = InputSanitizer.sanitize_text_input(long_text, max_length=100)
        assert len(result) <= 100
    
    def test_sanitize_text_input_preserves_newlines_tabs(self):
        """Test that newlines and tabs are preserved."""
        text_with_whitespace = "Line 1\\nLine 2\\tTabbed"
        result = InputSanitizer.sanitize_text_input(text_with_whitespace)
        assert "\\n" in result
        assert "\\t" in result
    
    def test_validate_file_hash_correct(self):
        """Test file hash validation with correct hash."""
        content = b"test content"
        expected_hash = "1eebdf4fdc9fc7bf283031b93f9aef3338de9052fb2538b3a42c6d7dd1c7e8f9"  # SHA256 of "test content"
        
        result = InputSanitizer.validate_file_hash(content, expected_hash)
        assert result == expected_hash
    
    def test_validate_file_hash_incorrect(self):
        """Test file hash validation with incorrect hash."""
        content = b"test content"
        wrong_hash = "wrong_hash_value"
        
        with pytest.raises(ValidationError, match="hash mismatch"):
            InputSanitizer.validate_file_hash(content, wrong_hash)
    
    def test_validate_file_hash_no_expected(self):
        """Test file hash calculation without expected hash."""
        content = b"test content"
        result = InputSanitizer.validate_file_hash(content)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex length


class TestSecurityAuditor:
    """Test security auditing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.auditor = SecurityAuditor()
    
    def test_log_security_event_basic(self):
        """Test basic security event logging."""
        event_details = {"user": "test", "action": "login"}
        self.auditor.log_security_event("login_attempt", event_details)
        
        assert len(self.auditor.audit_log) == 1
        event = self.auditor.audit_log[0]
        assert event["event_type"] == "login_attempt"
        assert event["details"] == event_details
        assert event["severity"] == "info"  # default
    
    def test_log_security_event_with_severity(self):
        """Test security event logging with custom severity."""
        event_details = {"error": "authentication failed"}
        self.auditor.log_security_event("login_failed", event_details, severity="error")
        
        event = self.auditor.audit_log[0]
        assert event["severity"] == "error"
    
    def test_audit_log_size_limit(self):
        """Test that audit log respects size limit."""
        # Add more than max_audit_entries
        for i in range(self.auditor.max_audit_entries + 100):
            self.auditor.log_security_event(f"event_{i}", {"index": i})
        
        assert len(self.auditor.audit_log) == self.auditor.max_audit_entries
        # Should keep the most recent entries
        last_event = self.auditor.audit_log[-1]
        assert "event_" in last_event["event_type"]
    
    def test_get_security_summary(self):
        """Test security summary generation."""
        # Add some test events
        self.auditor.log_security_event("login", {}, severity="info")
        self.auditor.log_security_event("login_failed", {}, severity="error")
        self.auditor.log_security_event("file_upload", {}, severity="info")
        
        summary = self.auditor.get_security_summary(hours=24)
        
        assert summary["total_events"] == 3
        assert "login" in summary["event_types"]
        assert "login_failed" in summary["event_types"]
        assert "file_upload" in summary["event_types"]
        assert "info" in summary["severity_levels"]
        assert "error" in summary["severity_levels"]
        assert len(summary["recent_events"]) <= 10


class TestSecureFileProcessing:
    """Test secure file processing integration."""
    
    def test_secure_file_processing_success(self):
        """Test successful secure file processing."""
        content = b"%PDF-1.4\\ntest content"
        filename = "test.pdf"
        
        with patch('security.security_manager.create_secure_temp_file') as mock_create:
            mock_create.return_value = "/tmp/secure_test.pdf"
            
            temp_path, file_hash = secure_file_processing(content, filename)
            
            assert temp_path == "/tmp/secure_test.pdf"
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64  # SHA256 length
    
    def test_secure_file_processing_sanitizes_filename(self):
        """Test that secure file processing sanitizes filename."""
        content = b"%PDF-1.4\\ntest"
        dangerous_filename = "../../../malicious.pdf"
        
        with patch('security.security_manager.create_secure_temp_file') as mock_create:
            mock_create.return_value = "/tmp/secure_test.pdf"
            
            temp_path, file_hash = secure_file_processing(content, dangerous_filename)
            
            # Should have called create_secure_temp_file with sanitized filename
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert "content" in kwargs
            assert kwargs["content"] == content
    
    def test_secure_file_processing_logs_event(self):
        """Test that secure file processing logs security events."""
        content = b"%PDF-1.4\\ntest content"
        filename = "test.pdf"
        
        with patch('security.security_manager.create_secure_temp_file') as mock_create, \
             patch('security.security_auditor.log_security_event') as mock_log:
            
            mock_create.return_value = "/tmp/secure_test.pdf"
            
            secure_file_processing(content, filename)
            
            # Should log the file upload event
            mock_log.assert_called()
            args, kwargs = mock_log.call_args
            assert args[0] == "file_upload"  # event type
            assert "filename" in args[1]  # details
            assert "size_bytes" in args[1]
            assert "file_hash" in args[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])