"""Utility functions for DeepseekOllamaRag application."""

import os
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import streamlit as st
from config import config
from exceptions import FileProcessingError, ValidationError


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_file(uploaded_file) -> bool:
    """Validate uploaded file."""
    if not uploaded_file:
        raise ValidationError("No file uploaded")
    
    # Check file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.app.allowed_file_types:
        raise ValidationError(
            f"File type '{file_extension}' not allowed. "
            f"Allowed types: {', '.join(config.app.allowed_file_types)}"
        )
    
    # Check file size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > config.app.max_file_size_mb:
        raise ValidationError(
            f"File size ({file_size_mb:.1f}MB) exceeds limit of {config.app.max_file_size_mb}MB"
        )
    
    return True


def save_uploaded_file(uploaded_file, directory: Optional[str] = None) -> str:
    """Save uploaded file and return path."""
    try:
        validate_file(uploaded_file)
        
        if directory is None:
            directory = config.app.temp_dir
        
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file_hash}_{uploaded_file.name}"
        file_path = os.path.join(directory, filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return file_path
    
    except Exception as e:
        raise FileProcessingError(f"Failed to save uploaded file: {str(e)}")


def cleanup_temp_files(directory: Optional[str] = None, max_age_hours: int = 24):
    """Clean up temporary files older than specified age."""
    if directory is None:
        directory = config.app.temp_dir
    
    if not os.path.exists(directory):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might be in use


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information."""
    try:
        stat = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "extension": Path(file_path).suffix.lower()
        }
    except Exception as e:
        raise FileProcessingError(f"Failed to get file info: {str(e)}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def init_session_state():
    """Initialize session state variables."""
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()


def check_session_timeout() -> bool:
    """Check if session has timed out."""
    if "session_start" in st.session_state:
        session_duration = datetime.now() - st.session_state.session_start
        timeout_duration = timedelta(minutes=config.app.session_timeout_minutes)
        return session_duration > timeout_duration
    return False


def reset_session():
    """Reset session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        logging.error(f"Error executing {func.__name__}: {str(e)}")
        return None, str(e)


def check_ollama_connection() -> bool:
    """Check if Ollama is accessible."""
    try:
        import requests
        response = requests.get(f"{config.model.ollama_base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        import requests
        response = requests.get(f"{config.model.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception:
        return []