"""Configuration management for DeepseekOllamaRag application."""

import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    llm_model: str = "deepseek-r1:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    similarity_search_k: int = 3
    search_type: str = "similarity"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    upload_dir: str = "uploads"
    temp_dir: str = "temp"
    max_file_size_mb: int = 50
    allowed_file_types: list = None
    session_timeout_minutes: int = 30
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ["pdf"]


@dataclass
class UIConfig:
    """UI customization configuration."""
    app_title: str = "📄 DeepSeek RAG System"
    page_title: str = "DeepSeek Ollama RAG"
    primary_color: str = "#007BFF"
    secondary_color: str = "#FFC107"
    background_color: str = "#F8F9FA"
    sidebar_background: str = "#2C2F33"


class Config:
    """Main configuration class that loads from environment variables."""
    
    def __init__(self):
        self.model = ModelConfig(
            llm_model=os.getenv("LLM_MODEL", "deepseek-r1:1.5b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        )
        
        self.vector_store = VectorStoreConfig(
            similarity_search_k=int(os.getenv("SIMILARITY_SEARCH_K", "3")),
            search_type=os.getenv("SEARCH_TYPE", "similarity"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        self.app = AppConfig(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
        )
        
        self.ui = UIConfig(
            app_title=os.getenv("APP_TITLE", "📄 DeepSeek RAG System"),
            page_title=os.getenv("PAGE_TITLE", "DeepSeek Ollama RAG"),
            primary_color=os.getenv("PRIMARY_COLOR", "#007BFF"),
            secondary_color=os.getenv("SECONDARY_COLOR", "#FFC107"),
            background_color=os.getenv("BACKGROUND_COLOR", "#F8F9FA"),
            sidebar_background=os.getenv("SIDEBAR_BACKGROUND", "#2C2F33")
        )
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.app.upload_dir).mkdir(exist_ok=True)
        Path(self.app.temp_dir).mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "vector_store": self.vector_store.__dict__,
            "app": self.app.__dict__,
            "ui": self.ui.__dict__
        }


# Global configuration instance
config = Config()