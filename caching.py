"""Advanced caching system for DeepseekOllamaRag application."""

import os
import pickle
import hashlib
import time
import threading
from typing import Any, Dict, Optional, Callable, Union, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from functools import wraps
import sqlite3

from config import config
from logging_config import global_logger as logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    tags: Optional[List[str]] = None


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if self._is_expired(entry):
                self._remove_entry(key)
                return None
            
            # Update access info
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Set value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl
            size_bytes = self._calculate_size(value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl,
                tags=tags or []
            )
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            
            # Evict if necessary
            self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def delete_by_tags(self, tags: List[str]):
        """Delete entries with any of the specified tags."""
        with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if entry.tags and any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self._remove_entry(key)
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "hit_ratio": self._calculate_hit_ratio(),
                "oldest_entry": min((e.created_at for e in self.cache.values()), default=None),
                "newest_entry": max((e.created_at for e in self.cache.values()), default=None)
            }
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and access order."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl_seconds is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full."""
        while len(self.cache) > self.max_size:
            if self.access_order:
                lru_key = self.access_order[0]
                self._remove_entry(lru_key)
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return len(str(obj).encode('utf-8'))
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This is a simplified calculation
        # In production, you'd track hits/misses separately
        if not self.cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return min(1.0, total_accesses / (len(self.cache) * 2))


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    created_at TEXT,
                    accessed_at TEXT,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    ttl_seconds INTEGER,
                    tags TEXT
                )
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path, created_at, ttl_seconds, access_count FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                file_path, created_at_str, ttl_seconds, access_count = row
                created_at = datetime.fromisoformat(created_at_str)
                
                # Check TTL
                if ttl_seconds and (datetime.now() - created_at).total_seconds() > ttl_seconds:
                    self.delete(key)
                    return None
                
                # Load value from file
                file_path = Path(file_path)
                if not file_path.exists():
                    self.delete(key)
                    return None
                
                try:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access info
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at = ?, access_count = ? WHERE key = ?",
                        (datetime.now().isoformat(), access_count + 1, key)
                    )
                    conn.commit()
                    
                    return value
                    
                except Exception as e:
                    logger.error(f"Error loading cached value for key {key}: {str(e)}")
                    self.delete(key)
                    return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        """Set value in disk cache."""
        with self.lock:
            # Generate file path
            key_hash = hashlib.md5(key.encode()).hexdigest()
            file_path = self.cache_dir / f"{key_hash}.pkl"
            
            try:
                # Save value to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = file_path.stat().st_size
                
                # Save metadata to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO cache_entries 
                           (key, file_path, created_at, accessed_at, access_count, size_bytes, ttl_seconds, tags)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            key,
                            str(file_path),
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                            1,
                            size_bytes,
                            ttl,
                            json.dumps(tags) if tags else None
                        )
                    )
                    conn.commit()
                
                # Cleanup if needed
                self._cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Error caching value for key {key}: {str(e)}")
                # Clean up partial files
                if file_path.exists():
                    file_path.unlink()
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT file_path FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    file_path = Path(row[0])
                    if file_path.exists():
                        file_path.unlink()
                    
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return True
                
                return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT file_path FROM cache_entries")
                for (file_path,) in cursor.fetchall():
                    path = Path(file_path)
                    if path.exists():
                        path.unlink()
                
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
    
    def _cleanup_if_needed(self):
        """Clean up old entries if cache is too large."""
        total_size = self._get_total_size()
        if total_size > self.max_size_bytes:
            # Remove oldest entries
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key FROM cache_entries ORDER BY accessed_at ASC"
                )
                keys_to_remove = []
                removed_size = 0
                
                for (key,) in cursor.fetchall():
                    cursor2 = conn.execute(
                        "SELECT size_bytes FROM cache_entries WHERE key = ?", (key,)
                    )
                    size_row = cursor2.fetchone()
                    if size_row:
                        keys_to_remove.append(key)
                        removed_size += size_row[0]
                        
                        if total_size - removed_size <= self.max_size_bytes * 0.8:
                            break
                
                for key in keys_to_remove:
                    self.delete(key)
                
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries ({removed_size} bytes)")
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
            result = cursor.fetchone()
            return result[0] or 0


class SmartCache:
    """Intelligent cache that combines memory and disk caching."""
    
    def __init__(self, 
                 memory_max_size: int = 100,
                 disk_max_size_mb: int = 500,
                 memory_ttl: int = 1800,  # 30 minutes
                 disk_ttl: int = 86400):  # 24 hours
        
        self.memory_cache = MemoryCache(memory_max_size, memory_ttl)
        self.disk_cache = DiskCache("cache", disk_max_size_mb)
        self.memory_ttl = memory_ttl
        self.disk_ttl = disk_ttl
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_running = True
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value, self.memory_ttl)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            memory_only: bool = False, tags: Optional[List[str]] = None):
        """Set value in cache."""
        memory_ttl = ttl or self.memory_ttl
        disk_ttl = ttl or self.disk_ttl
        
        # Always cache in memory
        self.memory_cache.set(key, value, memory_ttl, tags)
        
        # Cache on disk unless memory_only is True
        if not memory_only:
            self.disk_cache.set(key, value, disk_ttl, tags)
    
    def delete(self, key: str) -> bool:
        """Delete key from both caches."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        return memory_deleted or disk_deleted
    
    def clear(self):
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "memory": self.memory_cache.get_stats(),
            "disk": {
                "total_size_bytes": self.disk_cache._get_total_size(),
                "max_size_bytes": self.disk_cache.max_size_bytes
            }
        }
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                self.memory_cache.cleanup_expired()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        self.cleanup_running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)


# Caching decorators
def cached(cache_key_func: Optional[Callable] = None, ttl: int = 3600, 
          cache_instance: Optional[SmartCache] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = cache_instance or global_cache
            
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {key}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}: {key}")
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cache_embedding_key(text: str, model: str) -> str:
    """Generate cache key for embeddings."""
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return f"embedding:{model}:{content_hash}"


def cache_qa_key(question: str, doc_hash: str, model: str) -> str:
    """Generate cache key for Q&A results."""
    question_hash = hashlib.sha256(question.encode()).hexdigest()
    return f"qa:{model}:{doc_hash}:{question_hash}"


# Global cache instance
global_cache = SmartCache(
    memory_max_size=50,
    disk_max_size_mb=200,
    memory_ttl=1800,  # 30 minutes
    disk_ttl=86400    # 24 hours
)


def get_cache_status() -> Dict[str, Any]:
    """Get global cache status."""
    return {
        "cache_stats": global_cache.get_stats(),
        "timestamp": datetime.now().isoformat()
    }