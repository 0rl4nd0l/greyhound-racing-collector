#!/usr/bin/env python3
"""
Endpoint Cache System
=====================

High-performance caching layer for Flask endpoints with ETag support and 
automatic cache invalidation. Optimizes system_status and model_registry
endpoints by reducing database hits and providing client-side caching.

Features:
- In-memory cache with configurable TTL
- ETag generation and validation
- Thread-safe operations
- Cache statistics and monitoring
- Automatic cleanup of expired entries

Author: AI Assistant
Date: August 2, 2025
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from functools import wraps

import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    etag: str
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0


class EndpointCache:
    """
    High-performance in-memory cache for Flask endpoints
    """
    
    def __init__(self, default_ttl: int = 30, max_entries: int = 1000):
        """
        Initialize the cache system
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_entries: Maximum number of cache entries
        """
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cache_size': 0
        }
        
        logger.info(f"🚀 EndpointCache initialized (TTL: {default_ttl}s, Max entries: {max_entries})")
    
    def _generate_etag(self, data: Any) -> str:
        """Generate ETag hash from data"""
        try:
            # Convert data to JSON string for consistent hashing
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate ETag: {e}")
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at < current_time
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full"""
        if len(self._cache) < self.max_entries:
            return
        
        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].created_at)
        )
        
        entries_to_remove = len(sorted_entries) - self.max_entries + 10  # Remove extra for buffer
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[key]
            self._stats['evictions'] += 1
        
        logger.debug(f"🧹 Evicted {entries_to_remove} LRU cache entries")
    
    def get(self, key: str, if_none_match: Optional[str] = None) -> Tuple[Optional[Any], Optional[str], bool]:
        """
        Get cached data with ETag support
        
        Args:
            key: Cache key
            if_none_match: Client's ETag for conditional requests
            
        Returns:
            Tuple of (data, etag, is_not_modified)
        """
        with self._lock:
            current_time = time.time()
            
            # Clean up expired entries periodically
            if len(self._cache) > 0 and current_time % 60 < 1:
                self._cleanup_expired()
            
            entry = self._cache.get(key)
            
            if entry is None or entry.expires_at < current_time:
                self._stats['misses'] += 1
                return None, None, False
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = current_time
            self._stats['hits'] += 1
            
            # Check if client has current version (ETag match)
            if if_none_match and if_none_match == entry.etag:
                return None, entry.etag, True  # Not modified
            
            return entry.data, entry.etag, False
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> str:
        """
        Store data in cache with ETag generation
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Generated ETag
        """
        if ttl is None:
            ttl = self.default_ttl
        
        current_time = time.time()
        etag = self._generate_etag(data)
        
        with self._lock:
            # Evict entries if cache is full
            self._evict_lru()
            
            entry = CacheEntry(
                data=data,
                etag=etag,
                created_at=current_time,
                expires_at=current_time + ttl,
                last_accessed=current_time
            )
            
            self._cache[key] = entry
            self._stats['cache_size'] = len(self._cache)
        
        logger.debug(f"📝 Cached data for key '{key}' (ETag: {etag[:8]}..., TTL: {ttl}s)")
        return etag
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['cache_size'] = len(self._cache)
                logger.debug(f"🗑️ Invalidated cache key '{key}'")
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        with self._lock:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
            
            self._stats['cache_size'] = len(self._cache)
            
            if keys_to_remove:
                logger.debug(f"🗑️ Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")
            
            return len(keys_to_remove)
    
    def clear(self) -> int:
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats['cache_size'] = 0
            logger.info(f"🧹 Cleared all {count} cache entries")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': round(hit_rate, 2),
                'evictions': self._stats['evictions'],
                'cache_size': len(self._cache),
                'max_entries': self.max_entries,
                'default_ttl': self.default_ttl
            }
    
    def get_next_refresh_time(self, ttl: Optional[int] = None) -> str:
        """Get the next refresh timestamp for client polling"""
        if ttl is None:
            ttl = self.default_ttl
        
        next_refresh = datetime.now() + timedelta(seconds=ttl)
        return next_refresh.isoformat()


# Global cache instance
_cache_instance = None
_cache_lock = threading.Lock()


def get_endpoint_cache() -> EndpointCache:
    """Get the global cache instance (singleton)"""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = EndpointCache()
    return _cache_instance


def _json_safe(value):
    """Recursively convert values to JSON-serializable forms.
    - bytes -> utf-8 string (with errors='replace')
    - datetime -> isoformat string
    - sets/tuples -> lists
    - objects with __str__ -> string fallback
    """
    try:
        from datetime import datetime as _dt
    except Exception:
        _dt = None

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return str(value)
    if _dt is not None and isinstance(value, _dt):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return { _json_safe(k): _json_safe(v) for k, v in value.items() }
    if isinstance(value, (list, tuple, set)):
        return [ _json_safe(v) for v in value ]
    # Fallback to string representation
    try:
        return str(value)
    except Exception:
        return None


def cached_endpoint(key_func=None, ttl=30):
    """
    Decorator for caching Flask endpoint responses with ETag support
    
    Args:
        key_func: Function to generate cache key (receives request args)
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, jsonify, Response
            
            cache = get_endpoint_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(request)
            else:
                cache_key = f"{func.__name__}:{request.endpoint}:{request.query_string.decode()}"
            
            # Check for ETag in request headers
            if_none_match = request.headers.get('If-None-Match')
            
            # Try to get cached data
            cached_data, etag, not_modified = cache.get(cache_key, if_none_match)
            
            if not_modified:
                # Return 304 Not Modified
                response = Response()
                response.status_code = 304
                response.headers['ETag'] = etag
                response.headers['Cache-Control'] = f'max-age={ttl}'
                return response
            
            if cached_data is not None:
                # Return cached data with ETag
                response = jsonify(cached_data)
                response.headers['ETag'] = etag
                response.headers['Cache-Control'] = f'max-age={ttl}'
                response.headers['X-Cache'] = 'HIT'
                return response
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Extract data from response for caching (support dicts and (dict, status))
            data = None
            status = None
            headers = None
            
            if hasattr(result, 'get_json'):
                try:
                    data = result.get_json()
                except Exception:
                    data = None
            
            if data is None and isinstance(result, tuple) and len(result) >= 1:
                first = result[0]
                status = result[1] if len(result) > 1 else None
                headers = result[2] if len(result) > 2 else None
                if hasattr(first, 'get_json'):
                    try:
                        data = first.get_json()
                    except Exception:
                        data = None
                if data is None:
                    # If first element is already a dict, use it directly
                    if isinstance(first, dict):
                        data = first
            
            if data is None:
                # Fallback: if the view returned a plain dict, accept it
                if isinstance(result, dict):
                    data = result
                else:
                    # As a last resort, do not cache unknown types
                    return result
            
            # Add next_refresh timestamp to response
            if isinstance(data, dict):
                data['next_refresh'] = cache.get_next_refresh_time(ttl)
                data['cache_info'] = {
                    'cached_at': datetime.now().isoformat(),
                    'ttl': ttl,
                    'cache_key': cache_key[:32]  # Truncated for security
                }
            
            # Sanitize to JSON-safe payload before caching/returning
            safe_data = _json_safe(data)

            # Cache the data
            etag = cache.set(cache_key, safe_data, ttl)
            
            # Return response with cache headers
            response = jsonify(safe_data)
            if status is not None:
                response.status_code = status
            if headers:
                for k, v in headers.items():
                    response.headers[k] = v
            response.headers['ETag'] = etag
            response.headers['Cache-Control'] = f'max-age={ttl}'
            response.headers['X-Cache'] = 'MISS'
            
            return response
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the cache system
    cache = get_endpoint_cache()
    
    # Test basic functionality
    test_data = {"message": "Hello, World!", "timestamp": datetime.now().isoformat()}
    etag = cache.set("test_key", test_data)
    print(f"Stored data with ETag: {etag}")
    
    # Test retrieval
    data, retrieved_etag, not_modified = cache.get("test_key")
    print(f"Retrieved data: {data}")
    print(f"ETag match: {etag == retrieved_etag}")
    
    # Test ETag matching
    data, retrieved_etag, not_modified = cache.get("test_key", etag)
    print(f"Not modified (ETag match): {not_modified}")
    
    # Test statistics
    stats = cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
