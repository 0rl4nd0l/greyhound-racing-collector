#!/usr/bin/env python3
"""
Database Performance Optimizer
==============================

This module implements connection pooling and lazy loading optimizations
for the SQLite database to improve concurrent access performance.

Features:
- SQLite connection pooling with threading support
- Lazy loading for model relationships
- Query optimization and caching
- Connection management for high concurrency
"""

import sqlite3
import threading
import time
import contextlib
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
import logging
from functools import wraps
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """
    Thread-safe SQLite connection pool for improved concurrency.
    
    SQLite has limitations with concurrent writes, so this pool manages
    connections efficiently while respecting SQLite's threading model.
    """
    
    def __init__(self, database_path: str, pool_size: int = 20, 
                 timeout: float = 30.0, check_same_thread: bool = False):
        self.database_path = database_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        
        # Connection pool
        self._pool = Queue(maxsize=pool_size)
        self._pool_lock = threading.Lock()
        self._created_connections = 0
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'active_connections': 0
        }
        
        # Initialize pool
        self._initialize_pool()
        
        logger.info(f"🔗 SQLite Connection Pool initialized:")
        logger.info(f"   📊 Pool size: {pool_size}")
        logger.info(f"   📁 Database: {database_path}")
        logger.info(f"   ⏱️ Timeout: {timeout}s")

    def _initialize_pool(self):
        """Pre-populate the connection pool"""
        for _ in range(min(5, self.pool_size)):  # Start with 5 connections
            try:
                conn = self._create_connection()
                self._pool.put(conn, block=False)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimizations"""
        conn = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            check_same_thread=self.check_same_thread,
            isolation_level=None  # Autocommit mode for better concurrency
        )
        
        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Balanced durability/performance
        conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp storage
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        
        # Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        self.stats['connections_created'] += 1
        self._created_connections += 1
        
        logger.debug(f"✅ Created new database connection (total: {self._created_connections})")
        return conn

    @contextlib.contextmanager
    def get_connection(self):
        """Get a connection from the pool with context manager"""
        conn = None
        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get(block=True, timeout=5.0)
                self.stats['pool_hits'] += 1
                self.stats['connections_reused'] += 1
                logger.debug("🔄 Reused connection from pool")
            except Empty:
                # Pool is empty, create new connection
                conn = self._create_connection()
                self.stats['pool_misses'] += 1
                logger.debug("🆕 Created new connection (pool empty)")
            
            self.stats['active_connections'] += 1
            yield conn
            
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                try:
                    # Return connection to pool if pool not full
                    if self._pool.qsize() < self.pool_size:
                        self._pool.put(conn, block=False)
                        logger.debug("♻️ Connection returned to pool")
                    else:
                        conn.close()
                        logger.debug("🗑️ Connection closed (pool full)")
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
                
                self.stats['active_connections'] -= 1

    def execute_query(self, query: str, params: tuple = None, fetchall: bool = True):
        """Execute a query using pooled connection"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetchall:
                return cursor.fetchall()
            else:
                return cursor.fetchone()

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        self.stats['pool_size'] = self._pool.qsize()
        return self.stats.copy()

    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

class LazyModelLoader:
    """
    Lazy loading implementation for database models to reduce initial load time
    and memory usage during high concurrency scenarios.
    """
    
    def __init__(self, db_pool: SQLiteConnectionPool):
        self.db_pool = db_pool
        self._cache = {}
        self._cache_ttl = {}
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_ttl:
            return False
        return datetime.now() < self._cache_ttl[key]
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache with TTL"""
        self._cache[key] = data
        self._cache_ttl[key] = datetime.now() + self.cache_duration
    
    def get_race_metadata(self, race_id: str = None, venue: str = None, limit: int = 100):
        """Lazy load race metadata with caching"""
        cache_key = f"race_metadata_{race_id}_{venue}_{limit}"
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"📦 Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        # Build query based on parameters
        if race_id:
            query = "SELECT * FROM race_metadata WHERE race_id = ? LIMIT ?"
            params = (race_id, limit)
        elif venue:
            query = "SELECT * FROM race_metadata WHERE venue = ? ORDER BY race_date DESC LIMIT ?"
            params = (venue, limit)
        else:
            query = "SELECT * FROM race_metadata ORDER BY race_date DESC LIMIT ?"
            params = (limit,)
        
        start_time = time.time()
        results = self.db_pool.execute_query(query, params)
        end_time = time.time()
        
        # Convert to list of dicts for JSON serialization
        data = [dict(row) for row in results]
        self._set_cache(cache_key, data)
        
        logger.info(f"🔍 Loaded {len(data)} race metadata records in {(end_time - start_time)*1000:.2f}ms")
        return data
    
    def get_dog_data(self, dog_name: str = None, limit: int = 100):
        """Lazy load dog data with caching"""
        cache_key = f"dog_data_{dog_name}_{limit}"
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"📦 Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        if dog_name:
            query = """
                SELECT d.*, COUNT(drd.race_id) as race_count 
                FROM dogs d 
                LEFT JOIN dog_race_data drd ON d.dog_name = drd.dog_name 
                WHERE d.dog_name LIKE ? 
                GROUP BY d.dog_name 
                LIMIT ?
            """
            params = (f"%{dog_name}%", limit)
        else:
            query = """
                SELECT d.*, COUNT(drd.race_id) as race_count 
                FROM dogs d 
                LEFT JOIN dog_race_data drd ON d.dog_name = drd.dog_name 
                GROUP BY d.dog_name 
                ORDER BY d.total_wins DESC 
                LIMIT ?
            """
            params = (limit,)
        
        start_time = time.time()
        results = self.db_pool.execute_query(query, params)
        end_time = time.time()
        
        data = [dict(row) for row in results]
        self._set_cache(cache_key, data)
        
        logger.info(f"🐕 Loaded {len(data)} dog records in {(end_time - start_time)*1000:.2f}ms")
        return data
    
    def get_recent_race_performance(self, dog_name: str, days: int = 30):
        """Get recent performance data for a dog with caching"""
        cache_key = f"recent_performance_{dog_name}_{days}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        query = """
            SELECT drd.*, rm.race_date, rm.venue, rm.distance 
            FROM dog_race_data drd 
            JOIN race_metadata rm ON drd.race_id = rm.race_id 
            WHERE drd.dog_name = ? 
            AND rm.race_date >= date('now', '-{} days') 
            ORDER BY rm.race_date DESC
        """.format(days)
        
        start_time = time.time()
        results = self.db_pool.execute_query(query, (dog_name,))
        end_time = time.time()
        
        data = [dict(row) for row in results]
        self._set_cache(cache_key, data)
        
        logger.info(f"📈 Loaded {len(data)} recent performance records for {dog_name} in {(end_time - start_time)*1000:.2f}ms")
        return data

def query_performance_decorator(func):
    """Decorator to measure query performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000
        logger.info(f"⏱️ {func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper

# Global instances
_connection_pool = None
_lazy_loader = None

def initialize_db_optimization(database_path: str = "greyhound_racing_data.db", 
                             pool_size: int = 20):
    """Initialize database optimization components"""
    global _connection_pool, _lazy_loader
    
    _connection_pool = SQLiteConnectionPool(database_path, pool_size)
    _lazy_loader = LazyModelLoader(_connection_pool)
    
    logger.info("🚀 Database optimization initialized")
    return _connection_pool, _lazy_loader

def get_db_pool() -> SQLiteConnectionPool:
    """Get the global connection pool"""
    if _connection_pool is None:
        initialize_db_optimization()
    return _connection_pool

def get_lazy_loader() -> LazyModelLoader:
    """Get the global lazy loader"""
    if _lazy_loader is None:
        initialize_db_optimization()
    return _lazy_loader

def benchmark_db_performance():
    """Benchmark database performance with optimizations"""
    print("🔬 Running Database Performance Benchmark")
    print("=" * 50)
    
    pool = get_db_pool()
    loader = get_lazy_loader()
    
    # Test connection pool performance
    start_time = time.time()
    for i in range(10):
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            result = cursor.fetchone()
    end_time = time.time()
    
    print(f"📊 10 pooled queries: {(end_time - start_time)*1000:.2f}ms")
    
    # Test lazy loading performance
    start_time = time.time()
    race_data = loader.get_race_metadata(limit=100)
    end_time = time.time()
    
    print(f"🏁 Lazy load 100 races: {(end_time - start_time)*1000:.2f}ms")
    
    # Test cached performance
    start_time = time.time()
    race_data_cached = loader.get_race_metadata(limit=100)  # Should hit cache
    end_time = time.time()
    
    print(f"📦 Cached 100 races: {(end_time - start_time)*1000:.2f}ms")
    
    # Print pool stats
    stats = pool.get_stats()
    print(f"\n📈 Connection Pool Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    # Initialize and benchmark
    initialize_db_optimization()
    benchmark_db_performance()
