#!/usr/bin/env python3
"""
Race Time Cache System
=====================

High-performance L1 (memory) + L2 (disk) cache for race times scraped from thedogs.com.au.
Designed to dramatically reduce network requests and improve interactive races tab performance.

Author: AI Assistant
Date: September 2, 2025
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from urllib.parse import urlparse
import hashlib


class RaceTimeCache:
    """
    Two-tier caching system for race times:
    - L1 Cache: In-memory dict with TTL and size cap (fast access)
    - L2 Cache: SQLite database on disk (persistent storage)
    
    Cache value schema:
    {
        'race_time': '2:15 PM',
        'time_source': 'live_scraped' | 'cache' | 'estimated',
        'scraped_at_iso': '2025-09-02T06:00:00Z',
        'etag': 'W/"abc123"',
        'last_modified': 'Mon, 02 Sep 2025 06:00:00 GMT',
        'url': 'https://www.thedogs.com.au/racing/venue/2025-09-02/1',
        'date': '2025-09-02',
        'venue': 'VENUE_CODE',
        'race_number': 1,
        'parser_version': '1.0'
    }
    """
    
    def __init__(self, cache_dir: str = "cache/race_times", max_l1_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_l1_size = max_l1_size
        self.parser_version = "1.0"
        
        # L1 Cache: {cache_key: (value, expires_at_timestamp)}
        self._l1_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._l1_lock = threading.RLock()
        
        # L2 Cache: SQLite database
        self.db_path = self.cache_dir / "race_times.db"
        self._l2_lock = threading.RLock()
        
        self._init_l2_cache()
        
        # TTL settings (in minutes)
        self.ttl_settings = {
            'same_day': int(os.environ.get('RACE_TIME_CACHE_TTL_SAME_DAY', 120)),  # 2 hours
            'future_races': int(os.environ.get('RACE_TIME_CACHE_TTL_FUTURE', 360)),  # 6 hours  
            'past_grace': int(os.environ.get('RACE_TIME_CACHE_TTL_PAST', 1440)),  # 24 hours
            'missing': int(os.environ.get('RACE_TIME_CACHE_TTL_MISSING', 30))  # 30 minutes
        }
        
        print(f"🏎️ Race Time Cache initialized: {self.cache_dir}")
        print(f"   📊 L1 max size: {max_l1_size} entries")
        print(f"   ⏰ TTL settings: same_day={self.ttl_settings['same_day']}m, future={self.ttl_settings['future_races']}m, past={self.ttl_settings['past_grace']}m")
    
    def _init_l2_cache(self):
        """Initialize SQLite database for L2 cache"""
        with self._l2_lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS race_times (
                    cache_key TEXT PRIMARY KEY,
                    race_time TEXT,
                    time_source TEXT,
                    scraped_at_iso TEXT,
                    expires_at_timestamp REAL,
                    etag TEXT,
                    last_modified TEXT,
                    url TEXT,
                    date TEXT,
                    venue TEXT,
                    race_number INTEGER,
                    parser_version TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    accessed_at REAL DEFAULT (julianday('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date_venue ON race_times(date, venue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON race_times(expires_at_timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON race_times(url)")
            conn.commit()
            conn.close()
    
    def _generate_cache_key(self, url: str, date: str = None) -> str:
        """Generate consistent cache key from URL and optional date"""
        # Normalize URL
        parsed = urlparse(url)
        normalized_url = f"{parsed.netloc}{parsed.path}"
        
        # Include date if provided for additional uniqueness
        key_parts = [normalized_url]
        if date:
            key_parts.append(date)
            
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _calculate_ttl_minutes(self, date: str) -> int:
        """Calculate TTL in minutes based on race date"""
        try:
            race_date = datetime.strptime(date, '%Y-%m-%d').date()
            today = datetime.now().date()
            
            if race_date == today:
                return self.ttl_settings['same_day']
            elif race_date > today:
                days_ahead = (race_date - today).days
                if days_ahead <= 7:
                    return self.ttl_settings['future_races']
                else:
                    return self.ttl_settings['future_races'] * 2  # Longer TTL for far future
            else:
                return self.ttl_settings['past_grace']
        except (ValueError, TypeError):
            return self.ttl_settings['same_day']  # Default fallback
    
    def _is_expired(self, expires_at_timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() > expires_at_timestamp
    
    def _evict_l1_if_needed(self):
        """Evict oldest entries from L1 cache if size limit exceeded"""
        with self._l1_lock:
            if len(self._l1_cache) >= self.max_l1_size:
                # Sort by expiration time and remove oldest 10%
                sorted_items = sorted(
                    self._l1_cache.items(),
                    key=lambda x: x[1][1]  # Sort by expires_at_timestamp
                )
                
                num_to_evict = max(1, len(sorted_items) // 10)
                for i in range(num_to_evict):
                    key_to_evict = sorted_items[i][0]
                    del self._l1_cache[key_to_evict]
    
    def get(self, url: str, date: str = None) -> Optional[Dict[str, Any]]:
        """
        Get cached race time data.
        
        Args:
            url: Race page URL
            date: Race date in YYYY-MM-DD format (optional)
            
        Returns:
            Cached race data dict or None if not found/expired
        """
        cache_key = self._generate_cache_key(url, date)
        
        # Try L1 cache first
        with self._l1_lock:
            if cache_key in self._l1_cache:
                value, expires_at = self._l1_cache[cache_key]
                if not self._is_expired(expires_at):
                    # L1 cache hit
                    result = value.copy()
                    result['time_source'] = 'cache_l1'
                    return result
                else:
                    # Expired, remove from L1
                    del self._l1_cache[cache_key]
        
        # Try L2 cache
        with self._l2_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM race_times WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row and not self._is_expired(row['expires_at_timestamp']):
                    # L2 cache hit - promote to L1
                    value = {
                        'race_time': row['race_time'],
                        'time_source': 'cache_l2',
                        'scraped_at_iso': row['scraped_at_iso'],
                        'etag': row['etag'],
                        'last_modified': row['last_modified'],
                        'url': row['url'],
                        'date': row['date'],
                        'venue': row['venue'],
                        'race_number': row['race_number'],
                        'parser_version': row['parser_version']
                    }
                    
                    # Update L1 cache
                    self._evict_l1_if_needed()
                    with self._l1_lock:
                        self._l1_cache[cache_key] = (value, row['expires_at_timestamp'])
                    
                    # Update access time in L2
                    conn = sqlite3.connect(self.db_path)
                    conn.execute(
                        "UPDATE race_times SET accessed_at = julianday('now') WHERE cache_key = ?",
                        (cache_key,)
                    )
                    conn.commit()
                    conn.close()
                    
                    return value
                elif row:
                    # Expired entry in L2 - cleanup
                    conn = sqlite3.connect(self.db_path)
                    conn.execute("DELETE FROM race_times WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    conn.close()
                    
            except Exception as e:
                print(f"   ⚠️ L2 cache read error for {cache_key}: {e}")
        
        return None
    
    def put(self, url: str, race_data: Dict[str, Any], date: str = None) -> bool:
        """
        Store race time data in both L1 and L2 cache.
        
        Args:
            url: Race page URL
            race_data: Race data dict with race_time, time_source, etc.
            date: Race date in YYYY-MM-DD format (optional, extracted from race_data)
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Extract/validate required fields
            if not race_data.get('race_time'):
                return False
                
            cache_date = date or race_data.get('date', '')
            if not cache_date:
                # Try to extract date from URL or use today
                cache_date = datetime.now().strftime('%Y-%m-%d')
                
            cache_key = self._generate_cache_key(url, cache_date)
            ttl_minutes = self._calculate_ttl_minutes(cache_date)
            expires_at = time.time() + (ttl_minutes * 60)
            
            # Prepare cache value
            cache_value = {
                'race_time': race_data['race_time'],
                'time_source': race_data.get('time_source', 'unknown'),
                'scraped_at_iso': race_data.get('scraped_at_iso', datetime.now().isoformat()),
                'etag': race_data.get('etag', ''),
                'last_modified': race_data.get('last_modified', ''),
                'url': url,
                'date': cache_date,
                'venue': race_data.get('venue', ''),
                'race_number': race_data.get('race_number', 0),
                'parser_version': self.parser_version
            }
            
            # Store in L1 cache
            self._evict_l1_if_needed()
            with self._l1_lock:
                self._l1_cache[cache_key] = (cache_value, expires_at)
            
            # Store in L2 cache
            with self._l2_lock:
                conn = sqlite3.connect(self.db_path)
                conn.execute("""
                    INSERT OR REPLACE INTO race_times (
                        cache_key, race_time, time_source, scraped_at_iso,
                        expires_at_timestamp, etag, last_modified, url,
                        date, venue, race_number, parser_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, cache_value['race_time'], cache_value['time_source'],
                    cache_value['scraped_at_iso'], expires_at, cache_value['etag'],
                    cache_value['last_modified'], url, cache_date,
                    cache_value['venue'], cache_value['race_number'], self.parser_version
                ))
                conn.commit()
                conn.close()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Cache put error for {url}: {e}")
            return False
    
    def get_conditional_headers(self, url: str, date: str = None) -> Dict[str, str]:
        """
        Get conditional request headers (If-None-Match, If-Modified-Since) for HTTP requests.
        
        Args:
            url: Race page URL
            date: Race date (optional)
            
        Returns:
            Dict with conditional headers
        """
        cached_data = self.get(url, date)
        headers = {}
        
        if cached_data:
            if cached_data.get('etag'):
                headers['If-None-Match'] = cached_data['etag']
            if cached_data.get('last_modified'):
                headers['If-Modified-Since'] = cached_data['last_modified']
                
        return headers
    
    def handle_304_response(self, url: str, date: str = None) -> Optional[Dict[str, Any]]:
        """
        Handle HTTP 304 Not Modified response by refreshing cache TTL.
        
        Args:
            url: Race page URL  
            date: Race date (optional)
            
        Returns:
            Refreshed cache data or None
        """
        cached_data = self.get(url, date)
        if cached_data:
            # Refresh the cache entry with updated TTL
            refreshed_data = cached_data.copy()
            refreshed_data['time_source'] = 'cache_304'
            refreshed_data['scraped_at_iso'] = datetime.now().isoformat()
            
            if self.put(url, refreshed_data, date):
                return refreshed_data
                
        return None
    
    def prefill_race_times(self, races: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prefill race times from cache for a list of races.
        
        Args:
            races: List of race dicts with 'url', 'date', etc.
            
        Returns:
            List of races with cached times filled in where available
        """
        enriched_races = []
        cache_hits = 0
        
        for race in races:
            enriched_race = race.copy()
            
            cached_data = self.get(race.get('url', ''), race.get('date'))
            if cached_data:
                # Use cached time
                enriched_race['race_time'] = cached_data['race_time']
                enriched_race['time_source'] = cached_data['time_source'] 
                cache_hits += 1
            
            enriched_races.append(enriched_race)
        
        if races:
            hit_rate = (cache_hits / len(races)) * 100
            print(f"   📊 Cache prefill: {cache_hits}/{len(races)} hits ({hit_rate:.1f}%)")
            
        return enriched_races
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries from both L1 and L2 caches.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        current_time = time.time()
        
        # Clean L1 cache
        with self._l1_lock:
            expired_keys = [
                key for key, (_, expires_at) in self._l1_cache.items()
                if self._is_expired(expires_at)
            ]
            for key in expired_keys:
                del self._l1_cache[key]
                cleaned_count += 1
        
        # Clean L2 cache
        with self._l2_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute(
                    "DELETE FROM race_times WHERE expires_at_timestamp < ?",
                    (current_time,)
                )
                l2_cleaned = cursor.rowcount
                conn.commit()
                conn.close()
                cleaned_count += l2_cleaned
            except Exception as e:
                print(f"   ⚠️ L2 cleanup error: {e}")
        
        if cleaned_count > 0:
            print(f"   🧹 Cache cleanup: removed {cleaned_count} expired entries")
            
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._l1_lock:
            l1_count = len(self._l1_cache)
        
        l2_count = 0
        l2_size_mb = 0
        
        with self._l2_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM race_times")
                l2_count = cursor.fetchone()[0]
                
                # Get database file size
                if self.db_path.exists():
                    l2_size_mb = self.db_path.stat().st_size / (1024 * 1024)
                    
                conn.close()
            except Exception:
                pass
        
        return {
            'l1_entries': l1_count,
            'l1_max_size': self.max_l1_size,
            'l2_entries': l2_count,
            'l2_size_mb': round(l2_size_mb, 2),
            'cache_dir': str(self.cache_dir),
            'ttl_settings': self.ttl_settings
        }
    
    def cache_missing_result(self, url: str, date: str = None) -> bool:
        """
        Cache a 'missing' result (404, no time found) to avoid repeated requests.
        
        Args:
            url: Race page URL
            date: Race date (optional)
            
        Returns:
            True if cached successfully
        """
        missing_data = {
            'race_time': None,
            'time_source': 'missing',
            'scraped_at_iso': datetime.now().isoformat(),
            'venue': '',
            'race_number': 0
        }
        
        return self.put(url, missing_data, date)


# Global cache instance
_global_cache: Optional[RaceTimeCache] = None
_cache_lock = threading.Lock()


def get_race_time_cache() -> RaceTimeCache:
    """Get or create the global race time cache instance"""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = RaceTimeCache()
                
    return _global_cache


def clear_global_cache():
    """Clear the global cache instance (for testing)"""
    global _global_cache
    with _cache_lock:
        _global_cache = None


if __name__ == "__main__":
    # Simple test
    cache = RaceTimeCache()
    
    # Test data
    test_url = "https://www.thedogs.com.au/racing/test-venue/2025-09-02/1"
    test_data = {
        'race_time': '2:15 PM',
        'time_source': 'live_scraped',
        'venue': 'TEST',
        'race_number': 1,
        'date': '2025-09-02'
    }
    
    print("🧪 Testing cache operations...")
    
    # Test put
    success = cache.put(test_url, test_data)
    print(f"   Put result: {success}")
    
    # Test get
    cached = cache.get(test_url, '2025-09-02')
    print(f"   Get result: {cached}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"   Stats: {stats}")
    
    print("✅ Cache test completed")
