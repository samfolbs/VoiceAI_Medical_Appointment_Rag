"""
LRU Cache Implementation
Memory-efficient caching for embeddings and API responses
"""
from collections import OrderedDict
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) cache with size and time limits
    Thread-safe implementation for embedding caching
    """
    
    def __init__(
        self, 
        maxsize: int = 100,
        ttl_seconds: Optional[int] = 3600
    ):
        """
        Initialize LRU cache
        
        Args:
            maxsize: Maximum number of items to cache
            ttl_seconds: Time-to-live in seconds (None for no expiry)
        """
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(
            f"LRUCache initialized: maxsize={maxsize}, ttl={ttl_seconds}s"
        )
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """
        Check if cache entry is expired
        
        Args:
            timestamp: Entry timestamp
            
        Returns:
            True if expired
        """
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def _make_key(self, key: Any) -> str:
        """
        Create hashable cache key
        
        Args:
            key: Any hashable object
            
        Returns:
            String key
        """
        if isinstance(key, str):
            return key
        
        # Hash complex objects
        try:
            key_str = json.dumps(key, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        except (TypeError, ValueError):
            return str(key)
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            cache_key = self._make_key(key)
            
            if cache_key not in self.cache:
                self.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Get entry
            entry = self.cache[cache_key]
            timestamp, value = entry
            
            # Check expiry
            if self._is_expired(timestamp):
                logger.debug(f"Cache expired: {key}")
                del self.cache[cache_key]
                self.misses += 1
                return None
            
            # Move most recently used to the end 
            self.cache.move_to_end(cache_key)
            self.hits += 1
            
            logger.debug(f"Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: Any, value: Any) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._make_key(key)
            
            # Update existing entry
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                self.cache[cache_key] = (datetime.now(), value)
                logger.debug(f"Cache updated: {key}")
                return True
            
            # Add new entry
            self.cache[cache_key] = (datetime.now(), value)
            
            # Evict oldest if over size limit
            if len(self.cache) > self.maxsize:
                evicted_key, _ = self.cache.popitem(last=False)
                self.evictions += 1
                logger.debug(f"Cache evicted: {evicted_key}")
            
            logger.debug(f"Cache set: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: Any) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        try:
            cache_key = self._make_key(key)
            
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.debug(f"Cache deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared: {size} entries removed")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0
        
        try:
            expired_keys = []
            
            for cache_key, (timestamp, _) in self.cache.items():
                if self._is_expired(timestamp):
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                del self.cache[cache_key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Statistics dictionary
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'evictions': self.evictions,
            'ttl_seconds': self.ttl_seconds
        }
    
    def __len__(self) -> int:
        """Get cache size"""
        return len(self.cache)
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None


class MultiLevelCache:
    """
    Two-level cache: Embedding caching with hot/cold data -fast in-memory L1, slower but larger L2
    """
    
    def __init__(
        self,
        l1_size: int = 50,
        l2_size: int = 500,
        ttl_seconds: int = 3600
    ):
        """
        Initialize multi-level cache
        
        Args:
            l1_size: L1 cache size (hot data)
            l2_size: L2 cache size (warm data)
            ttl_seconds: Time-to-live for entries
        """
        self.l1_cache = LRUCache(maxsize=l1_size, ttl_seconds=ttl_seconds)
        self.l2_cache = LRUCache(maxsize=l2_size, ttl_seconds=ttl_seconds)
        self.l1_hits = 0
        self.l2_hits = 0
        self.total_misses = 0
        
        logger.info(
            f"MultiLevelCache initialized: L1={l1_size}, L2={l2_size}"
        )
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache (L1 first, then L2)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            self.l2_hits += 1
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        # Miss
        self.total_misses += 1
        return None
    
    def set(self, key: Any, value: Any) -> bool:
        """
        Set value in cache (both L1 and L2)
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        # Set in both levels
        l1_success = self.l1_cache.set(key, value)
        l2_success = self.l2_cache.set(key, value)
        
        return l1_success and l2_success
    
    def delete(self, key: Any) -> bool:
        """Delete from both cache levels"""
        l1_deleted = self.l1_cache.delete(key)
        l2_deleted = self.l2_cache.delete(key)
        return l1_deleted or l2_deleted
    
    def clear(self) -> None:
        """Clear both cache levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        logger.info("MultiLevelCache cleared")
    
    def cleanup_expired(self) -> int:
        """Cleanup expired entries in both levels"""
        l1_cleaned = self.l1_cache.cleanup_expired()
        l2_cleaned = self.l2_cache.cleanup_expired()
        return l1_cleaned + l2_cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        total_requests = self.l1_hits + self.l2_hits + self.total_misses
        
        if total_requests > 0:
            l1_hit_rate = (self.l1_hits / total_requests) * 100
            l2_hit_rate = (self.l2_hits / total_requests) * 100
            total_hit_rate = ((self.l1_hits + self.l2_hits) / total_requests) * 100
        else:
            l1_hit_rate = l2_hit_rate = total_hit_rate = 0
        
        return {
            'l1_cache': self.l1_cache.get_stats(),
            'l2_cache': self.l2_cache.get_stats(),
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'total_misses': self.total_misses,
            'l1_hit_rate': f"{l1_hit_rate:.2f}%",
            'l2_hit_rate': f"{l2_hit_rate:.2f}%",
            'total_hit_rate': f"{total_hit_rate:.2f}%"
        }