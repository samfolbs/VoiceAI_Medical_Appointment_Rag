"""
rag/cache.py
LRU cache implementations for embedding results.
"""
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with optional TTL expiry."""

    def __init__(self, maxsize: int = 100, ttl_seconds: Optional[int] = 3600) -> None:
        self._cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.hits = self.misses = self.evictions = 0

    def _key(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        try:
            return hashlib.md5(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()
        except (TypeError, ValueError):
            return str(raw)

    def _expired(self, ts: datetime) -> bool:
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - ts).total_seconds() > self.ttl_seconds

    def get(self, key: Any) -> Optional[Any]:
        k = self._key(key)
        if k not in self._cache:
            self.misses += 1
            return None
        ts, value = self._cache[k]
        if self._expired(ts):
            del self._cache[k]
            self.misses += 1
            return None
        self._cache.move_to_end(k)
        self.hits += 1
        return value

    def set(self, key: Any, value: Any) -> bool:
        k = self._key(key)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = (datetime.now(), value)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
            self.evictions += 1
        return True

    def delete(self, key: Any) -> bool:
        k = self._key(key)
        if k in self._cache:
            del self._cache[k]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def cleanup_expired(self) -> int:
        expired = [k for k, (ts, _) in self._cache.items() if self._expired(ts)]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "evictions": self.evictions,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None


class MultiLevelCache:
    """Two-level LRU cache (L1 = hot / small, L2 = warm / large)."""

    def __init__(
        self,
        l1_size: int = 50,
        l2_size: int = 500,
        ttl_seconds: int = 3600,
    ) -> None:
        self.l1 = LRUCache(maxsize=l1_size, ttl_seconds=ttl_seconds)
        self.l2 = LRUCache(maxsize=l2_size, ttl_seconds=ttl_seconds)
        self.l1_hits = self.l2_hits = self.total_misses = 0

    def get(self, key: Any) -> Optional[Any]:
        value = self.l1.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        value = self.l2.get(key)
        if value is not None:
            self.l2_hits += 1
            self.l1.set(key, value)  # promote to L1
            return value
        self.total_misses += 1
        return None

    def set(self, key: Any, value: Any) -> bool:
        return self.l1.set(key, value) and self.l2.set(key, value)

    def delete(self, key: Any) -> bool:
        return self.l1.delete(key) or self.l2.delete(key)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def cleanup_expired(self) -> int:
        return self.l1.cleanup_expired() + self.l2.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        total = self.l1_hits + self.l2_hits + self.total_misses
        hit_rate = ((self.l1_hits + self.l2_hits) / total * 100) if total else 0
        return {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats(),
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "total_misses": self.total_misses,
            "total_hit_rate": f"{hit_rate:.1f}%",
        }
