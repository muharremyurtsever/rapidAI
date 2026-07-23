"""Segmented LRU cache with byte-cost accounting (equation factor: M_miss).

Byte totals are maintained as running counters — profiling (2026-07-24)
showed the original recompute-on-every-call `sum()` was 59% of decode time.
"""

from collections import OrderedDict


class SLRUCache:
    def __init__(self, capacity_bytes: int, protected_frac: float = 0.8):
        self.capacity = capacity_bytes
        self.protected_cap = int(capacity_bytes * protected_frac)
        self.probation: OrderedDict = OrderedDict()  # key -> (value, nbytes)
        self.protected: OrderedDict = OrderedDict()
        self._probation_bytes = 0
        self._protected_bytes = 0
        self.hits = self.misses = self.evictions = 0

    def _resident(self) -> int:
        return self._probation_bytes + self._protected_bytes

    def get(self, key):
        if key in self.protected:
            self.protected.move_to_end(key)
            self.hits += 1
            return self.protected[key][0]
        if key in self.probation:
            value, nbytes = self.probation.pop(key)
            self._probation_bytes -= nbytes
            self.hits += 1
            self._make_room_protected(nbytes)
            self.protected[key] = (value, nbytes)
            self._protected_bytes += nbytes
            return value
        self.misses += 1
        return None

    def put(self, key, value, nbytes: int):
        if nbytes > self.capacity:
            return
        if key in self.probation:
            _, old = self.probation.pop(key)
            self._probation_bytes -= old
        if key in self.protected:
            _, old = self.protected.pop(key)
            self._protected_bytes -= old
        while self._resident() + nbytes > self.capacity:
            self._evict_one()
        self.probation[key] = (value, nbytes)
        self._probation_bytes += nbytes

    def _make_room_protected(self, nbytes: int):
        # demote protected LRU entries while protected segment exceeds its cap
        while self._protected_bytes + nbytes > self.protected_cap and self.protected:
            k, (v, n) = self.protected.popitem(last=False)
            self._protected_bytes -= n
            self.probation[k] = (v, n)
            self.probation.move_to_end(k, last=False)
            self._probation_bytes += n
        while self._resident() + nbytes > self.capacity:
            self._evict_one()

    def _evict_one(self):
        if self.probation:
            _, (_, n) = self.probation.popitem(last=False)
            self._probation_bytes -= n
        elif self.protected:
            _, (_, n) = self.protected.popitem(last=False)
            self._protected_bytes -= n
        self.evictions += 1

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "resident_bytes": self._resident(),
        }
