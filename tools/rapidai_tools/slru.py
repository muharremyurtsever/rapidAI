"""Segmented LRU cache with byte-cost accounting (equation factor: M_miss)."""

from collections import OrderedDict


class SLRUCache:
    def __init__(self, capacity_bytes: int, protected_frac: float = 0.8):
        self.capacity = capacity_bytes
        self.protected_cap = int(capacity_bytes * protected_frac)
        self.probation: OrderedDict = OrderedDict()  # key -> (value, nbytes)
        self.protected: OrderedDict = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def _resident(self) -> int:
        return sum(n for _, n in self.probation.values()) + sum(
            n for _, n in self.protected.values()
        )

    def get(self, key):
        if key in self.protected:
            self.protected.move_to_end(key)
            self.hits += 1
            return self.protected[key][0]
        if key in self.probation:
            value, nbytes = self.probation.pop(key)
            self.hits += 1
            self._make_room_protected(nbytes)
            self.protected[key] = (value, nbytes)
            return value
        self.misses += 1
        return None

    def put(self, key, value, nbytes: int):
        if nbytes > self.capacity:
            return
        if key in self.probation:
            del self.probation[key]
        if key in self.protected:
            del self.protected[key]
        while self._resident() + nbytes > self.capacity:
            self._evict_one()
        self.probation[key] = (value, nbytes)

    def _make_room_protected(self, nbytes: int):
        # demote protected LRU entries while protected segment exceeds its cap
        while (
            sum(n for _, n in self.protected.values()) + nbytes > self.protected_cap
            and self.protected
        ):
            k, (v, n) = self.protected.popitem(last=False)
            self.probation[k] = (v, n)
            self.probation.move_to_end(k, last=False)
        while self._resident() + nbytes > self.capacity:
            self._evict_one()

    def _evict_one(self):
        if self.probation:
            self.probation.popitem(last=False)
        elif self.protected:
            self.protected.popitem(last=False)
        self.evictions += 1

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "resident_bytes": self._resident(),
        }
