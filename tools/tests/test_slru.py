from rapidai_tools.slru import SLRUCache


def test_hit_and_miss_counters():
    c = SLRUCache(capacity_bytes=100)
    assert c.get("a") is None
    c.put("a", "va", 10)
    assert c.get("a") == "va"
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1


def test_eviction_prefers_probationary():
    c = SLRUCache(capacity_bytes=30, protected_frac=0.5)
    c.put("a", "va", 10)
    c.get("a")  # promote a to protected
    c.put("b", "vb", 10)  # b probationary
    c.put("c", "vc", 20)  # needs 20 free -> evict probationary b (not protected a)
    assert c.get("a") == "va"
    assert c.get("b") is None


def test_capacity_respected():
    c = SLRUCache(capacity_bytes=25)
    for i in range(10):
        c.put(f"k{i}", i, 10)
    assert c.stats()["resident_bytes"] <= 25


def test_oversized_item_rejected():
    c = SLRUCache(capacity_bytes=10)
    c.put("big", "x", 50)
    assert c.get("big") is None
