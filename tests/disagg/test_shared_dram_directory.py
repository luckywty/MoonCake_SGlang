"""Metadata ownership tests require no GPU, socket or POSIX SHM allocation."""

import threading
from types import SimpleNamespace

import pytest
from minisgl.disagg.shared_dram_directory import SharedDramDirectory
from minisgl.disagg.shared_dram_pool import FREE, QUARANTINED, READY, STAGED, SharedPage


class HeaderPool:
    def __init__(self, counts):
        self.manifest = SimpleNamespace(
            node_slots=counts, name="history", registered_bytes=8192, capacity=sum(counts.values())
        )
        self.headers = {(n, s): (0, FREE) for n, size in counts.items() for s in range(size)}

    def set_header(self, node, slot, generation, state):
        self.headers[node, slot] = generation, state

    def header(self, node, slot):
        return self.headers[node, slot]


def digest(i):
    return f"{i:064x}"


def allocate(directory, i, node=0, owner="d"):
    result = directory.allocate_write(digest(i), node, owner)
    assert result["status"] == "allocated", result
    return SharedPage.from_object(result["page"])


def commit(directory, page):
    directory.commit_write(page.numa_id, page.slot_id, page.generation, page.page_hash)


@pytest.mark.parametrize("counts", [{0: 2}, {0: 1, 1: 1}, {0: 1, 2: 1, 4: 1}])
def test_local_then_other_nodes_then_eviction_and_full_when_pinned(counts):
    pool = HeaderPool(counts)
    directory = SharedDramDirectory(pool)
    pages = [allocate(directory, i + 1) for i in range(sum(counts.values()))]
    assert pages[0].numa_id == 0
    assert [p.numa_id for p in pages] == [n for n in sorted(counts) for _ in range(counts[n])]
    for p in pages:
        commit(directory, p)
    leases = directory.pin_prefix([p.page_hash for p in pages], "p")
    assert len(leases) == len(pages)
    assert directory.allocate_write(digest(99), 0, "d") == {"status": "full"}
    directory.unpin(leases[:1], "p")
    new = allocate(directory, 99)
    assert (new.numa_id, new.slot_id, new.generation) == (pages[0].numa_id, pages[0].slot_id, 2)
    assert directory.lookup_prefix([pages[0].page_hash]) == []
    directory.unpin(leases, "p")  # A repeated already-released lease has no effect.


def test_staged_visibility_duplicate_commit_and_abort():
    pool = HeaderPool({0: 2, 1: 1})
    directory = SharedDramDirectory(pool)
    p = allocate(directory, 1)
    assert pool.header(0, p.slot_id) == (1, STAGED)
    assert directory.lookup_prefix([digest(1)]) == []
    assert directory.pin_prefix([digest(1)], "p") == []
    assert directory.allocate_write(digest(1), 1, "other") == {"status": "duplicate"}
    commit(directory, p)
    assert directory.lookup_prefix([digest(1)]) == [p.to_object()]
    assert pool.header(0, p.slot_id) == (1, READY)
    assert directory.allocate_write(digest(1), 1, "other") == {"status": "duplicate"}
    q = allocate(directory, 2)
    directory.abort_write(q.numa_id, q.slot_id, q.generation, q.page_hash)
    assert pool.header(q.numa_id, q.slot_id) == (q.generation, FREE)
    with pytest.raises(ValueError, match="generation"):
        commit(directory, q)


@pytest.mark.parametrize("policy", ["lru", "lfu"])
def test_eviction_prefers_local_even_when_remote_is_colder(policy):
    clock = [1.0]
    directory = SharedDramDirectory(HeaderPool({0: 2, 1: 1}), policy=policy, clock=lambda: clock[0])
    a = allocate(directory, 1, node=1)
    commit(directory, a)
    clock[0] += 1
    b = allocate(directory, 2)
    commit(directory, b)
    clock[0] += 1
    c = allocate(directory, 3)
    commit(directory, c)
    for _ in range(3):
        clock[0] += 1
        pins = directory.pin_prefix([b.page_hash], "p")
        directory.unpin(pins, "p")
    d = allocate(directory, 4)
    assert d.numa_id == 0
    assert directory.lookup_prefix([a.page_hash])
    assert directory.lookup_prefix([b.page_hash])
    assert directory.lookup_prefix([c.page_hash]) == []
    commit(directory, d)
    pins = directory.pin_prefix([b.page_hash, d.page_hash], "p")
    e = allocate(directory, 5)
    assert e.numa_id == 1  # Local pages pinned; only the remote victim is safe.
    directory.unpin(pins, "p")


def test_lfu_selects_lower_frequency_and_decays():
    now = [1.0]
    directory = SharedDramDirectory(HeaderPool({0: 2}), policy="lfu", clock=lambda: now[0])
    a, b = allocate(directory, 1), allocate(directory, 2)
    commit(directory, a)
    commit(directory, b)
    for _ in range(3):
        directory.unpin(directory.pin_prefix([a.page_hash], "p"), "p")
    now[0] = 121
    allocate(directory, 3)
    assert directory.lookup_prefix([a.page_hash])
    assert directory.lookup_prefix([b.page_hash]) == []
    assert directory.entries[a.page_hash].freq < 4


def test_generation_and_owner_checks_reject_late_mutations():
    directory = SharedDramDirectory(HeaderPool({0: 1}))
    a = allocate(directory, 1)
    commit(directory, a)
    leases = directory.pin_prefix([a.page_hash], "p")
    with pytest.raises(ValueError, match="mismatch"):
        directory.unpin(leases, "wrong-owner")
    directory.unpin(leases, "p")
    b = allocate(directory, 2)
    assert b.generation == a.generation + 1
    for method in (directory.commit_write, directory.abort_write):
        with pytest.raises(ValueError, match="generation"):
            method(a.numa_id, a.slot_id, a.generation, a.page_hash)
    with pytest.raises(ValueError, match="another owner"):
        directory.commit_write(b.numa_id, b.slot_id, b.generation, b.page_hash, writer_id="other")


def test_socket_loss_retains_readers_and_quarantines_writers_until_process_exit():
    directory = SharedDramDirectory(HeaderPool({0: 2}))
    a = allocate(directory, 1)
    commit(directory, a)
    b = allocate(directory, 2)
    directory.pin_prefix([a.page_hash], "p")
    directory.disconnect_owner("p")
    directory.disconnect_owner("d")
    assert directory.health()["readers"] == 1
    assert directory.pool.header(b.numa_id, b.slot_id)[1] == QUARANTINED
    assert directory.allocate_write(digest(3), 0, "d2")["status"] == "full"
    directory.disconnect_owner("p", process_dead=True)
    directory.disconnect_owner("d", process_dead=True)
    assert directory.health()["readers"] == 0
    assert directory.pool.header(b.numa_id, b.slot_id)[1] == FREE
    assert directory.lookup_prefix([a.page_hash])  # Committed data survives its writer.


def test_lock_free_snapshot_lookup_and_concurrent_duplicate_allocation():
    directory = SharedDramDirectory(HeaderPool({0: 4, 1: 4}))
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(directory.allocate_write(digest(1), 0, "d")))
        for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r["status"] == "allocated" for r in results) == 1
    p = SharedPage.from_object(next(r["page"] for r in results if r["status"] == "allocated"))
    commit(directory, p)

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("lookup must not acquire the directory mutation lock")

        def __exit__(self, *args):
            pass

    directory.lock = ForbiddenLock()
    assert directory.lookup_prefix([digest(1), digest(2)]) == [p.to_object()]
