"""Bounded host KV pages. A deadline revokes a lease, never proves DMA stopped."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

import torch


@dataclass(eq=False)
class DramPage:
    digest: bytes
    parent: bytes
    depth: int
    slot: int
    generation: int
    writing: bool = True
    ready: bool = False
    committed: bool = False
    failed: bool = False
    readers: int = 0
    exports: int = 0
    touched: float = 0.0
    frequency: float = 1.0
    transactions: set[str] = field(default_factory=set)


@dataclass
class DramLease:
    id: str
    peer: str
    pages: tuple[DramPage, ...]
    expires: float
    expired: bool = False


class DramPagePool:
    """One startup allocation; all returned page views borrow this allocation.

    Host bookkeeping is protected independently of the GPU scheduler. Exported
    pages are immutable and remain pinned until a drained receipt or a native
    fence proves the peer cannot access their addresses anymore.
    """

    def __init__(
        self,
        page_bytes,
        capacity_bytes,
        *,
        policy="lru",
        pin_memory=True,
        max_leases=32,
        lease_ttl=30.0,
        transaction_ttl=300.0,
        max_transactions=256,
        clock=time.monotonic,
        observer=None,
    ):
        if (
            type(page_bytes) is not int
            or page_bytes <= 0
            or type(capacity_bytes) is not int
            or capacity_bytes <= 0
            or policy not in ("lru", "lfu")
            or max_leases < 1
            or lease_ttl <= 0
            or transaction_ttl <= 0
            or max_transactions < 1
        ):
            raise ValueError("invalid DRAM cache configuration")
        self.page_bytes = page_bytes
        self.stride = (page_bytes + 255) // 256 * 256
        self.capacity = capacity_bytes // self.stride
        if not self.capacity:
            raise ValueError("DRAM cache cannot hold one aligned KV page")
        self.buffer = torch.empty(
            (self.capacity, self.stride), dtype=torch.uint8, device="cpu", pin_memory=pin_memory
        )
        self.policy, self.clock, self.observer = policy, clock, observer
        self.max_leases, self.lease_ttl = max_leases, lease_ttl
        self.transaction_ttl, self.max_transactions = transaction_ttl, max_transactions
        self.pages: OrderedDict[bytes, DramPage] = OrderedDict()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.generations = [0] * self.capacity
        self.transactions = OrderedDict()
        self.transaction_pages = {}
        self.active_transactions = set()
        self.closed_transactions = OrderedDict()
        self.leases: dict[str, DramLease] = {}
        self.closed_leases = OrderedDict()
        self.quarantined_slots = set()
        self.lock = threading.RLock()
        self.last_decay = clock()
        self.stats = {"evictions": 0, "hits": 0, "staged": 0, "committed": 0, "aborted": 0}

    @property
    def registered_bytes(self):
        return self.buffer.numel()

    def view(self, page):
        return self.buffer[page.slot, : self.page_bytes]

    def _notify(self, op, page):
        if self.observer is not None:
            try:
                self.observer(op, page)
            except Exception:
                # An optional routing directory must not affect payload ownership.
                self.observer = None

    def begin(self, transaction):
        with self.lock:
            if transaction in self.closed_transactions:
                return False
            if transaction in self.transactions:
                return self.transactions[transaction][0] != "abort"
            if len(self.transactions) >= self.max_transactions:
                return False
            self.transactions[transaction] = ("pending", self.clock() + self.transaction_ttl)
            self.transaction_pages[transaction] = set()
            self.active_transactions.add(transaction)
            return True

    def end(self, transaction):
        """The scheduler has released this request; no further page admissions."""
        with self.lock:
            self.active_transactions.discard(transaction)
            state = self.transactions.get(transaction)
            if state is not None and state[0] != "pending":
                self._retire_transaction(transaction)

    def _retire_transaction(self, transaction):
        self.transactions.pop(transaction, None)
        for digest in self.transaction_pages.pop(transaction, ()):
            page = self.pages.get(digest)
            if page is not None:
                page.transactions.discard(transaction)
        self.closed_transactions[transaction] = None
        while len(self.closed_transactions) > self.max_transactions:
            self.closed_transactions.popitem(last=False)

    def keepalive(self, transaction):
        with self.lock:
            old = self.transactions.get(transaction)
            if old is not None and old[0] == "pending":
                self.transactions[transaction] = (old[0], self.clock() + self.transaction_ttl)

    def reserve(self, digest, parent, depth, transaction):
        """Return a writable new page, or None for a duplicate/admission miss."""
        if len(digest) != 32 or len(parent) != 32 or type(depth) is not int or depth < 1:
            raise ValueError("invalid DRAM page identity")
        with self.lock:
            tx = self.transactions.get(transaction)
            if tx is None or tx[0] == "abort":
                return None
            existing = self.pages.get(digest)
            if existing is not None:
                if (existing.parent, existing.depth) != (parent, depth):
                    raise ValueError("conflicting DRAM page ancestry")
                if existing.failed:
                    return None
                if not existing.ready:
                    existing.transactions.add(transaction)
                    self.transaction_pages[transaction].add(digest)
                    existing.committed |= tx[0] == "commit"
                    self._publish(existing)
                return None
            if not self.free and not self._evict_one():
                return None
            slot = self.free.pop()
            self.generations[slot] += 1
            page = DramPage(
                digest,
                parent,
                depth,
                slot,
                self.generations[slot],
                committed=tx[0] == "commit",
                touched=self.clock(),
                transactions={transaction},
            )
            self.pages[digest] = page
            self.transaction_pages[transaction].add(digest)
            self.stats["staged"] += 1
            return page

    def written(self, page, *, success=True):
        with self.lock:
            if self.pages.get(page.digest) is not page or not page.writing:
                raise ValueError("stale DRAM writer")
            page.writing = False
            if not success:
                page.failed = True
                for transaction in page.transactions:
                    self.transaction_pages.get(transaction, set()).discard(page.digest)
                page.transactions.clear()
                page.committed = False
                if not page.readers and not page.exports:
                    self._remove(page)
            elif page.committed:
                self._publish(page)
            elif not page.transactions and not page.readers and not page.exports:
                self._remove(page)

    def _publish(self, page):
        if not page.writing and not page.failed and page.committed and not page.ready:
            page.ready = True
            self.stats["committed"] += 1
            self._notify("add", page)

    def finish(self, transaction, *, success):
        with self.lock:
            old = self.transactions.get(transaction)
            if old is None or old[0] != "pending":
                return
            self.transactions[transaction] = (
                "commit" if success else "abort",
                self.clock() + self.transaction_ttl,
            )
            self.stats["aborted"] += not success
            for digest in tuple(self.transaction_pages.get(transaction, ())):
                page = self.pages.get(digest)
                if page is None or transaction not in page.transactions:
                    continue
                page.transactions.discard(transaction)
                self.transaction_pages.get(transaction, set()).discard(digest)
                page.committed |= success
                self._publish(page)
                if (
                    not page.committed
                    and not page.transactions
                    and not page.writing
                    and not page.readers
                    and not page.exports
                ):
                    self._remove(page)
            if transaction not in self.active_transactions:
                self._retire_transaction(transaction)

    def acquire(self, digests, *, touch=True):
        """Pin the longest ready prefix. Missing ancestors need not live here."""
        with self.lock:
            result = []
            for digest in digests:
                page = self.pages.get(digest)
                if page is None or not page.ready or page.slot in self.quarantined_slots:
                    break
                page.readers += 1
                if touch:
                    self.touch(page)
                result.append(page)
            return tuple(result)

    def release(self, pages):
        with self.lock:
            for page in pages:
                if self.pages.get(page.digest) is not page or page.readers < 1:
                    raise ValueError("unbalanced DRAM reader")
                page.readers -= 1
                if not (
                    page.readers
                    or page.exports
                    or page.writing
                    or page.transactions
                    or page.committed
                ):
                    self._remove(page)

    def touch(self, page):
        with self.lock:
            self.stats["hits"] += 1
            page.touched = self.clock()
            page.frequency += 1
            self.pages.move_to_end(page.digest)

    def pin_dram(self, digests, peer, ttl=None):
        with self.lock:
            if len(self.leases) >= self.max_leases:
                return None
            ttl = self.lease_ttl if ttl is None else min(ttl, self.lease_ttl)
            if ttl <= 0 or not peer:
                raise ValueError("invalid DRAM lease")
            pages = self.acquire(digests, touch=False)
            if not pages:
                return None
            for page in pages:
                page.readers -= 1
                page.exports += 1
            lease = DramLease(str(uuid4()), peer, pages, self.clock() + ttl)
            self.leases[lease.id] = lease
            return lease

    def unpin_dram(self, lease_id, peer, *, drained=False):
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                return self.closed_leases.get(lease_id) == peer
            if lease.peer != peer:
                raise ValueError("DRAM lease belongs to another reader")
            if not drained:
                lease.expired = True
                return False
            self._release_lease(lease)
            return True

    def _release_lease(self, lease):
        for page in lease.pages:
            page.exports -= 1
        del self.leases[lease.id]
        self.closed_leases[lease.id] = lease.peer
        while len(self.closed_leases) > self.max_leases * 4:
            self.closed_leases.popitem(last=False)

    def fence_complete(self, peers):
        """Call only after native revocation, including retry paths, has drained."""
        with self.lock:
            for lease in list(self.leases.values()):
                if lease.peer in peers:
                    self._release_lease(lease)

    def expire(self):
        with self.lock:
            now = self.clock()
            for transaction, (status, deadline) in list(self.transactions.items()):
                if deadline <= now:
                    if status == "pending":
                        self.finish(transaction, success=False)
                    self.transactions.pop(transaction, None)
                    self.transaction_pages.pop(transaction, None)
                    self.active_transactions.discard(transaction)
            peers = set()
            for lease in self.leases.values():
                lease.expired |= lease.expires <= now
                if lease.expired:
                    peers.add(lease.peer)
            return peers

    def _evict_one(self):
        now = self.clock()
        if self.policy == "lfu" and now - self.last_decay >= 60:
            factor = 0.5 ** min(32, (now - self.last_decay) / 60)
            for page in self.pages.values():
                page.frequency *= factor
            self.last_decay = now
        candidates = (
            p
            for p in self.pages.values()
            if not (p.writing or p.readers or p.exports) and p.slot not in self.quarantined_slots
        )
        victim = (
            next(candidates, None)
            if self.policy == "lru"
            else min(candidates, key=lambda p: (p.frequency, p.touched), default=None)
        )
        if victim is None:
            return False
        self._remove(victim)
        self.stats["evictions"] += 1
        return True

    def _remove(self, page):
        if page.writing or page.readers or page.exports:
            raise ValueError("cannot recycle an owned DRAM page")
        if page.ready:
            self._notify("remove", page)
        del self.pages[page.digest]
        for transaction in page.transactions:
            self.transaction_pages.get(transaction, set()).discard(page.digest)
        self.free.append(page.slot)

    def health(self):
        with self.lock:
            return dict(
                capacity_pages=self.capacity,
                free_pages=len(self.free),
                ready_pages=sum(p.ready for p in self.pages.values()),
                writing_pages=sum(p.writing for p in self.pages.values()),
                readers=sum(p.readers for p in self.pages.values()),
                leases=len(self.leases),
                expired_leases=sum(l.expired for l in self.leases.values()),
                registered_bytes=self.registered_bytes,
                quarantined_pages=len(self.quarantined_slots),
                **self.stats,
            )
