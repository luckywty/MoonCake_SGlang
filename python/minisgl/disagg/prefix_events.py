"""Bounded metadata journal owned by one P cache; contains no GPU references."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import islice
from uuid import uuid4

from .prefix_hash import PREFIX_INDEX_VERSION, PrefixHashSpec, hash_page

SYNC_PAGES = 512
SYNC_EVENTS = 512
SNAPSHOT_TTL = 30.0


@dataclass(frozen=True)
class PageRecord:
    digest: bytes
    parent: bytes
    depth: int
    node_id: int

    def wire(self):
        return [self.digest.hex(), self.parent.hex(), self.depth]


class PrefixPublisher:
    def __init__(self, spec: PrefixHashSpec, instance_id: str, max_pages: int, event_limit=32768):
        if max_pages < 1 or event_limit < 1:
            raise ValueError("prefix metadata limits must be positive")
        self.spec, self.instance_id = spec, instance_id
        self.max_pages, self.event_limit = max_pages, event_limit
        self.records: dict[bytes, PageRecord] = {}
        self.by_node: dict[int, set[bytes]] = {}
        self.children: dict[bytes, set[bytes]] = {}
        self.leaves: OrderedDict[bytes, None] = OrderedDict()
        self.events = deque(maxlen=event_limit)
        self.snapshots = OrderedDict()
        self.seq = 0
        self.enabled = True
        self.error = None
        self.lock = threading.RLock()

    def health(self):
        with self.lock:
            return {
                "version": PREFIX_INDEX_VERSION, "enabled": self.enabled,
                "instance_id": self.instance_id, "domain_id": self.spec.domain_id,
                "manifest": self.spec.manifest, "seq": self.seq,
                "pages": len(self.records), "max_pages": self.max_pages, "reason": self.error,
                "event_count": len(self.events),
                "oldest_seq": self.events[0]["seq"] if self.events else self.seq + 1,
            }

    def _emit(self, op, record=None):
        self.seq += 1
        self.events.append({"seq": self.seq, "op": op, "page": record.wire() if record else None})

    def disable(self, reason):
        with self.lock:
            self.enabled, self.error = False, str(reason)
            self.records.clear()
            self.by_node.clear()
            self.children.clear()
            self.leaves.clear()
            self.snapshots.clear()
            self.events.clear()
            self._emit("reset")

    def _remove_leaf(self, digest):
        record = self.records.pop(digest)
        self.leaves.pop(digest, None)
        self.children.pop(digest, None)
        siblings = self.children.get(record.parent)
        if siblings is not None:
            siblings.discard(digest)
            if not siblings:
                self.children.pop(record.parent, None)
                if record.parent in self.records:
                    self.leaves[record.parent] = None
        owned = self.by_node[record.node_id]
        owned.remove(digest)
        if not owned:
            del self.by_node[record.node_id]
        self._emit("remove", record)

    def _add(self, record):
        if record.digest in self.records:
            if record.digest in self.leaves:
                self.leaves.move_to_end(record.digest)
            return
        if record.parent != self.spec.root and record.parent not in self.records:
            return  # Never export an orphan or infer an unseen prefix.
        if len(self.records) >= self.max_pages:
            victim = next((key for key in self.leaves if key != record.parent), None)
            if victim is None:
                return  # A chain longer than the directory cap is under-reported.
            self._remove_leaf(victim)
        self.records[record.digest] = record
        self.by_node.setdefault(record.node_id, set()).add(record.digest)
        self.children.setdefault(record.parent, set()).add(record.digest)
        self.leaves.pop(record.parent, None)
        self.leaves[record.digest] = None
        self._emit("add", record)

    def publish(self, handle):
        """Called only after CacheManager commits an already-computed prefix."""
        nodes, node = [], handle.node
        while not node.is_root():
            nodes.append(node)
            node = node.parent
        parent, depth, page = self.spec.root, 0, self.spec.page_size
        with self.lock:
            if not self.enabled:
                return
            for node in reversed(nodes):
                count = node.length // page
                if (node.prefix_end_digest is not None
                        and len(self.by_node.get(node.uuid, ())) == count):
                    parent = node.prefix_end_digest
                    depth += count
                    continue
                # Tokens are CPU metadata; do not copy KV or physical indices from HBM.
                tokens = node._key.tolist()
                for offset in range(0, count * page, page):
                    digest = hash_page(parent, tokens[offset:offset + page], page)
                    depth += 1
                    self._add(PageRecord(digest, parent, depth, node.uuid))
                    parent = digest
                node.prefix_end_digest = parent

    def split(self, child, new_parent, boundary_depth):
        """A structural split changes bookkeeping, not page residency or seq."""
        with self.lock:
            if not self.enabled:
                return
            keys = tuple(self.by_node.get(child.uuid, ()))
            for key in keys:
                record = self.records[key]
                if record.depth <= boundary_depth:
                    self.by_node[child.uuid].remove(key)
                    self.by_node.setdefault(new_parent.uuid, set()).add(key)
                    self.records[key] = PageRecord(key, record.parent, record.depth, new_parent.uuid)
            if child.uuid in self.by_node and not self.by_node[child.uuid]:
                del self.by_node[child.uuid]
            if child.prefix_end_digest is not None:
                ancestor = new_parent.parent
                parent = self.spec.root if ancestor.is_root() else ancestor.prefix_end_digest
                if parent is None:
                    raise RuntimeError("hashed node has an unhashed ancestor")
                tokens = new_parent._key.tolist()
                page = self.spec.page_size
                for offset in range(0, len(tokens), page):
                    parent = hash_page(parent, tokens[offset:offset + page], page)
                new_parent.prefix_end_digest = parent

    def remove_node(self, node_id):
        with self.lock:
            # Eviction removes a real radix leaf; remove its pages tail first.
            keys = sorted(self.by_node.get(node_id, ()), key=lambda k: self.records[k].depth, reverse=True)
            for key in keys:
                if self.children.get(key):
                    raise RuntimeError("evicted radix leaf still has exported descendants")
                self._remove_leaf(key)

    def sync(self, message):
        """RPC threads read frozen CPU records; GPU scheduling never serializes JSON."""
        expected = message.get("instance_id")
        domain = message.get("domain_id")
        after = message.get("after_seq")
        if type(after) is not int or after < -1:
            raise ValueError("invalid prefix cursor")
        with self.lock:
            base = {"instance_id": self.instance_id, "domain_id": self.spec.domain_id}
            if expected != self.instance_id or domain != self.spec.domain_id or not self.enabled:
                return {**base, "status": "prefix_reset"}
            now = time.monotonic()
            for key, (_, expires, _) in list(self.snapshots.items()):
                if expires <= now:
                    del self.snapshots[key]
            if after == -1:
                snapshot_id = message.get("snapshot_id")
                offset = message.get("offset", 0)
                if type(offset) is not int or offset < 0:
                    raise ValueError("invalid prefix snapshot offset")
                if snapshot_id is None:
                    if offset:
                        raise ValueError("initial snapshot offset must be zero")
                    oldest = self.events[0]["seq"] if self.events else self.seq + 1
                    snapshot_id = next((key for key, (seq, _, _) in reversed(self.snapshots.items())
                                        if seq >= oldest - 1), None)
                    if snapshot_id is None:
                        snapshot_id = str(uuid4())
                        # Frozen CPU records can be shared by independent Routers.
                        # Insertion order remains parent-before-child; no GPU refs.
                        records = tuple(self.records.values())
                        self.snapshots[snapshot_id] = (self.seq, now + SNAPSHOT_TTL, records)
                        while len(self.snapshots) > 2:
                            self.snapshots.popitem(last=False)
                snapshot = self.snapshots.get(snapshot_id)
                if snapshot is None:
                    return {**base, "status": "prefix_reset"}
                seq, _, records = snapshot
                if offset > len(records):
                    raise ValueError("snapshot offset exceeds record count")
                chunk = records[offset:offset + SYNC_PAGES]
                end = offset + len(chunk)
                complete = end == len(records)
                return {**base, "status": "prefix_snapshot", "seq": seq,
                        "snapshot_id": snapshot_id, "offset": offset, "next_offset": end,
                        "complete": complete, "pages": [r.wire() for r in chunk]}
            if after > self.seq or (self.events and after < self.events[0]["seq"] - 1):
                return {**base, "status": "prefix_reset"}
            start = max(0, after + 1 - self.events[0]["seq"]) if self.events else 0
            events = list(islice(self.events, start, start + SYNC_EVENTS))
            end = events[-1]["seq"] if events else after
            return {**base, "status": "prefix_events", "seq": end,
                    "head_seq": self.seq, "events": events, "more": end < self.seq}
