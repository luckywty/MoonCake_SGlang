"""A bounded, instance-scoped Router mirror. Its answers are routing hints only."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .prefix_events import SYNC_EVENTS, SYNC_PAGES
from .prefix_hash import PREFIX_INDEX_VERSION, PreparedPrefix, PrefixHashSpec


@dataclass(eq=False)
class PrefixMirror:
    instance_id: str
    spec: PrefixHashSpec
    cursor: int = -1
    ready: bool = False
    updated_at: float = 0.0
    snapshot_id: str | None = None
    snapshot_seq: int | None = None
    offset: int = 0
    pages: OrderedDict = field(default_factory=OrderedDict)
    children: dict = field(default_factory=dict)


class PrefixIndex:
    def __init__(self, max_entries=262144, max_worker_entries=65536, max_age=6.0):
        if max_entries < 1 or max_worker_entries < 1 or max_age <= 0:
            raise ValueError("prefix index bounds must be positive")
        self.max_entries, self.max_worker_entries, self.max_age = (
            max_entries, max_worker_entries, max_age
        )
        self.states: dict[str, PrefixMirror] = {}
        self.lru = OrderedDict()
        self.lock = threading.RLock()
        self.capacity_dropped_pages = 0
        self.resets = 0

    def forget(self, worker_id):
        with self.lock:
            state = self.states.pop(worker_id, None)
            if state is not None:
                for digest in state.pages:
                    self.lru.pop((worker_id, digest), None)
                self.resets += 1

    def observe(self, worker_id, health):
        with self.lock:
            info = health.get("prefix_index")
            try:
                if (health.get("role") != "prefill" or not isinstance(info, dict)
                        or info.get("enabled") is not True
                        or info.get("version") != PREFIX_INDEX_VERSION
                        or info.get("instance_id") != health.get("instance_id")):
                    raise ValueError("no compatible P prefix publisher")
                instance = info["instance_id"]
                if not isinstance(instance, str) or not 1 <= len(instance) <= 64:
                    raise ValueError("invalid prefix instance identity")
                spec = PrefixHashSpec(info["manifest"])
                if spec.domain_id != info["domain_id"]:
                    raise ValueError("prefix domain mismatch")
                # The hash manifest must describe the actual healthy KV layout.
                expected = PrefixHashSpec.from_layout(
                    health["layout"], health.get("cache_namespace", "default")
                )
                if spec != expected:
                    raise ValueError("prefix manifest differs from Worker layout")
            except (KeyError, TypeError, ValueError):
                self.forget(worker_id)
                return
            old = self.states.get(worker_id)
            if old is None or old.instance_id != instance or old.spec != spec:
                self.forget(worker_id)
                self.states[worker_id] = PrefixMirror(instance, spec)

    def _clear(self, worker_id, state):
        for digest in state.pages:
            self.lru.pop((worker_id, digest), None)
        state.pages.clear()
        state.children.clear()
        state.cursor, state.ready, state.updated_at = -1, False, 0.0
        state.snapshot_id = state.snapshot_seq = None
        state.offset = 0
        self.resets += 1

    def request(self, worker_id):
        with self.lock:
            state = self.states.get(worker_id)
            if state is None:
                return None
            message = {"op": "prefix_sync", "instance_id": state.instance_id,
                       "domain_id": state.spec.domain_id, "after_seq": state.cursor}
            if state.cursor == -1:
                message.update(snapshot_id=state.snapshot_id, offset=state.offset)
            return state, message

    @staticmethod
    def _record(value):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("invalid prefix page record")
        digest, parent, depth = value
        if (not isinstance(digest, str) or not isinstance(parent, str)
                or len(digest) != 64 or len(parent) != 64
                or type(depth) is not int or not 1 <= depth < 2**31):
            raise ValueError("invalid prefix page identity")
        raw, ancestor = bytes.fromhex(digest), bytes.fromhex(parent)
        if raw.hex() != digest or ancestor.hex() != parent:
            raise ValueError("noncanonical prefix digest")
        return raw, ancestor, depth

    def _drop_subtree(self, worker_id, state, digest):
        # Never re-anchor orphan suffixes; other Workers' owners remain intact.
        pending = [digest]
        removed = 0
        while pending:
            key = pending.pop()
            pending.extend(state.children.pop(key, ()))
            record = state.pages.pop(key, None)
            if record is None:
                continue
            removed += 1
            parent, _ = record
            siblings = state.children.get(parent)
            if siblings is not None:
                siblings.discard(key)
                if not siblings:
                    state.children.pop(parent, None)
            self.lru.pop((worker_id, key), None)
        return removed

    def _add(self, worker_id, state, value):
        digest, parent, depth = self._record(value)
        if digest in state.pages:
            if state.pages[digest] != (parent, depth):
                raise ValueError("conflicting prefix ancestry")
            return
        if depth == 1:
            if parent != state.spec.root:
                raise ValueError("prefix page has the wrong root seed")
        elif state.pages.get(parent, (None, None))[1] != depth - 1:
            return  # Missing parents (including local LRU drops) never imply hits.
        while len(state.pages) >= self.max_worker_entries:
            self.capacity_dropped_pages += self._drop_subtree(
                worker_id, state, next(iter(state.pages))
            )
        while len(self.lru) >= self.max_entries:
            other, key = next(iter(self.lru))
            self.capacity_dropped_pages += self._drop_subtree(other, self.states[other], key)
        if depth > 1 and parent not in state.pages:
            return
        state.pages[digest] = (parent, depth)
        state.children.setdefault(parent, set()).add(digest)
        self.lru[(worker_id, digest)] = None

    def apply(self, worker_id, expected_state, reply):
        with self.lock:
            state = self.states.get(worker_id)
            if state is not expected_state:
                return False  # Late response from an old connection/instance.
            try:
                if (reply.get("instance_id") != state.instance_id
                        or reply.get("domain_id") != state.spec.domain_id):
                    raise ValueError("prefix response belongs to another instance")
                status = reply.get("status")
                if status == "prefix_reset":
                    self._clear(worker_id, state)
                    return False
                seq = reply.get("seq")
                if type(seq) is not int or seq < 0:
                    raise ValueError("invalid prefix sequence")
                if status == "prefix_snapshot":
                    pages = reply.get("pages")
                    snapshot_id = reply.get("snapshot_id")
                    if (state.cursor != -1 or reply.get("offset") != state.offset
                            or not isinstance(pages, list) or len(pages) > SYNC_PAGES
                            or not isinstance(snapshot_id, str) or len(snapshot_id) > 64
                            or type(reply.get("complete")) is not bool
                            or reply.get("next_offset") != state.offset + len(pages)):
                        raise ValueError("invalid prefix snapshot page")
                    if state.snapshot_id is not None and (
                        state.snapshot_id != snapshot_id or state.snapshot_seq != seq
                    ):
                        raise ValueError("prefix snapshot changed during pagination")
                    state.snapshot_id, state.snapshot_seq = snapshot_id, seq
                    for value in pages:
                        self._add(worker_id, state, value)
                    state.offset += len(pages)
                    if reply["complete"]:
                        state.cursor = seq
                        state.snapshot_id = state.snapshot_seq = None
                        state.offset = 0
                    return True  # Catch up deltas before allowing a snapshot to route.
                if status != "prefix_events" or state.cursor < 0:
                    raise ValueError("unexpected prefix synchronization state")
                events, head = reply.get("events"), reply.get("head_seq")
                if (not isinstance(events, list) or len(events) > SYNC_EVENTS
                        or type(head) is not int or head < seq
                        or type(reply.get("more")) is not bool or reply["more"] != (seq < head)):
                    raise ValueError("invalid prefix delta batch")
                cursor = state.cursor
                for event in events:
                    if type(event.get("seq")) is not int or event["seq"] != cursor + 1:
                        raise ValueError("prefix sequence gap")
                    if event.get("op") == "add":
                        self._add(worker_id, state, event.get("page"))
                    elif event.get("op") == "remove":
                        digest, _, _ = self._record(event.get("page"))
                        self._drop_subtree(worker_id, state, digest)
                    elif event.get("op") == "reset":
                        self._clear(worker_id, state)
                        return False
                    else:
                        raise ValueError("unknown prefix event")
                    cursor += 1
                if seq != cursor:
                    raise ValueError("prefix cursor does not match its events")
                state.cursor = cursor
                state.ready = not reply["more"]
                state.updated_at = time.monotonic()
                return reply["more"]
            except (AttributeError, KeyError, TypeError, ValueError):
                self._clear(worker_id, state)
                return False

    def failed(self, worker_id, expected_state):
        with self.lock:
            if self.states.get(worker_id) is expected_state:
                self._clear(worker_id, expected_state)

    def spec_for(self, nodes):
        with self.lock:
            for node in nodes:
                state = self.states.get(node.worker_id)
                if state is not None and node.healthy:
                    return state.spec
        return None

    def hits(self, nodes, prepared: PreparedPrefix):
        result = {}
        now = time.monotonic()
        with self.lock:
            for node in nodes:
                state = self.states.get(node.worker_id)
                if (state is None or not node.healthy or not state.ready
                        or now - state.updated_at > self.max_age
                        or node.health.get("instance_id") != state.instance_id
                        or prepared.domain_id != state.spec.domain_id
                        or len(prepared.page_hashes) % 32
                        or len(prepared.page_hashes) // 32
                        != len(prepared.token_ids) // state.spec.page_size):
                    continue
                parent, matched = state.spec.root, 0
                for depth, digest in enumerate(prepared.digests(), start=1):
                    if state.pages.get(digest) != (parent, depth):
                        break
                    matched += 1
                    parent = digest
                    state.pages.move_to_end(digest)
                    self.lru.move_to_end((node.worker_id, digest))
                result[node.worker_id] = matched * state.spec.page_size
        return result
