"""Instance-scoped HBM/DRAM directory. Suffix pages need no resident ancestor."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from uuid import uuid4

from .dram_protocol import MAX_DRAM_PULL_PAGES, TIER_INDEX_VERSION, RemotePullPlan, digest_bytes
from .prefix_hash import PrefixHashSpec


class TierPublisher:
    def __init__(self, spec, instance_id, *, max_entries=65536, event_limit=32768):
        self.spec, self.instance_id = spec, instance_id
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.events = deque(maxlen=event_limit)
        self.snapshots = OrderedDict()
        self.seq = 0
        self.lock = threading.RLock()

    def update(self, op, tier, digest, parent, depth, generation=0):
        with self.lock:
            key = (tier, digest)
            record = [tier, digest.hex(), parent.hex(), depth, generation]
            if op == "add":
                if self.entries.get(key) == record:
                    return
                if key not in self.entries and len(self.entries) >= self.max_entries:
                    _, old = self.entries.popitem(last=False)
                    self._emit("remove", old)
                self.entries[key] = record
            elif op == "remove":
                if self.entries.pop(key, None) is None:
                    return
            else:
                raise ValueError("invalid tier directory operation")
            self._emit(op, record)

    def _emit(self, op, record):
        self.seq += 1
        self.events.append({"seq": self.seq, "op": op, "page": record})

    def legacy_event(self, op, record):
        if op == "reset":
            with self.lock:
                for key, value in list(self.entries.items()):
                    if key[0] == "HBM":
                        self.update(
                            "remove",
                            "HBM",
                            bytes.fromhex(value[1]),
                            bytes.fromhex(value[2]),
                            value[3],
                        )
        elif record is not None:
            self.update(op, "HBM", record.digest, record.parent, record.depth)

    def dram_event(self, op, page):
        self.update(op, "DRAM", page.digest, page.parent, page.depth, page.generation)

    def health(self):
        with self.lock:
            return {
                "version": TIER_INDEX_VERSION,
                "enabled": True,
                "instance_id": self.instance_id,
                "manifest": self.spec.manifest,
                "domain_id": self.spec.domain_id,
                "seq": self.seq,
                "pages": len(self.entries),
            }

    def sync(self, message):
        with self.lock:
            base = {"instance_id": self.instance_id, "domain_id": self.spec.domain_id}
            after = message.get("after_seq")
            if (
                message.get("instance_id") != self.instance_id
                or message.get("domain_id") != self.spec.domain_id
            ):
                return dict(base, status="prefix_reset")
            if type(after) is not int or after < -1:
                raise ValueError("invalid tier cursor")
            now = time.monotonic()
            for key, (_, expires, _) in list(self.snapshots.items()):
                if expires <= now:
                    del self.snapshots[key]
            if after == -1:
                sid, offset = message.get("snapshot_id"), message.get("offset", 0)
                if type(offset) is not int or offset < 0:
                    raise ValueError("invalid tier snapshot offset")
                if sid is None:
                    if offset:
                        raise ValueError("initial tier snapshot offset must be zero")
                    sid = str(uuid4())
                    self.snapshots[sid] = (self.seq, now + 30, tuple(self.entries.values()))
                    while len(self.snapshots) > 2:
                        self.snapshots.popitem(last=False)
                snapshot = self.snapshots.get(sid)
                if snapshot is None:
                    return dict(base, status="prefix_reset")
                seq, _, records = snapshot
                if offset > len(records):
                    raise ValueError("tier offset exceeds snapshot")
                pages = list(records[offset : offset + 512])
                end = offset + len(pages)
                return dict(
                    base,
                    status="prefix_snapshot",
                    seq=seq,
                    snapshot_id=sid,
                    offset=offset,
                    next_offset=end,
                    complete=end == len(records),
                    pages=pages,
                )
            if after > self.seq or (self.events and after < self.events[0]["seq"] - 1):
                return dict(base, status="prefix_reset")
            events = [e for e in self.events if e["seq"] > after][:512]
            end = events[-1]["seq"] if events else after
            return dict(
                base,
                status="prefix_events",
                seq=end,
                head_seq=self.seq,
                more=end < self.seq,
                events=events,
            )


@dataclass(eq=False)
class TierMirror:
    instance_id: str
    spec: PrefixHashSpec
    cursor: int = -1
    ready: bool = False
    updated: float = 0
    snapshot_id: str | None = None
    snapshot_seq: int | None = None
    offset: int = 0
    entries: OrderedDict = field(default_factory=OrderedDict)


@dataclass(frozen=True)
class TierHit:
    hbm_tokens: int = 0
    dram_tokens: int = 0
    remote_tokens: int = 0
    remote_plan: RemotePullPlan | None = None
    restore_cost_ms: float = 0

    @property
    def total(self):
        return self.hbm_tokens + self.dram_tokens + self.remote_tokens


class TierIndex:
    def __init__(self, max_entries=262144, max_worker_entries=65536, max_age=6.0):
        self.states = {}
        self.lru = OrderedDict()
        self.max_entries, self.max_worker_entries, self.max_age = (
            max_entries,
            max_worker_entries,
            max_age,
        )
        self.lock = threading.RLock()

    def forget(self, worker_id):
        with self.lock:
            state = self.states.pop(worker_id, None)
            if state:
                for key in state.entries:
                    self.lru.pop((worker_id, key), None)

    def observe(self, worker_id, health):
        with self.lock:
            info = health.get("tier_index")
            try:
                if (
                    not isinstance(info, dict)
                    or info.get("version") != TIER_INDEX_VERSION
                    or info.get("enabled") is not True
                    or info.get("instance_id") != health.get("instance_id")
                ):
                    raise ValueError("no tier directory")
                spec = PrefixHashSpec(info["manifest"])
                if (
                    spec
                    != PrefixHashSpec.from_layout(
                        health["layout"], health.get("cache_namespace", "default")
                    )
                    or spec.domain_id != info["domain_id"]
                ):
                    raise ValueError("tier identity mismatch")
            except (KeyError, TypeError, ValueError):
                self.forget(worker_id)
                return
            old = self.states.get(worker_id)
            if old is None or old.instance_id != info["instance_id"] or old.spec != spec:
                self.forget(worker_id)
                self.states[worker_id] = TierMirror(info["instance_id"], spec)

    def request(self, worker_id):
        with self.lock:
            state = self.states.get(worker_id)
            if state is None:
                return None
            return state, {
                "op": "tier_sync",
                "instance_id": state.instance_id,
                "domain_id": state.spec.domain_id,
                "after_seq": state.cursor,
                "snapshot_id": state.snapshot_id,
                "offset": state.offset,
            }

    def failed(self, worker_id, expected):
        with self.lock:
            if self.states.get(worker_id) is expected:
                spec, instance = expected.spec, expected.instance_id
                self.forget(worker_id)
                self.states[worker_id] = TierMirror(instance, spec)

    def spec_for(self, nodes):
        with self.lock:
            for node in nodes:
                state = self.states.get(node.worker_id)
                if state is not None and node.healthy:
                    return state.spec
        return None

    def _add(self, worker_id, state, value):
        if (
            not isinstance(value, list)
            or len(value) != 5
            or value[0] not in ("HBM", "DRAM")
            or type(value[3]) is not int
            or value[3] < 1
            or type(value[4]) is not int
            or value[4] < 0
        ):
            raise ValueError("invalid tier page")
        tier, digest, parent, depth, generation = value
        digest, parent = digest_bytes(digest), digest_bytes(parent)
        if depth == 1 and parent != state.spec.root:
            raise ValueError("invalid tier root")
        key = tier, digest
        while key not in state.entries and len(state.entries) >= self.max_worker_entries:
            old, _ = state.entries.popitem(last=False)
            self.lru.pop((worker_id, old), None)
        while (worker_id, key) not in self.lru and len(self.lru) >= self.max_entries:
            (other, old), _ = self.lru.popitem(last=False)
            self.states[other].entries.pop(old, None)
        state.entries[key] = (parent, depth, generation)
        self.lru[(worker_id, key)] = None

    def apply(self, worker_id, expected, reply):
        with self.lock:
            state = self.states.get(worker_id)
            if state is not expected:
                return False
            try:
                if (
                    reply["instance_id"] != state.instance_id
                    or reply["domain_id"] != state.spec.domain_id
                    or reply["status"] == "prefix_reset"
                ):
                    raise ValueError("tier reset")
                seq = reply["seq"]
                if type(seq) is not int or seq < 0:
                    raise ValueError("invalid tier sequence")
                if reply["status"] == "prefix_snapshot":
                    pages = reply["pages"]
                    if (
                        state.cursor != -1
                        or reply["offset"] != state.offset
                        or not isinstance(pages, list)
                        or len(pages) > 512
                        or type(reply["complete"]) is not bool
                        or reply["next_offset"] != state.offset + len(pages)
                        or not isinstance(reply["snapshot_id"], str)
                        or (
                            state.snapshot_id is not None
                            and (
                                state.snapshot_id != reply["snapshot_id"]
                                or state.snapshot_seq != seq
                            )
                        )
                    ):
                        raise ValueError("invalid tier snapshot")
                    state.snapshot_id, state.snapshot_seq = reply["snapshot_id"], seq
                    for page in pages:
                        self._add(worker_id, state, page)
                    state.offset = reply["next_offset"]
                    if reply["complete"]:
                        state.cursor = seq
                        state.snapshot_id = state.snapshot_seq = None
                        state.offset = 0
                    return True
                if reply["status"] != "prefix_events" or state.cursor < 0:
                    raise ValueError("unexpected tier reply")
                events = reply["events"]
                if (
                    not isinstance(events, list)
                    or len(events) > 512
                    or type(reply["more"]) is not bool
                    or type(reply["head_seq"]) is not int
                    or reply["head_seq"] < seq
                    or reply["more"] != (seq < reply["head_seq"])
                ):
                    raise ValueError("invalid tier delta")
                for event in events:
                    if event["seq"] != state.cursor + 1:
                        raise ValueError("tier sequence gap")
                    if event["op"] == "add":
                        self._add(worker_id, state, event["page"])
                    elif event["op"] == "remove":
                        value = event["page"]
                        key = value[0], digest_bytes(value[1])
                        state.entries.pop(key, None)
                        self.lru.pop((worker_id, key), None)
                    else:
                        raise ValueError("unknown tier delta")
                    state.cursor += 1
                if seq != state.cursor:
                    raise ValueError("tier cursor mismatch")
                state.ready, state.updated = not reply["more"], time.monotonic()
                return reply["more"]
            except (KeyError, TypeError, ValueError, IndexError):
                self.failed(worker_id, state)
                return False

    def hits(
        self,
        prefill_nodes,
        decode_nodes,
        prepared,
        *,
        min_remote_tokens=1024,
        h2d_gbps=12.0,
        rdma_gbps=10.0,
        prefill_ms_per_token=0.01,
        max_remote_bytes=64 * 1024**2,
    ):
        result = {}
        with self.lock:
            now = time.monotonic()

            def valid(node):
                s = self.states.get(node.worker_id)
                return (
                    s
                    if s is not None
                    and node.healthy
                    and s.ready
                    and now - s.updated <= self.max_age
                    and s.spec.domain_id == prepared.domain_id
                    and s.instance_id == node.health.get("instance_id")
                    else None
                )

            for p in prefill_nodes:
                state = valid(p)
                if state is None:
                    continue
                page = state.spec.page_size
                hashes = tuple(prepared.digests())[: (len(prepared.token_ids) - 1) // page]
                parent, i = state.spec.root, 0

                def advance(s, tier, start, parent):
                    j = start
                    while j < len(hashes):
                        record = s.entries.get((tier, hashes[j]))
                        if record is None or record[:2] != (parent, j + 1):
                            break
                        parent, j = hashes[j], j + 1
                    return j, parent

                h, parent = advance(state, "HBM", i, parent)
                c, parent = advance(state, "DRAM", h, parent)
                best, remote, plan = 0, 0, None
                layout = p.health["layout"]
                bpt = (
                    2
                    * layout["layers"]
                    * layout["kv_heads"]
                    * layout["head_dim"]
                    * {"float16": 2, "bfloat16": 2, "float32": 4}[layout["dtype"]]
                )
                for d in (
                    decode_nodes
                    if p.health.get("dram_cache", {}).get("remote_ready", False)
                    else ()
                ):
                    source = valid(d)
                    if source is None or not d.health.get("dram_cache", {}).get(
                        "remote_ready", False
                    ):
                        continue
                    end, _ = advance(source, "DRAM", c, parent)
                    end = min(end, c + max_remote_bytes // (page * bpt), c + MAX_DRAM_PULL_PAGES)
                    tokens = (end - c) * page
                    cost = tokens * bpt / 1e6 * (1 / h2d_gbps + 1 / rdma_gbps)
                    if tokens < min_remote_tokens or cost >= tokens * prefill_ms_per_token:
                        continue
                    benefit = tokens * prefill_ms_per_token - cost
                    if benefit > best:
                        best, remote = benefit, tokens
                        plan = RemotePullPlan(
                            d.worker_id,
                            source.instance_id,
                            d.endpoint,
                            prepared.domain_id,
                            c,
                            hashes[c:end],
                        )
                local = (c - h) * page
                cost = (local + remote) * bpt / 1e6 / h2d_gbps + remote * bpt / 1e6 / rdma_gbps
                result[p.worker_id] = TierHit(h * page, local, remote, plan, cost)
        return result
