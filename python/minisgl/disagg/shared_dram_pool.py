"""NUMA-segmented historical KV data plane, separate from the P->D relay.

One slot holds one complete all-layer token page. The directory owns slot
headers; Workers only access payloads while holding write grants/read leases.
"""

from __future__ import annotations

import json
import mmap
import os
import re
from dataclasses import asdict, dataclass

from .numa import host_id
from .shm_relay_pool import SharedRelayPool, align

FREE, STAGED, READY, QUARANTINED = range(4)
SHARED_DRAM_VERSION = 1


@dataclass(frozen=True)
class SharedDramManifest:
    name: str
    host_id: str
    page_bytes: int
    page_size: int
    cache_namespace: str
    node_slots: dict[int, int]
    worker_numa: dict[str, int]
    directory_socket: str
    secret: str
    version: int = SHARED_DRAM_VERSION

    @property
    def group_bytes(self):
        # SharedRelayPool's resource/mapping primitives accept this geometry.
        return self.page_bytes

    @property
    def stride(self):
        return 256 + align(self.page_bytes)

    @property
    def registered_bytes(self):
        return sum(self.segment_bytes(n) for n in self.node_slots)

    @property
    def capacity(self):
        return sum(self.node_slots.values())

    def segment_name(self, node):
        return f"/{self.name}_numa_{node}"

    def segment_bytes(self, node):
        return align(self.node_slots[node] * self.stride, mmap.PAGESIZE)

    def validate(self, *, budget=None, local_host=None):
        if self.version != SHARED_DRAM_VERSION or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,96}", self.name
        ):
            raise ValueError("invalid shared history pool name/version")
        if not self.host_id or (local_host is not None and self.host_id != local_host):
            raise ValueError("shared_shm requires the same host boot identity")
        if any(type(v) is not int or v < 1 for v in (self.page_bytes, self.page_size)):
            raise ValueError("shared history page dimensions must be positive integers")
        if not self.node_slots or any(
            type(n) is not int or n < 0 or type(s) is not int or s < 1
            for n, s in self.node_slots.items()
        ):
            raise ValueError("each shared history NUMA segment needs at least one slot")
        if not self.worker_numa or any(
            not w or n not in self.node_slots for w, n in self.worker_numa.items()
        ):
            raise ValueError("shared history worker NUMA mapping is invalid")
        if (
            not os.path.isabs(self.directory_socket)
            or len(os.fsencode(self.directory_socket)) > 103
            or not re.fullmatch(r"[a-f0-9]{32,64}", self.secret)
            or not 1 <= len(self.cache_namespace.encode()) <= 1024
        ):
            raise ValueError("invalid shared history directory identity")
        if budget is not None and (type(budget) is not int or budget < self.registered_bytes):
            raise ValueError(
                f"shared history requires {self.registered_bytes} registered bytes; budget={budget}"
            )
        return self

    def check_space(self, *, additional_bytes=0):
        info = os.statvfs("/dev/shm")
        required = self.registered_bytes + additional_bytes
        if required > info.f_bavail * info.f_frsize:
            raise ValueError(f"/dev/shm requires {required} free bytes for history and relay pools")

    def to_json(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        data["node_slots"] = {int(k): v for k, v in data["node_slots"].items()}
        return cls(**data).validate()


def build_shared_manifest(
    *,
    name,
    page_bytes,
    page_size,
    capacity_bytes,
    numa_nodes,
    worker_numa,
    directory_socket,
    secret,
    cache_namespace="default",
    node_bytes=None,
):
    nodes = sorted(set(numa_nodes))
    if not nodes or type(capacity_bytes) is not int or capacity_bytes < 1:
        raise ValueError("shared_shm needs a positive host-wide --pd-dram-cache-bytes budget")
    budgets = (
        {n: capacity_bytes // len(nodes) for n in nodes}
        if node_bytes is None
        else {int(n): v for n, v in node_bytes.items()}
    )
    if (
        set(budgets) != set(nodes)
        or any(type(b) is not int or b < 1 for b in budgets.values())
        or sum(budgets.values()) > capacity_bytes
    ):
        raise ValueError(
            "NUMA history budgets must match the node inventory and fit the total budget"
        )
    stride = 256 + align(page_bytes)
    slots = {n: (b // mmap.PAGESIZE * mmap.PAGESIZE) // stride for n, b in budgets.items()}
    result = SharedDramManifest(
        name,
        host_id(),
        page_bytes,
        page_size,
        cache_namespace,
        slots,
        worker_numa,
        directory_socket,
        secret,
    )
    return result.validate(budget=capacity_bytes)


class SharedDramPool(SharedRelayPool):
    """Reuse POSIX SHM, strict mbind/first-touch, CUDA registration and cleanup.

    Inheriting resource primitives does not reuse the relay's segments, allocator,
    tickets or ownership. The manifest accepts a dynamic NUMA-node list.
    """

    def view(self, page):
        generation, state = self.header(page.numa_id, page.slot_id)
        expected = READY if page.lease_id else STAGED
        if (generation, state) != (page.generation, expected):
            raise ValueError("shared history slot generation/state changed before DMA")
        return self.tensor(page.numa_id, page.slot_id, self.manifest.page_bytes)


@dataclass(frozen=True)
class SharedPage:
    page_hash: str
    numa_id: int
    slot_id: int
    generation: int
    lease_id: str = ""

    def to_object(self):
        return asdict(self)

    @classmethod
    def from_object(cls, data):
        if not isinstance(data, dict) or set(data) != {
            "page_hash",
            "numa_id",
            "slot_id",
            "generation",
            "lease_id",
        }:
            raise ValueError("invalid shared history page descriptor")
        result = cls(**data)
        if (
            not isinstance(result.page_hash, str)
            or not re.fullmatch(r"[a-f0-9]{64}", result.page_hash)
            or any(type(v) is not int or v < 0 for v in (result.numa_id, result.slot_id))
            or type(result.generation) is not int
            or result.generation < 1
            or not isinstance(result.lease_id, str)
        ):
            raise ValueError("invalid shared history page identity")
        return result
