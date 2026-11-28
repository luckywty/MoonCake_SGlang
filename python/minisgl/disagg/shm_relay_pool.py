"""Launcher-owned POSIX SHM. One P writer per static partition; no shared allocator.

Only the P sender pump changes slot state. TCP READY notifications publish bytes;
H2D completion receipts permit reuse. Neither a timeout nor an expired lease is a
completion receipt. This pool is unrelated to the persistent DRAM KV cache.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import multiprocessing
import os
import re
import struct
from dataclasses import asdict, dataclass

from .numa import bind_cpu, bind_memory, host_id

ALIGNMENT = 256
HEADER = struct.Struct("<QQ")  # generation, state (writer is always the owning P)
FREE, WRITING, READY, QUARANTINED = range(4)


def align(n, alignment=ALIGNMENT):
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Partition:
    worker_id: str
    numa_id: int
    local_slots: tuple[int, ...]
    overflow_numa: int | None = None
    overflow_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class RelayManifest:
    name: str
    host_id: str
    group_bytes: int
    max_tokens: int
    bytes_per_token: int
    layers: int
    layers_per_chunk: int
    node_slots: dict[int, int]
    partitions: tuple[Partition, ...]

    @property
    def stride(self):
        return ALIGNMENT + self.group_bytes

    @property
    def registered_bytes(self):
        return sum(self.segment_bytes(node) for node in self.node_slots)

    def segment_bytes(self, node):
        return align(max(1, self.node_slots[node]) * self.stride, mmap.PAGESIZE)

    def segment_name(self, node):
        return f"/{self.name}_numa_{node}"

    def partition(self, worker_id):
        return next(p for p in self.partitions if p.worker_id == worker_id)

    def validate(self, *, budget=None, local_host=None):
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("shared relay registration budget must be a positive integer")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", self.name):
            raise ValueError("invalid POSIX shared relay name")
        if not self.host_id or (local_host is not None and self.host_id != local_host):
            raise ValueError("shm_dram requires the launcher's shared pool on the same host")
        if len(self.node_slots) not in (1, 2) or any(
            type(n) is not int or n < 0 or type(s) is not int or s < 0
            for n, s in self.node_slots.items()
        ):
            raise ValueError("shm_dram supports one or two NUMA nodes")
        if (
            any(
                type(v) is not int or v <= 0
                for v in (
                    self.group_bytes,
                    self.max_tokens,
                    self.bytes_per_token,
                    self.layers,
                    self.layers_per_chunk,
                )
            )
            or self.group_bytes % ALIGNMENT
            or self.bytes_per_token % self.layers
        ):
            raise ValueError("invalid shared relay geometry/alignment")
        expected = align(
            self.max_tokens
            * (self.bytes_per_token // self.layers)
            * min(self.layers, self.layers_per_chunk)
        )
        if self.group_bytes != expected:
            raise ValueError("shared relay group size differs from the model geometry")
        assigned = {n: set() for n in self.node_slots}
        owners = set()
        for p in self.partitions:
            if p.worker_id in owners or not p.local_slots or p.numa_id not in assigned:
                raise ValueError("invalid or duplicate shared relay owner")
            owners.add(p.worker_id)
            if (p.overflow_numa is None) != (not p.overflow_slots):
                raise ValueError("overflow slots need an overflow NUMA node")
            for overflow, (node, slots) in enumerate(
                ((p.numa_id, p.local_slots), (p.overflow_numa, p.overflow_slots))
            ):
                if not slots:
                    continue
                if node not in assigned or (overflow and node == p.numa_id):
                    raise ValueError("overflow must belong to the other NUMA node")
                for slot in slots:
                    if (
                        type(slot) is not int
                        or not 0 <= slot < self.node_slots[node]
                        or slot in assigned[node]
                    ):
                        raise ValueError("shared relay partitions overlap or exceed their segment")
                    assigned[node].add(slot)
        if not owners or any(len(assigned[n]) != s for n, s in self.node_slots.items()):
            raise ValueError("all shared relay slots must have exactly one P owner")
        if budget is not None and self.registered_bytes > budget:
            raise ValueError(
                f"shared relay registrations require {self.registered_bytes} bytes "
                f"including local/overflow segments, exceeding budget {budget}"
            )
        return self

    def to_json(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        data["node_slots"] = {int(k): v for k, v in data["node_slots"].items()}
        data["partitions"] = tuple(
            Partition(
                **{
                    **p,
                    "local_slots": tuple(p["local_slots"]),
                    "overflow_slots": tuple(p["overflow_slots"]),
                }
            )
            for p in data["partitions"]
        )
        return cls(**data).validate()


def build_manifest(
    name,
    workers,
    plan,
    *,
    bytes_per_token,
    layers,
    layers_per_chunk,
    local_slots,
    overflow_slots,
    numa_nodes,
    machine=None,
):
    nodes = sorted(set(numa_nodes))
    counters = dict.fromkeys(nodes, 0)
    partitions = []
    for worker_id, node in workers:
        if node not in counters:
            raise ValueError("P NUMA node is absent from the shared relay topology")
        slots = tuple(range(counters[node], counters[node] + local_slots))
        counters[node] += local_slots
        other = next((n for n in nodes if n != node), None)
        overflow = ()
        if other is not None and overflow_slots:
            overflow = tuple(range(counters[other], counters[other] + overflow_slots))
            counters[other] += overflow_slots
        partitions.append(Partition(worker_id, node, slots, other if overflow else None, overflow))
    return RelayManifest(
        name,
        machine or host_id(),
        plan.group_bytes,
        plan.max_tokens,
        bytes_per_token,
        layers,
        layers_per_chunk,
        counters,
        tuple(partitions),
    ).validate()


def _libc():
    lib = ctypes.CDLL(None, use_errno=True)
    lib.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    lib.shm_open.restype = ctypes.c_int
    lib.shm_unlink.argtypes = [ctypes.c_char_p]
    return lib


def _map(name, size, create=False):
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
    fd = _libc().shm_open(name.encode(), flags, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), f"shm_open({name}) failed")
    try:
        if create:
            os.ftruncate(fd, size)
        if os.fstat(fd).st_size != size:
            raise ValueError("POSIX relay segment has an unexpected size")
        return mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    except BaseException:
        if create:
            _libc().shm_unlink(name.encode())
        raise
    finally:
        os.close(fd)


def _initialize(name, size, node, pipe):
    try:
        bind_cpu(node)
        region = _map(name, size)
        address = ctypes.addressof(ctypes.c_char.from_buffer(region))
        bind_memory(address, size, node)
        ctypes.memset(address, 0, size)  # first touch AFTER mbind, on the target node
        region.close()
        pipe.send(None)
    except BaseException as exc:
        pipe.send(f"{type(exc).__name__}: {exc}")
    finally:
        pipe.close()


class SharedRelayPool:
    def __init__(self, manifest, *, create=False, initialize=True):
        self.manifest = manifest.validate(local_host=host_id())
        self.regions, self._registered, self._created = {}, [], []
        try:
            if create:
                free = os.statvfs("/dev/shm")
                if manifest.registered_bytes > free.f_bavail * free.f_frsize:
                    raise RuntimeError(
                        "/dev/shm is too small for the shared relay; increase its mount size"
                    )
            for node in manifest.node_slots:
                name, size = manifest.segment_name(node), manifest.segment_bytes(node)
                self.regions[node] = _map(name, size, create)
                if create:
                    self._created.append(name)
                    if initialize:
                        ctx = multiprocessing.get_context("spawn")
                        parent, child = ctx.Pipe(duplex=False)
                        process = ctx.Process(target=_initialize, args=(name, size, node, child))
                        process.start()
                        child.close()
                        try:
                            if not parent.poll(60):
                                raise RuntimeError(
                                    "NUMA relay first-touch timed out or initializer crashed"
                                )
                            error = parent.recv()
                            process.join(5)
                            if error or process.exitcode != 0:
                                raise RuntimeError(f"NUMA relay initialization failed: {error}")
                        finally:
                            if process.is_alive():
                                process.kill()
                            process.join()
                            parent.close()
        except BaseException:
            self.close()
            raise

    def register_cuda(self):
        import resource

        import torch

        runtime = torch.cuda.cudart()
        try:
            for region in self.regions.values():
                address = ctypes.addressof(ctypes.c_char.from_buffer(region))
                status = runtime.cudaHostRegister(
                    address, len(region), 0
                )  # cudaHostRegisterDefault
                if int(status[0] if isinstance(status, tuple) else status):
                    raise RuntimeError(f"cudaHostRegister returned {status}")
                self._registered.append(address)
        except BaseException as exc:
            self._unregister()
            raise RuntimeError(
                "shm_dram requires pinned shared memory; cudaHostRegister failed "
                f"(RLIMIT_MEMLOCK={resource.getrlimit(resource.RLIMIT_MEMLOCK)}, "
                f"bytes={self.manifest.registered_bytes}). Check memlock/driver support; "
                "no unpinned fallback is allowed"
            ) from exc

    def _unregister(self):
        if self._registered:
            import torch

            for address in self._registered[:]:
                status = torch.cuda.cudart().cudaHostUnregister(address)
                if int(status[0] if isinstance(status, tuple) else status):
                    raise RuntimeError("cudaHostUnregister failed; retain shared mapping")
                self._registered.remove(address)

    def offset(self, node, slot):
        if (
            node not in self.regions
            or type(slot) is not int
            or not 0 <= slot < self.manifest.node_slots[node]
        ):
            raise ValueError("invalid shared relay slot")
        return slot * self.manifest.stride

    def header(self, node, slot):
        return HEADER.unpack_from(self.regions[node], self.offset(node, slot))

    def set_header(self, node, slot, generation, state):
        HEADER.pack_into(self.regions[node], self.offset(node, slot), generation, state)

    def tensor(self, node, slot, nbytes):
        import torch

        if type(nbytes) is not int or not 0 < nbytes <= self.manifest.group_bytes:
            raise ValueError("shared relay payload exceeds its slot")
        return torch.frombuffer(
            self.regions[node],
            dtype=torch.uint8,
            count=nbytes,
            offset=self.offset(node, slot) + ALIGNMENT,
        )

    def close(self):
        self._unregister()
        for region in self.regions.values():
            region.close()
        self.regions.clear()
        for name in self._created:
            if _libc().shm_unlink(name.encode()) and ctypes.get_errno() != 2:
                raise OSError(ctypes.get_errno(), f"shm_unlink({name}) failed")
        self._created.clear()


@dataclass(frozen=True)
class SlotLease:
    slot_numa_id: int
    slot_id: int
    generation: int
    request_id: str


class RelayAllocator:
    """Called only by a P's sender thread; D never mutates the allocation map."""

    def __init__(self, pool, worker_id):
        self.pool, self.partition = pool, pool.manifest.partition(worker_id)
        self.leases = {}

    def _take(self, node, slots, request_id):
        for slot in slots:
            generation, state = self.pool.header(node, slot)
            if state == FREE:
                lease = SlotLease(node, slot, generation + 1, request_id)
                self.pool.set_header(node, slot, lease.generation, WRITING)
                self.leases[node, slot] = lease
                return lease
        return None

    def acquire(self, request_id, d_numa, *, elapsed_ms=0, local_wait_ms=1):
        p = self.partition
        lease = self._take(p.numa_id, p.local_slots, request_id)
        if lease is not None:
            return lease
        if p.overflow_slots and (d_numa != p.numa_id or elapsed_ms >= local_wait_ms):
            return self._take(p.overflow_numa, p.overflow_slots, request_id)
        return None  # bounded wait/retry belongs to the sender pump; never overwrite

    def _check(self, lease):
        if self.leases.get((lease.slot_numa_id, lease.slot_id)) != lease:
            raise ValueError("stale shared relay lease/generation (ABA)")
        generation, _ = self.pool.header(lease.slot_numa_id, lease.slot_id)
        if generation != lease.generation:
            raise ValueError("shared relay generation changed while leased")

    def ready(self, lease, *, copied):
        self._check(lease)
        if not copied:
            raise ValueError("D2H must finish before publishing READY")
        self.pool.set_header(lease.slot_numa_id, lease.slot_id, lease.generation, READY)

    def release(self, lease, *, consumer_complete):
        self._check(lease)
        if not consumer_complete:
            raise ValueError("H2D drain proof is required before slot reuse")
        self.pool.set_header(lease.slot_numa_id, lease.slot_id, lease.generation, FREE)
        del self.leases[lease.slot_numa_id, lease.slot_id]

    def reclaim_request(self, request_id, *, consumer_complete=False):
        for lease in list(self.leases.values()):
            if lease.request_id != request_id:
                continue
            if consumer_complete:
                self.release(lease, consumer_complete=True)
            else:
                self.pool.set_header(
                    lease.slot_numa_id, lease.slot_id, lease.generation, QUARANTINED
                )
