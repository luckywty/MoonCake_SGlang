"""Host-local metadata coordinator. No KV payload traverses this service.

Lookup uses immutable sharded snapshots and takes no directory lock. Pin/write
operations serialize ownership changes. Socket loss alone never proves DMA
drain: disconnected owners are reaped after verified process exit, or resume
and explicitly release their leases after their local CUDA fences complete.
"""

from __future__ import annotations

import array
import os
import select
import socket
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .control import Channel
from .shared_dram_pool import FREE, QUARANTINED, READY, STAGED, SharedPage


def _digest(value):
    value = value.hex() if isinstance(value, bytes) else value
    if not isinstance(value, str) or len(value) != 64 or bytes.fromhex(value).hex() != value:
        raise ValueError("invalid shared history page hash")
    return value


@dataclass
class Entry:
    page: SharedPage
    writer: str
    state: int = STAGED
    refcount: int = 0
    last_access: float = 0.0
    freq: float = 1.0


class SharedDramDirectory:
    def __init__(self, pool, *, policy="lru", clock=time.monotonic):
        if policy not in ("lru", "lfu"):
            raise ValueError("invalid shared history eviction policy")
        self.pool, self.policy, self.clock = pool, policy, clock
        self.lock = threading.RLock()
        self.entries, self.slots, self.leases = {}, {}, {}
        self.free = {n: deque(range(count)) for n, count in pool.manifest.node_slots.items()}
        if any(
            pool.header(n, s) != (0, FREE)
            for n, count in pool.manifest.node_slots.items()
            for s in range(count)
        ):
            raise ValueError(
                "shared history coordinator requires fresh segments; existing DMA owners cannot be forgotten"
            )
        self.generations = {
            n: array.array("Q", [0]) * count for n, count in pool.manifest.node_slots.items()
        }
        self._snapshot = tuple({} for _ in range(64))
        self.last_decay = clock()
        self.domain_id = None
        self.stats = {
            "allocations": 0,
            "duplicates": 0,
            "full": 0,
            "evictions": 0,
            "commits": 0,
            "aborts": 0,
            "process_reclaims": 0,
        }

    def bind_domain(self, domain_id):
        domain_id = _digest(domain_id)
        with self.lock:
            if self.domain_id is not None and self.domain_id != domain_id:
                raise ValueError("shared history model/cache namespace differs")
            self.domain_id = domain_id

    def _publish(self, key, page=None):
        shard = int(key[:2], 16) % len(self._snapshot)
        updated = dict(self._snapshot[shard])
        if page is None:
            updated.pop(key, None)
        else:
            updated[key] = page
        snapshots = list(self._snapshot)
        snapshots[shard] = updated
        self._snapshot = tuple(snapshots)

    def _touch(self, entry):
        entry.last_access = self.clock()
        entry.freq += 1

    def _drop(self, entry):
        p = entry.page
        if entry.refcount:
            raise ValueError("cannot recycle a pinned shared history page")
        self._publish(p.page_hash)
        self.entries.pop(p.page_hash, None)
        self.slots.pop((p.numa_id, p.slot_id), None)
        self.pool.set_header(p.numa_id, p.slot_id, p.generation, FREE)
        self.free[p.numa_id].append(p.slot_id)

    def _victim(self, preferred):
        now = self.clock()
        if self.policy == "lfu" and now - self.last_decay >= 60:
            factor = 0.5 ** min(32, (now - self.last_decay) / 60)
            for entry in self.entries.values():
                entry.freq *= factor
            self.last_decay = now
        eligible = [e for e in self.entries.values() if e.state == READY and not e.refcount]
        local = [e for e in eligible if e.page.numa_id == preferred]
        choices = local or eligible
        key = (
            (lambda e: (e.last_access, e.page.numa_id, e.page.slot_id))
            if self.policy == "lru"
            else (lambda e: (e.freq, e.last_access, e.page.numa_id, e.page.slot_id))
        )
        return min(choices, key=key, default=None)

    def allocate_write(self, page_hash, preferred_numa_id, writer_id):
        key = _digest(page_hash)
        if (
            type(preferred_numa_id) is not int
            or preferred_numa_id not in self.free
            or not writer_id
        ):
            raise ValueError("invalid shared history writer/NUMA identity")
        with self.lock:
            old = self.entries.get(key)
            if old is not None:
                if old.state == QUARANTINED:
                    self.stats["full"] += 1
                    return {"status": "full"}
                self._touch(old)
                self.stats["duplicates"] += 1
                return {"status": "duplicate"}
            order = [preferred_numa_id, *(n for n in sorted(self.free) if n != preferred_numa_id)]
            node = next((n for n in order if self.free[n]), None)
            if node is None:
                victim = self._victim(preferred_numa_id)
                if victim is None:
                    self.stats["full"] += 1
                    return {"status": "full"}
                node = victim.page.numa_id
                self._drop(victim)
                self.stats["evictions"] += 1
            slot = self.free[node].popleft()
            self.generations[node][slot] += 1
            page = SharedPage(key, node, slot, self.generations[node][slot])
            entry = Entry(page, writer_id, last_access=self.clock())
            self.entries[key] = self.slots[node, slot] = entry
            self.pool.set_header(node, slot, page.generation, STAGED)
            self.stats["allocations"] += 1
            return {"status": "allocated", "page": page.to_object()}

    def _entry(self, numa_id, slot_id, generation, page_hash):
        entry = self.slots.get((numa_id, slot_id))
        if (
            entry is None
            or entry.page.generation != generation
            or entry.page.page_hash != _digest(page_hash)
        ):
            raise ValueError("stale shared history generation (ABA)")
        if self.pool.header(numa_id, slot_id)[0] != generation:
            raise ValueError("shared history slot header generation differs")
        return entry

    def commit_write(self, numa_id, slot_id, generation, page_hash, *, writer_id=None):
        with self.lock:
            entry = self._entry(numa_id, slot_id, generation, page_hash)
            if writer_id is not None and entry.writer != writer_id:
                raise ValueError("shared history write grant belongs to another owner")
            if entry.state == READY:
                return True
            if entry.state not in (STAGED, QUARANTINED):
                raise ValueError("shared history page is not staged")
            entry.state = READY
            self.pool.set_header(numa_id, slot_id, generation, READY)
            self._publish(entry.page.page_hash, entry.page)
            self.stats["commits"] += 1
            return True

    def abort_write(self, numa_id, slot_id, generation, page_hash, *, writer_id=None, drained=True):
        with self.lock:
            entry = self._entry(numa_id, slot_id, generation, page_hash)
            if writer_id is not None and entry.writer != writer_id:
                raise ValueError("shared history write grant belongs to another owner")
            if entry.state == READY:
                raise ValueError("cannot abort an already published shared history page")
            if drained:
                self._drop(entry)
                self.stats["aborts"] += 1
            else:
                entry.state = QUARANTINED
                self.pool.set_header(numa_id, slot_id, generation, QUARANTINED)
            return True

    def lookup_prefix(self, page_hashes):
        snapshot = self._snapshot  # Lock-free hint; pin_prefix revalidates ownership.
        result = []
        for value in page_hashes:
            key = _digest(value)
            page = snapshot[int(key[:2], 16) % len(snapshot)].get(key)
            if page is None:
                break
            result.append(page.to_object())
        return result

    def pin_prefix(self, page_hashes, reader_id):
        if not reader_id:
            raise ValueError("shared history pin needs a reader id")
        keys = [_digest(h) for h in page_hashes]
        with self.lock:
            result = []
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or entry.state != READY:
                    break
                self._touch(entry)
                entry.refcount += 1
                lease = SharedPage(**{**entry.page.to_object(), "lease_id": uuid4().hex})
                self.leases[lease.lease_id] = (reader_id, lease)
                result.append(lease.to_object())
            return result

    def unpin(self, leases, reader_id):
        pages = [SharedPage.from_object(p) if isinstance(p, dict) else p for p in leases]
        if len({p.lease_id for p in pages}) != len(pages):
            raise ValueError("duplicate shared history unpin token")
        with self.lock:
            for page in pages:
                record = self.leases.get(page.lease_id)
                if record is None:
                    continue  # Idempotent replay; a unique lease token cannot affect a new owner.
                if record != (reader_id, page):
                    raise ValueError("shared history reader lease mismatch")
                entry = self._entry(page.numa_id, page.slot_id, page.generation, page.page_hash)
                if entry.refcount < 1:
                    raise ValueError("shared history reader underflow")
            for page in pages:
                if self.leases.pop(page.lease_id, None) is not None:
                    self.slots[page.numa_id, page.slot_id].refcount -= 1
            return True

    def disconnect_owner(self, owner_id, *, process_dead=False, drained=False):
        """A live disconnected process may still DMA. Retain its reader refs."""
        with self.lock:
            if process_dead or drained:
                pages = [p.to_object() for owner, p in self.leases.values() if owner == owner_id]
                self.unpin(pages, owner_id)
            for entry in list(self.entries.values()):
                if entry.writer == owner_id and entry.state in (STAGED, QUARANTINED):
                    if process_dead or drained:
                        self._drop(entry)
                    else:
                        entry.state = QUARANTINED
                        p = entry.page
                        self.pool.set_header(p.numa_id, p.slot_id, p.generation, QUARANTINED)
            self.stats["process_reclaims"] += bool(process_dead)

    def health(self):
        with self.lock:
            return {
                "mode": "shared_shm",
                "pool": self.pool.manifest.name,
                "registered_bytes": self.pool.manifest.registered_bytes,
                "capacity_pages": self.pool.manifest.capacity,
                "ready_pages": sum(e.state == READY for e in self.entries.values()),
                "staged_pages": sum(e.state == STAGED for e in self.entries.values()),
                "quarantined_pages": sum(e.state == QUARANTINED for e in self.entries.values()),
                "readers": len(self.leases),
                **self.stats,
            }


def _process_start(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return tail[19], tail[0]  # starttime (field 22), state
    except FileNotFoundError:
        return None
    except (OSError, IndexError) as exc:
        raise RuntimeError("cannot verify shared history peer process exit") from exc


@dataclass
class Owner:
    pid: int
    start: object
    pidfd: int | None
    connection: int = 0
    connected: bool = True
    seq: int = 0
    request: dict | None = None
    response: dict | None = None
    lock: object = field(default_factory=threading.Lock)

    def dead(self):
        if self.pidfd is not None:
            return bool(select.select([self.pidfd], [], [], 0)[0])
        try:
            current = _process_start(self.pid)
        except RuntimeError:
            return False  # Failure to inspect /proc is not a DMA-drain proof.
        return current is None or current[0] != self.start or current[1] in ("Z", "X")


class SharedDramServer:
    def __init__(self, directory):
        self.directory, self.manifest = directory, directory.pool.manifest
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.owners, self.connections = {}, set()
        self.handlers = ThreadPoolExecutor(max_workers=32, thread_name_prefix="shared-kv-rpc")
        self.slots = threading.BoundedSemaphore(32)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.manifest.directory_socket)
        os.chmod(self.manifest.directory_socket, 0o600)
        self.listener.listen(64)
        self.listener.settimeout(0.1)
        self.thread = threading.Thread(target=self._serve, name="shared-kv-directory", daemon=True)
        self.thread.start()

    def _serve(self):
        while not self.stop.is_set():
            self.reap()
            try:
                sock, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self.slots.acquire(blocking=False):
                sock.close()
                continue
            with self.lock:
                self.connections.add(sock)
            self.handlers.submit(self._handle, sock)

    def reap(self):
        with self.lock:
            for key, owner in list(self.owners.items()):
                if owner.dead():
                    with owner.lock:
                        # A buffered final RPC must not allocate after reaping.
                        owner.connection += 1
                        owner.connected = False
                        self.directory.disconnect_owner(key, process_dead=True)
                    if owner.pidfd is not None:
                        os.close(owner.pidfd)
                    del self.owners[key]

    def _hello(self, sock, msg):
        pid, uid, _ = struct.unpack(
            "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if (
            uid != os.getuid()
            or msg.get("secret") != self.manifest.secret
            or msg.get("host_id") != self.manifest.host_id
            or msg.get("pool") != self.manifest.name
        ):
            raise ValueError("shared history requires the same host/pool identity")
        reader = msg.get("reader_id")
        if not isinstance(reader, str) or not 1 <= len(reader) <= 256:
            raise ValueError("invalid shared history owner id")
        if msg.get("domain_id"):
            self.directory.bind_domain(msg["domain_id"])
        start = _process_start(pid)
        if start is None:
            raise ValueError("cannot verify shared history peer process")
        with self.lock:
            owner = self.owners.get(reader)
            if owner is None:
                try:
                    pidfd = os.pidfd_open(pid)
                except (AttributeError, OSError):
                    pidfd = None
                owner = self.owners[reader] = Owner(pid, start[0], pidfd)
            elif owner.pid != pid or owner.start != start[0]:
                raise ValueError("shared history owner cannot be taken over by another process")
            owner.connection += 1
            owner.connected = True
            return reader, owner, owner.connection

    def _dispatch(self, owner, msg):
        op = msg.get("op")
        d = self.directory
        if op == "lookup":
            result = d.lookup_prefix(msg["hashes"])
        elif op == "pin":
            result = d.pin_prefix(msg["hashes"], owner)
        elif op == "unpin":
            result = d.unpin(msg["pages"], owner)
        elif op == "allocate":
            hashes = [_digest(h) for h in msg["hashes"]]
            result = [d.allocate_write(h, msg["preferred_numa_id"], owner) for h in hashes]
        elif op in ("commit", "abort"):
            result = []
            for data in msg["pages"]:
                page = SharedPage.from_object(data)
                method = d.commit_write if op == "commit" else d.abort_write
                options = {"drained": msg.get("drained", True)} if op == "abort" else {}
                result.append(
                    method(
                        page.numa_id,
                        page.slot_id,
                        page.generation,
                        page.page_hash,
                        writer_id=owner,
                        **options,
                    )
                )
        elif op == "close":
            d.disconnect_owner(owner, drained=bool(msg.get("drained")))
            result = True
        elif op == "health":
            result = d.health()
        else:
            raise ValueError("unknown shared history RPC")
        return {"status": "ok", "result": result, "health": d.health()}

    def _handle(self, sock):
        key = owner = connection = None
        channel = Channel(sock, 30, self.stop)
        try:
            hello = channel.receive()
            try:
                key, owner, connection = self._hello(sock, hello)
            except ValueError as exc:
                channel.send({"status": "error", "error": str(exc)})
                return
            channel.send({"status": "ready"})
            while not self.stop.is_set():
                msg = channel.receive()
                with owner.lock:
                    if owner.connection != connection:
                        raise ValueError("shared history connection was superseded")
                    seq = msg.get("seq")
                    if type(seq) is not int or seq < 1:
                        raise ValueError("shared history RPC sequence is invalid")
                    if seq == owner.seq:
                        if owner.request != msg:
                            raise ValueError("shared history RPC replay changed its operation")
                        reply = owner.response
                    elif seq == owner.seq + 1:
                        try:
                            reply = self._dispatch(key, msg)
                        except (ValueError, KeyError, TypeError) as exc:
                            reply = {"status": "error", "error": str(exc)}
                        owner.seq, owner.request, owner.response = seq, msg, reply
                    else:
                        raise ValueError("shared history RPC sequence skipped an operation")
                    channel.send(reply)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            pass
        finally:
            with self.lock:
                if owner is not None and owner.connection == connection:
                    owner.connected = False
                    self.directory.disconnect_owner(key)
                self.connections.discard(sock)
            sock.close()
            self.slots.release()

    def close(self):
        self.stop.set()
        self.listener.close()
        with self.lock:
            for sock in self.connections:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.thread.join(2)
        self.handlers.shutdown(wait=True)
        with self.lock:
            for owner in self.owners.values():
                if owner.pidfd is not None:
                    os.close(owner.pidfd)
            self.owners.clear()
        Path(self.manifest.directory_socket).unlink(missing_ok=True)


class SharedDramClient:
    def __init__(self, manifest, *, reader_id=None, domain_id=None, timeout=0.5):
        self.manifest, self.reader_id = manifest, reader_id or uuid4().hex
        self.domain_id, self.timeout = domain_id, timeout
        self.lock = threading.Lock()
        self.seq, self.sock, self.channel, self.last_health = 0, None, None, {}

    def _connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.manifest.directory_socket)
        self.channel = Channel(self.sock, self.timeout)
        self.channel.send(
            {
                "secret": self.manifest.secret,
                "host_id": self.manifest.host_id,
                "pool": self.manifest.name,
                "reader_id": self.reader_id,
                "domain_id": self.domain_id,
            }
        )
        reply = self.channel.receive()
        if reply != {"status": "ready"}:
            raise ValueError(f"shared history directory rejected this Worker: {reply.get('error')}")

    def disconnect(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = self.channel = None

    def call(self, op, **args):
        with self.lock:
            self.seq += 1
            message = {"seq": self.seq, "op": op, **args}
            for attempt in range(2):
                try:
                    if self.channel is None:
                        self._connect()
                    self.channel.send(message)
                    reply = self.channel.receive()
                    if reply.get("status") != "ok":
                        raise ValueError(reply.get("error", "shared history operation rejected"))
                    self.last_health = reply.get("health", self.last_health)
                    return reply["result"]
                except (OSError, RuntimeError):
                    self.disconnect()
                    if attempt:
                        raise

    def lookup_prefix(self, hashes):
        return self.call("lookup", hashes=[_digest(h) for h in hashes])

    def pin_prefix(self, hashes):
        return tuple(
            SharedPage.from_object(p) for p in self.call("pin", hashes=[_digest(h) for h in hashes])
        )

    def unpin(self, pages):
        return self.call("unpin", pages=[p.to_object() for p in pages])

    def allocate(self, hashes, numa_id):
        return self.call("allocate", hashes=[_digest(h) for h in hashes], preferred_numa_id=numa_id)

    def written(self, pages, *, success=True, drained=True):
        return self.call(
            "commit" if success else "abort",
            pages=[p.to_object() for p in pages],
            **({} if success else {"drained": drained}),
        )

    def close(self, *, drained=False):
        try:
            if self.channel is not None:
                self.call("close", drained=drained)
        finally:
            self.disconnect()
