"""Scheduler-owned GPU copies with asynchronous shared-directory metadata RPCs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from threading import BoundedSemaphore

import torch

from .dram_controller import CopyBatch, DramController
from .prefix_hash import hash_page
from .shared_dram_directory import SharedDramClient
from .shared_dram_pool import READY, SharedDramPool, SharedPage
from .transfer import TransferBackendUnavailable


class SharedPoolAdapter:
    """Small adapter for existing all-layer GPU copy primitives, not a private pool.

    Payload views point directly at the resident NUMA segment. There is no local
    Host payload allocation, promotion or memcpy on the P restoration path.
    """

    def __init__(self, manifest, spec, device, *, numa_id, slots=4):
        self.manifest, self.numa_id = manifest, numa_id
        self.shared = SharedDramPool(manifest)
        self.client = SharedDramClient(manifest, domain_id=spec.domain_id)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shared-kv-metadata")
        self.optional_slots = BoundedSemaphore(slots)
        self.operations = []
        try:
            if device.type == "cuda":
                self.shared.register_cuda()
            self.client.call("health")  # Validate model domain/host before serving.
        except BaseException:
            self.client.disconnect()
            self.executor.shutdown(wait=True)
            self.shared.close()
            raise

    def submit(self, method, *args, optional=False, **kwargs):
        if optional and not self.optional_slots.acquire(blocking=False):
            return None

        def run():
            try:
                return method(*args, **kwargs)
            finally:
                if optional:
                    self.optional_slots.release()

        try:
            return self.executor.submit(run)
        except BaseException:
            if optional:
                self.optional_slots.release()
            raise

    def allocate(self, hashes):
        return self.submit(self.client.allocate, hashes, self.numa_id, optional=True)

    def pin(self, hashes):
        return self.submit(self.client.pin_prefix, hashes, optional=True)

    def view(self, page):
        return self.shared.view(page)

    def written(self, page, *, success=True):
        self.operations.append(self.submit(self.client.written, (page,), success=success))

    def release(self, pages):
        if not pages:
            return
        for page in pages:
            if self.shared.header(page.numa_id, page.slot_id) != (page.generation, READY):
                raise TransferBackendUnavailable(
                    "shared history generation changed before H2D unpin"
                )
        self.operations.append(self.submit(self.client.unpin, tuple(pages)))

    def progress(self):
        for future in list(self.operations):
            if future.done():
                try:
                    future.result()
                except Exception as exc:
                    # Retiring the Worker gives the directory a process-exit
                    # completion boundary; never silently strand live DMA owners.
                    raise TransferBackendUnavailable(
                        "shared history directory ownership is unknown"
                    ) from exc
                self.operations.remove(future)

    def health(self):
        return {
            "mode": "shared_shm",
            "capacity_bytes": self.manifest.registered_bytes,
            "capacity_pages": self.manifest.capacity,
            **self.client.last_health,
        }

    def begin(self, transaction):
        return True

    def finish(self, transaction, *, success):
        # shared_shm publishes individually fenced, fully computed pages. Client
        # inference transaction messages do not revoke another reader's page.
        return None

    def end(self, transaction):
        return None

    def close(self):
        self.executor.shutdown(wait=True)
        self.progress()
        self.client.close(drained=True)
        self.shared.close()


@dataclass
class Allocation:
    ctx: object
    future: object
    starts: tuple[int, ...]
    until: int
    grants: tuple | None = None


class SharedDramController(DramController):
    def __init__(self, scheduler, pool, spec, config):
        super().__init__(scheduler, pool, spec, config)
        self.allocations = {}

    def begin(self, request, *, decode=False, cancelled=None):
        # P may already publish prompt pages; D's complete imported pages are
        # harmless duplicates. Include them, as well as new Decode/seam pages.
        return super().begin(request, decode=False, cancelled=cancelled)

    def offload(self, ctx):
        state, req = ctx.dram, ctx.req
        if state is None or req is None or state.terminal:
            return True
        if ctx.request.transfer_id in self.allocations:
            return False
        end = req.cached_len // self.page * self.page
        if end <= state.cursor:
            return True
        if self.prepared(state) is None or not self.free["write"]:
            return False
        until = min(end, state.cursor + self.config.batch_pages * self.page)
        starts, hashes = tuple(range(state.cursor, until, self.page)), []
        for start in starts:
            depth = start // self.page + 1
            parent = (
                bytes(state.hashes[(depth - 2) * 32 : (depth - 1) * 32])
                if depth > 1
                else self.spec.root
            )
            if len(state.hashes) < depth * 32:
                state.hashes.extend(
                    hash_page(parent, req.input_ids[start : start + self.page].tolist(), self.page)
                )
            hashes.append(bytes(state.hashes[(depth - 1) * 32 : depth * 32]))
        future = self.pool.allocate(hashes)
        if future is None:
            state.cursor = until
            self.stats["skipped_pages"] += len(starts)
            return until >= end  # Metadata admission is optional; never stall Decode.
        state.pending_writes += 1  # The source row must survive a pending allocation.
        self.allocations[ctx.request.transfer_id] = Allocation(ctx, future, starts, until)
        return False

    def _install_grants(self, allocation):
        ctx, state = allocation.ctx, allocation.ctx.dram
        if allocation.grants is None:
            try:
                records = allocation.future.result()
                allocation.grants = tuple(
                    (SharedPage.from_object(record["page"]), start)
                    for start, record in zip(allocation.starts, records, strict=True)
                    if record["status"] == "allocated"
                )
            except Exception as exc:
                raise TransferBackendUnavailable(
                    "shared history write allocation completion is unknown"
                ) from exc
            self.stats["skipped_pages"] += len(allocation.starts) - len(allocation.grants)
        pages = tuple(page for page, _ in allocation.grants)
        if state.terminal or ctx.session.cancelled.is_set():
            for page in pages:
                self.pool.written(page, success=False)  # No CUDA writer was launched.
            state.pending_writes -= 1
            return True
        if pages and not self.free["write"]:
            return False
        state.cursor = allocation.until
        if not pages:
            state.pending_writes -= 1
            return True
        slot = self.free["write"].pop()
        event = self.events["write"][slot]
        batch = CopyBatch("write", slot, event, tuple(pages), state)
        self.pending.append(batch)
        stream, req = self.offload_stream, ctx.req
        try:
            if self.cuda:
                stream.wait_stream(self.s.engine.stream)
            with torch.cuda.stream(stream) if self.cuda else nullcontext():
                for i, (page, start) in enumerate(allocation.grants):
                    indices = self.indices["write"][slot, i]
                    indices.copy_(
                        self.s.engine.page_table[req.table_idx, start : start + self.page]
                    )
                    stage = self.buffers["write"][slot, i]
                    self.s.engine.kv_cache.export_page_into(stage, indices)
                    self.pool.view(page).copy_(stage, non_blocking=self.cuda)
        except Exception:
            batch.kind = "failed_write"
        finally:
            self._record(event, stream)
        return True

    def progress(self):
        super().progress()  # written/unpin only run after the local CUDA event.
        self.pool.progress()
        for key, work in list(self.allocations.items()):
            if work.future.done() and self._install_grants(work):
                del self.allocations[key]

    def cancel(self, ctx):
        if ctx.dram is not None:
            ctx.dram.terminal = True

    def close(self):
        if self.allocations:
            raise TransferBackendUnavailable("shared history allocations still own source KV")
        super().close()
        self.pool.close()
