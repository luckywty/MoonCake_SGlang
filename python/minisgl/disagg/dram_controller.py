"""Scheduler-owned asynchronous copies using fixed GPU/host page allocations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from threading import BoundedSemaphore

import torch

from .prefix_hash import hash_page, prepare_prefix
from .transfer import TransferBackendUnavailable


class ImmediateEvent:
    def query(self):
        return True


@dataclass
class DramRequestState:
    prepared: object
    cursor: int
    hashes: bytearray | None = None
    pending_writes: int = 0
    terminal: bool = False
    started: bool = False


@dataclass
class RestoreTask:
    row: int
    start: int
    pages: tuple
    launched: int = 0
    completed: int = 0
    pending: int = 0
    error: str | None = None

    @property
    def done(self):
        return self.completed == len(self.pages) and self.pending == 0


@dataclass
class CopyBatch:
    kind: str
    slot: int
    event: object
    pages: tuple
    owner: object


class DramController:
    def _record(self, event, stream):
        if self.cuda:
            try:
                event.record(stream)
            except Exception as exc:
                raise TransferBackendUnavailable(
                    "cannot fence DRAM CUDA copy; retain its source and destination pages"
                ) from exc

    def __init__(self, scheduler, pool, spec, config):
        self.s, self.pool, self.spec, self.config = scheduler, pool, spec, config
        self.cuda = scheduler.device.type == "cuda"
        self.page = scheduler.layout.page_size
        self.page_bytes = self.page * scheduler.layout.bytes_per_token
        self.offload_stream = torch.cuda.Stream(device=scheduler.device) if self.cuda else None
        self.restore_stream = torch.cuda.Stream(device=scheduler.device) if self.cuda else None
        shape = (config.staging_slots, config.batch_pages, self.page_bytes)
        self.buffers = {
            kind: torch.empty(shape, dtype=torch.uint8, device=scheduler.device)
            for kind in ("write", "read")
        }
        self.indices = {
            kind: torch.empty(
                (config.staging_slots, config.batch_pages, self.page),
                dtype=torch.int64,
                device=scheduler.device,
            )
            for kind in ("write", "read")
        }
        self.events = {
            kind: [
                torch.cuda.Event() if self.cuda else ImmediateEvent()
                for _ in range(config.staging_slots)
            ]
            for kind in ("write", "read")
        }
        self.free = {kind: list(range(config.staging_slots)) for kind in ("write", "read")}
        self.pending = []
        self.hash_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-dram-hash")
        self.hash_capacity = scheduler.table_manager._max_running_reqs
        self.hash_slots = BoundedSemaphore(self.hash_capacity)
        self.stats = {
            "offloaded_tokens": 0,
            "restored_tokens": 0,
            "skipped_pages": 0,
            "copy_failed_pages": 0,
            "hash_skips": 0,
        }

    def begin(self, request, *, decode=False, cancelled=None):
        base = len(request.prompt_ids) // self.page * self.page if decode else 0
        if not self.hash_slots.acquire(blocking=False):
            self.stats["hash_skips"] += 1
            return DramRequestState(None, base, terminal=True)

        def prepare():
            try:
                if cancelled is not None and cancelled.is_set():
                    return None
                return prepare_prefix(request.prompt_ids, self.spec)
            finally:
                self.hash_slots.release()

        try:
            return DramRequestState(self.hash_executor.submit(prepare), base)
        except RuntimeError:
            self.hash_slots.release()
            return DramRequestState(None, base, terminal=True)

    def prepared(self, state):
        if state.terminal or not state.prepared.done():
            return None
        prepared = state.prepared.result()
        if prepared is None:
            return None
        if state.hashes is None:
            state.hashes = bytearray(prepared.page_hashes)
        return prepared

    def offload(self, ctx):
        """True once all currently valid full pages were copied or declined.

        Keep the request row/handle alive while False or pending_writes > 0.
        This protects both shared radix prefixes and private source pages.
        """
        state, req = ctx.dram, ctx.req
        if state is None or req is None or state.terminal:
            return True
        end = req.cached_len // self.page * self.page
        if end <= state.cursor:
            return True
        if self.prepared(state) is None:
            return False
        if not self.free["write"]:
            return False
        slot = self.free["write"].pop()
        pages, starts = [], []
        until = min(end, state.cursor + self.config.batch_pages * self.page)
        for start in range(state.cursor, until, self.page):
            depth = start // self.page + 1
            parent = (
                bytes(state.hashes[(depth - 2) * 32 : (depth - 1) * 32])
                if depth > 1
                else self.spec.root
            )
            if len(state.hashes) < depth * 32:
                digest = hash_page(
                    parent, req.input_ids[start : start + self.page].tolist(), self.page
                )
                state.hashes.extend(digest)
            digest = bytes(state.hashes[(depth - 1) * 32 : depth * 32])
            page = self.pool.reserve(digest, parent, depth, ctx.request.transfer_id)
            if page is None:
                self.stats["skipped_pages"] += 1
            else:
                pages.append(page)
                starts.append(start)
        state.cursor = until
        if not pages:
            self.free["write"].append(slot)
            return state.cursor >= end
        event = self.events["write"][slot]
        batch = CopyBatch("write", slot, event, tuple(pages), state)
        state.pending_writes += 1
        self.pending.append(batch)  # Ownership precedes the first GPU submission.
        stream = self.offload_stream
        try:
            if self.cuda:
                stream.wait_stream(self.s.engine.stream)
            with torch.cuda.stream(stream) if self.cuda else nullcontext():
                for i, (page, start) in enumerate(zip(pages, starts, strict=True)):
                    indices = self.indices["write"][slot, i]
                    indices.copy_(
                        self.s.engine.page_table[req.table_idx, start : start + self.page]
                    )
                    stage = self.buffers["write"][slot, i]
                    self.s.engine.kv_cache.export_page_into(stage, indices)
                    self.pool.view(page).copy_(stage, non_blocking=self.cuda)
        except Exception:
            batch.kind = "failed_write"
            raise
        finally:
            self._record(event, stream)
        return state.cursor >= end

    def restore(self, row, start, pages):
        return RestoreTask(row, start, tuple(pages))

    def advance_restore(self, task):
        if task.error or task.launched == len(task.pages) or not self.free["read"]:
            return
        slot = self.free["read"].pop()
        end = min(len(task.pages), task.launched + self.config.batch_pages)
        pages = task.pages[task.launched : end]
        stream, event = self.restore_stream, self.events["read"][slot]
        batch = CopyBatch("read", slot, event, pages, task)
        task.pending += 1
        self.pending.append(batch)
        try:
            if self.cuda:
                stream.wait_stream(self.s.engine.stream)
            with torch.cuda.stream(stream) if self.cuda else nullcontext():
                for i, page in enumerate(pages):
                    start = task.start + (task.launched + i) * self.page
                    stage = self.buffers["read"][slot, i]
                    stage.copy_(self.pool.view(page), non_blocking=self.cuda)
                    indices = self.indices["read"][slot, i]
                    indices.copy_(self.s.engine.page_table[task.row, start : start + self.page])
                    self.s.engine.kv_cache.import_page_from(stage, indices)
        except Exception as exc:
            task.error = str(exc)
            batch.kind = "failed_read"
        finally:
            task.launched = end
            self._record(event, stream)

    def progress(self):
        for batch in list(self.pending):
            try:
                ready = batch.event.query()
            except Exception as exc:
                raise TransferBackendUnavailable(
                    "DRAM CUDA copy completion is unknown; retain its pages"
                ) from exc
            if not ready:
                continue
            if batch.kind.endswith("write"):
                for page in batch.pages:
                    self.pool.written(page, success=batch.kind == "write")
                batch.owner.pending_writes -= 1
                if batch.kind == "write":
                    self.stats["offloaded_tokens"] += len(batch.pages) * self.page
                self.free["write"].append(batch.slot)
            else:
                task = batch.owner
                task.pending -= 1
                task.completed += len(batch.pages)
                self.pool.release(batch.pages)
                if batch.kind == "read":
                    self.stats["restored_tokens"] += len(batch.pages) * self.page
                self.free["read"].append(batch.slot)
            self.pending.remove(batch)
            if batch.kind.startswith("failed_"):
                self.stats["copy_failed_pages"] += len(batch.pages)

    def abandon_restore(self, task):
        self.pool.release(task.pages[task.launched :])
        task.completed += len(task.pages) - task.launched
        task.launched = len(task.pages)

    def close(self):
        if self.pending:
            raise TransferBackendUnavailable("DRAM copies are still consuming KV pages")
        self.hash_executor.shutdown(wait=True, cancel_futures=True)
