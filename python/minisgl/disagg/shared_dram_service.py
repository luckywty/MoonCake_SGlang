"""Optional shared history lifecycle; private DRAM and Host TE stay separate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.utils import PendingReq

from .prefix_hash import PrefixHashSpec
from .shared_dram_controller import SharedDramController, SharedPoolAdapter
from .shared_dram_pool import SharedDramManifest
from .transfer import TransferBackendUnavailable


class SharedDramServiceMixin:
    def _init_shared_history(self, config):
        self.shared_history = True
        self.shared_manifest_path = config["shared_dram_manifest"]
        self.shared_manifest = SharedDramManifest.from_json(
            Path(self.shared_manifest_path).read_text()
        )
        self.shared_spec = PrefixHashSpec.from_layout(asdict(self.s.layout), self.cache_namespace)
        if (
            self.shared_manifest.page_bytes
            != self.s.layout.bytes_per_token * self.s.layout.page_size
        ):
            raise ValueError("shared history layout differs from effective Worker KV layout")
        self.tier_publisher = None
        self.dram_pool = SharedPoolAdapter(
            self.shared_manifest,
            self.shared_spec,
            self.s.device,
            numa_id=self.numa_id,
            slots=max(2, self.dram_config.staging_slots * 2),
        )
        self.dram_controller = SharedDramController(
            self.s, self.dram_pool, self.shared_spec, self.dram_config
        )

    def _shared_identity(self):
        if not getattr(self, "shared_history", False):
            return None
        return {
            "mode": "shared_shm",
            "host_id": self.shared_manifest.host_id,
            "pool": self.shared_manifest.name,
            "manifest": self.shared_manifest_path,
            "spec": self.shared_spec.manifest,
        }

    def _shared_begin(self, ctx):
        ctx.dram = self.dram_controller.begin(
            ctx.request, decode=not self.is_prefill, cancelled=ctx.session.cancelled
        )
        if ctx.dram.terminal:
            return False
        if self.is_prefill:
            ctx.phase = "prefix_restore"
            return True
        return False

    def _shared_start_pin(self, ctx):
        prepared = self.dram_controller.prepared(ctx.dram)
        if prepared is None:
            return
        s, page = self.s, self.s.layout.page_size
        if self.is_prefill and ctx.req is None:
            handle = s.cache_manager.match_req(
                PendingReq(
                    ctx.uid,
                    torch.tensor(ctx.request.prompt_ids, dtype=torch.int32),
                    SamplingParams(max_tokens=1, ignore_eos=True),
                )
            ).cuda_handle
            s.cache_manager.lock(handle)
            h = handle.cached_len
            needed = self._aligned(len(ctx.request.prompt_ids) + 1) - h
            future = sum(
                max(0, self._aligned(len(c.request.prompt_ids) + 1) - self._aligned(c.allocated))
                for c in self.contexts.values()
                if c.req is not None and c.phase in ("prefix_restore", "prefill")
            )
            if (
                not s.table_manager.available_size
                or needed + future > s.cache_manager.available_size
            ):
                s.cache_manager.unlock(handle)
                ctx.dram.cursor = h
                self._dram_enqueue_prefill(ctx)
                return
            row = None
            try:
                row = s.table_manager.allocate()
                if h:
                    s.engine.page_table[row, :h] = handle.get_matched_indices()
                ctx.req = Req(
                    torch.tensor(ctx.request.prompt_ids, dtype=torch.int32),
                    row,
                    h,
                    1,
                    ctx.uid,
                    SamplingParams(max_tokens=1, ignore_eos=True),
                    handle,
                )
                ctx.allocated = ctx.restore_ready_len = ctx.dram.cursor = h
            except Exception:
                s.cache_manager.unlock(handle)
                if row is not None:
                    s.table_manager.free(row)
                raise
        if not self.is_prefill:
            ctx.restore_ready_len = ctx.decode_prefix_len
        limit = (len(ctx.request.prompt_ids) - (1 if self.is_prefill else 0)) // page
        hashes = tuple(prepared.digests())[ctx.restore_ready_len // page : limit][:16384]
        ctx.shared_pin_future = self.dram_pool.pin(hashes) if hashes else None
        if ctx.shared_pin_future is None:
            self._shared_finish_restore(ctx)
        else:
            ctx.restore_kind = "shared_pin"

    def _shared_finish_restore(self, ctx):
        self._dram_trim_prefill(ctx)
        if self.is_prefill:
            self._dram_enqueue_prefill(ctx)
        else:
            ctx.decode_prefix_len = ctx.restore_ready_len
            self._dram_finish_reserve(ctx)

    def _shared_progress_context(self, ctx):
        if ctx.restore_kind != "shared_pin" and ctx.restore_task is None:
            self._shared_start_pin(ctx)
            return
        future = ctx.shared_pin_future
        if future is not None:
            if not future.done():
                return
            try:
                pages = future.result()
            except Exception as exc:
                raise TransferBackendUnavailable("shared history pin ownership is unknown") from exc
            ctx.shared_pin_future = None
            if not pages:
                self._shared_finish_restore(ctx)
                return
            try:
                self._dram_restore_pages(ctx, pages, ctx.restore_ready_len)
            except Exception:
                if ctx.restore_task is None:
                    self.dram_pool.release(pages)
                raise
            ctx.restore_kind = "shared_read"
        task = ctx.restore_task
        if task is not None:
            self.dram_controller.advance_restore(task)
            if task.error:
                self.dram_controller.abandon_restore(task)
            if not task.done:
                return
            if not task.error:
                ctx.restore_ready_len = task.start + len(task.pages) * self.s.layout.page_size
                local = sum(p.numa_id == self.numa_id for p in task.pages) * self.s.layout.page_size
                self.dram_stats["local_hit_tokens"] += local
                self.dram_stats["remote_hit_tokens"] += (
                    len(task.pages) * self.s.layout.page_size - local
                )
            ctx.restore_task = None
            self._shared_finish_restore(ctx)

    def _shared_progress(self):
        self.dram_controller.progress()
        for ctx in list(self.contexts.values()):
            if ctx.phase == "draining":
                if self._dram_can_free(ctx):
                    self._remove(ctx, complete=ctx.drain_complete)
            elif (
                ctx.phase in ("prefix_restore", "reserving_shared")
                and not ctx.session.cancelled.is_set()
            ):
                try:
                    self._shared_progress_context(ctx)
                except TransferBackendUnavailable:
                    raise
                except Exception as exc:
                    self.dram_error = str(exc)
                    self.dram_stats["pull_fallbacks"] += 1
                    if ctx.restore_task is not None:
                        ctx.restore_task.error = str(exc)
                        self.dram_controller.abandon_restore(ctx.restore_task)
                    else:
                        self._shared_finish_restore(ctx)

    def _shared_can_free(self, ctx):
        future = ctx.shared_pin_future
        if future is not None:
            if not future.done():
                return False
            try:
                self.dram_pool.release(future.result())
            except Exception as exc:
                raise TransferBackendUnavailable(
                    "shared history pin could not be safely cancelled"
                ) from exc
            ctx.shared_pin_future = None
        if ctx.restore_task is not None:
            self.dram_controller.abandon_restore(ctx.restore_task)
            if not ctx.restore_task.done:
                return False
            ctx.restore_task = None
        if ctx.dram is None:
            return True
        if ctx.session.cancelled.is_set():
            self.dram_controller.cancel(ctx)
        elif ctx.success and not self.dram_controller.offload(ctx):
            return False
        return ctx.dram.pending_writes == 0
