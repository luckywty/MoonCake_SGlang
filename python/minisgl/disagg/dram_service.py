"""Optional DRAM lifecycle mixed into the single-owner WorkerService loop."""

from __future__ import annotations

from dataclasses import asdict

import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.utils import PendingReq

from .dram_controller import DramController
from .dram_pool import DramPagePool
from .dram_protocol import DramConfig, RemotePullPlan
from .prefix_events import PrefixPublisher
from .prefix_hash import PrefixHashSpec
from .tier_index import TierPublisher
from .transfer import TransferBackendUnavailable


class DramServiceMixin:
    def _init_dram(self, config):
        self.dram_config = DramConfig(**config.get("dram", {}))
        self.dram_pool = self.dram_controller = self.dram_transfer = self.tier_publisher = None
        self.dram_error = None
        self.dram_stats = {"pull_fallbacks": 0, "remote_hit_tokens": 0, "local_hit_tokens": 0}
        if not self.dram_config.enabled:
            return
        spec = PrefixHashSpec.from_layout(asdict(self.s.layout), self.cache_namespace)
        self.tier_publisher = TierPublisher(
            spec,
            self.instance_id,
            max_entries=config.get("prefix_max_pages", 65536),
            event_limit=config.get("prefix_event_limit", 32768),
        )
        cache = self.s.cache_manager.prefix_cache
        if hasattr(cache, "page_observer"):
            if self.prefix_publisher is None:
                self.prefix_publisher = PrefixPublisher(
                    spec,
                    self.instance_id,
                    min(self.s.engine.num_pages, config.get("prefix_max_pages", 65536)),
                    config.get("prefix_event_limit", 32768),
                )
                cache.page_observer = self.prefix_publisher
            self.prefix_publisher.event_sink = self.tier_publisher.legacy_event
        self.dram_pool = DramPagePool(
            self.s.layout.page_size * self.s.layout.bytes_per_token,
            self.dram_config.capacity_bytes,
            policy=self.dram_config.policy,
            pin_memory=self.s.device.type == "cuda",
            max_leases=self.dram_config.max_leases,
            lease_ttl=self.dram_config.lease_ttl,
            transaction_ttl=self.dram_config.transaction_ttl,
            max_transactions=max(256, self.queue_limit * 2),
            observer=self.tier_publisher.dram_event,
        )
        self.dram_controller = DramController(self.s, self.dram_pool, spec, self.dram_config)
        if self.dram_config.remote_enabled:
            from .dram_transfer import DramTransfer

            try:
                self.dram_transfer = DramTransfer(self, self.dram_config)
            except Exception as exc:
                self.dram_error = str(exc)

    def _dram_health(self):
        if self.dram_pool is None:
            return {"enabled": False, "remote_ready": False}
        return dict(
            enabled=True,
            version=1,
            remote_ready=self.dram_transfer is not None and self.dram_transfer.ready,
            error=self.dram_error
            or (self.dram_transfer.reason if self.dram_transfer is not None else None),
            **self.dram_pool.health(),
            **self.dram_controller.stats,
            **self.dram_stats,
        )

    def _begin_dram_request(self, ctx, message):
        if self.dram_controller is not None and message.get("cache_transaction") is True:
            if self.dram_pool.begin(ctx.request.transfer_id):
                ctx.dram = self.dram_controller.begin(
                    ctx.request, decode=not self.is_prefill, cancelled=ctx.session.cancelled
                )
        if ctx.dram is not None and ctx.dram.terminal:
            self.dram_pool.finish(ctx.request.transfer_id, success=False)
            return False
        if self.is_prefill and self.dram_controller is not None:
            if ctx.dram is None:
                # Old clients cannot publish a transaction, but can use committed local pages.
                ctx.dram = self.dram_controller.begin(ctx.request, cancelled=ctx.session.cancelled)
            if ctx.dram.terminal:
                self.dram_pool.finish(ctx.request.transfer_id, success=False)
                return False
            if not self.dram_pool.pages and message.get("remote_pull_plan") is None:
                return False  # Cold-cache hashing overlaps Prefill instead of delaying it.
            ctx.phase = "prefix_restore"
            try:
                raw = message.get("remote_pull_plan")
                ctx.remote_plan = RemotePullPlan.from_object(raw) if raw is not None else None
            except (KeyError, TypeError, ValueError):
                self.dram_stats["pull_fallbacks"] += 1
            return True
        return False

    def _dram_rpc(self, message):
        if self.dram_pool is None:
            return {"status": "cache_miss"}
        op = message.get("op")
        if op in ("cache_commit", "cache_abort"):
            from .protocol import check_transfer_id

            check_transfer_id(message.get("transfer_id"))
            self.dram_pool.finish(message["transfer_id"], success=op == "cache_commit")
            return {"status": "cache_finalized"}
        if op == "tier_sync":
            return self.tier_publisher.sync(message)
        if op in ("dram_pin", "dram_unpin") and self.dram_transfer is not None:
            return self.dram_transfer.rpc(message)
        return {"status": "cache_miss"}

    def _stage_dram_live(self, ctx):
        if (
            self.dram_controller is not None
            and ctx.dram is not None
            and not ctx.session.cancelled.is_set()
        ):
            try:
                self.dram_controller.offload(ctx)
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self.dram_error = str(exc)
                self.dram_pool.finish(ctx.request.transfer_id, success=False)
                ctx.dram.terminal = True

    def _dram_decode_restore(self, ctx):
        if self.dram_controller is None or not self.decode_radix_cache:
            return False
        if not self.dram_pool.pages:
            return False
        if ctx.dram is None:
            ctx.dram = self.dram_controller.begin(
                ctx.request, decode=True, cancelled=ctx.session.cancelled
            )
        if ctx.dram.terminal:
            self.dram_pool.finish(ctx.request.transfer_id, success=False)
            return False
        ctx.restore_ready_len = ctx.decode_prefix_len
        ctx.phase = "reserving_hash"
        return True

    def _dram_start_prefill_restore(self, ctx):
        controller, s = self.dram_controller, self.s
        prepared = controller.prepared(ctx.dram)
        if prepared is None:
            return
        page = s.layout.page_size
        hashes = tuple(prepared.digests())[: (len(ctx.request.prompt_ids) - 1) // page]
        handle = s.cache_manager.match_req(
            PendingReq(
                ctx.uid,
                torch.tensor(ctx.request.prompt_ids, dtype=torch.int32),
                SamplingParams(max_tokens=1, ignore_eos=True),
            )
        ).cuda_handle
        s.cache_manager.lock(handle)
        h = handle.cached_len
        try:
            pages = self.dram_pool.acquire(hashes[h // page :])
        except Exception:
            s.cache_manager.unlock(handle)
            raise
        plan = ctx.remote_plan
        remote_possible = (
            plan is not None
            and self.dram_transfer is not None
            and self.dram_transfer.ready
            and plan.domain_id == prepared.domain_id
        )
        needed = self._aligned(len(ctx.request.prompt_ids) + 1) - h
        future_tokens = sum(
            max(0, self._aligned(len(c.request.prompt_ids) + 1) - self._aligned(c.allocated))
            for c in self.contexts.values()
            if c.req is not None and c.phase in ("prefix_restore", "prefill")
        )
        if (
            (not pages and not remote_possible)
            or not s.table_manager.available_size
            or needed + future_tokens > s.cache_manager.available_size
        ):
            self.dram_pool.release(pages)
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
            ctx.allocated = h
            ctx.dram.cursor = h
            ctx.restore_ready_len = h
        except Exception:
            self.dram_pool.release(pages)
            s.cache_manager.unlock(handle)
            if row is not None:
                s.table_manager.free(row)
            raise
        if pages:
            try:
                self._dram_restore_pages(ctx, pages, h)
                ctx.restore_kind = "local"
            except TransferBackendUnavailable:
                raise
            except Exception:
                if ctx.restore_task is None:
                    self.dram_pool.release(pages)
                raise
        else:
            self._dram_start_remote(ctx, prepared)

    def _dram_restore_pages(self, ctx, pages, start):
        end = start + len(pages) * self.s.layout.page_size
        if end > ctx.allocated:
            self.s.cache_manager.reserve_import(
                ctx.req.table_idx, end - ctx.allocated, start=ctx.allocated
            )
            ctx.allocated = end
        ctx.restore_task = self.dram_controller.restore(ctx.req.table_idx, start, pages)
        self.dram_controller.advance_restore(ctx.restore_task)

    def _dram_start_remote(self, ctx, prepared):
        plan = ctx.remote_plan
        ctx.remote_plan = None  # At most one remote source, including failed attempts.
        start = ctx.restore_ready_len // self.s.layout.page_size
        hashes = tuple(prepared.digests())[
            : (len(ctx.request.prompt_ids) - 1) // self.s.layout.page_size
        ]
        if (
            plan is None
            or self.dram_transfer is None
            or not self.dram_transfer.ready
            or plan.domain_id != prepared.domain_id
            or start < plan.start_page
        ):
            self._dram_enqueue_prefill(ctx)
            return
        end = min(len(hashes), plan.start_page + len(plan.page_hashes))
        if (
            end <= start
            or tuple(hashes[plan.start_page : end]) != plan.page_hashes[: end - plan.start_page]
            or (end - start) * self.s.layout.page_size < self.dram_config.min_remote_tokens
        ):
            self._dram_enqueue_prefill(ctx)
            return
        from .dram_transfer import RemotePullJob

        ctx.pull_job = RemotePullJob(
            self.dram_config.staging_slots,
            parent=hashes[start - 1] if start else self.tier_publisher.spec.root,
        )
        ctx.pull_future = self.dram_transfer.start_pull(
            ctx.request.transfer_id,
            plan,
            start,
            hashes[start:end],
            ctx.pull_job,
        )
        if ctx.pull_future is None:
            ctx.pull_job = None
            self.dram_stats["pull_fallbacks"] += 1
            self._dram_enqueue_prefill(ctx)
            return
        ctx.restore_kind = "remote_wait"

    def _dram_enqueue_prefill(self, ctx):
        if ctx.req is not None:
            ctx.req.cached_len = ctx.restore_ready_len
            ctx.dram.cursor = ctx.restore_ready_len
            if ctx.restore_ready_len:
                self.s.token_pool[ctx.req.table_idx, : ctx.restore_ready_len].copy_(
                    ctx.req.input_ids[: ctx.restore_ready_len],
                    non_blocking=self.s.device.type == "cuda",
                )
        ctx.phase = "prefill"
        ctx.restore_kind = None
        ctx.restore_task = None
        self.s.prefill_manager.pending_list.append(
            PendingReq(
                ctx.uid,
                torch.tensor(ctx.request.prompt_ids, dtype=torch.int32),
                SamplingParams(max_tokens=1, ignore_eos=True),
                prepared_prefix=ctx.req,
            )
        )

    def _dram_progress(self):
        if self.dram_controller is None:
            return
        self.dram_controller.progress()
        peers = self.dram_pool.expire()
        if peers and self.dram_transfer is not None:
            self.dram_transfer.request_fence(peers)
        if self.dram_transfer is not None:
            self.dram_transfer.progress()
        for ctx in list(self.contexts.values()):
            if ctx.dram is not None and ctx.phase not in ("draining", "removed"):
                self.dram_pool.keepalive(ctx.request.transfer_id)
            if ctx.phase == "draining":
                if self._dram_can_free(ctx):
                    self._remove(ctx, complete=ctx.drain_complete)
                continue
            if ctx.phase not in ("prefix_restore", "reserving_hash", "reserving_dram"):
                continue
            if ctx.session.cancelled.is_set():
                continue
            try:
                self._dram_progress_context(ctx)
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self.dram_error = str(exc)
                self.dram_stats["pull_fallbacks"] += 1
                if ctx.restore_task is not None:
                    ctx.restore_task.error = str(exc)
                    self.dram_controller.abandon_restore(ctx.restore_task)
                    continue
                if ctx.pull_future is not None:
                    ctx.pull_job.cancelled.set()
                    continue
                self._dram_trim_prefill(ctx)
                if self.is_prefill:
                    self._dram_enqueue_prefill(ctx)
                else:
                    self._dram_finish_reserve(ctx)

    def _dram_trim_prefill(self, ctx):
        if self.is_prefill and ctx.req is not None and ctx.allocated > ctx.restore_ready_len:
            self.s.cache_manager._free(
                self.s.engine.page_table[ctx.req.table_idx, ctx.restore_ready_len : ctx.allocated]
            )
            ctx.allocated = ctx.restore_ready_len

    def _dram_finish_reserve(self, ctx):
        try:
            self._complete_reserve(ctx)
        except TransferBackendUnavailable:
            raise
        except Exception as exc:
            self._fail(ctx.session, exc)

    def _dram_progress_context(self, ctx):
        if ctx.phase == "reserving_hash":
            prepared = self.dram_controller.prepared(ctx.dram)
            if prepared is None:
                return
            start = ctx.decode_prefix_len
            pages = self.dram_pool.acquire(
                tuple(prepared.digests())[start // self.s.layout.page_size :]
            )
            if pages:
                try:
                    self._dram_restore_pages(ctx, pages, start)
                except TransferBackendUnavailable:
                    raise
                except Exception:
                    if ctx.restore_task is None:
                        self.dram_pool.release(pages)
                    raise
                ctx.phase = "reserving_dram"
                ctx.restore_kind = "local"
            else:
                self._dram_finish_reserve(ctx)
            return
        if ctx.phase == "prefix_restore" and ctx.req is None:
            self._dram_start_prefill_restore(ctx)
            return
        if ctx.pull_future is not None and ctx.restore_task is None:
            if ctx.pull_job.cancelled.is_set():
                while (unused := ctx.pull_job.take()) is not None:
                    self.dram_pool.release(unused)
            pages = ctx.pull_job.take()
            if pages is not None:
                try:
                    self._dram_restore_pages(ctx, pages, ctx.restore_ready_len)
                    ctx.restore_kind = "remote"
                except TransferBackendUnavailable:
                    raise
                except Exception:
                    if ctx.restore_task is None:
                        self.dram_pool.release(pages)
                    ctx.pull_job.cancelled.set()
                    raise
            elif ctx.pull_future.done():
                try:
                    if not ctx.pull_future.result():
                        self.dram_stats["pull_fallbacks"] += 1
                except Exception as exc:
                    self.dram_error = str(exc)
                    self.dram_stats["pull_fallbacks"] += 1
                ctx.pull_future = ctx.pull_job = None
                self._dram_trim_prefill(ctx)
                self._dram_enqueue_prefill(ctx)
            return
        task = ctx.restore_task
        if task is None:
            return
        self.dram_controller.advance_restore(task)
        if task.error:
            self.dram_controller.abandon_restore(task)
        if not task.done:
            return
        if not task.error:
            ctx.restore_ready_len = task.start + len(task.pages) * self.s.layout.page_size
            key = "remote_hit_tokens" if ctx.restore_kind == "remote" else "local_hit_tokens"
            self.dram_stats[key] += len(task.pages) * self.s.layout.page_size
        else:
            self._dram_trim_prefill(ctx)
            if ctx.pull_job is not None:
                ctx.pull_job.cancelled.set()
        ctx.restore_task = None
        if ctx.phase == "reserving_dram":
            ctx.decode_prefix_len = ctx.restore_ready_len
            self._dram_finish_reserve(ctx)
        elif ctx.restore_kind == "local":
            self._dram_start_remote(ctx, self.dram_controller.prepared(ctx.dram))
        elif ctx.pull_future is None:
            self._dram_enqueue_prefill(ctx)

    def _dram_can_free(self, ctx):
        if self.dram_controller is None:
            return True
        if ctx.pull_future is not None:
            ctx.pull_job.cancelled.set()
            while (pages := ctx.pull_job.take()) is not None:
                self.dram_pool.release(pages)
            if not ctx.pull_future.done():
                return False
            try:
                ctx.pull_future.result()
            except Exception:
                pass
            while (pages := ctx.pull_job.take()) is not None:
                self.dram_pool.release(pages)
            ctx.pull_future = ctx.pull_job = None
        if ctx.restore_task is not None:
            self.dram_controller.abandon_restore(ctx.restore_task)
            if not ctx.restore_task.done:
                return False
            ctx.restore_task = None
        if ctx.dram is None:
            return True
        if ctx.session.cancelled.is_set():
            self.dram_pool.finish(ctx.request.transfer_id, success=False)
        elif ctx.success:
            try:
                if not self.dram_controller.offload(ctx):
                    return False
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self.dram_error = str(exc)
                self.dram_pool.finish(ctx.request.transfer_id, success=False)
                ctx.dram.terminal = True
        return ctx.dram.pending_writes == 0
