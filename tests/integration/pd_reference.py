"""CPU reference model for Phase 2 tests. It uses real attention tensors and the real KV pool.

The GPU Engine is replaced only in this test fixture. CUDA readiness, kernels and model loading
must be verified separately with tools/pd/verify_phase2_gpu.py.
"""

import socket
import time
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import minisgl.core as core
import minisgl.distributed.info as tp_info
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.disagg.control import Channel, endpoint_name
from minisgl.disagg.protocol import KVLayout
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.tcp_backend import TcpTransferBackend
from minisgl.disagg.worker import PDWorkerScheduler, serve_worker
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.table import TableManager


class TinyEngine:
    def __init__(self, page_size, delay=0.0):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_pages = 32
        self.max_seq_len = 64
        self.page_table = torch.zeros((3, 64), dtype=torch.int32)
        self.kv_cache = MHAKVCache(1, 2, 8, 33, page_size, self.dtype, self.device)
        self.kv_cache._kv_buffer.zero_()
        self.graph_runner = SimpleNamespace(
            pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs)
        )
        self.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
        self.sampler = SimpleNamespace(prepare=lambda batch: None)
        generator = torch.Generator().manual_seed(812)
        self.embedding = torch.randn(32, 8, generator=generator)
        self.weights = torch.randn(2, 4, 8, 8, generator=generator) / 3
        self.head = torch.randn(8, 32, generator=generator)
        self.delay = delay

    def forward_batch(self, batch, _sample_args):
        if self.delay:
            time.sleep(self.delay)
        x = self.embedding[batch.input_ids.to(torch.long)]
        row = batch.reqs[0].table_idx
        for layer in range(2):
            q, k, v = (x @ self.weights[layer, i] for i in range(3))
            keys = self.kv_cache.k_cache(layer).view(-1, 8)
            values = self.kv_cache.v_cache(layer).view(-1, 8)
            keys[batch.out_loc.to(torch.long)] = k
            values[batch.out_loc.to(torch.long)] = v
            if self.kv_cache.layer_ready is not None:
                self.kv_cache.layer_ready(layer)
            results = []
            for i, position in enumerate(batch.positions.tolist()):
                slots = self.page_table[row, : position + 1].to(torch.long)
                scores = (keys[slots] @ q[i]) / (8**0.5)
                results.append(torch.softmax(scores, dim=0) @ values[slots])
            x = torch.tanh(x + torch.stack(results) @ self.weights[layer, 3])
        token = torch.argmax(x[-1] @ self.head).reshape(1).to(torch.int32)
        for req in batch.reqs:
            req.complete_one()
        return SimpleNamespace(
            next_tokens_gpu=token,
            next_tokens_cpu=token,
            copy_done_event=SimpleNamespace(synchronize=lambda: None),
        )


def cpu_scheduler(role, page_size=4, delay=0.0):
    tp_info._TP_INFO = tp_info.DistributedInfo(0, 1)
    core._GLOBAL_CTX = core.Context(page_size)
    scheduler = PDWorkerScheduler.__new__(PDWorkerScheduler)
    scheduler.engine = TinyEngine(page_size, delay)
    scheduler.device = torch.device("cpu")
    scheduler.role = WorkerRole(role)
    scheduler.layout = KVLayout("a" * 64, "float32", 2, 1, 8, page_size)
    scheduler.vocab_size = 32
    scheduler.eos_token_id = 31
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.table_manager = TableManager(2, scheduler.engine.page_table)
    scheduler.cache_manager = CacheManager(32, page_size, scheduler.engine.page_table, "naive")
    if scheduler.role is WorkerRole.DECODE:
        scheduler.cache_manager.free_slots = scheduler.cache_manager.free_slots.flip(0)
    scheduler.decode_manager = DecodeManager(page_size)
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
    )
    scheduler.token_pool = scheduler.table_manager.token_pool
    scheduler.prefill_budget = 3
    scheduler._owned_req = None
    scheduler._allocated_len = 0
    return scheduler


def cpu_pinned_allocations():
    stack = ExitStack()
    original_empty, original_tensor = torch.empty, torch.tensor

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    def tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    stack.enter_context(patch("torch.empty", side_effect=empty))
    stack.enter_context(patch("torch.tensor", side_effect=tensor))
    stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
    return stack


def reference_tokens(prompt, max_tokens, ignore_eos=True):
    tp_info._TP_INFO = tp_info.DistributedInfo(0, 1)
    engine = TinyEngine(1)
    sequence = list(prompt)
    outputs = []
    for _ in range(max_tokens):
        req = Req(torch.tensor(sequence), 0, 0, 1, 0, SamplingParams(), None)
        engine.page_table[0, : len(sequence)] = torch.arange(len(sequence), dtype=torch.int32)
        batch = Batch([req], "prefill")
        batch.input_ids = torch.tensor(sequence, dtype=torch.int32)
        batch.positions = torch.arange(len(sequence), dtype=torch.int32)
        batch.out_loc = engine.page_table[0, : len(sequence)]
        token = int(engine.forward_batch(batch, None).next_tokens_cpu[0])
        outputs.append(token)
        sequence.append(token)
        if not ignore_eos and token == 31:
            break
    return outputs


def run_cpu_worker(role, ready, stop, delay=0.0):
    torch.set_num_threads(1)
    with cpu_pinned_allocations():
        scheduler = cpu_scheduler(role, delay=delay)
        with socket.create_server(("127.0.0.1", 0)) as listener:
            bind = ("127.0.0.1", 0) if role == "decode" else None
            with TcpTransferBackend(bind, timeout=2) as backend:
                ready.put(endpoint_name(*listener.getsockname()))
                serve_worker(scheduler, listener, backend, 2, stop)


def health(endpoint):
    from minisgl.disagg.control import endpoint_address

    with socket.create_connection(endpoint_address(endpoint), timeout=3) as sock:
        channel = Channel(sock, 3)
        channel.send({"op": "health"})
        return channel.receive()
