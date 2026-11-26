"""CPU adapters for cache tests; keep cache ownership, hashing and attention real."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import minisgl.core as core
import minisgl.distributed.info as tp_info
import pytest
import torch
from minisgl.disagg.control import StartRequest
from minisgl.disagg.prefix_events import PrefixPublisher
from minisgl.disagg.prefix_hash import PrefixHashSpec
from minisgl.disagg.protocol import KVLayout
from minisgl.disagg.service import Session
from minisgl.kvcache.radix_cache import RadixPrefixCache, RadixTreeNode
from test_pd_batching import drive, receive


@pytest.fixture
def cpu_radix_compare(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core._GLOBAL_CTX)
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info._TP_INFO)

    def compare(node, ids):
        length = 0
        for a, b in zip(node._key.tolist(), ids.tolist()):
            if a != b:
                break
            length += 1
        return length

    monkeypatch.setattr(RadixTreeNode, "get_match_len", compare)


class RecordingSession(Session):
    def __post_init__(self):
        super().__post_init__()
        self.messages = []

    def emit(self, message, *, final=False):
        self.messages.append(message)
        super().emit(message, final=final)


def start_cached_request(p, d, prompt, tokens):
    request = StartRequest(str(uuid4()), 0, tuple(prompt), tokens, True)
    ds = RecordingSession()
    d.submit(ds, {"op": "reserve", "request": request.to_object()})
    d.step()
    messages = receive(ds)
    assert len(messages) == 1 and messages[0]["status"] == "ready", messages
    ready = messages[0]
    ps = RecordingSession()
    p.submit(
        ps,
        {
            "op": "prefill",
            "request": request.to_object(),
            "ticket": ready["ticket"],
            "decode_prefix_len": ready["decode_prefix_len"],
        },
    )
    return SimpleNamespace(
        request=request, ps=ps, ds=ds, ready=ready, item=(request, ds, ready["ticket"])
    )


def complete_cached_requests(p, d, requests):
    return drive(p, [d], [r.item for r in requests], [r.ps for r in requests])


def cached_round_trip(p, d, prompt, tokens):
    result = start_cached_request(p, d, prompt, tokens)
    result.output = complete_cached_requests(p, d, [result])[result.request.transfer_id]
    result.handoff = next(m["handoff"] for m in result.ps.messages if m["status"] == "handoff")
    return result


def prefix_owner(
    worker_id="p-0", *, page=4, tokens=(), namespace="default", max_pages=65536, event_limit=32768
):
    core._GLOBAL_CTX = core.Context(page)
    layout = asdict(KVLayout("a" * 64, "float32", 2, 1, 8, page))
    spec = PrefixHashSpec.from_layout(layout, namespace)
    instance = str(uuid4())
    publisher = PrefixPublisher(spec, instance, max_pages, event_limit)
    cache = RadixPrefixCache(torch.device("cpu"))
    cache.page_observer = publisher
    node = SimpleNamespace(
        worker_id=worker_id,
        healthy=True,
        endpoint="worker:9000",
        health={
            "role": "prefill",
            "instance_id": instance,
            "layout": layout,
            "cache_namespace": namespace,
            "prefix_index": publisher.health(),
        },
    )
    owner = SimpleNamespace(cache=cache, publisher=publisher, node=node, spec=spec)
    if tokens:
        publish_prefix(owner, tokens)
    return owner


def publish_prefix(owner, tokens):
    result = owner.cache.insert_prefix(
        torch.tensor(tokens, dtype=torch.int32), torch.arange(len(tokens), dtype=torch.int32)
    )
    owner.cache.publish_handle(result.handle)
    return result.handle


def sync_prefix(index, owner):
    owner.node.health["prefix_index"] = owner.publisher.health()
    index.observe(owner.node.worker_id, owner.node.health)
    for _ in range(32):
        state, request = index.request(owner.node.worker_id)
        reply = json.loads(json.dumps(owner.publisher.sync(request)))
        index.apply(owner.node.worker_id, state, reply)
        if state.ready:
            return state
    raise AssertionError("prefix mirror did not catch up")
