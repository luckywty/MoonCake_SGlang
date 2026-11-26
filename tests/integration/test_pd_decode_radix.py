"""Independent CPU regression for 5e1e21e: D HBM sharing and suffix handoff."""

import json
import time
from unittest.mock import Mock

import pytest
import torch
from layered_reference import wait_for
from minisgl.disagg.layered_protocol import LayeredReceiveTicket
from minisgl.disagg.routing_policy import RoutingBucket, RoutingConfig, WorkerGroup
from pd_cache_reference import (
    cached_round_trip,
    complete_cached_requests,
    cpu_radix_compare,  # noqa: F401
    start_cached_request,
)
from pd_reference import reference_tokens
from test_pd_batching import cluster, fake_cuda  # noqa: F401
from test_pd_routing import running_cluster

pytestmark = pytest.mark.usefixtures("cpu_radix_compare")


def assert_idle_cache(service):
    assert not service.contexts
    assert service.s.table_manager.available_size == 8
    assert service.s.cache_manager.prefix_cache.size_info.protected_size == 0
    service.s.cache_manager.check_integrity()
    free = service.s.cache_manager.free_slots.tolist()
    assert len(free) == len(set(free))
    assert not service.backend._leases


@pytest.mark.parametrize("page", [1, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_multiturn_generated_prefix_transfers_exactly_the_reserved_suffix(page, enabled):
    prompt = tuple(range(1, 11))
    with cluster(page_size=page, decode_radix_cache=enabled) as (p, decoders, _):
        d = decoders[0]
        first = cached_round_trip(p, d, prompt, 7)
        assert first.output == reference_tokens(prompt, 7)
        assert first.ready["decode_prefix_len"] == 0
        cached = (len(prompt) + len(first.output) - 1) // page * page if enabled else 0
        assert d.s.cache_manager.prefix_cache.size_info.total_size == cached
        # The newly sampled last token has no KV until a later forward.
        history = prompt + tuple(first.output) + (20, 21)
        previous = len(p.backend.fake_engine.submissions)
        second = cached_round_trip(p, d, history, 3)
        assert second.output == reference_tokens(history, 3)
        assert second.ready["decode_prefix_len"] == cached
        metadata = second.handoff["kv"]
        assert metadata["prefix_len"] == cached and metadata["kv_tokens"] == len(history) - cached
        suffix_bytes = (len(history) - cached) * d.s.layout.bytes_per_token
        ticket = LayeredReceiveTicket.from_wire(json.dumps(second.ready["ticket"]).encode())
        assert ticket.nbytes == suffix_bytes and ticket.prefix_len == cached
        assert sum(t.length for t in p.backend.fake_engine.submissions[previous:]) == suffix_bytes
        assert d.stats["received_kv_tokens"] == len(prompt) + len(history) - cached
        for service in (p, d):
            assert_idle_cache(service)
        if not enabled:
            assert len(d.s.cache_manager.free_slots) == d.s.engine.num_pages


def test_full_decode_prefix_hit_has_no_payload_but_prefill_still_computes(monkeypatch):
    prompt = tuple(range(1, 13))
    with cluster(decode_radix_cache=True) as (p, decoders, _):
        d = decoders[0]
        cached_round_trip(p, d, prompt, 5)
        calls = Mock(wraps=p.s._forward)
        monkeypatch.setattr(p.s, "_forward", calls)
        sent = len(p.backend.fake_engine.submissions)
        received = d.stats["received_kv_tokens"]
        second = cached_round_trip(p, d, prompt, 3)
        assert second.output == reference_tokens(prompt, 3)
        assert second.ready["decode_prefix_len"] == len(prompt)
        assert second.ready["ticket"] is None
        assert second.handoff["kv"]["kv_tokens"] == 0
        assert len(p.backend.fake_engine.submissions) == sent
        assert d.stats["received_kv_tokens"] == received
        calls.assert_called_once()
        assert_idle_cache(p)
        assert_idle_cache(d)


def test_concurrent_identical_installs_adopt_canonical_pages_and_free_duplicates(monkeypatch):
    prompt = tuple(range(1, 11))
    with cluster(decode_radix_cache=True) as (p, decoders, _):
        d = decoders[0]
        installed = []
        original = d.s.cache_manager.cache_req

        def publish(req, *, finished, rebind=False):
            original(req, finished=finished, rebind=rebind)
            if rebind and not finished:
                installed.append(d.s.engine.page_table[req.table_idx, :8].clone())

        monkeypatch.setattr(d.s.cache_manager, "cache_req", publish)
        requests = [start_cached_request(p, d, prompt, 6) for _ in range(2)]
        assert all(r.ready["decode_prefix_len"] == 0 for r in requests)
        output = complete_cached_requests(p, d, requests)
        assert all(output[r.request.transfer_id] == reference_tokens(prompt, 6) for r in requests)
        assert len(installed) == 2 and torch.equal(installed[0], installed[1])
        assert d.stats["max_decode_batch"] == 2
        assert d.s.cache_manager.prefix_cache.size_info.total_size == 12
        assert len(d.s.cache_manager.free_slots) == 29
        assert_idle_cache(d)


def test_cancelled_suffix_reservation_preserves_cached_prefix_and_releases_private_pages():
    prompt = tuple(range(1, 11))
    with cluster(decode_radix_cache=True) as (p, decoders, _):
        d = decoders[0]
        cached_round_trip(p, d, prompt, 3)
        before = d.s.cache_manager.prefix_cache.size_info.total_size
        # A different page tail matches the first eight tokens only.
        pending = start_cached_request(p, d, prompt + (20, 21), 4)
        assert pending.ready["decode_prefix_len"] == 8
        assert d.s.cache_manager.prefix_cache.size_info.protected_size >= 8
        pending.ds.cancelled.set()
        pending.ps.cancelled.set()
        until = time.monotonic() + 3
        while (p.contexts or d.contexts) and time.monotonic() < until:
            p.step()
            d.step()
            time.sleep(0.001)
        assert d.s.cache_manager.prefix_cache.size_info.total_size == before
        assert_idle_cache(p)
        assert_idle_cache(d)


def test_install_failure_preserves_shared_prefix_then_a_retry_succeeds(monkeypatch):
    config = RoutingConfig(
        policy="round_robin",
        decode_policy="round_robin",
        buckets=(
            RoutingBucket(
                prefill=WorkerGroup(worker_ids=("p-0",)), decode=WorkerGroup(worker_ids=("d-0",))
            ),
        ),
    )
    with running_cluster(monkeypatch, routing_config=config, decode_radix_cache=True) as (
        client,
        services,
        _,
        _,
    ):
        prompt = tuple(range(1, 11))
        assert client.generate(prompt, 3, ignore_eos=True) == reference_tokens(prompt, 3)
        d = services[2]
        wait_for(lambda: not d.contexts)
        before = d.s.cache_manager.prefix_cache.size_info.total_size
        importer = d.s.engine.kv_cache.import_layered_tensor
        failed = Mock(side_effect=ValueError("injected cached suffix install failure"))
        monkeypatch.setattr(d.s.engine.kv_cache, "import_layered_tensor", failed)
        followup = prompt + (20, 21)
        with pytest.raises(RuntimeError, match="injected cached suffix install failure"):
            client.generate(followup, 4, ignore_eos=True)
        failed.assert_called_once()
        wait_for(lambda: all(not s.contexts for s in services))
        assert d.s.cache_manager.prefix_cache.size_info.total_size == before
        assert_idle_cache(d)
        assert not client.router.inflight
        monkeypatch.setattr(d.s.engine.kv_cache, "import_layered_tensor", importer)
        assert client.generate(followup, 4, ignore_eos=True) == reference_tokens(followup, 4)
        wait_for(lambda: all(not s.contexts for s in services))
        assert_idle_cache(d)


def test_routed_tcp_protocol_sends_only_the_real_decode_suffix(monkeypatch):
    config = RoutingConfig(
        policy="round_robin",
        decode_policy="round_robin",
        buckets=(
            RoutingBucket(
                prefill=WorkerGroup(worker_ids=("p-0",)), decode=WorkerGroup(worker_ids=("d-0",))
            ),
        ),
    )
    with running_cluster(
        monkeypatch, mode="tcp", routing_config=config, decode_radix_cache=True
    ) as (client, services, _, _):
        prompt = tuple(range(1, 11))
        output = client.generate(prompt, 7, ignore_eos=True)
        assert output == reference_tokens(prompt, 7)
        wait_for(lambda: all(not s.contexts for s in services))
        p, d = services[0], services[2]
        sender = Mock(wraps=p.backend.send)
        monkeypatch.setattr(p.backend, "send", sender)
        followup = prompt + tuple(output) + (20, 21)
        assert client.generate(followup, 3, ignore_eos=True) == reference_tokens(followup, 3)
        sender.assert_called_once()
        metadata, payload, _ = sender.call_args.args
        assert metadata.prefix_len == 16 and metadata.kv_tokens == len(followup) - 16
        assert len(payload) == metadata.nbytes == (len(followup) - 16) * d.s.layout.bytes_per_token
        assert d.stats["received_kv_tokens"] == len(prompt) + len(followup) - 16
        wait_for(lambda: all(not s.contexts for s in services))
        d.s.cache_manager.check_integrity()
        assert not client.router.inflight
