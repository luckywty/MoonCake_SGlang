"""Independent CPU regression for 4d15539: identities, journals and routing hints."""

import copy
from types import SimpleNamespace

import pytest
import torch
from layered_reference import wait_for
from minisgl.disagg.prefix_events import SYNC_PAGES
from minisgl.disagg.prefix_hash import PrefixHashSpec, hash_page, prepare_prefix
from minisgl.disagg.prefix_index import PrefixIndex
from minisgl.disagg.prompt_prepare import prepare_with_tokenizer
from minisgl.disagg.routing_policy import RoutingConfig
from pd_cache_reference import (  # noqa: F401
    cpu_radix_compare,
    prefix_owner,
    publish_prefix,
    sync_prefix,
)
from pd_reference import reference_tokens
from test_pd_batching import fake_cuda  # noqa: F401
from test_pd_routing import running_cluster

pytestmark = pytest.mark.usefixtures("cpu_radix_compare")


def test_hash_chain_matches_fixed_interoperability_vectors_and_ignores_partial_page():
    owner = prefix_owner()
    prepared = prepare_prefix(tuple(range(1, 10)), owner.spec)
    # Fixed vectors computed independently with SHA-256 and little-endian int32.
    assert (
        owner.spec.root.hex() == "81ed4bbc91791e67d14957fe6bae3d7e904f8306926e7bd876b5141983f6d4e8"
    )
    assert [h.hex() for h in prepared.digests()] == [
        "bd5ffa09a723e240cb65e312926de057847ea0a8762c7b346ebe11e6af3bd9d6",
        "ca8fa29c234ab4aeb826e5e2358d887b604ee78482cdce56b5d1b8f32c06a0d3",
    ]
    assert prepared.page_hashes == prepare_prefix(tuple(range(1, 9)), owner.spec).page_hashes
    changed = tuple(prepare_prefix((1, 2, 3, 4, 99, 6, 7, 8), owner.spec).digests())
    original = tuple(prepared.digests())
    assert changed[0] == original[0] and changed[1] != original[1]


@pytest.mark.parametrize(
    "change",
    [
        {"model_fingerprint": "b" * 64},
        {"dtype": "bfloat16"},
        {"head_dim": 16},
        {"layers": 3},
        {"page_size": 1},
        {"tp_size": 2},
        {"tp_size": 2, "tp_rank": 1},
    ],
)
def test_layout_changes_isolate_the_entire_hash_domain(change):
    owner = prefix_owner()
    alternate = PrefixHashSpec.from_layout({**owner.node.health["layout"], **change})
    assert alternate.domain_id != owner.spec.domain_id
    assert (
        prepare_prefix(tuple(range(8)), alternate).page_hashes
        != prepare_prefix(tuple(range(8)), owner.spec).page_hashes
    )


def test_namespace_and_worker_identity_have_distinct_lifetimes():
    a, b = prefix_owner("p-0"), prefix_owner("p-1")
    assert a.node.health["instance_id"] != b.node.health["instance_id"]
    assert a.spec.root == b.spec.root
    tenant = prefix_owner(namespace="another-model-deployment")
    assert tenant.spec.root != a.spec.root


@pytest.mark.parametrize("bad", [True, -1, 2**31, 1.0, "1"])
def test_hashing_rejects_non_int32_token_identities(bad):
    spec = prefix_owner().spec
    with pytest.raises(ValueError):
        prepare_prefix((1, 2, 3, bad), spec)
    with pytest.raises(ValueError):
        hash_page(spec.root, (1, 2, 3, bad), 4)


def test_radix_publication_is_committed_and_split_eviction_keeps_ancestors():
    owner = prefix_owner()
    ids = torch.arange(12, dtype=torch.int32)
    handle = owner.cache.insert_prefix(ids, ids).handle
    assert owner.publisher.health()["pages"] == 0
    owner.cache.publish_handle(handle)
    assert owner.publisher.health()["pages"] == 3
    seq = owner.publisher.seq
    parent = owner.cache.match_prefix(ids[:8]).cuda_handle
    assert parent.cached_len == 8 and owner.publisher.seq == seq
    owner.cache.publish_handle(handle)
    assert owner.publisher.seq == seq  # Structural split/duplicate publish adds no page.
    assert len(owner.cache.evict(4)) == 4
    assert sorted(p.depth for p in owner.publisher.records.values()) == [1, 2]
    owner.cache.lock_handle(parent)
    with pytest.raises(AssertionError):
        owner.cache.evict(8)
    owner.cache.lock_handle(parent, unlock=True)
    owner.cache.evict(8)
    assert not owner.publisher.records and not owner.publisher.by_node


def test_paginated_snapshot_stays_cold_until_post_snapshot_evictions_are_applied():
    count = SYNC_PAGES + 3
    owner = prefix_owner(page=1, tokens=tuple(range(count)))
    index = PrefixIndex()
    index.observe(owner.node.worker_id, owner.node.health)
    state, request = index.request(owner.node.worker_id)
    reply = owner.publisher.sync(request)
    assert len(reply["pages"]) == SYNC_PAGES and not reply["complete"]
    index.apply(owner.node.worker_id, state, reply)
    prepared = prepare_prefix(tuple(range(count)), owner.spec)
    assert not index.hits([owner.node], prepared)
    owner.cache.match_prefix(torch.arange(count - 2, dtype=torch.int32))
    owner.cache.evict(2)
    _, request = index.request(owner.node.worker_id)
    frozen = owner.publisher.sync(request)
    assert frozen["seq"] == reply["seq"] and len(frozen["pages"]) == 3
    index.apply(owner.node.worker_id, state, frozen)
    assert not index.hits([owner.node], prepared)
    sync_prefix(index, owner)
    assert index.hits([owner.node], prepared)[owner.node.worker_id] == count - 2


def test_journal_overflow_forces_a_new_snapshot_instead_of_reusing_stale_pages():
    owner = prefix_owner(tokens=tuple(range(12)), event_limit=2)
    index = PrefixIndex()
    state = sync_prefix(index, owner)
    old = prepare_prefix(tuple(range(12)), owner.spec)
    owner.cache.evict(12)
    publish_prefix(owner, tuple(range(100, 112)))
    _, request = index.request(owner.node.worker_id)
    reply = owner.publisher.sync(request)
    assert reply["status"] == "prefix_reset"
    index.apply(owner.node.worker_id, state, reply)
    assert not state.ready and not state.pages
    sync_prefix(index, owner)
    assert index.hits([owner.node], old)[owner.node.worker_id] == 0
    new = prepare_prefix(tuple(range(100, 112)), owner.spec)
    assert index.hits([owner.node], new)[owner.node.worker_id] == 12


@pytest.mark.parametrize("fault", ["gap", "wrong_instance", "wrong_domain", "unknown_op"])
def test_bad_delta_clears_the_hint_and_can_rebuild(fault):
    owner = prefix_owner(tokens=tuple(range(8)))
    index = PrefixIndex()
    state = sync_prefix(index, owner)
    publish_prefix(owner, tuple(range(12)))
    _, request = index.request(owner.node.worker_id)
    reply = copy.deepcopy(owner.publisher.sync(request))
    if fault == "gap":
        reply["events"][0]["seq"] += 1
    elif fault == "wrong_instance":
        reply["instance_id"] = "old-instance"
    elif fault == "wrong_domain":
        reply["domain_id"] = "0" * 64
    else:
        reply["events"][0]["op"] = "unknown"
    assert not index.apply(owner.node.worker_id, state, reply)
    assert not state.ready and not state.pages
    assert owner.node.healthy
    sync_prefix(index, owner)
    assert (
        index.hits([owner.node], prepare_prefix(tuple(range(12)), owner.spec))[owner.node.worker_id]
        == 12
    )


def test_restart_ignores_late_old_snapshot_and_stale_or_unhealthy_hints(monkeypatch):
    owner = prefix_owner(tokens=tuple(range(8)))
    index = PrefixIndex(max_age=1)
    old = sync_prefix(index, owner)
    _, request = index.request(owner.node.worker_id)
    reply = owner.publisher.sync(request)
    restarted = prefix_owner(owner.node.worker_id)
    index.observe(restarted.node.worker_id, restarted.node.health)
    assert not index.apply(owner.node.worker_id, old, reply)
    assert not index.states[owner.node.worker_id].pages
    state = sync_prefix(index, restarted)
    publish_prefix(restarted, tuple(range(8)))
    sync_prefix(index, restarted)
    prepared = prepare_prefix(tuple(range(8)), restarted.spec)
    assert index.hits([restarted.node], prepared)[restarted.node.worker_id] == 8
    with monkeypatch.context() as clock:
        clock.setattr(
            "minisgl.disagg.prefix_index.time",
            SimpleNamespace(monotonic=lambda: state.updated_at + 2),
        )
        assert not index.hits([restarted.node], prepared)
    assert index.hits([restarted.node], prepared)[restarted.node.worker_id] == 8
    restarted.node.healthy = False
    assert not index.hits([restarted.node], prepared)
    index.forget(restarted.node.worker_id)
    assert index.request(restarted.node.worker_id) is None and not index.lru


def test_directory_capacity_drops_orphan_chains_without_dropping_other_owners():
    a = prefix_owner("p-0", tokens=tuple(range(12)))
    b = prefix_owner("p-1", tokens=tuple(range(12)))
    index = PrefixIndex(max_entries=4, max_worker_entries=3)
    sync_prefix(index, a)
    sync_prefix(index, b)
    prepared = prepare_prefix(tuple(range(12)), a.spec)
    assert len(index.lru) <= 4 and not index.states[a.node.worker_id].pages
    assert index.hits([a.node, b.node], prepared) == {a.node.worker_id: 0, b.node.worker_id: 12}
    bounded = PrefixIndex(max_worker_entries=2)
    sync_prefix(bounded, a)
    assert len(bounded.states[a.node.worker_id].pages) <= 2
    assert bounded.hits([a.node], prepared)[a.node.worker_id] == 0


@pytest.mark.parametrize("chat", [False, True])
def test_preparation_preserves_tokenizer_semantics_and_hashes_the_returned_ids(chat):
    from unittest.mock import Mock

    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(9))
    tokenizer.apply_chat_template.return_value = list(range(1, 10))
    messages = [{"role": "user", "content": "hello"}] if chat else None
    spec = prefix_owner().spec
    prepared = prepare_with_tokenizer(tokenizer, messages, None if chat else "hello", spec)
    assert prepared.prefix == prepare_prefix(prepared.token_ids, spec)
    if chat:
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=True, add_generation_prompt=True
        )
        tokenizer.encode.assert_not_called()
    else:
        tokenizer.encode.assert_called_once_with("hello")
        tokenizer.apply_chat_template.assert_not_called()
    assert prepare_with_tokenizer(tokenizer, messages, "hello", None).prefix is None


def test_routed_hbm_index_selects_the_owner_and_stale_hint_never_skips_real_compute(monkeypatch):
    config = RoutingConfig(policy="cache_aware", cache_min_tokens=1, prefix_sync_interval=0.01)
    with running_cluster(monkeypatch, routing_config=config, prefill_cache_type="radix") as (
        client,
        services,
        _,
        frames,
    ):
        prompt = tuple(range(1, 11))
        assert client.generate(prompt, 3, ignore_eos=True) == reference_tokens(prompt, 3)
        wait_for(lambda: all(not s.contexts for s in services))
        owner = next(s for s in services[:2] if s.s.cache_manager.prefix_cache.size_info.total_size)
        spec = owner.prefix_publisher.spec
        followup = prompt + (20, 21)
        prepared = prepare_prefix(followup, spec)
        wait_for(
            lambda: client.pool.prefix_index.hits(client.pool.healthy("prefill"), prepared).get(
                owner.s.worker_id
            )
            == 8
        )
        starts = []
        original = owner._prepare

        def prepare(batch):
            starts.extend(r.cached_len for r in batch.reqs)
            return original(batch)

        monkeypatch.setattr(owner, "_prepare", prepare)
        assert client.generate(followup, 3, ignore_eos=True) == reference_tokens(followup, 3)
        assert starts == [8]
        wait_for(lambda: all(not s.contexts for s in services))
        cache = owner.s.cache_manager
        cache._free(cache.prefix_cache.evict(cache.prefix_cache.size_info.evictable_size))
        monkeypatch.setattr(
            client, "_prefix_matches", lambda *args, **kwargs: {owner.s.worker_id: 8}
        )
        assert client.generate(followup, 3, ignore_eos=True) == reference_tokens(followup, 3)
        assert starts == [8, 0]
        assert any(frame.get("message", {}).get("op") == "prefix_sync" for frame in frames)
        assert not any(frame.get("message", {}).get("op") == "prefix_match" for frame in frames)
        assert all(
            frame["message"]["decode_prefix_len"] == 0
            for frame in frames
            if frame.get("message", {}).get("op") == "prefill"
        )
