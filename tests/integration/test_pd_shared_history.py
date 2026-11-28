"""Real shared segments/coordinator plus CPU KV copies and controlled CUDA fences."""

import multiprocessing
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from minisgl.disagg import launcher
from minisgl.disagg.control import StartRequest
from minisgl.disagg.dram_protocol import DramConfig
from minisgl.disagg.numa import host_id
from minisgl.disagg.prefix_hash import PrefixHashSpec, prepare_prefix
from minisgl.disagg.service import WorkerService
from minisgl.disagg.shared_dram_controller import SharedDramController, SharedPoolAdapter
from minisgl.disagg.shared_dram_directory import (
    SharedDramClient,
    SharedDramDirectory,
    SharedDramServer,
)
from minisgl.disagg.shared_dram_pool import (
    READY,
    STAGED,
    SharedDramManifest,
    SharedDramPool,
    SharedPage,
)
from minisgl.disagg.transfer import TransferBackendUnavailable
from pd_cache_reference import cached_round_trip
from pd_reference import cpu_scheduler, reference_tokens
from test_pd_shm import backend_pair, fake_cuda  # noqa: F401


def until(predicate, step=lambda: None, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        step()
        assert time.monotonic() < deadline, "shared history operation did not complete"
        time.sleep(0.001)


@contextmanager
def history(layout, counts=None, policy="lru"):
    with tempfile.TemporaryDirectory(prefix="msgl-history-") as root:
        manifest = SharedDramManifest(
            name=f"minisgl_history_test_{uuid4().hex}",
            host_id=host_id(),
            page_bytes=layout.page_size * layout.bytes_per_token,
            page_size=layout.page_size,
            cache_namespace="default",
            node_slots=counts or {0: 8, 1: 8},
            worker_numa={"p-0": 0, "p-1": 1, "d-0": 1},
            directory_socket=str(Path(root) / "dir.sock"),
            secret=uuid4().hex,
        )
        if 1 not in manifest.node_slots:
            manifest = SharedDramManifest(
                **{**asdict(manifest), "worker_numa": {"p-0": 0, "d-0": 0}}
            )
        manifest.validate()
        path = Path(root) / "manifest.json"
        path.write_text(manifest.to_json())
        owner = SharedDramPool(manifest, create=True, initialize=False)
        directory = SharedDramDirectory(owner, policy=policy)
        server = SharedDramServer(directory)
        try:
            yield manifest, path, owner, directory, server
        finally:
            server.close()
            owner.close()


@contextmanager
def services(counts=None):
    p = cpu_scheduler("prefill", max_running_req=8, prefill_budget=5, cache_type="naive")
    d = cpu_scheduler("decode", max_running_req=8)
    with history(p.layout, counts) as cache:
        manifest, path, _, _, _ = cache
        with backend_pair(p, d, d_numa=1) as (_, pb, db):
            for scheduler in (p, d):
                scheduler.pd_service_config = {
                    "dram": asdict(
                        DramConfig(capacity_bytes=manifest.registered_bytes, batch_pages=2)
                    ),
                    "dram_cache_mode": "shared_shm",
                    "shared_dram_manifest": str(path),
                    "batch_wait_ms": 0,
                }
            ps, ds = WorkerService(p, pb, 2), WorkerService(d, db, 2)
            try:
                yield ps, ds, cache
            finally:
                ps.close()
                ds.close()


@pytest.mark.parametrize("counts", [{0: 8, 1: 8}, {0: 16, 1: 1}])
def test_decode_generated_pages_are_restored_directly_by_later_prefill(counts):
    with services(counts) as (p, d, (_, _, owner, directory, _)):
        prompt = tuple(range(1, 11))
        first = cached_round_trip(p, d, prompt, 7)
        assert first.output == reference_tokens(prompt, 7)
        until(
            lambda: not p.contexts
            and not d.contexts
            and not p.dram_pool.operations
            and not d.dram_pool.operations,
            step=lambda: (p.step(), d.step()),
        )
        history_ids = prompt + tuple(first.output)
        hashes = tuple(prepare_prefix(history_ids, p.shared_spec).digests())
        assert len(directory.lookup_prefix(hashes)) == 4
        before = {n: bytes(region) for n, region in owner.regions.items()}
        observed = []
        original = p.dram_pool.view

        def view(page):
            observed.append((page.numa_id, page.slot_id, page.generation))
            assert page.lease_id
            return original(page)

        p.dram_pool.view = view
        calls = []
        forward = p.s._forward

        def record_forward(batch):
            calls.append(sum(r.extend_len for r in batch.batch.reqs))
            return forward(batch)

        p.s._forward = record_forward
        second = cached_round_trip(p, d, history_ids + (20,), 3)
        assert second.output == reference_tokens(history_ids + (20,), 3)
        assert second.ready["decode_prefix_len"] == 0  # D Radix toggle still controls P->D suffix.
        assert sum(calls) == 2  # First 16 tokens restored, only two newly computed by P.
        assert any(n == 1 for n, _, _ in observed)
        assert not hasattr(p.dram_pool, "buffer")  # No private/local Host payload cache.
        for node, slot, generation in observed:
            begin = slot * owner.manifest.stride + 256
            end = begin + owner.manifest.page_bytes
            if owner.header(node, slot)[0] == generation:
                assert bytes(owner.regions[node][begin:end]) == before[node][begin:end]
        until(lambda: directory.health()["readers"] == 0, step=lambda: (p.step(), d.step()))


def test_all_pinned_cache_skips_offload_without_stalling_decode():
    with services({0: 1, 1: 1}) as (p, d, (_, _, _, directory, _)):
        first = cached_round_trip(p, d, (1, 2, 3, 4, 5, 6, 7, 8), 4)
        assert len(first.output) == 4
        until(
            lambda: not p.contexts
            and not d.contexts
            and not p.dram_pool.operations
            and not d.dram_pool.operations,
            step=lambda: (p.step(), d.step()),
        )
        pinned = []
        for entry in list(directory.entries.values()):
            if entry.state == READY:
                pinned.extend(directory.pin_prefix([entry.page.page_hash], "test-hold"))
        assert len(pinned) == 2
        try:
            output = cached_round_trip(p, d, (11, 12, 13, 14, 15, 16, 17, 18), 6)
            assert output.output == reference_tokens(output.request.prompt_ids, 6)
            assert directory.stats["full"] > 0
            assert d.dram_controller.stats["skipped_pages"] > 0
        finally:
            directory.unpin(pinned, "test-hold")


def test_controller_writes_local_then_remote_and_reads_remote_without_host_copy():
    d = cpu_scheduler("decode", max_running_req=2)
    p = cpu_scheduler("prefill", max_running_req=2)
    spec = PrefixHashSpec.from_layout(asdict(d.layout))
    with history(d.layout, {0: 2, 1: 1}) as (manifest, _, owner, directory, _):
        config = DramConfig(capacity_bytes=manifest.registered_bytes, batch_pages=2)
        dp = SharedPoolAdapter(manifest, spec, d.device, numa_id=1)
        pp = SharedPoolAdapter(manifest, spec, p.device, numa_id=0)
        dc, pc = SharedDramController(d, dp, spec, config), SharedDramController(
            p, pp, spec, config
        )
        try:
            d.engine.kv_cache._kv_buffer.fill_(37)
            d.engine.page_table[0, :12] = torch.arange(12)
            p.engine.page_table[0, :12] = torch.arange(12)
            request = StartRequest(str(uuid4()), 0, (0, 1, 2, 3), 10, True)
            ctx = SimpleNamespace(
                request=request,
                session=SimpleNamespace(cancelled=threading.Event()),
                req=SimpleNamespace(cached_len=12, input_ids=torch.arange(12), table_idx=0),
                dram=dc.begin(request, decode=True),
            )
            until(
                lambda: directory.health()["ready_pages"] == 3,
                step=lambda: (dc.progress(), dc.offload(ctx)),
            )
            prepared = prepare_prefix(tuple(range(12)), spec)
            located = directory.lookup_prefix(tuple(prepared.digests()))
            assert [r["numa_id"] for r in located] == [1, 0, 0]
            pages = pp.client.pin_prefix(tuple(prepared.digests()))
            host_before = {n: bytes(region) for n, region in owner.regions.items()}
            task = pc.restore(0, 0, pages)
            until(lambda: task.done, step=lambda: (pc.advance_restore(task), pc.progress()))
            assert host_before == {n: bytes(region) for n, region in owner.regions.items()}
            for layer in range(p.layout.layers):
                assert torch.all(p.engine.kv_cache.k_cache(layer).view(-1, 8)[:12] == 37)
                assert torch.all(p.engine.kv_cache.v_cache(layer).view(-1, 8)[:12] == 37)
            until(lambda: directory.health()["readers"] == 0, step=pc.progress)
        finally:
            dc.close()
            pc.close()


def test_h2d_event_protects_shared_pages_through_request_cancellation():
    p = cpu_scheduler("prefill", max_running_req=2)
    spec = PrefixHashSpec.from_layout(asdict(p.layout))
    with history(p.layout, {0: 1, 1: 1}) as (manifest, _, owner, directory, _):
        hashes = tuple(prepare_prefix(tuple(range(8)), spec).digests())
        for h in hashes:
            record = directory.allocate_write(h, 1, "seed")
            page = SharedPage.from_object(record["page"])
            owner.view(page).zero_()
            directory.commit_write(page.numa_id, page.slot_id, page.generation, page.page_hash)
        pool = SharedPoolAdapter(manifest, spec, p.device, numa_id=0)
        controller = SharedDramController(
            p, pool, spec, DramConfig(capacity_bytes=8192, batch_pages=1)
        )
        gate = threading.Event()

        class Event:
            def query(self):
                return gate.is_set()

        controller.events["read"] = [Event(), Event()]
        try:
            p.engine.page_table[0, :8] = torch.arange(8)
            pages = pool.client.pin_prefix(hashes)
            task = controller.restore(0, 0, pages)
            controller.advance_restore(task)
            controller.abandon_restore(task)  # Cancel: unsent page can unpin now.
            controller.progress()
            until(lambda: directory.health()["readers"] == 1, step=controller.progress)
            assert not task.done
            held = pages[0]
            assert owner.header(held.numa_id, held.slot_id) == (held.generation, READY)
            assert directory.entries[held.page_hash].refcount == 1
            gate.set()
            until(
                lambda: task.done and directory.health()["readers"] == 0, step=controller.progress
            )
        finally:
            gate.set()
            controller.progress()
            controller.close()


@pytest.mark.parametrize("counts", [{0: 2}, {0: 1, 1: 1}])
def test_completed_d2h_event_is_required_before_shared_page_becomes_visible(counts):
    d = cpu_scheduler("decode", max_running_req=2)
    spec = PrefixHashSpec.from_layout(asdict(d.layout))
    with history(d.layout, counts) as (manifest, _, owner, directory, _):
        pool = SharedPoolAdapter(manifest, spec, d.device, numa_id=0)
        controller = SharedDramController(
            d, pool, spec, DramConfig(capacity_bytes=manifest.registered_bytes)
        )
        gate = threading.Event()

        class Event:
            def query(self):
                return gate.is_set()

        controller.events["write"] = [Event(), Event()]
        request = StartRequest(str(uuid4()), 0, (1, 2, 3, 4), 2, True)
        ctx = SimpleNamespace(
            request=request,
            session=SimpleNamespace(cancelled=threading.Event()),
            req=SimpleNamespace(cached_len=4, input_ids=torch.arange(1, 5), table_idx=0),
            dram=controller.begin(request, decode=True),
        )
        d.engine.page_table[0, :4] = torch.arange(4)
        d.engine.kv_cache._kv_buffer.fill_(29)
        hashes = tuple(prepare_prefix(request.prompt_ids, spec).digests())
        try:
            until(
                lambda: bool(controller.pending),
                step=lambda: (controller.progress(), controller.offload(ctx)),
            )
            page = controller.pending[0].pages[0]
            assert owner.header(page.numa_id, page.slot_id) == (page.generation, STAGED)
            controller.progress()
            assert not directory.lookup_prefix(hashes)
            assert not pool.client.pin_prefix(hashes)
            assert ctx.dram.pending_writes == 1
            gate.set()
            until(lambda: bool(directory.lookup_prefix(hashes)), step=controller.progress)
            assert owner.header(page.numa_id, page.slot_id) == (page.generation, READY)
            assert ctx.dram.pending_writes == 0
        finally:
            gate.set()
            controller.progress()
            controller.close()


def test_generation_is_rechecked_before_unpin_and_model_domains_are_isolated():
    s = cpu_scheduler("prefill")
    spec = PrefixHashSpec.from_layout(asdict(s.layout))
    with history(s.layout) as (manifest, _, owner, directory, _):
        pool = SharedPoolAdapter(manifest, spec, s.device, numa_id=0)
        key = "c" * 64
        page = SharedPage.from_object(directory.allocate_write(key, 0, "seed")["page"])
        directory.commit_write(page.numa_id, page.slot_id, page.generation, key)
        leases = pool.client.pin_prefix([key])
        wrong_domain = SharedDramClient(manifest, domain_id="d" * 64)
        try:
            with pytest.raises(ValueError, match="rejected"):
                wrong_domain.call("health")
            owner.set_header(page.numa_id, page.slot_id, page.generation + 1, READY)
            with pytest.raises(TransferBackendUnavailable, match="generation"):
                pool.release(leases)
            assert directory.health()["readers"] == 1
        finally:
            owner.set_header(page.numa_id, page.slot_id, page.generation, READY)
            pool.release(leases)
            pool.close()
            wrong_domain.disconnect()


def _lost_owner(encoded, hashes, write, pipe):
    manifest = SharedDramManifest.from_json(encoded)
    client = SharedDramClient(manifest)
    result = client.allocate(hashes, 0) if write else client.pin_prefix(hashes)
    assert len(result) == 1
    client.disconnect()  # A socket can close while a GPU would still own these pages.
    pipe.send(True)
    pipe.recv()


@pytest.mark.parametrize("write", [False, True])
def test_disconnected_live_process_retains_ownership_but_crash_is_reaped(write):
    s = cpu_scheduler("prefill")
    with history(s.layout, {0: 1, 1: 1}) as (manifest, _, _, directory, server):
        key = "a" * 64
        if not write:
            page = SharedPage.from_object(directory.allocate_write(key, 0, "seed")["page"])
            directory.commit_write(page.numa_id, page.slot_id, page.generation, key)
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_lost_owner, args=(manifest.to_json(), [key], write, child))
        try:
            process.start()
            child.close()
            assert parent.poll(15) and parent.recv()
            until(lambda: directory.health()["quarantined_pages" if write else "readers"] == 1)
            server.reap()
            assert directory.health()["quarantined_pages" if write else "readers"] == 1
            process.terminate()
            process.join(10)
            until(
                lambda: directory.health()["quarantined_pages" if write else "readers"] == 0,
                step=server.reap,
            )
            if write:
                assert not directory.entries
            else:
                assert directory.lookup_prefix([key])
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            parent.close()


def _cache_owner(encoded, manifest_path, pipe):
    manifest = SharedDramManifest.from_json(encoded)
    args, _ = launcher.parser().parse_known_args(["--model", "tiny"])
    args.shared_history_manifest_path = manifest_path
    supervisor = launcher.ClusterSupervisor(args, [], [], [], [])
    try:
        supervisor.history_pool = SharedDramPool(manifest, create=True, initialize=False)
        supervisor.history_server = SharedDramServer(SharedDramDirectory(supervisor.history_pool))
        signal.signal(signal.SIGTERM, lambda *_: supervisor.stop.set())
        pipe.send("ready")
        supervisor.stop.wait(20)
    finally:
        supervisor.close()
        pipe.close()


def test_launcher_sigterm_cleans_history_segments_socket_and_manifest():
    s = cpu_scheduler("prefill")
    root = Path(tempfile.mkdtemp(prefix="msgl-cleanup-"))
    manifest = SharedDramManifest(
        f"minisgl_cleanup_{uuid4().hex}",
        host_id(),
        s.layout.bytes_per_token * 4,
        4,
        "default",
        {0: 1, 1: 1},
        {"p-0": 0, "d-0": 1},
        str(root / "dir.sock"),
        uuid4().hex,
    )
    path = root / "manifest.json"
    path.write_text(manifest.to_json())
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_cache_owner, args=(manifest.to_json(), str(path), child))
    try:
        process.start()
        child.close()
        assert parent.poll(15) and parent.recv() == "ready"
        process.terminate()
        process.join(10)
        assert process.exitcode == 0
        assert not root.exists()
        assert all(
            not Path("/dev/shm", manifest.segment_name(n)[1:]).exists() for n in manifest.node_slots
        )
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        parent.close()


def test_lost_pin_response_replays_same_lease_without_double_refcount(monkeypatch):
    from minisgl.disagg import shared_dram_directory as module

    s = cpu_scheduler("prefill")
    with history(s.layout) as (manifest, _, _, directory, _):
        key = "b" * 64
        page = SharedPage.from_object(directory.allocate_write(key, 0, "seed")["page"])
        directory.commit_write(page.numa_id, page.slot_id, page.generation, key)
        lost = threading.Event()
        original = module.Channel.send

        def send(channel, message):
            result = message.get("result")
            if (
                threading.current_thread().name.startswith("shared-kv-rpc")
                and isinstance(result, list)
                and result
                and isinstance(result[0], dict)
                and result[0].get("lease_id")
                and not lost.is_set()
            ):
                lost.set()
                raise ConnectionError("lost pin receipt")
            return original(channel, message)

        monkeypatch.setattr(module.Channel, "send", send)
        client = SharedDramClient(manifest)
        try:
            pins = client.pin_prefix([key])
            assert lost.is_set() and len(pins) == 1 and directory.health()["readers"] == 1
            client.unpin(pins)
            assert directory.health()["readers"] == 0
        finally:
            client.close(drained=True)
