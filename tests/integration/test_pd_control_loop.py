"""Exercise real concurrent handlers and PDClient with in-memory channel framing.

No OS sockets are created. Real TCP framing/network acceptance remains separate.
"""

import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from minisgl.disagg import client as client_module
from minisgl.disagg import service as service_module
from minisgl.disagg.client import PDClient
from pd_reference import reference_tokens
from test_pd_batching import cluster, fake_cuda  # noqa: F401

_EOF = object()


class MemorySocket:
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.outgoing.put(_EOF)

    def shutdown(self, *args):
        self.close()
        self.incoming.put(_EOF)


class MemoryListener:
    def __init__(self):
        self.accepts = queue.Queue()

    def settimeout(self, timeout):
        pass

    def connect(self):
        inbound, outbound = queue.Queue(), queue.Queue()
        self.accepts.put(MemorySocket(inbound, outbound))
        return MemorySocket(outbound, inbound)

    def accept(self):
        try:
            return self.accepts.get(timeout=0.02), None
        except queue.Empty as exc:
            raise socket.timeout() from exc


class MemoryChannel:
    def __init__(self, sock, timeout, cancel=None):
        self.socket, self.timeout, self.cancel = sock, timeout, cancel

    def send(self, message):
        if self.socket.closed:
            raise ConnectionError("memory channel is closed")
        self.socket.outgoing.put(message)

    def receive(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.cancel is not None and self.cancel.is_set():
                raise InterruptedError("request cancelled")
            try:
                message = self.socket.incoming.get(timeout=0.01)
            except queue.Empty:
                continue
            if message is _EOF:
                raise ConnectionError("memory peer closed")
            return message
        raise TimeoutError("memory control message timed out")


def readable(readers, writers, errors, timeout):
    ready = [sock for sock in readers if not sock.incoming.empty()]
    if not ready:
        time.sleep(min(timeout, 0.001))
    return ready, [], []


@pytest.mark.parametrize("cancel_one", [False, True])
def test_worker_control_connections_continue_while_gpu_batches_run(monkeypatch, cancel_one):
    prompts = [(1, 2), (3, 4, 5), (6, 7), (8, 9, 10)]
    expected = [reference_tokens(prompt, 6) for prompt in prompts]
    with cluster(rows=8) as (p, decoders, _):
        d = decoders[0]
        p.batch_wait = 0.02
        p.s.engine.delay = d.s.engine.delay = 0.01
        listeners = {("127.0.0.1", 4001): MemoryListener(), ("127.0.0.1", 4002): MemoryListener()}
        monkeypatch.setattr(
            client_module,
            "socket",
            SimpleNamespace(
                create_connection=lambda address, timeout: listeners[address].connect()
            ),
        )
        monkeypatch.setattr(client_module, "Channel", MemoryChannel)
        monkeypatch.setattr(service_module, "Channel", MemoryChannel)
        monkeypatch.setattr(service_module, "select", SimpleNamespace(select=readable))
        stop, errors = threading.Event(), []

        def run(service, listener):
            try:
                service.run(listener, stop)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=(service, listener), daemon=True)
            for service, listener in zip((p, d), listeners.values())
        ]
        for thread in threads:
            thread.start()
        client = PDClient("127.0.0.1:4001", "127.0.0.1:4002", timeout=3)
        barrier = threading.Barrier(len(prompts))

        def generate(index):
            barrier.wait(timeout=2)
            iterator = client.stream(prompts[index], 6, ignore_eos=True)
            try:
                if cancel_one and index == 0:
                    return [next(iterator)]
                return list(iterator)
            finally:
                iterator.close()

        try:
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                outputs = list(executor.map(generate, range(len(prompts))))
            if cancel_one:
                expected[0] = expected[0][:1]
            assert outputs == expected
            for address in listeners:
                with listeners[address].connect() as sock:
                    channel = MemoryChannel(sock, 1)
                    channel.send({"op": "health"})
                    assert channel.receive()["status"] == "ready"
            assert p.stats["max_prefill_batch"] > 1
            assert d.stats["max_decode_batch"] > 1
            assert client.admission.requests == client.admission.tokens == 0
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=4)
            assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert not p.contexts and not d.contexts
        assert not d.backend._leases
        assert p.s.health()["free_pages"] == d.s.health()["free_pages"] == 32
