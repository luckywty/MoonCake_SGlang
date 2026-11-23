"""CPU adapter for the real layer sender, runtime and socket control protocol.

CUDA events and native DMA are controllable fakes. This validates ownership and
ordering, not actual GPU stream execution, RDMA visibility, or performance.
"""

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from minisgl.disagg.layered_mooncake import LayeredMooncakeBackend
from minisgl.disagg.layered_protocol import LayeredReceiveTicket


class FakeStream:
    def __init__(self, device=None):
        pass

    def synchronize(self):
        pass

    def wait_event(self, event):
        assert event.recorded


class FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        assert threading.current_thread().name == "pd-layer-sender"


class FakeCuda:
    info = {"fake": True}

    def __init__(self):
        self.visible = 0

    def activate(self):
        return nullcontext()

    def prepare_buffer(self, tensor):
        pass

    def make_visible(self):
        self.visible += 1


class FakeTicket:
    def __init__(self, engine, source, target, length):
        self.engine, self.source, self.target, self.length = engine, source, target, length
        self.snapshot = source[:length].clone()
        self.drained = False
        self._status = "unknown"

    def drain(self, timeout_ms):
        assert threading.current_thread().name == "pd-layer-sender" or self.engine.cleaning
        if self.drained:
            return self._status
        if self.engine.gate is not None and not self.engine.gate.is_set():
            return "unknown"
        # A ring overwrite while DMA owns this slot is observable even though
        # the fake's final copy itself executes synchronously on the CPU.
        assert torch.equal(self.source[: self.length], self.snapshot), "in-flight slot overwritten"
        self.drained = True
        if self.engine.fail:
            self._status = "failed"
        else:
            self.target[: self.length].copy_(self.source[: self.length])
            self._status = "complete"
        return self._status


class FakeEngine:
    registrations = {}
    next_port = 45000

    def __init__(self, gate=None, fail=False):
        self.gate, self.fail = gate, fail
        self.submissions = []
        self.register_calls = 0
        self.unregister_calls = 0
        self.cleaning = False

    def register_memory(self, address, length, device, remote):
        assert address not in self.registrations
        self.register_calls += 1
        return 0

    def unregister_memory(self, address):
        self.unregister_calls += 1
        self.registrations.pop(address)
        return 0

    def submit_scatter_write(
        self,
        session,
        sources,
        capacities,
        targets,
        remote_capacities,
        source_offsets,
        target_offsets,
        lengths,
    ):
        source = self.registrations[sources[0]]
        target = next(
            tensor[targets[0] - address :]
            for address, tensor in self.registrations.items()
            if address <= targets[0] < address + tensor.numel()
        )
        length = lengths[0][0]
        assert length <= capacities[0] and length <= remote_capacities[0]
        assert source_offsets == target_offsets == [[0]]
        ticket = FakeTicket(self, source, target, length)
        self.submissions.append(ticket)
        return ticket


class CpuLayeredBackend(LayeredMooncakeBackend):
    def __init__(self, *args, gate=None, fail=False, **kwargs):
        self.fake_engine = FakeEngine(gate, fail)
        super().__init__(*args, **kwargs)

    def bind_device(self, device):
        self.runtime.device = torch.device("cpu")
        self.runtime.cuda = FakeCuda()
        self.runtime.module = SimpleNamespace(
            ScatterTransferCompletionStatus=SimpleNamespace(
                COMPLETED="complete",
                FAILED_DRAINED="failed",
                COMPLETION_UNKNOWN="unknown",
            )
        )
        self.runtime.engine = self.fake_engine
        FakeEngine.next_port += 1
        self.runtime.session = f"127.0.0.1:{FakeEngine.next_port}"
        original = self.runtime.register_tensor

        def register(tensor, key):
            buffer = original(tensor, key)
            FakeEngine.registrations[buffer.address] = tensor
            return buffer

        self.runtime.register_tensor = register
        self._bound.set()

    def cleanup_fake(self):
        # Test teardown only: finish all fake DMA before dropping CPU backing
        # storage. This is not a production recovery path or a remote proof.
        self.fake_engine.cleaning = True
        if self.fake_engine.gate is not None:
            self.fake_engine.gate.set()
        for operation in list(self.runtime._operations.values()):
            self.runtime.poll(operation, timeout_ms=0)
        for buffer in list(self.runtime._buffers.values()):
            self.runtime.mark_safe(buffer)
        self.close()


class MemoryControl:
    """Replace only socket framing in ownership tests, without opening a port."""

    def __init__(self, receiver, lose_receipt=None):
        self.receiver = receiver
        self.lose_receipt = lose_receipt
        self.message = None

    def send(self, message):
        self.message = message
        if message["op"] == "layer_open":
            import json

            ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
            self.reply = self.receiver.open_layers(ticket, message["source"])
        else:
            self.reply = self.receiver._settle_layers(message)

    def receive(self):
        if (
            self.message["op"] == "layer_terminal"
            and self.lose_receipt is not None
            and not self.lose_receipt.is_set()
        ):
            self.lose_receipt.set()
            raise ConnectionError("injected lost terminal receipt")
        return dict(self.reply)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.005)
