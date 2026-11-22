import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, ReceiveTicket
from minisgl.disagg.transfer import (
    KVTransferBackend,
    MockFabric,
    MockTransferBackend,
    TransferError,
)


class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.fabric = MockFabric()
        self.p = MockTransferBackend("prefill-0", self.fabric)
        self.d = MockTransferBackend("decode-0", self.fabric)
        self.layout = KVLayout("b" * 64, "float16", 2, 1, 2, 4)
        self.meta = KVTransferMetadata(str(uuid4()), self.layout, 5, (7, 3))
        self.destination = bytearray(self.meta.nbytes)
        self.source = bytes(range(self.meta.nbytes))

    def reserve(self):
        target = self.d.prepare_receive(self.meta, self.layout, self.destination)
        self.addCleanup(self.d.discard_receive, target)
        return target

    def test_two_endpoints_and_wire_ticket(self):
        backend: KVTransferBackend[bytes | bytearray, bytearray] = self.p
        target = ReceiveTicket.from_wire(self.reserve().to_wire())
        backend.send(KVTransferMetadata.from_wire(self.meta.to_wire()), self.source, target)
        self.d.finish_receive(target)
        self.assertEqual(self.destination, self.source)
        self.destination.extend(b"released")
        with self.assertRaises(TransferError):
            backend.send(self.meta, self.source, target)

    def test_reservation_pins_destination_size(self):
        target = self.reserve()
        with self.assertRaises(BufferError):
            self.destination.extend(b"cannot resize while registered")
        self.d.discard_receive(target)
        self.destination.extend(b"released")

    def test_reject_incompatible_layout_before_registering(self):
        for change in (
            {"model_fingerprint": "c" * 64},
            {"dtype": "bfloat16"},
            {"layers": 1},
            {"kv_heads": 2},
            {"head_dim": 4},
            {"page_size": 1},
            {"tp_size": 2},
        ):
            with self.subTest(change=change), self.assertRaises(TransferError):
                self.d.prepare_receive(self.meta, replace(self.layout, **change), self.destination)
        self.destination.extend(b"no registration leaked")

    def test_reject_size_and_destination_type(self):
        for destination in (bytearray(), bytearray(self.meta.nbytes + 1), b"readonly"):
            with self.subTest(size=len(destination)), self.assertRaises(TransferError):
                self.d.prepare_receive(self.meta, self.layout, destination)
        self.reserve()

    def test_reject_duplicate_reservations(self):
        self.reserve()
        with self.assertRaises(TransferError):
            self.d.prepare_receive(self.meta, self.layout, bytearray(self.meta.nbytes))
        with self.assertRaises(TransferError):
            self.d.prepare_receive(
                replace(self.meta, transfer_id=str(uuid4())), self.layout, self.destination
            )

    def test_receive_before_send_is_incomplete(self):
        target = self.reserve()
        with self.assertRaises(TransferError):
            self.d.finish_receive(target)
        self.assertEqual(self.destination, bytearray(self.meta.nbytes))
        self.p.send(self.meta, self.source, target)
        self.d.finish_receive(target)
        self.assertEqual(self.destination, self.source)

    def test_invalid_send_does_not_modify_destination(self):
        target = self.reserve()
        for source in (self.source[:-1], self.source + b"x", memoryview(self.source)):
            with self.subTest(length=len(source)), self.assertRaises(TransferError):
                self.p.send(self.meta, source, target)
        with self.assertRaises(TransferError):
            self.p.send(replace(self.meta, source_pages=(1, 2)), self.source, target)
        with self.assertRaises(TransferError):
            self.p.send(self.meta, self.source, replace(target, locator="forged"))
        self.assertEqual(self.destination, bytearray(self.meta.nbytes))
        self.p.send(self.meta, self.source, target)

    def test_second_send_rejected_without_overwrite(self):
        target = self.reserve()
        self.p.send(self.meta, self.source, target)
        with self.assertRaises(TransferError):
            self.p.send(self.meta, bytes(self.meta.nbytes), target)
        self.assertEqual(self.destination, self.source)

    def test_cancel_rejects_late_writes(self):
        target = self.reserve()
        self.d.discard_receive(target)
        self.d.discard_receive(target)
        with self.assertRaises(TransferError):
            self.p.send(self.meta, self.source, target)
        self.assertEqual(self.destination, bytearray(self.meta.nbytes))

    def test_old_ticket_cannot_write_or_discard_new_reservation(self):
        old = self.reserve()
        self.d.discard_receive(old)
        new = self.reserve()
        self.assertNotEqual(old.locator, new.locator)
        self.d.discard_receive(old)
        with self.assertRaises(TransferError):
            self.p.send(self.meta, self.source, old)
        self.p.send(self.meta, self.source, new)
        self.d.finish_receive(new)
        self.assertEqual(self.destination, self.source)

    def test_only_receiver_releases_registration(self):
        target = self.reserve()
        with self.assertRaises(TransferError):
            self.p.discard_receive(target)
        self.p.send(self.meta, self.source, target)
        with self.assertRaises(TransferError):
            self.p.finish_receive(target)
        self.d.finish_receive(target)

    def test_send_and_discard_race_drains_before_return(self):
        for _ in range(20):
            target = self.reserve()
            start = Barrier(2)

            def send():
                start.wait(timeout=5)
                try:
                    self.p.send(self.meta, self.source, target)
                    return True
                except TransferError:
                    return False

            def discard():
                start.wait(timeout=5)
                self.d.discard_receive(target)

            self.destination[:] = bytes(self.meta.nbytes)
            with ThreadPoolExecutor(max_workers=2) as pool:
                sent = pool.submit(send)
                discarded = pool.submit(discard)
                discarded.result(timeout=10)
                did_send = sent.result(timeout=10)
            self.assertEqual(self.destination, self.source if did_send else bytes(self.meta.nbytes))
            snapshot = bytes(self.destination)
            with self.assertRaises(TransferError):
                self.p.send(self.meta, self.source, target)
            self.assertEqual(self.destination, snapshot)
            self.destination.extend(b"x")
            self.destination.pop()


if __name__ == "__main__":
    unittest.main()
