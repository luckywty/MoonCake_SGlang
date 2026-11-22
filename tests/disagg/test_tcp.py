import hashlib
import json
import multiprocessing as mp
import socket
import struct
import time
import unittest
from threading import Event
from uuid import uuid4

from minisgl.disagg.control import MAX_CONTROL_BYTES, CancelledError, Channel, endpoint_address
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, ReceiveTicket
from minisgl.disagg.tcp_backend import TcpTransferBackend
from minisgl.disagg.transfer import TransferError


def fixture():
    layout = KVLayout("a" * 64, "float16", 2, 1, 2, 4)
    return KVTransferMetadata(str(uuid4()), layout, 5, (7, 3))


def receive_process(meta_wire, ready, done, result):
    meta = KVTransferMetadata.from_wire(meta_wire)
    destination = bytearray(meta.nbytes)
    with TcpTransferBackend(("127.0.0.1", 0), timeout=3) as backend:
        ticket = backend.prepare_receive(meta, meta.layout, destination)
        ready.put(ticket.to_wire())
        if not done.wait(5):
            raise TimeoutError("sender did not complete")
        backend.finish_receive(ticket)
        result.put(bytes(destination))


class TestTcp(unittest.TestCase):
    def test_two_process_transfer(self):
        ctx = mp.get_context("spawn")
        meta = fixture()
        ready, result, done = ctx.Queue(), ctx.Queue(), ctx.Event()
        process = ctx.Process(target=receive_process, args=(meta.to_wire(), ready, done, result))
        process.start()
        try:
            ticket = ReceiveTicket.from_wire(ready.get(timeout=10))
            payload = bytes(range(meta.nbytes))
            with TcpTransferBackend(timeout=3) as sender:
                sender.send(meta, payload, ticket)
            done.set()
            self.assertEqual(result.get(timeout=5), payload)
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            ready.close()
            result.close()

    def test_checksum_failure_and_retry(self):
        meta = fixture()
        target = bytearray(meta.nbytes)
        with TcpTransferBackend(("127.0.0.1", 0), timeout=2) as receiver:
            ticket = receiver.prepare_receive(meta, meta.layout, target)
            with socket.create_connection(endpoint_address(ticket.endpoint)) as sock:
                channel = Channel(sock, 2)
                channel.send(
                    {
                        "metadata": json.loads(meta.to_wire()),
                        "ticket": json.loads(ticket.to_wire()),
                        "sha256": "bad",
                    }
                )
                self.assertEqual(channel.receive(), {"status": "ready"})
                sock.sendall(bytes(meta.nbytes))
                self.assertEqual(channel.receive()["status"], "error")
            self.assertEqual(target, bytearray(meta.nbytes))
            with TcpTransferBackend(timeout=2) as sender:
                sender.send(meta, bytes(range(meta.nbytes)), ticket)
                with self.assertRaises(TransferError):
                    sender.send(meta, bytes(meta.nbytes), ticket)
            receiver.finish_receive(ticket)

    def test_cancel_during_network_receive(self):
        meta = fixture()
        destination = bytearray(meta.nbytes)
        payload = bytes(range(meta.nbytes))
        with TcpTransferBackend(("127.0.0.1", 0), timeout=2) as receiver:
            ticket = receiver.prepare_receive(meta, meta.layout, destination)
            with socket.create_connection(endpoint_address(ticket.endpoint)) as sock:
                channel = Channel(sock, 2)
                channel.send(
                    {
                        "metadata": json.loads(meta.to_wire()),
                        "ticket": json.loads(ticket.to_wire()),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                self.assertEqual(channel.receive(), {"status": "ready"})
                sock.sendall(payload[:3])
                receiver.discard_receive(ticket)
                destination.extend(b"released")
                self.assertEqual(channel.receive()["status"], "error")
            self.assertEqual(destination, bytes(meta.nbytes) + b"released")

    def test_disconnect_before_payload_complete(self):
        meta = fixture()
        destination = bytearray(meta.nbytes)
        with TcpTransferBackend(("127.0.0.1", 0), timeout=2) as receiver:
            ticket = receiver.prepare_receive(meta, meta.layout, destination)
            with socket.create_connection(endpoint_address(ticket.endpoint)) as sock:
                channel = Channel(sock, 2)
                channel.send(
                    {
                        "metadata": json.loads(meta.to_wire()),
                        "ticket": json.loads(ticket.to_wire()),
                        "sha256": "unused",
                    }
                )
                channel.receive()
                sock.sendall(b"partial")
            with self.assertRaises(TransferError):
                receiver.finish_receive(ticket)
            receiver.discard_receive(ticket)
            self.assertEqual(destination, bytes(meta.nbytes))

    def test_close_drains_receiver_and_releases_views(self):
        meta = fixture()
        destination = bytearray(meta.nbytes)
        receiver = TcpTransferBackend(("127.0.0.1", 0), timeout=2)
        receiver.prepare_receive(meta, meta.layout, destination)
        receiver.close()
        receiver.close()
        destination.extend(b"released")

    def test_cancel_header_race_keeps_receiver_thread_alive(self):
        meta = fixture()
        destination = bytearray(meta.nbytes)
        with TcpTransferBackend(("127.0.0.1", 0), timeout=2) as receiver:
            ticket = receiver.prepare_receive(meta, meta.layout, destination)
            # Pause discard between revoking the read and removing the host reservation.
            with receiver._fabric._lock:
                receiver._cancellations.pop(ticket.locator).set()
            with TcpTransferBackend(timeout=2) as sender:
                with self.assertRaises(TransferError):
                    sender.send(meta, bytes(meta.nbytes), ticket)
                receiver.discard_receive(ticket)
                new_ticket = receiver.prepare_receive(meta, meta.layout, destination)
                sender.send(meta, bytes(meta.nbytes), new_ticket)
                receiver.finish_receive(new_ticket)


class TestControl(unittest.TestCase):
    def test_frames_and_endpoint(self):
        self.assertEqual(endpoint_address("localhost:1234"), ("localhost", 1234))
        left, right = socket.socketpair()
        with left, right:
            Channel(left, 1).send({"op": "health"})
            self.assertEqual(Channel(right, 1).receive(), {"op": "health"})

    def test_reject_oversized_frame(self):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack("!Q", MAX_CONTROL_BYTES + 1))
            with self.assertRaises(ValueError):
                Channel(right, 1).receive()

    def test_cancel_and_deadline(self):
        left, right = socket.socketpair()
        with left, right:
            cancel = Event()
            cancel.set()
            with self.assertRaises(CancelledError):
                Channel(right, 1, cancel).receive()
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                Channel(left, 0.05).receive()
            self.assertLess(time.monotonic() - start, 1)


if __name__ == "__main__":
    unittest.main()
