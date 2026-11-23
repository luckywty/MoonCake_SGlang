import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from minisgl.disagg.admission import AdmissionBudget, AdmissionError
from minisgl.disagg.allocator import ReceiveAllocator
from minisgl.disagg.control import CancelledError
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, ReceiveTicket
from minisgl.disagg.tcp_backend import TcpTransferBackend
from minisgl.disagg.transfer import TransferError


class TestAdmission(unittest.TestCase):
    def test_tcp_budget_rejects_payload_before_creating_a_socket(self):
        layout = KVLayout("a" * 64, "float32", 1, 1, 1, 1)
        meta = KVTransferMetadata(str(uuid4()), layout, 1, (0,))
        target = ReceiveTicket(meta.transfer_id, "127.0.0.1:9999", str(uuid4()), meta.nbytes)
        with TcpTransferBackend(max_transfer_bytes=4) as sender:
            with self.assertRaisesRegex(TransferError, "transfer limit"):
                sender.send(meta, bytes(meta.nbytes), target)

    def test_receive_pool_fragmentation_alignment_and_coalescing(self):
        allocator = ReceiveAllocator(1024)
        self.assertEqual(allocator.allocate("a", 1), 0)
        self.assertEqual(allocator.allocate("b", 257), 256)
        self.assertEqual(allocator.allocate("c", 256), 768)
        allocator.release("b")
        with self.assertRaisesRegex(TransferError, "space"):
            allocator.allocate("too-large", 513)
        self.assertEqual(allocator.allocations, {"a": (0, 256), "c": (768, 256)})
        allocator.release("a")
        self.assertEqual(allocator.allocate("combined", 700), 0)
        allocator.release("c")
        allocator.release("combined")
        self.assertEqual(allocator.free_ranges, [(0, 1024)])

    def test_frontend_budget_admits_multiple_streams_and_restores_after_error(self):
        budget = AdmissionBudget(2, 10)
        barrier = threading.Barrier(3)
        finish = threading.Event()

        def run():
            with budget.reserve(5, 1):
                barrier.wait(timeout=1)
                finish.wait(timeout=1)

        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(run) for _ in range(2)]
            try:
                barrier.wait(timeout=1)
                self.assertEqual((budget.requests, budget.tokens), (2, 10))
                with self.assertRaisesRegex(AdmissionError, "busy"), budget.reserve(1, 0.01):
                    pass
                cancelled = threading.Event()
                cancelled.set()
                with self.assertRaises(CancelledError), budget.reserve(1, 1, cancelled):
                    pass
            finally:
                finish.set()
            for future in futures:
                future.result(timeout=1)
        self.assertEqual((budget.requests, budget.tokens), (0, 0))
        with self.assertRaises(ValueError), budget.reserve(10, 1):
            raise ValueError("stream failed")
        self.assertEqual((budget.requests, budget.tokens), (0, 0))

    def test_request_larger_than_frontend_budget_is_rejected_before_wait(self):
        with self.assertRaisesRegex(AdmissionError, "exceeds"), AdmissionBudget(2, 4).reserve(5, 1):
            pass
