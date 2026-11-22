import unittest
from dataclasses import replace
from uuid import uuid4

from minisgl.disagg.lifecycle import RequestLifecycle, RequestPhase
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, RequestHandoff


def request():
    layout = KVLayout("a" * 64, "float16", 2, 1, 4, 1)
    meta = KVTransferMetadata(str(uuid4()), layout, 2, (7, 2))
    return RequestHandoff(17, (1, 2), 3, 4, False, 99, meta)


def reach(state, req, phase):
    actions = (
        state.begin_prefill,
        lambda: state.complete_prefill(req),
        state.begin_transfer,
        lambda: state.accept_receive(req.kv.transfer_id),
        state.begin_decode,
        state.finish_decode,
    )
    for action in actions:
        if state.phase is phase:
            return
        action()


class TestLifecycle(unittest.TestCase):
    def test_full_lifecycle(self):
        req = request()
        state = RequestLifecycle(req.uid, req.kv.transfer_id)
        reach(state, req, RequestPhase.FINISHED)
        self.assertIs(state.phase, RequestPhase.FINISHED)
        self.assertFalse(state.cancel())
        self.assertFalse(state.fail())

    def test_first_token_completes_request(self):
        for req in (replace(request(), max_tokens=1), replace(request(), first_token_id=99)):
            with self.subTest(req=req):
                state = RequestLifecycle(req.uid, req.kv.transfer_id)
                state.begin_prefill()
                state.complete_prefill(req)
                self.assertIs(state.phase, RequestPhase.FINISHED)
                with self.assertRaises(RuntimeError):
                    state.begin_transfer()

    def test_invalid_order_and_duplicate_ack(self):
        req = request()
        state = RequestLifecycle(req.uid, req.kv.transfer_id)
        with self.assertRaises(RuntimeError):
            state.begin_decode()
        with self.assertRaises(RuntimeError):
            state.accept_receive(req.kv.transfer_id)
        reach(state, req, RequestPhase.DECODE_READY)
        with self.assertRaises(RuntimeError):
            state.accept_receive(req.kv.transfer_id)

    def test_attempt_and_uid_mismatch(self):
        req = request()
        state = RequestLifecycle(req.uid, req.kv.transfer_id)
        state.begin_prefill()
        for other in (replace(req, uid=req.uid + 1), request()):
            with self.assertRaises(ValueError):
                state.complete_prefill(other)
        self.assertIs(state.phase, RequestPhase.PREFILLING)
        state.complete_prefill(req)
        state.begin_transfer()
        with self.assertRaises(ValueError):
            state.accept_receive(str(uuid4()))
        self.assertIs(state.phase, RequestPhase.TRANSFERRING)

    def test_cancellation_and_failure_are_terminal(self):
        phases = list(RequestPhase)[:6]
        for phase in phases:
            for stop, terminal in (
                ("cancel", RequestPhase.CANCELLED),
                ("fail", RequestPhase.FAILED),
            ):
                with self.subTest(phase=phase, stop=stop):
                    req = request()
                    state = RequestLifecycle(req.uid, req.kv.transfer_id)
                    reach(state, req, phase)
                    self.assertTrue(getattr(state, stop)())
                    self.assertFalse(state.cancel())
                    self.assertFalse(state.fail())
                    with self.assertRaises(RuntimeError):
                        state.accept_receive(req.kv.transfer_id)
                    self.assertIs(state.phase, terminal)

    def test_invalid_request_identity(self):
        for uid, transfer_id in ((True, str(uuid4())), (-1, str(uuid4())), (1, "attempt")):
            with self.subTest(uid=uid), self.assertRaises(ValueError):
                RequestLifecycle(uid, transfer_id)


if __name__ == "__main__":
    unittest.main()
