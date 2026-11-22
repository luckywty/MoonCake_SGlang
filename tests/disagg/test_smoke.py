import contextlib
import io
import json
import unittest
from unittest.mock import patch

from minisgl.disagg.__main__ import main
from minisgl.disagg.lifecycle import RequestLifecycle, RequestPhase
from minisgl.disagg.transfer import MockTransferBackend, TransferError


class TestSmoke(unittest.TestCase):
    def test_control_handoff(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["state"], "decode_ready")
        self.assertEqual(result["kv_tokens"], 5)
        self.assertEqual(result["remaining_tokens"], 3)

    def test_failure_after_copy_revokes_ticket_and_keeps_request_failed(self):
        states = []
        destinations = []
        original_prepare = MockTransferBackend.prepare_receive
        original_send = MockTransferBackend.send

        def create_state(*args):
            state = RequestLifecycle(*args)
            states.append(state)
            return state

        def prepare(receiver, meta, local_layout, destination):
            destinations.append(destination)
            return original_prepare(receiver, meta, local_layout, destination)

        def fail_after_send(sender, *args):
            original_send(sender, *args)
            raise TransferError("injected completion acknowledgment failure")

        with (
            patch("minisgl.disagg.__main__.RequestLifecycle", side_effect=create_state),
            patch.object(
                MockTransferBackend, "prepare_receive", autospec=True, side_effect=prepare
            ),
            patch.object(MockTransferBackend, "send", autospec=True, side_effect=fail_after_send),
            self.assertRaises(TransferError),
        ):
            main()
        self.assertEqual(len(states), 1)
        self.assertIs(states[0].phase, RequestPhase.FAILED)
        destinations[0].extend(b"registration released after failure")


if __name__ == "__main__":
    unittest.main()
