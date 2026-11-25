import json
import unittest
from dataclasses import replace
from uuid import uuid4

from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, ReceiveTicket, RequestHandoff


def layout(**kwargs):
    return KVLayout("a" * 64, "bfloat16", 2, 3, 4, 4, **kwargs)


def metadata():
    return KVTransferMetadata.from_slots(
        str(uuid4()), layout(), (28, 29, 30, 31, 12), num_pool_pages=8
    )


def handoff():
    return RequestHandoff(7, (1, 2, 3, 4, 5), 11, 4, False, 99, metadata())


class TestProtocol(unittest.TestCase):
    def test_wire_round_trip(self):
        request = handoff()
        meta = request.kv
        target = ReceiveTicket(meta.transfer_id, "decode-0", "opaque-locator", meta.nbytes)
        self.assertEqual(RequestHandoff.from_wire(request.to_wire()), request)
        self.assertEqual(KVTransferMetadata.from_wire(meta.to_wire()), meta)
        self.assertEqual(ReceiveTicket.from_wire(target.to_wire()), target)
        self.assertIsInstance(json.loads(request.to_wire())["prompt_ids"], list)
        self.assertEqual(meta.source_pages, (7, 3))
        self.assertEqual(meta.nbytes, 5 * 2 * 2 * 3 * 4 * 2)

    def test_valid_raw_slot_snapshots(self):
        for page_size in (1, 2, 4, 16):
            for count in (1, page_size, page_size + 1):
                with self.subTest(page_size=page_size, tokens=count):
                    local = replace(layout(), page_size=page_size)
                    slots = tuple(
                        (7 - pos // page_size) * page_size + pos % page_size for pos in range(count)
                    )
                    meta = KVTransferMetadata.from_slots(
                        str(uuid4()), local, slots, num_pool_pages=8
                    )
                    self.assertEqual(meta.kv_tokens, count)
                    self.assertEqual(meta.source_pages[0], 7)

    def test_invalid_slots_rejected(self):
        for slots in ((), (32,), (-1,), (29,), (28, 30), (28, 13), (28, 29, 30, 31, 28), (True,)):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                KVTransferMetadata.from_slots(str(uuid4()), layout(), slots, num_pool_pages=8)
        with self.assertRaises(ValueError):
            KVTransferMetadata.from_slots(str(uuid4()), layout(), (0,), num_pool_pages=0)

    def test_layout_validation(self):
        for change in (
            {"model_fingerprint": "Qwen3-0.6B"},
            {"model_fingerprint": "A" * 64},
            {"dtype": "float8_e4m3fn"},
            {"dtype": []},
            {"quantization": "static-head-scale"},
            {"packing": "raw-pool"},
            {"layers": 0},
            {"kv_heads": True},
            {"head_dim": -1},
            {"page_size": 0},
            {"tp_size": 0},
            {"tp_rank": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(layout(), **change)

    def test_metadata_validation(self):
        for change in (
            {"transfer_id": "attempt-0"},
            {"version": True},
            {"version": 3},
            {"kv_tokens": 0},
            {"kv_tokens": 4.0},
            {"source_pages": (1,)},
            {"source_pages": (1, 1)},
            {"source_pages": [1, 2]},
            {"source_pages": (1, -1)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(metadata(), **change)

    def test_first_token_and_budget(self):
        request = handoff()
        self.assertEqual(request.remaining_tokens, 3)
        self.assertEqual(replace(request, max_tokens=1).remaining_tokens, 0)
        self.assertEqual(replace(request, first_token_id=99).remaining_tokens, 0)
        self.assertEqual(replace(request, first_token_id=99, ignore_eos=True).remaining_tokens, 3)
        self.assertEqual(replace(request, eos_token_id=None).remaining_tokens, 3)
        self.assertEqual(request.kv.kv_tokens, len(request.prompt_ids))

    def test_invalid_handoff(self):
        for change in (
            {"prompt_ids": ()},
            {"prompt_ids": (1, 2)},
            {"prompt_ids": (1, 2, 3, 4, -1)},
            {"uid": -1},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"ignore_eos": 0},
            {"first_token_id": 2**31},
            {"eos_token_id": -1},
            {"sampling": "random"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(handoff(), **change)

    def test_strict_json_schema(self):
        original = json.loads(handoff().to_wire())
        for key in ("uid", "kv", "sampling"):
            data = dict(original)
            del data[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                RequestHandoff.from_wire(json.dumps(data).encode())
        for payload in (b"[]", b"null", b"{", b'{"uid":0,"uid":1}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                RequestHandoff.from_wire(payload)
        for change in (
            {"extra_field": 0},
            {"version": 3},
            {"prompt_ids": "12345"},
            {"ignore_eos": 1},
            {"kv": {**original["kv"], "source_pages": "73"}},
            {"kv": {**original["kv"], "version": 3}},
            {"kv": {**original["kv"], "layout": {}}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                RequestHandoff.from_wire(json.dumps({**original, **change}).encode())

    def test_ticket_validation(self):
        meta = metadata()
        target = ReceiveTicket(meta.transfer_id, "d", "a", meta.nbytes)
        for change in ({"endpoint": ""}, {"locator": " "}, {"nbytes": 0}, {"version": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(target, **change)


if __name__ == "__main__":
    unittest.main()
