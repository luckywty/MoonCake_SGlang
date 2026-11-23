import unittest
from dataclasses import replace
from uuid import uuid4

from minisgl.disagg.layered_protocol import LayeredReceiveTicket
from minisgl.disagg.protocol import KVLayout


def ticket(layers_per_chunk=2):
    layout = KVLayout("a" * 64, "bfloat16", 5, 2, 8, 4)
    layer_bytes = 7 * layout.bytes_per_token // layout.layers
    return LayeredReceiveTicket(
        str(uuid4()),
        "127.0.0.1:1931",
        str(uuid4()),
        "127.0.0.1:5000",
        4096,
        layout,
        7,
        layers_per_chunk,
        tuple(i * layer_bytes for i in range(5)),
        31,
    )


class TestLayeredProtocol(unittest.TestCase):
    def test_round_trip_and_tail_group(self):
        original = ticket()
        self.assertEqual(LayeredReceiveTicket.from_wire(original.to_wire()), original)
        self.assertEqual(original.groups, ((0, 2), (2, 4), (4, 5)))
        self.assertEqual(original.nbytes, 7 * original.layout.bytes_per_token)

    def test_reject_overlapping_gapped_or_mistyped_offsets(self):
        original = ticket()
        for offsets in (
            (0,) * 5,
            (0, 1, 2, 3, 4),
            original.layer_offsets[:-1],
            (False, *original.layer_offsets[1:]),
        ):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                replace(original, layer_offsets=offsets)

    def test_invalid_ticket_fields(self):
        for changes in (
            {"address": 0},
            {"layers_per_chunk": 0},
            {"kv_tokens": True},
            {"session": "0.0.0.0"},
            {"version": 1},
            {"packing": "old"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ticket(), **changes)


if __name__ == "__main__":
    unittest.main()
