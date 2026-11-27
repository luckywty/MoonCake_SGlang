import json
import unittest
from dataclasses import replace
from uuid import uuid4

from minisgl.disagg.layered_protocol import LayeredReceiveTicket
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, RequestHandoff


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
    def test_explicit_host_mode_round_trip_and_legacy_gpu_wire(self):
        original = ticket()
        self.assertNotIn("transfer_mode", json.loads(original.to_wire()))
        host = replace(original, transfer_mode="host_dram", prefix_len=4)
        self.assertEqual(json.loads(host.to_wire())["transfer_mode"], "host_dram")
        self.assertEqual(LayeredReceiveTicket.from_wire(host.to_wire()), host)
        meta = KVTransferMetadata.from_slots(
            host.transfer_id,
            host.layout,
            (4, 5, 6),
            num_pool_pages=8,
            prefix_len=4,
            transfer_mode="host_dram",
        )
        handoff = RequestHandoff(0, tuple(range(7)), 1, 3, True, 31, meta)
        self.assertEqual(RequestHandoff.from_wire(handoff.to_wire()), handoff)
        self.assertEqual(json.loads(handoff.to_wire())["kv"]["transfer_mode"], "host_dram")
        gpu = replace(handoff, kv=replace(meta, transfer_mode="gpu_direct"))
        self.assertNotIn("transfer_mode", json.loads(gpu.to_wire())["kv"])
        self.assertEqual(RequestHandoff.from_wire(gpu.to_wire()), gpu)

    def test_unknown_modes_and_host_legacy_versions_are_rejected(self):
        for mode in (None, True, "automatic", "tcp"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                replace(ticket(), transfer_mode=mode)
        with self.assertRaises(ValueError):
            replace(ticket(), version=2, transfer_mode="host_dram")
        meta = KVTransferMetadata.from_slots(
            ticket().transfer_id, ticket().layout, (4,), num_pool_pages=8
        )
        with self.assertRaises(ValueError):
            replace(meta, version=1, transfer_mode="host_dram")

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
