"""Run the Phase 1 control-plane smoke test with standard Python only."""

import hashlib
import json
from uuid import uuid4

from .lifecycle import RequestLifecycle
from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket, RequestHandoff
from .transfer import MockFabric, MockTransferBackend


def main() -> None:
    layout = KVLayout(hashlib.sha256(b"phase1-test-fixture").hexdigest(), "float16", 2, 1, 2, 4)
    meta = KVTransferMetadata.from_slots(
        str(uuid4()), layout, (28, 29, 30, 31, 12), num_pool_pages=8
    )
    handoff = RequestHandoff(0, (1, 2, 3, 4, 5), 9, 4, False, 99, meta)
    state = RequestLifecycle(handoff.uid, meta.transfer_id)
    state.begin_prefill()
    state.complete_prefill(handoff)
    fabric = MockFabric()
    prefill = MockTransferBackend("prefill-0", fabric)
    decode = MockTransferBackend("decode-0", fabric)
    incoming = RequestHandoff.from_wire(handoff.to_wire())
    destination = bytearray(incoming.kv.nbytes)
    target = decode.prepare_receive(incoming.kv, layout, destination)
    source = bytes(range(meta.nbytes))
    try:
        state.begin_transfer()
        prefill.send(meta, source, ReceiveTicket.from_wire(target.to_wire()))
        decode.finish_receive(target)
        if destination != source:
            raise RuntimeError("mock payload mismatch")
        state.accept_receive(target.transfer_id)
    except Exception:
        state.fail()
        raise
    finally:
        decode.discard_receive(target)
    print(
        json.dumps(
            {
                "test": "phase1-control-contracts",
                "result": "PASS",
                "state": state.phase.value,
                "kv_tokens": meta.kv_tokens,
                "nbytes": meta.nbytes,
                "remaining_tokens": handoff.remaining_tokens,
            }
        )
    )


if __name__ == "__main__":
    main()
