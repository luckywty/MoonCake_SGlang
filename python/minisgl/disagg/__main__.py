"""Run the Phase 1 control-plane smoke test with standard Python only."""

import hashlib
import json
from uuid import uuid4

from .lifecycle import RequestLifecycle
from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket, RequestHandoff
from .transfer import MockFabric, MockTransferBackend


def main() -> None:
    #作用是描述发送端和接收端应该如何理解 KV Cache 数据。包括模型层数、head数量、维度等
    layout = KVLayout(hashlib.sha256(b"phase1-test-fixture").hexdigest(), "float16", 2, 1, 2, 4)
    meta = KVTransferMetadata.from_slots(
        str(uuid4()), layout, (28, 29, 30, 31, 12), num_pool_pages=8
    )#描述了需要发送哪些 KV Token、这些 Token 原本存储在哪里、数据总量是多少。
    handoff = RequestHandoff(0, (1, 2, 3, 4, 5), 9, 4, False, 99, meta) #Prefill 交给 Decode 的请求交接信息

    #状态机的作用就是约束这些时序关系，例如避免尚未完成 Prefill 就开始传输、未完成接收就确认交接等
    state = RequestLifecycle(handoff.uid, meta.transfer_id) #管理请求生命周期
    state.begin_prefill()
    state.complete_prefill(handoff)


    '''
    prefill-0 是 Prefill 传输端点的逻辑名称。

    decode-0 是 Decode 传输端点的逻辑名称。
    '''
    fabric = MockFabric()
    prefill = MockTransferBackend("prefill-0", fabric)
    decode = MockTransferBackend("decode-0", fabric)

    incoming = RequestHandoff.from_wire(handoff.to_wire())
    destination = bytearray(incoming.kv.nbytes) #分配一个指定大小的字节缓冲区
    target = decode.prepare_receive(incoming.kv, layout, destination)   #Decode 节点注册接收操作，并获得一个接收凭证 target
    source = bytes(range(meta.nbytes))  #生成要发送的测试数据
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
