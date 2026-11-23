from __future__ import annotations

from enum import Enum

from .protocol import RequestHandoff, check_transfer_id


class RequestPhase(str, Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    PREFILLING_AND_TRANSFERRING = "prefilling_and_transferring"
    AWAITING_TRANSFER = "awaiting_transfer"
    TRANSFERRING = "transferring"
    DECODE_READY = "decode_ready"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL = {RequestPhase.FINISHED, RequestPhase.CANCELLED, RequestPhase.FAILED}


class RequestLifecycle:
    '''P/D 分离系统中的请求生命周期状态机

    记录一个推理请求当前执行到了哪个阶段，并限制请求只能按照合法的顺序进行状态转换。
    整体交接与逐层重叠使用不同状态；Decode 必须在接收确认之后才能执行。
    '''
    """Owned by one scheduler event loop for one frontend request and transfer attempt."""

    def __init__(self, uid: int, transfer_id: str) -> None:
        if type(uid) is not int or uid < 0:
            raise ValueError("invalid frontend uid")
        check_transfer_id(transfer_id)
        self.uid = uid
        self.transfer_id = transfer_id
        self._phase = RequestPhase.QUEUED

    @property
    def phase(self) -> RequestPhase:
        return self._phase

    def _step(self, expected: RequestPhase, new: RequestPhase) -> None:
        if self.phase is not expected:
            raise RuntimeError(f"cannot enter {new.value} from {self.phase.value}")
        self._phase = new

    def begin_prefill(self) -> None:
        self._step(RequestPhase.QUEUED, RequestPhase.PREFILLING)

    def complete_prefill(self, handoff: RequestHandoff) -> None:
        if self.uid != handoff.uid or self.transfer_id != handoff.kv.transfer_id:
            raise ValueError("handoff belongs to another request or attempt")
        if self.phase is RequestPhase.PREFILLING_AND_TRANSFERRING:
            # Even EOS must await the cancellation/drain receipt before resources
            # can be retired. There may already be native writes in flight.
            self._step(self.phase, RequestPhase.TRANSFERRING)
        else:
            new = RequestPhase.AWAITING_TRANSFER if handoff.remaining_tokens else RequestPhase.FINISHED
            self._step(RequestPhase.PREFILLING, new)

    def begin_prefill_transfer(self) -> None:
        self._step(RequestPhase.PREFILLING, RequestPhase.PREFILLING_AND_TRANSFERRING)

    def finish_without_decode(self) -> None:
        self._step(RequestPhase.TRANSFERRING, RequestPhase.FINISHED)

    def begin_transfer(self) -> None:
        self._step(RequestPhase.AWAITING_TRANSFER, RequestPhase.TRANSFERRING)

    def accept_receive(self, transfer_id: str) -> None:
        """Coordinator consumes D's acknowledgment after D validates the installed KV."""
        if transfer_id != self.transfer_id:
            raise ValueError("stale receive acknowledgment")
        self._step(RequestPhase.TRANSFERRING, RequestPhase.DECODE_READY)

    def begin_decode(self) -> None:
        self._step(RequestPhase.DECODE_READY, RequestPhase.DECODING)

    def finish_decode(self) -> None:
        self._step(RequestPhase.DECODING, RequestPhase.FINISHED)

    def _stop(self, new: RequestPhase) -> bool:
        if self.phase in _TERMINAL:
            return False
        self._phase = new
        return True

    def cancel(self) -> bool:
        return self._stop(RequestPhase.CANCELLED)

    def fail(self) -> bool:
        return self._stop(RequestPhase.FAILED)
