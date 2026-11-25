"""Server-side request demultiplexing; all GPU mutations stay on WorkerService."""

import queue
import select
import time
from collections import OrderedDict

from .multiplex import MUX_VERSION, envelope, unpack
from .protocol import check_transfer_id


def serve_multiplex(service, sock, channel, handshake, stopping):
    from .service import Session

    if (
        set(handshake) != {"op", "version", "request_id"}
        or type(handshake["version"]) is not int
        or handshake["version"] != MUX_VERSION
    ):
        raise ValueError("unsupported multiplex handshake")
    check_transfer_id(handshake["request_id"])
    channel.send(
        {"status": "multiplex", "version": MUX_VERSION, "request_id": handshake["request_id"]}
    )
    sessions, activity, recent = {}, {}, OrderedDict()

    def retire(request_id):
        sessions.pop(request_id, None)
        activity.pop(request_id, None)
        recent[request_id] = True
        while len(recent) > 4096:
            recent.popitem(last=False)

    try:
        while not stopping.is_set():
            for request_id, session in list(sessions.items()):
                try:
                    message = session.outbox.get_nowait()
                except queue.Empty:
                    if session.done.is_set():
                        channel.send(envelope(request_id, {"status": "finished"}, terminal=True))
                        retire(request_id)
                    elif time.monotonic() - activity[request_id] >= service.timeout:
                        session.cancelled.set()
                        session.emit(
                            {
                                "status": "error",
                                "code": "timeout",
                                "error": "request control deadline expired",
                            },
                            final=True,
                        )
                    continue
                terminal = session.done.is_set() and session.outbox.empty()
                channel.send(envelope(request_id, message, terminal=terminal))
                activity[request_id] = time.monotonic()
                if terminal:
                    retire(request_id)
            readable, _, _ = select.select([sock], [], [], 0.005 if sessions else 0.1)
            if not readable:
                continue
            request_id, message, terminal = unpack(channel.receive())
            if terminal:
                raise ValueError("clients cannot declare server request completion")
            if request_id in recent:
                if message.get("op") != "cancel":
                    channel.send(
                        envelope(
                            request_id,
                            {
                                "status": "error",
                                "code": "failed",
                                "error": "request_id already completed",
                            },
                            terminal=True,
                        )
                    )
                continue
            session = sessions.get(request_id)
            if session is None:
                if message == {"op": "health"}:
                    channel.send(envelope(request_id, service.health(), terminal=True))
                    retire(request_id)
                    continue
                if message == {"op": "capabilities"}:
                    channel.send(envelope(request_id, service.capabilities(), terminal=True))
                    retire(request_id)
                    continue
                if message.get("op") == "prefix_sync":
                    try:
                        reply = service.prefix_sync(message)
                    except (KeyError, TypeError, ValueError) as exc:
                        reply = {"status": "error", "code": "failed", "error": str(exc)}
                    channel.send(envelope(request_id, reply, terminal=True))
                    retire(request_id)
                    continue
                if message.get("op") == "cancel":
                    continue
                if len(sessions) >= service.queue_limit:
                    channel.send(
                        envelope(
                            request_id,
                            {
                                "status": "error",
                                "code": "busy",
                                "error": "connection request budget is busy",
                            },
                            terminal=True,
                        )
                    )
                    retire(request_id)
                    continue
                session = Session(service.output_limit, request_id=request_id)
                sessions[request_id] = session
            activity[request_id] = time.monotonic()
            service.submit(session, message)
    finally:
        for session in sessions.values():
            session.cancelled.set()
