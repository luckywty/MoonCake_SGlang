from __future__ import annotations

from dataclasses import dataclass, field
from minisgl.disagg.role import WorkerRole
from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    worker_role: WorkerRole = field(default=WorkerRole.UNIFIED, kw_only=True)

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    #让 Scheduler 能够知道当前 Worker 应该以 UNIFIED、PREFILL 还是 DECODE 角色运行
    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_role", WorkerRole(self.worker_role))

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True

