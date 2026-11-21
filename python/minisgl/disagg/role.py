from enum import Enum


class WorkerRole(str, Enum):
    UNIFIED = "unified"
    PREFILL = "prefill"
    DECODE = "decode"


def require_serving_role(role: WorkerRole | str) -> None:
    """Reject reserved roles before creating workers, CUDA contexts or IPC endpoints."""
    role = WorkerRole(role)
    if role is not WorkerRole.UNIFIED:
        raise NotImplementedError(
            f"worker_role={role.value}: split serving requires Phase 2; "
            "Phase 1 supports the unified worker and standalone handoff tests."
        )
