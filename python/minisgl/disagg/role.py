# from enum import Enum


# class WorkerRole(str, Enum):
#     UNIFIED = "unified"
#     PREFILL = "prefill"
#     DECODE = "decode"

# '''为 mini-SGLang 定义三种 Worker 运行角色，
# 并在 Phase 1 阶段禁止启动尚未实现的 Prefill/Decode 独立 Worker'''
# def require_serving_role(role: WorkerRole | str) -> None:
#     """Reject reserved roles before creating workers, CUDA contexts or IPC endpoints."""
#     role = WorkerRole(role)
#     if role is not WorkerRole.UNIFIED:
#         raise NotImplementedError(
#             f"worker_role={role.value}: split serving requires Phase 2; "
#             "Phase 1 supports the unified worker and standalone handoff tests."
#         )


from enum import Enum


class WorkerRole(str, Enum):
    UNIFIED = "unified"
    PREFILL = "prefill"
    DECODE = "decode"

# 允许启动 Prefill/Decode 独立 Worker，
# 但必须通过专用的 PDWorkerScheduler 入口启动，
# 不能继续使用原来的 Unified Scheduler 入口。
def require_serving_role(role: WorkerRole | str, *, split_worker: bool = False) -> None:
    """Keep split requests on the dedicated Phase 2 worker entrypoint."""
    role = WorkerRole(role)
    if role is not WorkerRole.UNIFIED and not split_worker:
        raise NotImplementedError(
            f"worker_role={role.value}: Phase 2 split roles use PDWorkerScheduler; "
            "launch them with python -m minisgl --worker-role prefill/decode."
        )
