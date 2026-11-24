# P/D 多请求调度与逐层批量传输

本版在 `e18a111` 基础上实现多请求调度。P/D 保持 TP=1、greedy、MHA KV、eager，
支持并发请求、Prefill batch、Continuous Batching Decode，以及批内按请求分层传输。
实际 batch 大小由到达时间、token 预算、请求表行和 KV 页共同决定，不保证每轮大于 1。

## 线程与执行边界

`worker.serve_worker()` 调用 `service.WorkerService.run()`。监听线程持续接受连接，
有上限的控制线程处理各自请求的消息和输出队列；所有请求表、页分配、PrefillManager、
DecodeManager 与模型前向仍由一个 GPU 调度线程管理。控制线程不直接调用模型或释放 KV 页。

控制线程将 `prefill/reserve/commit/release/cancel` 放入命令队列，断连设置该请求的取消标记。
GPU 循环每轮推进命令、传输完成与安装状态，然后组成下一批。健康检查读取调度快照，
无需等待整个请求结束，也不会对含活跃私有页的 cache 调用空闲态完整性检查。
输出队列满只取消相应慢客户端，不阻塞 GPU 调度线程。

## 六步数据流程

1. D 在前置 `reserve` 时可匹配并锁住 GPU Radix 前缀，只为缺失后缀分配私有 KV 页，并预留输出页预算。
   `ReceiveAllocator` 从单个启动期注册的连续池切出 256 字节对齐的非重叠区间，释放时合并空闲区间。
   v3 票据 `address` 已包含切片偏移，`layer_offsets` 相对后缀接收区的基地址；完整命中不分配接收切片。
   每个 locator/transfer_id 有独立状态、完成事件、取消标记和终态证明。
2. P 调用 `prefill_manager.schedule_next_batch(prefill_budget)`。请求使用 Worker 内唯一 UID，
   对外保留客户端 UID，多个客户端都传 `uid=0` 也不会在调度器内混淆。
   `BatchTransferPlan` 记录本批每个请求的 slots、最后 token chunk 标记与发送任务；任务持有目标票据。
3. 每个层组完成时，计算流只为整个 batch 记录一个 CUDA event 并入队。
   发送线程只打包 `is_last_chunk=True` 的请求，先等待计算 event，再 gather 到空闲注册槽位，
   在后台等待打包完成 event 后提交 `submit_scatter_write`。两个就绪屏障均保留。
4. 发送器限制各活跃请求同时占用的槽位数、轮流推进请求，检查所有在途 operation；不因队首请求未完成而忽略别的完成事件。
   同目标 D 复用短 RPC 连接，不在该连接上等待整个 RDMA；不同目标使用独立控制执行器。
   共享槽位用完会产生背压，槽位只能在原生终态且 `ticket.drained` 为真后复用。
5. P 在一次 batch 采样后分别返回首 token。后台按请求独立发完成证明、确认 `sent/retired`；
   客户端收到该请求的 `sent` 后向其 D 发送 `commit`。
   `sources_done` 表示不再读取该请求的原 KV 页，允许提前归还源页；独立 Staging 槽位仍保持到 DMA 排空。
6. D 收到 commit 后非阻塞查询接收完成，在 `install_stream` 中 scatter KV、写入首 token，记录 event。
   主循环用 `event.query()` 检查完成，归还该接收切片，发送 `installed` 并加入 DecodeManager。
   每次 `_forward()` 为当前 batch 的每个请求各生成一个 token；新安装请求动态加入，结束请求单独移出。

原有 `PDClient` 保留单组端点，`RoutedPDClient` 支持健康地址池负载/缓存感知选路、长连接和 reserve 重选。
不同客户端或路由请求可向同一个 P 提供不同 D 的票据，详见 [多 Worker 说明](multi_worker.md)。
路由新增 Bucket 和 Worker 准入筛选，保留本文的组批、页分配与逐层流水线，详见[选路说明](routing.md)。
`n=1` 指每个 HTTP 请求返回一个候选，与服务并发及 GPU batch 大小无关。

## 准入与容量参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| Worker `--max-running-requests` | 沿用引擎默认值 | 请求表行/活跃 GPU 请求上限 |
| Worker `--max-prefill-length` | 沿用引擎默认值 | 每个 Prefill batch 的 token 预算，长 prompt 可分块 |
| Worker `--pd-max-queued-requests` | 128 | 控制面已准入且尚未回收的请求总上限 |
| Worker `--pd-max-queued-tokens` | KV token 容量的 4 倍 | 已准入请求 prompt+max_tokens 总预算 |
| Worker `--pd-batch-wait-ms` | 2 | 新 Prefill 请求的短暂聚合窗口；分块续算不等待此窗口 |
| Worker `--pd-output-queue-size` | 128 | 每个连接待发送消息数上限 |
| Worker `--pd-max-pending-transfers` | 32 | Mooncake 活跃发送任务、接收租约和目标控制连接的上限；注册数也受此参数限制 |
| Gateway `--max-inflight-requests` | 128 | 同一个 PDClient 的并发请求上限 |
| Gateway `--max-inflight-tokens` | 1048576 | 同一个 PDClient 的 prompt+max_tokens 总预算 |

客户端准入只短暂锁住预算计数，等待时释放条件锁并响应取消；各流独立持有连接。
Worker 是显存容量的最终准入点：P 按页对齐估算批次，D 将后续输出尚需的页也计入预留。
容量不足返回 `code=busy`；非流式 HTTP 映射为 429，已开始的 SSE 通过错误事件返回。
网关保留每个请求自己的 iterator 锁，防止 `next()` 与 `close()` 竞争，不串行化不同请求。

单请求最大 KV 字节仍受 `--pd-max-transfer-bytes` 限制。D 注册池容量为
`min(max_registered_bytes, max_running_requests * (max_request_KV_bytes + 255))`，其中
`max_request_KV_bytes = min(max_seq_len, max_transfer_bytes // bytes_per_token) * bytes_per_token`。
同一池里能同时接纳多少请求取决于实际 prompt 长度、对齐、碎片和隔离切片；注册预算不会自动扩大。

## 取消、异常与验证

取消只停止本请求的未提交任务。已提交写入仍要排空；未获得成功/安全取消证明的切片保留隔离，
其他空闲区间仍可分配。正在安装的切片和 KV 页必须等 GPU reader event 完成才能回收。
原生完成未知、CUDA 消费完成未知或资源释放失败要求 Worker 停止服务，不以业务取消替代完成证明。

无网络回归使用真实管理器、KV 池、页表及确定性 CPU attention，替换 CUDA 和原生 DMA：

```bash
PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/disagg/test_admission.py tests/integration/test_pd_batching.py \
  tests/integration/test_pd_failures.py tests/integration/test_pd_control_loop.py
```

覆盖多请求 token 与独立参考的一致性、Prefill/Decode 实际 batch>1、不同 D 的独立完成、
连接复用、分块取消、同批 EOS、双槽位慢 DMA 隔离、安装 reader 的 event.query、页/输出预算、切片回收与常量注册次数。
整体 payload 路径也通过 Mock 传输验证组批，真实 TCP 两进程测试保留在 `test_pd_pipeline.py`。

当前环境是 CPU PyTorch；真实 socket/HTTP 回归受沙箱限制，沙箱外执行申请未获批准。
真实 CUDA 多请求、RDMA 可见性、双机及时间线仍待验收，CPU 测试不证明性能收益。

## 本轮回归记录（2026-11-23）

CPU 子集结果为 **131 passed、10 skipped、5 deselected、113 subtests passed**，
报告保存于本机 `/tmp/minisgl-pd-batching-cpu.xml`。Ruff、Python 语法编译、
shell 语法与 `git diff --check` 另行检查。复现命令：

```bash
PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/disagg tests/integration/test_pd_batching.py \
  tests/integration/test_pd_control_loop.py tests/integration/test_pd_tensor_cache.py \
  tests/integration/test_pd_entrypoints.py tests/integration/test_pd_failures.py \
  tests/integration/test_pd_layered.py tests/integration/test_pd_layered_cuda.py \
  tests/integration/test_pd_gpu.py tests/core/test_cache_allocate.py tests/misc/test_serialize.py \
  --ignore=tests/disagg/test_tcp.py \
  -k 'not early_handshake and not client_disconnect and not reservation_disconnect'
```

其中 10 项 GPU 用例跳过，5 项真实逐层 socket 用例被排除；TCP 与 HTTP 文件没有纳入本子集。
`test_pd_control_loop.py` 使用真实控制线程、调度循环和 PDClient，替换 socket/framing 为内存通道，
验证 4 个请求并发完成及关闭其中一个 generator 后其余请求继续完成，不代替真实网络验收。
