# 主 Mooncake 通道的 Host DRAM 逐层中转

更新日期：2026-11-27（Asia/Shanghai）。基线：`1899a5d`。

`--pd-mooncake-transport-mode host_dram` 显式启用主 P→D 通道的 pinned host 中转，
适用于 CUDA 推理可用、HCA/RDMA 可用但 GPUDirect 不可用的机器。默认仍为 `gpu_direct`；
缺少 GDR、CUDA 扩展或注册失败不会自动切换模式，也不会切换为 TCP。
Host 模式仍要求 CUDA 模型、`cuda-bindings`、classic RDMA HCA 和带 scatter completion API 的 Mooncake binding。
主通道只注册 CPU 内存，因此不要求该 binding 的 `SUPPORT_CUDA=true`，也不查询 GDR 能力或使用 peermem 注册。
容器仍须具有 HCA 访问及 pinned/MR 内存权限。

## 配置

P 和 D 必须使用相同模式。多 Worker 启动器把该参数传给所有本机 Worker：

```bash
python -m minisgl --model /models/model --dtype bfloat16 \
  --num-prefill-workers 1 --num-decode-workers 1 \
  --prefill-gpus 0 --decode-gpus 1 \
  --pd-transfer-backend mooncake --pd-mooncake-transport-mode host_dram \
  --pd-mooncake-host 127.0.0.1 \
  --pd-layers-per-chunk 1 --pd-staging-slots 3 \
  --pd-max-transfer-bytes 67108864 --pd-max-registered-bytes 134217728
```

模型、HCA、地址及预算按部署调整。示例限制单请求缺失 KV 为 64 MiB；大模型或长 prompt 需要提高预算。
独立启动时在两端 Worker 命令均增加 `--pd-transfer-backend mooncake --pd-mooncake-transport-mode host_dram`。
固定主 TE 端口继续使用 `--pd-mooncake-port`，本地启动器使用 `--base-mooncake-port`。
参数用于 split Mooncake Worker，`host_dram` 与 TCP/Unified 组合会在加载模型前报错。

此模式不要求启用 `--pd-dram-cache-bytes` 或 `--pd-dram-remote-pull`。
后者仍控制独立的 D→P 缓存读取 TE，使用 `--pd-dram-port` / `--base-dram-port`，需要原有 reader fence API。

## 数据流与隔离

```text
P HBM → 固定 GPU gather 缓冲 → P pinned 环形槽位
                              ↓ 主 P→D TE / RDMA Write
D HBM ← 固定 GPU scatter 缓冲 ← D pinned 接收池切片

D 缓存 DRAM → 独立 Host TE / RDMA Read → P 缓存 DRAM → P HBM
```

两条通道使用不同的 TE 实例、端口、MR 和物理分配。主通道 ring/receive pool 不经过 `DramPagePool`，
不受 LRU/LFU 控制，也不会被 D→P 的 `fence_remote_readers()` 撤销。
独立 Host TE 退役后，主 Host 中转仍可继续推理。

P→D 范围继续以 D reserve 的真实 `d_prefix_len` 为准，仅传 `[d_prefix_len, prompt_len)`。
完整命中时不创建 receive ticket、不分配切片、不提交零字节写入；P 仍计算首 token/logits。
D 收到的完整 prompt 在安装后只进入 GPU KV 页；不会批量写入本地 DRAM 缓存。
Decode 后新产生的满页/接缝页继续走原有异步 offload 和事务提交逻辑。

## 启动预算与固定分配

令 `N=min(max_seq_len, max_transfer_bytes // bytes_per_token)`，`L` 为层数，`G=min(layers_per_chunk,L)`：

```text
payload_bytes = N * bytes_per_token
group_bytes = align_up(N * bytes_per_token / L * G, 256)
P registered_bytes = staging_slots * group_bytes
D registered_bytes = min(align_down(max_registered_bytes, 256),
                         max_running_req * align_up(payload_bytes, 256))
GPU scratch bytes = group_bytes + 8 * N
```

P 注册所有 pinned ring 槽位；D 只注册一个连续 pinned 接收池，由 `ReceiveAllocator` 切出 256 字节对齐区间。
每个 pinned 分配还保留最多 255 字节的基址对齐余量；注册预算按实际 MR 区间计数。
两端各有一个 GPU group 缓冲及 int64 索引缓冲，均在启动期分配，不注册给 TE。
共享 scratch 在各自单一 transfer/install stream 上按顺序复用。

P ring 必须不超过 `--pd-max-registered-bytes`，D 预算至少能容纳一份最大、对齐后的 payload。
校验发生在权重读取/Engine 构造前，并在实际 pipeline 配置时复核。
Host 模式给 Engine 预留 GPU scratch 的逻辑字节量，不再把整个 host MR 预算扣成 HBM。
与推理引擎的其他分配相同，CUDA allocator/context 开销仍需由显存余量覆盖。
热路径不创建 KV 中转存储、不注册或注销 MR；请求事件及索引/控制元数据沿用现有机制。

## 两道 DMA 屏障

1. 模型 layer-ready 回调只记录 compute event 并排队。后台发送线程的 `transfer_stream` 等待该 event，
   gather 后将缺失 KV 异步 D2H 至空闲 pinned 槽位，再在后台执行 D2H event 的 `synchronize()`。
   此后才允许提交主 TE write；ring 槽位持续保留到 native ticket 确认 `drained`。
2. 所有层 D2H 完成后，`sources_done` 才允许归还 P 源 HBM；开启本地 DRAM offload 时还须满足其 reader 条件。
   RDMA 可以继续读取 pinned ring，源 HBM 已可独立复用。
3. D 校验全部层写入成功及 commit 后，在独立 `install_stream` 分组 H2D→scatter。
   索引复制也使用预分配 GPU int64 缓冲；该路径不调用会把 GPU slots 转回 CPU 的索引校验。
4. 主服务循环只 `install_event.query()`；就绪后立即释放 Host 切片，并把请求并入 Continuous Batching Decode。
   不等待 Decode 结束。取消或部分安装失败也必须等该 event，无法证明完成时保留资源并要求 Worker 退役。

D 的 host receive 不调用 GDR visibility/context 全局同步。Host H2D 本身及 install event 建立 GPU 消费顺序。
这里描述生产用的并发 `WorkerService`；保留的旧同步 scheduler 适配 API 仍使用其原有同步请求边界。

## 协议与失败处理

health/capabilities/reserve 发布 `transfer_mode`，Router 将其纳入 P/D 兼容身份；单端点客户端也在 reserve 前比较两端能力。
层组票据及 handoff KV 元数据携带 Host 模式，P 发送前、D 安装前再次比较。完整命中也检查模式。
handoff v2、层组票据 v3 的后缀语义保留。默认 GPU 模式的 wire 省略新字段并按 legacy GPU 解析；
Host 模式必须显式编码，旧 Worker 无法把它误当作 GPU 传输。
Host 控制包标识为 `mooncake-layered-host-dram-v3`，GPU 仍为 `mooncake-layered-gpu-v3`。

`COMPLETION_UNKNOWN` 保留 pinned MR、ticket 和 scratch；`FAILED_DRAINED` 不批准 D 复用切片。
完成回执丢失仍只补送一次相同证明，不重新发送 KV。Host 模式保持原有隔离、终态重放及进程退出回收边界。

## 验证

完整 CPU 集合：**404 passed、19 skipped、117 subtests passed、0 failed**。
新增/扩展覆盖在 `test_pd_layered.py`、`test_pd_batching.py`、`test_pd_routing.py`、
`test_pd_startup_validation.py`、`test_pd_dram.py` 及 `tests/disagg/test_layered_protocol.py`。
包含真实本机 socket 控制握手、CPU attention token 对照、后缀/完整命中、D2H 前禁止写入、
源 HBM 复用、安装 event 前后切片归还、取消/EOS、隔离及独立 Host TE fence 后主通道续算。
报告位于执行机 `/tmp/minisgl-host-dram-cpu.xml`。

CPU 适配器替换 CUDA/pinned/native DMA 原语，不能证明真实硬件 RDMA、可见性或 TPOT 收益。
`test_pd_layered_cuda.py` 新增三种 dtype 的真实 pinned/CUDA 双流往返，无 CUDA 时明确 skip；不包含原生 RDMA。
实机还须验证无 peermem 双端 token 对照、实际 MR 数量固定、迟到/失败写入、并发 TPOT 和主/独立 TE 故障隔离。
