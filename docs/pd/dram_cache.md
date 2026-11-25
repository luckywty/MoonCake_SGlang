# P/D 本地 DRAM KV 缓存与 D→P 按需拉取

更新日期：2026-11-26（Asia/Shanghai）；下述测试记录执行于 2026-11-25。

本次在 `4d15539` 页面哈希索引、`5e1e21e` D Radix 后缀交接基础上实现。
功能默认关闭；`--pd-dram-cache-bytes` 大于零时启用本地缓存，
`--pd-dram-remote-pull` 额外启用独立 Mooncake host RDMA 引擎。
它不依赖 Mooncake Store，不提供持久化、自动发现或跨请求故障迁移。

## 内存与所有权

| 位置 | 保存的数据 | 生命周期 |
|---|---|---|
| P HBM | 活跃请求、已锁定的 Radix 前缀、从 DRAM 恢复的前缀、传输/拷贝 Staging | 沿用 GPU CacheManager；所有 gather/D2H reader 完成后才归还 |
| P Pinned DRAM | P 新算的完整满页、成功拉取的 D 满页 | 固定容量本地池；成功事务提交后可命中，LRU/LFU 淘汰 |
| D HBM | P 发送的缺失后缀、D 的活跃序列和新算 KV；可选 Radix 历史前缀 | Radix 关闭时，请求结束且 reader 排空后归还全部请求 KV；开启时沿用 Radix |
| D Pinned DRAM | 包含 D 新算 token 的完整满页，包含必要的 prompt/decode 接缝 | 独立于 D GPU Radix 开关；LRU/LFU 被动淘汰，远端租约期间不可覆写 |
| Router | 逻辑页哈希、worker/instance/role/tier、序号、快照和负载 | 有界且可过期的 Hint，不保存 KV Tensor 或物理页地址 |

设页大小为 `B`，prompt 长度为 `N`，结束时已计算 KV 的 token 数为 `C`：

- P 卸载本次新计算且未在 DRAM 中存在的完整页，复用已有 HBM/DRAM 前缀不重复卸载。
- D 从 `floor(N/B)*B` 开始，按整条序列的同一 SHA-256 链卸载到 `floor(C/B)*B`。
- D 不批量复制收到的完整 prompt 页。若 `N=10,B=4,C=17`，D 缓存 `[8,12)` 与 `[12,16)`；
  第一页包含两个 prompt token 与两个已计算的生成 token，最后一个刚采样、尚无 KV 的 token 不会被缓存。
- 容量、并发或缓存事务预算不足时允许少缓存，不影响推理。

`dram_pool.py` 只在启动时分配一次 CPU `uint8` pinned 池，各槽位按 256 字节对齐；
payload 严格为一个 MHA KV 页，布局为 `[layer,K/V,token,head,dim]`。
`DramController` 启动时分配固定 GPU Staging 环、indices 和 events，Engine 显存预算预扣两条拷贝流的 Staging。
热路径不新增 pinned payload 分配或 MR 注册。页哈希等 Python 元数据和正常请求 GPU 页分配仍会发生。
Worker 的哈希任务也有并发预算；排队达到请求行预算时直接回退，取消请求不会持续堆积无限哈希任务。

## 请求流程

1. Router 在候选 `P_i` 的镜像中从第 0 页探测：连续 P HBM → 连续 P DRAM → 单个健康 D 的连续 DRAM。
   中间缺页即停止，不将不同 D 的碎片拼接；远端长度须达到门槛且估计搬运成本低于重算收益。
2. 客户端始终先向目标 `D_j` 发 `reserve`。D GPU Radix 开启时，先锁定真实 HBM 前缀，
   再尝试从本地 DRAM 恢复其后连续页；恢复完成后才返回真实 `decode_prefix_len`。
   GPU Radix 关闭时，保留原行为：`decode_prefix_len=0`，P→D 传整个 prompt。
3. P 先锁定真实本地 Radix，再从本地 DRAM 异步 H2D。
   Router Hint 不能扩大 P 实际跳过的前缀；每一步都以已完成的真实载入页为准。
4. P 对计划中的唯一 D 申请 `dram_pin`，校验 instance/domain、页链、深度、generation 与长度。
   数据只从 D 的预注册 host MR RDMA Read 到 P 固定 DRAM 槽位；绝不读取 D HBM。
   按 `batch_pages` 分批读，已完成批交给 GPU 主线程做 H2D，同时后台读下一批；队列有界。
   拉取使用独立的两个后台线程，提交前即做并发准入；忙时直接重算，不占用原 P→D I/O 执行器排队。
5. RDMA 全部排空后立刻发送 `dram_unpin`；P 本地 H2D 可以继续使用自己的 DRAM 副本。
   已成功恢复的连续前缀之后由 P 正常计算，原有 P→D 层组边界只取 `D_j reserve` 的真实前缀。
   即使 `D_k=D_j`，也不把 D 已命中的 A+B 从 P 重传一遍。
6. P/D 各自在独立 `offload_stream` 异步 D2H。结束请求先退出 decode 调度，
   等既有 sender gather 与全部 D2H reader 完成，再缓存/释放 GPU 页和请求表行。

所有远端控制、pin、容量、描述符和读超时错误均回退到 P 从最后成功恢复位置重算。
远端完成状态未知时，目的槽位、native ticket 和租约信息保留在隔离区，不用于 H2D、不被重用；
后台取得 drained 证明后才释放。推理可先完成，无需等待这次不确定的 host DMA。
若 CUDA 自身的 event 无法提供完成证明，则仍按原工程约束停止该 Worker，防止复用在读 GPU 页。

## 成功提交与租约

新页先处于 STAGED 状态，D2H/RDMA 完成且请求成功提交后才进入 READY 并发布索引。
新客户端在整个 P/D 推理成功后以独立、有限队列的控制任务向两端发送 `cache_commit`；
取消或失败发送 `cache_abort`。消息丢失时，未提交事务按 TTL 清理，宁可少命中。
活跃请求续期 pending 事务。旧客户端可以使用已经 READY 的本地页，但不会发布本次新缓存。
这避免 P 先完成、D 后续失败时提前公开本次 DRAM 页。

每个 export 租约有 TTL，但 **TTL 到期本身不等于 DMA 停止**。
接收 drained 回执后可直接解除租约；回执遗失或 P 退出时，D 在后台永久 fence 独立 host TE：
拒绝新 RDMA 握手，将既有 QP 转入错误态并检查所有 QP 销毁结果。
只有 native fence 成功才自动解除剩余 export；失败时继续保留这些槽位，等待迟到回执或进程退出。
一次 fence 后该实例的远端缓存功能关闭，重启 Worker 才恢复；本地 DRAM 与原 GPU P→D TE 不受该 host fence 影响。

LRU 为默认策略；LFU 使用命中计数和按分钟衰减。成功远端读会更新 D 页的访问热度。
P 保存副本不会主动删除 D 副本，后续优先使用 P 本地缓存，由 D 自己的淘汰策略回收冗余副本。

## 配置

仅本地 DRAM，不需要 host RDMA 依赖：在 P、D 启动命令各追加

```bash
--pd-dram-cache-bytes 8589934592 --pd-dram-cache-policy lru
```

多 Worker 启动器模板（将模型和 GPU 映射替换为实际环境）：

```bash
python -m minisgl --model /models/model \
  --num-prefill-workers 1 --num-decode-workers 1 \
  --prefill-gpus 0 --decode-gpus 1 \
  --pd-transfer-backend mooncake \
  --pd-dram-cache-bytes 8589934592 --pd-dram-remote-pull \
  --base-dram-port 25000 \
  --pd-dram-min-remote-tokens 1024 --min-remote-pull-tokens 1024 \
  --pd-dram-max-pull-bytes 268435456 --dram-max-pull-bytes 268435456 \
  --routing-policy cache_aware --affinity-mode balanced \
  --disaggregation-decode-enable-radix-cache
```

启动器为每个 Worker 分配互异的 control/distributed/GPU TE/data/host TE 端口，禁止透传覆盖 `--pd-dram-port`。
单独启动 Worker 可显式设置 `--pd-dram-port`，省略时为 `control_port+3`；
host MR 只注册到独立 TE，通过 `initialize_with_rpc_port` 显式绑定独立端口，不修改 GPU TE 的进程级端口环境。
跨机需 `--pd-dram-host` 或正确的 advertise host；HCA 使用 `--pd-dram-rdma-device`，默认沿用 GPU TE 的 HCA 配置。
原 `--prefill-rdma-devices`/`--decode-rdma-devices` 映射也传入各子进程的 host TE。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--pd-dram-cache-bytes` | 0 | 每 Worker pinned host 池的总字节预算，0 禁用 |
| `--pd-dram-cache-policy` | lru | lru 或 lfu |
| `--pd-dram-staging-slots` / `--pd-dram-batch-pages` | 2 / 8 | 每条 GPU 拷贝流的固定槽位数 / 批大小 |
| `--pd-dram-remote-pull` | 关闭 | 独立 host RDMA 读/导出能力 |
| `--pd-dram-min-remote-tokens` | 1024 | P 与源 D 都会执行的最低命中门槛 |
| `--pd-dram-pull-timeout` | 1 秒 | 远端控制操作和整次分批读的期限 |
| `--pd-dram-max-pull-bytes` | 64 MiB | 每次拉取上限和令牌桶最大突发 |
| `--pd-dram-read-bandwidth` | 256 MiB/s | 每 Worker 读/导出的令牌桶补充速率 |
| `--pd-dram-max-leases` | 32 | 页池租约预算；传输层同时最多导出 2 个租约、发起 2 次拉取 |
| `--pd-dram-lease-ttl` | 30 秒 | export 租约期限，须大于读 timeout |
| `--pd-dram-transaction-ttl` | 300 秒 | 成功结果确认丢失时清理 STAGED 页的期限 |
| `--min-remote-pull-tokens` | 1024 | Router 规划门槛，建议与 Worker 一致 |
| `--dram-max-pull-bytes` | 64 MiB | Router 规划上限，建议与 Worker 一致 |
| `--dram-h2d-gbps` / `--dram-rdma-gbps` | 12 / 10 | 选路成本模型的有效带宽，需实测校准 |
| `--dram-prefill-ms-per-token` | 0.01 | 选路重算成本标定，需实测校准 |

成本按 KV 字节量估算，`balanced` 将 H2D、RDMA+H2D 分别转换成等效 prefill 开销；
不把远端命中与 HBM 命中等权。带宽限制是 host 引擎的并发与准入速率控制，并非 NIC 硬 QoS。
`max-pull-bytes` 必须足以容纳门槛 token 对应的 KV；例如 128 KiB/token、1024 token 至少需要 128 MiB。
否则该配置始终选择重算，属于正常降级。
另限制单次最多 16384 页，确保 pin 描述符不超过控制帧上限；超过部分由 P 计算。

## 原生依赖与观测

远端路径要求配套构建本工作区 `../Mooncake` 中的 Python binding，额外暴露
`initialize_with_rpc_port()`、`has_remote_reader_fence()` 和 `fence_remote_readers()`。
显式端口接口避开 classic TE 只初始化一次的全局 RPC 端口范围，保证同进程的 GPU/host TE 不争用固定端口。
后者贯穿 Python → TransferEngine → RdmaTransport → RdmaContext/EndpointStore。
仅 classic RDMA 支持；TENT、无 RDMA 或缺少 API 时 health 报 `remote_ready=false`，本地 DRAM 和正常推理继续运行。
PyPI 原有 `pd-mooncake` extra 仍可用于 GPU P→D，但不能假设它具有本次新 fence API。
配套原生改动已打包为 [host-dram.patch](../../patches/mooncake/host-dram.patch)，
基线和应用步骤见 [补丁说明](../../patches/mooncake/README.md)，从 GitHub 检出本项目后可据此重建依赖。

`health.dram_cache` 提供容量、READY/WRITING 页、readers、leases、隔离页、提交/淘汰计数，
以及 `local_hit_tokens`、`remote_hit_tokens`、`pull_fallbacks`、错误原因。
`health.tier_index` 为 v2，HBM/DRAM 共用 instance-scoped 单调 seq 与有界增量/快照。
v1 P HBM 索引继续兼容；重启、失联、retired 和 seq 缺口清空相关镜像，索引永远不直接授权跳过计算。

主要实现文件：`dram_pool.py`、`dram_protocol.py`、`dram_controller.py`、`dram_service.py`、
`dram_transfer.py`、`tier_index.py`；WorkerService、Router、客户端、scheduler/prefill、MHA 页导入导出及 Engine 预算接入这些模块。
双流 reader 保护通过持有请求行、私有页和 Radix handle，以及联合检查原 sender `sources_done`/D2H event 实现，
不在异步线程直接调用 GPU CacheManager。

## 验证与实机验收

CPU 核心用例：

```bash
PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/integration/test_pd_dram.py tests/integration/test_pd_batching.py \
  tests/integration/test_pd_tensor_cache.py tests/integration/test_pd_control_loop.py \
  tests/integration/test_pd_policy.py tests/integration/test_pd_cluster_config.py
```

`test_pd_dram.py` 使用真实 CPU attention 续算和真实缓存/调度/租约代码，只替换 CUDA/RDMA 原语，
覆盖接缝、取消提交、延迟 D2H、单源选路、D reserve 恢复、反向提升、超时隔离、租约 fence 失败和实例重启。
`test_pd_dram_cuda.py` 提供真实 pinned/CUDA 双流页往返用例，没有 GPU 时明确 skip。
本轮定向回归为 **116 passed、6 skipped（CUDA）、69 subtests passed**，结果保存在执行环境
`/tmp/minisgl-dram-validation.xml`。同时通过 Ruff、Black、Python 编译和两仓库差异空白检查。
原生层提供 `FixedRpcPortTest` 和 `ReaderFenceRejectsNewPassiveHandshakes` 用例，因下述依赖限制尚未运行。

扩大到 `tests/disagg`、`tests/integration`、`tests/core` 与 `tests/misc` 后，结果为
**211 passed、16 skipped、2 failed、5 deselected、113 subtests passed**，报告为
`/tmp/minisgl-dram-cpu-final.xml`，日志为 `/tmp/minisgl-dram-cpu-final.log`。
两个失败均已在修改前的 `4d15539` 副本复现，未将其算为本次回归通过。
该轮排除了不能在当前环境执行的 TCP/HTTP 用例和 5 个需要绑定端口的 layered 用例，具体命令为：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/disagg tests/integration tests/core tests/misc \
  --ignore=tests/disagg/test_tcp.py \
  --ignore=tests/integration/test_pd_gateway.py \
  --ignore=tests/integration/test_pd_pipeline.py \
  -k 'not early_handshake_real_scheduler_continuation_and_eos and not client_disconnect_after_first_token_reclaims_layer_reservation and not reservation_disconnect_before_prefill_reclaims_private_pages'
```

当前执行环境为 CPU PyTorch，没有运行真实 CUDA、RDMA 或 TPOT/TTFT 基准。
本地 Mooncake CMake 配置因缺少 `yaml-cpp` 等开发依赖未完成，native fence 修改尚未编译验收。
本机 socket 测试受沙箱限制；HTTP TestClient 在当前环境启动超时，未作为已通过项。
另已在修改前的 `4d15539` 源码副本复现两个既有 layered 用例失败：旧 quiescence 消息断言、旧 adapter install 失败时 lease 断言。

实机验收必须补齐：配套 Mooncake 编译；两轮同 P 请求和更换 P 的输出对照；
D GPU Radix 开/关；D0→P0→D0 的实际后缀字节数；P/D 取消与进程退出；MR 注册次数固定；
P 崩溃后 TTL fence；读超时后迟到 DMA；并发 decode 的 TPOT 与限速效果，以及成本参数标定。
