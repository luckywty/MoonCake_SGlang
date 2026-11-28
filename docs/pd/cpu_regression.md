# P/D 缓存 CPU 回归

验证日期：2026-11-27（Asia/Shanghai）。对应三个功能提交：`5e1e21e`（D Radix 与后缀传输）、
`4d15539`（P 页面哈希索引）、`deb7e11`（P/D DRAM 缓存）。首次补测修改测试、适配器、运行脚本和文档，
未放宽完成凭证校验；后续增加模型容量启动校验、health 参数告警及三方规划限额、统一端口防碰撞。
在 `1899a5d` 上进一步增加显式主通道 Host 中转，覆盖两道 DMA 屏障、混连拒绝及 TE/缓存池隔离。

## 结果与命令

环境：Python 3.12.3、PyTorch 2.9.1+cpu、pytest 9.2.0，无 CUDA。首次补测新增 46 项，
启动期防呆另增 63 项；Host 中转再增 48 项 CPU 回归和 3 项 CUDA 参数化用例。
该阶段扩大集合结果：**404 passed、19 skipped、117 subtests passed、0 failed**；未排除 TCP、HTTP 或逐层 socket 用例。
19 个跳过项需要 CUDA/GPU 环境，其中真实模型对照还需显式配置模型。
JUnit 报告：执行机 `/tmp/minisgl-host-dram-cpu.xml`，不随仓库发布。

### 2026-11-28：shm_dram 实现与验证进度

本轮先在原始 `8d693f8` 快照重跑完整集合，确认 **404 passed、19 skipped、117 subtests passed**。
加入共享中转后端后，完整集合曾通过 **431 passed、19 skipped、117 subtests passed**，
其中新增 27 项覆盖真实 POSIX 段、多 P 进程分区、独立 P/D 进程、逐组安装与槽位循环、
两道 CUDA fence 的 CPU 模拟、取消排空、完成未知保护、NUMA 选路与协议版本。

最终新增用例补至 38 项。之后沙箱内复验：既有配置/协议集合 **107 passed、69 subtests passed**，
新拓扑/路由/启动校验、直连客户端等纯逻辑集合 **19 passed**；Ruff 与 diff whitespace 检查通过。
最后补充的完整前缀命中和 SIGTERM 实际子进程清理用例尚未运行，最终完整集合仍待重跑；
不能把此前 431 的结果当作最终全部新增用例的验收结果。
环境仍为 CPU PyTorch，无 CUDA；没有执行真实双进程 CUDA 注册或 NUMA 带宽验收。

本轮报告在执行机 `/tmp/minisgl-shm-baseline.xml`、`/tmp/minisgl-shm-regression.xml`、
`/tmp/minisgl-shm-final-config.xml`、`/tmp/minisgl-shm-pure-final.xml`。
共享中转参数和真机预检见 [shm_dram 说明](shm_dram.md)。

在项目根目录、已安装测试依赖且允许本机 socket/子进程的环境中运行：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/disagg tests/integration tests/core tests/misc \
  --tb=short --junitxml=/tmp/minisgl-host-dram-cpu.xml
```

本机已有虚拟环境时把 `python` 换成 `.venv/bin/python`。
`TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 禁用 CPU 验证不需要的可选 Torch DLPack JIT。
统一入口也已改为按这四个目录发现测试，后续新增文件会自动纳入：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 FLUXINFER_TEST_PYTHON=python bash tools/pd/test_phase2.sh
```

## 两个基线失败的修复

两个旧测试在 `4d15539` 已失败，原因都是伪造的完成凭证仍写死 `mooncake-layered-gpu-v2`，
而该版本实现已经使用层组票据/完成协议 v3。实现提前拒绝凭证，测试没有进入预期路径。

| 原用例（`test_pd_layered.py`） | 修复与保留的约束 |
|---|---|
| `test_stale_or_incomplete_certificate_never_releases_another_lease` | 凭证随 `LAYERED_TICKET_VERSION` 构造，继续验证缺少 quiescence 证明不能回收、旧凭证重放不影响新租约、改写已记录的凭证必须拒绝 |
| `test_layered_decode_failure_rolls_back_private_pages[install]` | 使用当前凭证，明确断言安装注入函数调用一次、抛出指定异常，并验证私有页、表行和租约回收 |

另补四种非法 backend 凭证：旧版、未来版本、空值及布尔值。均须继续拒绝，接收缓冲区在获得有效排空证明前不得复用。

本轮补跑此前排除的网络集合时，还发现旧 `test_first_token_eos_short_circuits_decode` 把 D 指向不可达端口，
假定 EOS 请求不连接 D。这与现有 reserve-first 协议冲突。替换后的真实 TCP 用例连接有效 D，
检查 `reserve → prefill → release`、不发送 `prepare/send/commit`、接收 KV 计数不增长，且两端资源回到空闲。

## 覆盖映射

下表数量包含参数化展开，DRAM/路由文件还包含此前已有用例。

| 功能 | 文件 | 关键断言 |
|---|---|---|
| D Radix、后缀边界 | [test_pd_decode_radix.py](../../tests/integration/test_pd_decode_radix.py)，9 项 | 页大小 1/4、开关开/关、多轮生成 KV、最后采样 token 不缓存；实际传输字节数等于 `(N-d_prefix_len)*bytes_per_token`；完整命中不发 payload；并发安装重绑规范页；取消及安装失败后保留共享前缀并可重试 |
| P 页面哈希目录 | [test_pd_prefix_index.py](../../tests/integration/test_pd_prefix_index.py)，26 项 | 固定 SHA-256 向量、域/namespace 隔离、整页边界、真实 Radix 发布/拆分/淘汰；分页冻结快照及追赶、日志溢出、坏增量、重启旧回复、过期/失联、容量与孤儿链；实际路由命中，过期 Hint 强制回到真实计算 |
| 分词/哈希任务边界 | [test_pd_prompt_prepare.py](../../tests/integration/test_pd_prompt_prepare.py)，3 项 | 取消/超时后运行中任务继续占额度、完成后才接纳下一请求、迟到结果不串请求、submit 失败释放额度；不下载 tokenizer 或启动实际准备进程 |
| DRAM 与反向拉取 | [test_pd_dram.py](../../tests/integration/test_pd_dram.py)，25 项 | 标准接缝页、事务提交/取消、延迟 D2H reader、GPU Radix 关闭时 GPU 全归还；单源及最低收益/字节门槛；D DRAM→P 提升不改变 P→D 边界；超时隔离及 TTL fence；新增 LRU/LFU 实际淘汰选择和第二页读超时后的部分前缀续算 |
| 缓存介质成本 | [test_pd_policy.py](../../tests/integration/test_pd_policy.py) | `balanced` 对远端 RDMA+H2D 折算额外成本，同样的命中数量不能直接等同于 HBM 命中 |
| 启动期防呆 | [test_pd_startup_validation.py](../../tests/integration/test_pd_startup_validation.py)，63 项 | 默认 64 MiB 临界值、满页向上对齐、权重/Engine/pinned 池前拒绝无解配置；health 不一致去重告警、缺失字段降级；P/D/Router 三方限额裁剪；全种类跨 Worker 冲突、Host TE 默认偏移、网关冲突、TE 临时端口区间及首个子进程前拒绝 |
| 完成协议与网络 | [test_pd_layered.py](../../tests/integration/test_pd_layered.py)、[test_pd_pipeline.py](../../tests/integration/test_pd_pipeline.py) | 两个旧失败、非法协议下的 reader 保护、真实 socket/两进程续算及 EOS 预留释放 |
| 主通道 Host 中转 | [test_pd_layered.py](../../tests/integration/test_pd_layered.py)、[test_pd_batching.py](../../tests/integration/test_pd_batching.py)、[test_pd_routing.py](../../tests/integration/test_pd_routing.py) | D2H event 完成前禁止提交写、源 HBM 与在途 pinned ring 独立复用、安装 event 前后切片回收、两种客户端混连拒绝、后缀与完整命中、EOS/断连、UNKNOWN/FAILED_DRAINED 隔离、回执只补送一次 |

共享适配器 [pd_cache_reference.py](../../tests/integration/pd_cache_reference.py) 复用真实请求握手、
CacheManager、Radix、页表、MHAKVCache、哈希发布与镜像同步。CPU Radix 测试仅替换原生最长前缀比较原语，
保留树结构、引用、拆分、去重和淘汰实现。输出与独立逐步重算前缀的 CPU attention 参考结果对照。

## 验证边界

真实 TCP/HTTP/跨进程基础集合已运行。新增 D Radix 的 TCP 协议后缀测试使用内存控制通道及 Mock 字节传输；
逐层/DRAM 测试替换 CUDA event、pinned 分配和原生 DMA，并控制延迟、失败和迟到完成。
这些测试验证协议、KV 内容续算和资源所有权，不能证明真实 CUDA 流重叠、RDMA 可见性、TTFT/TPOT 或带宽收益。

配套 Mooncake 的原生编译、GPU 字节对照、跨服务器、多请求压力及进程崩溃后的 reader fence
仍需实机验证，见 [DRAM 验收清单](dram_cache.md#验证与实机验收)。
Host 模式的独立 pinned/CUDA 双流往返也提供 opt-in 检查；CPU 适配器不能证明主 TE 的真实 Host RDMA 或 TPOT，
配置与验收边界见 [Host 中转说明](host_dram_transfer.md)。
