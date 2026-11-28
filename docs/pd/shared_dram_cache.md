# 单机 NUMA 共享历史 KV 缓存

更新日期：2026-11-28（Asia/Shanghai）。基于 `688378a` 的共享中转后端，新增显式
`--pd-dram-cache-mode shared_shm`。默认仍为 `private`，现有私有 `dram_pool.py`
与独立 Mooncake Host TE 跨机拉取路径保持原有行为。

## 三种 Host 内存的区别

| 内存 | 管理者与用途 | 数据生命周期 |
|---|---|---|
| 共享历史池 `shared_dram_pool.py` | Launcher 建池，统一目录管理 P/D 的完整历史 KV 页 | 跨请求保留，LRU/LFU 淘汰；读保护和写入期间不可覆写 |
| 共享中转池 `shm_relay_pool.py` | P 的静态主/溢出段，逐层组 P→D 交接 | D 安装完成并归还后可循环使用 |
| 私有历史池 `dram_pool.py` | 每个 Worker 的 pinned 页池，可配独立 Host TE 拉取 | 沿用私有事务、引用、远端租约和淘汰规则 |

共享历史池与共享中转池使用不同名字、manifest、控制目录和槽位状态，不互相借用容量。
历史缓存模式与主传输后端独立，可搭配 `tcp`、Mooncake `gpu_direct` / `host_dram` 或
`shm_dram`。在 `shared_shm` 模式下禁止同时开启 `--pd-dram-remote-pull`；
跨机拉取继续使用 `private` 模式，不把共享历史页交给 Host TE。

## 数据和恢复路径

每个实际 NUMA 节点一个 POSIX SHM 段 `<pool>_numa_<node_id>`，支持动态节点列表，
包括没有选中 GPU 的宿主机 NUMA 节点。每槽位先放 256 字节控制头，随后为 256 字节
对齐的一个完整全层 KV 页 `[layer,K/V,token,head,dim]`。
控制头包含单调递增 `generation` 与 `FREE / STAGED / READY / QUARANTINED`。

P/D 均可保存已计算 KV 的完整页。D 包含收到的完整 prompt 页、Decode 新页和接缝页，
与旧私有模式“只缓存包含 D 新计算 token 的完整页”不同；全局页哈希去重会跳过已有页。
最后采样但尚未计算 KV 的 token 不缓存。不会因为 offload 成功而逐层释放活跃请求 HBM。

写入路径：

```text
D（或 P）正式 HBM KV 页
  → 独立 offload_stream 上的 GPU 打包缓冲
  → 目录分配的本地/对端 NUMA 共享历史槽位（D2H）
  → 本地 CUDA event 完成 → commit_write → READY
```

分配按“本地 FREE → 其他 NUMA FREE → 本地最冷 READY 未保护页 → 其他 NUMA 最冷
READY 未保护页”的顺序进行。页已 READY/STAGED 时跳过重复写；全局无安全槽位时返回
`full` 并少缓存，不等待容量、不阻塞 Decode。元数据 RPC 在独立有界准入的后台执行器运行，
CUDA 提交仍由调度器线程负责。已提交 D2H 的源 GPU 页在 event 完成前保留。

P 恢复路径：

```text
本地 HBM Radix 实际匹配并锁定
  → 对其后连续共享历史页 pin_prefix
  → 每页所在 NUMA 的共享 DRAM
  → 直接 H2D 到 P 的 GPU 恢复缓冲 → Scatter 到 P 正式 KV 页
  → event.query() 就绪且复核 (generation, READY) → unpin
  → 所有命中页恢复完成 → Prefill 计算缺失后缀
```

远端 NUMA 页直接跨 CPU 互连进入 P GPU，不先复制到 P 本地 NUMA DRAM；本流程没有
额外的 Host payload 池或 Host→Host memcpy。固定 GPU 恢复缓冲沿用现有完整页打包/Scatter
实现。本次没有加入“加载一层就计算一层”；P 仍等待本次命中的全部层/页恢复完毕再 Prefill。

D GPU Radix 开关及 reserve-first 协议保持原义：关闭时 `d_prefix_len=0`；开启时可在
实际锁定 HBM 后从共享历史池恢复连续页。P→D 只按 D reserve 返回的实际前缀发送缺失后缀，
不按 Router Hint 扩大跳过范围，全命中仍不写主传输 payload。

## 目录与故障边界

`SharedDramDirectory` 在 Launcher 内托管 Unix Domain Socket 服务，只传哈希、槽位和租约等
元数据，不经手 KV payload。`lookup_prefix` 使用不可变分片快照，无目录锁，结果仅为 Hint；
`pin_prefix` 原子复核连续 READY 页并增加引用。Router 对每个 P 计算本地/跨 NUMA 命中页的
恢复成本，`--pd-shared-dram-remote-h2d-factor` 默认 `1.5`，仅为估算系数。
P/D 必须属于同机、同共享池、同模型/布局/hash domain。

只有 GPU 写 event 完成才发布 READY；只有 GPU 读 event 完成并复核槽位头后才发 unpin。
取消时未发射页可立即归还，已发射 H2D/D2H 必须排干。与私有模式的整请求事务不同，
共享模式逐页发布已经完成的 KV，`cache_commit/cache_abort` 不撤销其他请求已保护的完整页。

Socket 断开本身不是 DMA 排空证明：仍存活的进程持有的读引用保留，未确认的写槽位隔离为
QUARANTINED。重连后可凭本地 event 完成提交/归还；确认进程退出后，目录通过 pidfd 或
`/proc` 的进程身份/状态自动回收未完成写槽位和未释放读引用。读取进程状态失败时保守保留。
RPC 序号与缓存回执防止丢失 pin/allocate 回执重试时重复增加引用或重新分配。
目录/回执状态无法确认时 Worker 停止服务，依靠进程退出边界回收，不继续冒险复用内存。
目录不能在保留旧 SHM/DMA 所有者的情况下无状态重启。

Launcher 独占创建和 unlink。先由目标 NUMA 初始化子进程执行 CPU 绑定、`mbind(MPOL_BIND)`、
first-touch，再启动服务和 Worker；每个 GPU Worker 映射并 `cudaHostRegister` 全部分段。
memlock、NUMA 策略或驱动不支持时显式报错，禁止未锁页降级。正常退出、异常和
SIGINT/SIGTERM 均先停止并等待所有子进程，再关闭目录和 unlink。SIGKILL 无法执行清理钩子。

## 启动与容量

通过多 Worker 启动器管理共享池，示例参数需结合模型和硬件容量调整：

```bash
PYTHONPATH=python python -m minisgl \
  --model /models/model --dtype bfloat16 \
  --num-prefill-workers 2 --num-decode-workers 2 \
  --prefill-gpus 0,2 --decode-gpus 1,3 \
  --pd-transfer-backend shm_dram \
  --pd-dram-cache-mode shared_shm \
  --pd-dram-cache-bytes 8589934592 --pd-dram-cache-policy lru \
  --pd-max-transfer-bytes 268435456 --pd-max-registered-bytes 12884901888 \
  --routing-policy cache_aware --affinity-mode balanced \
  --pd-shared-dram-remote-h2d-factor 1.5
```

`--pd-dram-cache-bytes` 在共享模式表示**整台主机的共享历史池总预算**，包含控制头和
OS 页对齐填充，不是每个 Worker 各分配一份。默认均分实际 NUMA 节点，可用
`--pd-shared-dram-numa-bytes '{"0":4294967296,"1":4294967296}'` 显式指定各节点预算；
节点必须匹配探测清单，且每节点至少容纳一页。未用满的对齐余量不另行分配。

`--pd-max-registered-bytes` 在本模式约束每 Worker 的**历史共享映射 + 主通道注册内存**
合计。主通道只能使用扣除共享历史池后的预算；主 `shm_dram` 计入完整 relay 段，
Mooncake `host_dram` 继续用 `host_staging_plan` 校验 sender ring / receive pool。
启动器检查 `/dev/shm` 可用空间能同时容纳历史池与 relay，Worker 在加载权重前检查
模型页几何、host/Worker/NUMA 身份、段尺寸及合计注册预算。

复用 `MINISGL_NUMA_NODES`、`MINISGL_GPU_NUMA_MAP` / 启动器对应参数注入 CPU 模拟拓扑，
不改变生产环境严格的 NUMA 绑定要求。Launcher 生成权限受限的 manifest 与本地 socket；
其中的 secret 不包含在 Worker health/capabilities 返回中。

## 验证

CPU 回归命令与实际结果见 [cpu_regression.md](cpu_regression.md)。新增用例覆盖
单/双/多 NUMA、溢出与 LRU/LFU、pin 保护、两道事件屏障、取消排空、generation 防 ABA、
直接读取远端页、同池选路、RPC 回执重放、进程崩溃回收及 Launcher SIGTERM 清理。

新增独立真机预检，不改动原中转预检脚本：

```bash
PYTHONPATH=python python tools/pd/verify_shared_dram_history.py \
  --prefill-gpu 0 --decode-gpu 4 --numa-nodes 0,1
```

选 NUMA 0 上的 P GPU、NUMA 1 上的 D GPU。两个进程同时注册全部 SHM，D 依次向各 NUMA
发布完整页，P 从原段直接 H2D/Scatter 并验证全层 K/V；输出同/跨 NUMA D2H、H2D 带宽与
设备信息。当前 CPU 环境仅检查脚本入口，真实 CUDA 注册、双路跨路正确性/带宽、
CUDA 流重叠、模型 TTFT/TPOT 和长期负载稳定性仍须真机验收。
