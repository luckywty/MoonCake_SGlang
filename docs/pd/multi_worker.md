# 多 P/D Worker、负载路由与长连接复用

在多请求组批版本 `16e4cea` 基础上增加本流程。每个 Worker 仍为 TP=1，保留内部 Prefill/Decode batch。
网关只配置地址池和路由，不复制模型或 KV；GPU 数据面仍由 P 直接发往所选 D。
当前默认 P/D 均使用两候选择优，可选 P 缓存亲和、Bucket 和单 Worker 准入，详见[选择算法](routing.md)。

## 单机启动

四块可见 GPU，启动 2P+2D 和一个 CPU tokenizer 网关：

```bash
PYTHONPATH=python python -m minisgl.disagg.launcher \
  --model /models/Qwen3-0.6B --dtype bfloat16 \
  --num-prefill-workers 2 --num-decode-workers 2 \
  --prefill-gpus 0,1 --decode-gpus 2,3 \
  --base-control-port 1919 --base-mooncake-port 19500 \
  --pd-transfer-backend tcp --routing-seed 42 \
  --max-running-requests 32 --max-prefill-length 8192 \
  --host 127.0.0.1 --port 1940
```

也可用 `python -m minisgl --num-prefill-workers ...`，其余参数相同。
`--host/--port` 属于网关；Worker 用 `--worker-host`、`--worker-advertise-host` 和起始端口。
模型路径按实际配置填写；GPU 映射支持逗号或空格列表，使用父进程 `CUDA_VISIBLE_DEVICES` 中的设备标识。
例如父进程可见 `2,4,6,7` 时，映射也填这些标识，子进程各只看到自己的一块设备。
N/M 默认各 1，必须均 >=1，总数不能超过实际可见 GPU 数，设备不能重复。

本例的分配结果：

| Worker | GPU | 控制端口 | 分布式端口 | TE RPC 端口 | 后端数据/握手端口 |
|---|---|---|---|---|---|
| p-0 | 0 | 1919 | 1920 | 19500 | 19501 |
| p-1 | 1 | 1921 | 1922 | 19502 | 19503 |
| d-0 | 2 | 1923 | 1924 | 19504 | 19505 |
| d-1 | 3 | 1925 | 1926 | 19506 | 19507 |

TCP 模式不启动 TE，P 不监听后端数据端口；端口仍保留在启动计划中以避免配置冲突。
Mooncake 模式的后端端口只传控制消息，KV 走 RDMA。端口分配会避开网关端口，非法或重叠配置先报错。
其他 Worker 参数（如页大小、层组、KV 容量）传给所有子进程；身份和端口不能被透传参数覆盖。

Mooncake 场景将后端改为 `--pd-transfer-backend mooncake`，并指定本机可达的 `--pd-mooncake-host`。
可选每卡亲和参数示例：

```text
--prefill-numa-nodes 0,0 --decode-numa-nodes 1,1
--prefill-rdma-devices mlx5_0,mlx5_1 --decode-rdma-devices mlx5_2,mlx5_3
```

NUMA 配置通过 `numactl --cpunodebind/--membind` 应用，需安装 numactl；数量与对应角色 Worker 数一致。
HCA 列表允许不同 GPU 共用 NIC；一个 Worker 使用多个 HCA 时以 `+` 连接，例如 `mlx5_0+mlx5_1`。
NIC、NUMA 与模型示例都需按实际拓扑调整。

固定 TE 端口使用每进程 `MC_MIN_RPC_PORT=MC_MAX_RPC_PORT`，并通过 `get_rpc_port()` 核实实际值。
本地 classic TE 排除 32768～60999 的临时端口区间，因此显式 TE 端口需为 1024～32767 或 61000～65535。
若安装扩展忽略该范围，Worker 明确报错，不把自动分配端口伪装成固定端口。

启动器用独立进程组和独立 `CUDA_VISIBLE_DEVICES` 启动各 Worker；本地节点全部就绪后再启动网关。
主进程收到 SIGINT/SIGTERM 或启动失败时，回收它创建的全部子进程；超出关闭时限则终止残留进程组。
运行中个别 Worker 退出会记录日志，健康池停止选择它；若某个角色全部退出或网关退出，启动器结束并清理本地子进程。
`--no-gateway` 只启动本机 Worker，由其他机器上的网关连接。纯 P 或纯 D 机器使用原有 `--worker-role` 入口。当前没有自动重启 Worker。

## 跨服务器地址池

先在各服务器独立启动 Worker，设好可达的控制/后端/TE 地址和端口，再运行网关：

```bash
PYTHONPATH=python python -m minisgl.disagg.gateway \
  --model /models/Qwen3-0.6B \
  --prefill-addrs 10.0.0.11:1919 10.0.0.12:1919 \
  --decode-addrs 10.0.0.21:1929 10.0.0.22:1929 \
  --routing-seed 42 --health-interval 2 --health-timeout 3 \
  --worker-ready-timeout 300 --max-quarantined-bytes 0 \
  --host 0.0.0.0 --port 1940
```

网关和 Token ID 客户端都接受 `--prefill-addrs/--decode-addrs`；可显式传 N/M 校验列表长度，否则由列表推导。
地址必须跨角色唯一。`launcher` 的远程地址模式只启动网关，不要求本机有 GPU，也不会管理远程 Worker 进程。
原有 Gateway `--prefill/--decode` 可作为单节点地址池简写；Python `PDClient` 保留原有单端点协议，
`RoutedPDClient` 提供多节点池。不能混用单端点与地址列表参数。

## 连接与请求流程

- 每个网关到每个 Worker 维持一条连接，初次协商 multiplex v1；只对完整帧写入加短锁。
  所有上下行帧携带 `request_id`，与该请求的 `transfer_id` 相同，响应严格分入各自有界队列。
  旧单请求连接仍兼容，但不会被多节点客户端当作复用连接使用。
- 先匹配长度 Bucket、筛选可准入节点，P/D 默认各随机抽两个候选按阶段负载择优。
  `--routing-policy cache_aware` 可优先选择前缀命中的 P；显式 `random/round_robin` 保留兼容策略。
  `pd_route` JSON 记录 `request_id`、`prefill_id`、`decode_id`、`bucket`、角色策略、`attempt`、`event`。
  Pool 的逻辑 ID 按地址列表顺序为 p-0/p-1、d-0/d-1；本地启动器使用同样的 Worker ID。
- 在 P 计算前对 D 发 `reserve`。D 先分配 prompt 私有页，计入所有活跃请求未来输出的页预算；
  Mooncake 返回预注册池切片票据，TCP 返回前置预留确认，并在 handoff 后通过 `prepare` 绑定接收票据。
- 只有容量不足、退役/不可用或连接失败才在尚未使用过的健康 D 中重选，每个 D 最多试一次，总重试不超过 M-1。
  跨 Bucket 共用已尝试 D 集合；Bucket 没有可用 P/D 对时退还 P 计数、尝试下一个匹配 Bucket。
  未确认的旧预留按原 request_id 取消；若取消未送达，Worker 的请求控制期限负责撤销。
  模型/请求校验错误直接返回；全部 Bucket 耗尽时不启动 Prefill。
- 成功 reserve 后才向所选 P 发 Prefill。后续首 token、sent、commit、installed、token、cancel 都走同一请求通道。
  不在 Prefill 开始后自动改选 P，也不在已输出首 token 后重放请求。
- P 的一个 batch 可以向不同 D 发 KV；不同 P 的 TE session 可以并发写入同一 D 的不同租约。
  源页、共享槽位、接收切片和 install event 的所有权规则沿用 [多请求调度](batching.md)。

种子固定的是相同候选与调用顺序下的抽样序列；并发到达顺序、负载或健康变化不同，逐请求分配可能不同。
选路只观察负载；D 的 reserve 和 P 的组批分配仍执行最终页容量检查，不使用 `Semaphore(1)` 串行化 Worker。

## 健康巡检与故障边界

默认每 2 秒通过同一复用连接发送 health，每次等待 3 秒。
仅接受角色、协议版本、KV 布局和传输模式兼容的节点；Mooncake 层组大小也需一致。
网关运行期间固定模型身份，重连必须重新 health；切换模型需要重启网关。地址池节点均需支持 multiplex v1。
`status/transfer.status/native.state` 为不可用、native reason 表示退役，或 native `quarantined_bytes` 超过阈值时，
节点从候选池移除；默认阈值 0。检测到节点重新健康后可重新加入。
这只影响新请求选择，不把活跃请求搬到其他节点；连接断开会失败该连接上的请求，客户端不透明重放。

`--max-inflight-requests/--max-inflight-tokens` 控制网关总准入；Worker 的页、请求、传输预算仍独立执行。
`--max-worker-inflight-requests` 控制单 Worker 的 Router 在途数，其他四种可选阈值读取新鲜 health 负载；
Bucket 格式、缓存亲和、过期指标处理和分配时机见[路由说明](routing.md)。
健康查询不占模型表行，也不等当前请求完整结束。有界输出队列防止单流无限堆积；
慢客户端溢出时取消该流。帧写入失败会关闭损坏的连接，防止后续帧混入半包。

## 实现与测试

入口：[launcher.py](../../python/minisgl/disagg/launcher.py)、[cluster_config.py](../../python/minisgl/disagg/cluster_config.py)。
控制面：[multiplex.py](../../python/minisgl/disagg/multiplex.py)、[mux_server.py](../../python/minisgl/disagg/mux_server.py)。
路由：[router.py](../../python/minisgl/disagg/router.py)、[routing_policy.py](../../python/minisgl/disagg/routing_policy.py)、[routed_client.py](../../python/minisgl/disagg/routed_client.py)。

```bash
PYTHONPATH=python python -m pytest -q -o addopts= \
  tests/integration/test_pd_cluster_config.py tests/integration/test_pd_routing.py \
  tests/integration/test_pd_policy.py
```

测试使用真实 Worker 调度、路由、复用连接分流及 CPU attention，替换 OS socket、CUDA event 和 native DMA。
覆盖资源/端口/NUMA/HCA 配置、子进程清理、原生端口参数核对、种子选择、2P→2D、reserve 重选、
全部 D 拒绝时 P 不计算、健康剔除/恢复、取消隔离和首 token 后失败不重放，另验证负载准入、Bucket 回退与缓存亲和。
本机缺少真实 CUDA/RDMA 环境；多进程 GPU、跨机器网络与性能仍需目标硬件验收。

当前无 socket 子集与复现命令见[项目交接文档](../../交接文档/项目交接文档.md)，本机报告 `/tmp/minisgl-policy-routing-cpu.xml`。
TCP/HTTP 真实网络集合不在该子集中，真实四卡/多进程 GPU、跨服务器与 RDMA 尚待验收。
