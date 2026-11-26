# D 本地 GPU Radix Cache

本功能控制 Decode Worker 是否跨请求保留、匹配和共享 GPU KV 前缀。
默认关闭；该开关仅控制 GPU Radix。独立的 [P/D DRAM 缓存](dram_cache.md)支持异步卸载和 D→P 按需拉取，
即使 GPU Radix 关闭也可启用；结束时须等 D2H reader 完成才归还 GPU 页。
P 和 Unified 沿用原有缓存配置。

## 配置

在现有 D Worker 启动命令中追加：

```bash
--disaggregation-decode-enable-radix-cache
```

独立启动时该参数只允许用于 `--worker-role decode`。
多 Worker 启动器也接受此参数，仅传给本机 D 子进程；远程地址模式须在远程 D 启动时设置。
D 关闭时使用 Naive 前缀缓存，开启时使用 Radix；D 的选择独立于原有 `--cache-type`。
开关在进程启动时确定，不支持运行中切换。

| 阶段 | 关闭 | 开启 |
|---|---|---|
| reserve | 分配完整 prompt 私有页 | 匹配并锁住完整页前缀，只分配后缀私有页 |
| P→D | 传完整 prompt KV | 仅传 D 缺失的后缀 |
| 安装 | scatter 完整 prompt | 保留只读前缀，scatter 后缀 |
| 生成 | 使用当前请求的上下文 KV | 使用共享前缀和请求私有的新 KV |
| 正常结束 | 归还请求 KV 页和表行 | 缓存有效前缀，归还尾部/重复页和表行、解除请求锁 |

## 范围与容量

设完整 prompt 长度为 N，D 命中前缀为 L，实际传输 token 数为 T：

```text
T = N - L
D page_table = 已有前缀 [0,L) + 私有后缀 [L,N) + 后续 Decode 页
```

L 按页向下对齐。D 先锁住前缀，再检查锁后容量，避免将命中页同时计入可淘汰容量。
沿用现有保守输出预算：新请求额外需求为 `align(N + max_tokens) - L`，
还需计入其他请求尚未分配的未来输出页。输出页仍随 Decode 逐步分配。
`--pd-max-transfer-bytes` 检查后缀字节数，序列长度和排队 token 预算仍按完整请求计算。

P 的 Prefill 起点由 P 自己的缓存决定。D 的 L 只控制发送范围，不会让 P 自动获得 D 的生成 KV。
未命中部分仍在 P 上计算；换到其他 D 时，只使用所选 D 的本地前缀。

## 协议与数据路径

当前客户端（单端点和 Routed）均在 P 计算前向 D reserve。
reserve 返回 `decode_prefix_len`、`prompt_tokens` 和协议版本；客户端把前缀边界转交给 P。
handoff/metadata 升级为 v2，`kv_tokens` 表示后缀长度，`prefix_len + kv_tokens` 必须等于完整 prompt 长度。
层组票据升级为 v3，层偏移按后缀长度计算，底层传输标识为 `mooncake-layered-gpu-v3`。
新客户端和路由健康池要求配套的新 Worker，不支持混合旧版进程。
编解码器仍能读取旧的全量 metadata/handoff；旧 `decode` 控制操作只允许全量传输，不参与前缀省传输。

```text
TCP：reserve → P Prefill/handoff → D prepare/ready → P send/sent → commit → installed
Mooncake：reserve/后缀票据 → P Prefill与逐层发送 → handoff/sent → commit → installed
```

两条数据路径均只 gather/export `[L,N)`，并只安装到 D 的 `[L,N)`。
Chunked Prefill 仍在最后一个 token chunk 按层组发送所需后缀。
`L=N` 时不创建接收切片、主机 payload 或发送任务，不提交零字节 RDMA；
P 仍按自己的缓存计算 logits、返回首 token，D 完成控制提交后以缓存 KV 续算。
首 token 已结束时，客户端释放 D 预留，不启动 Decode。

## 缓存与回收

所有缓存、页表和页分配修改仍由单个 GPU 调度线程完成。
新接收的 KV 在安装 event 确认完成前不发布到 Radix。
安装完成后 checkpoint prompt；并发相同前缀去重时，先重绑到缓存的规范页，再归还重复私有页。
请求持有新缓存 handle 的锁，保证 Decode 期间这些页不可淘汰。

正常结束仅缓存 `req.cached_len` 内已计算的、按完整页对齐的前缀，包含上下文和有效生成 KV。
最后刚采样出的 token 通常还没有 KV。页不足时可淘汰未受引用保护的缓存；
释放表示池容量可复用，不会逐字节清零，也不要求 `nvidia-smi` 占用下降。

取消/失败只归还本请求的私有页并解除锁，不清掉其他请求正在使用的前缀。
P 源页、原生 staging、D 安装 reader 继续遵循原有 event/drain/隔离条件。
缓存插入或页表重绑失败、无法确定资源所有权时要求 Worker 退役，避免二次释放。

## 观察与验证状态

health 增加：

- `decode_radix_cache`：D 开关状态；P 为 false。
- `handoff_version=2`、`decode_prefix_reuse=true`：协议能力。
- `decode_cache_hit_requests` / `decode_cache_hit_tokens`：成功 reserve 的命中请求/前缀 token 累计，包含后续主动释放的预留。
- `received_kv_tokens`：成功安装的后缀 token 累计；完整命中增加 0。

原有 `free_pages`、`protected_tokens`、`evictable_tokens` 仍可观察页占用和引用保护。
开启时空闲状态不要求全部页回到 free list，未被锁定的 Radix 页可以继续保留。

2026-11-26 新增独立 CPU 回归 `test_pd_decode_radix.py`，覆盖页大小 1/4、开关开/关、多轮生成 KV、
完整命中零 payload、实际后缀字节数、并发规范页去重、取消、安装失败及重试。
补跑真实 TCP 的首 token EOS 用例，验证 reserve-first 后释放预留且不提交传输或 Decode。
扩大集合为 **356 passed、16 skipped、0 failed**，命令和适配器边界见 [CPU 回归记录](cpu_regression.md)。
CPU 结果不替代真实 CUDA/RDMA 模型验收。
