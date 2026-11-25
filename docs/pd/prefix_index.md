# P 本地 HBM 页面哈希链索引

本文描述兼容保留的 v1 HBM 索引。启用本地 DRAM 后，还同步支持 P/D HBM/DRAM 的 v2 目录，
并支持单源 D→P 拉取和介质成本折算，见 [DRAM KV 缓存](dram_cache.md)。

`--routing-policy cache_aware` 使用 Router 的页面哈希镜像比较候选 P 的本地 HBM 前缀。
Worker 内的 token IDs Radix 树继续管理实际 KV 页；页面哈希索引只提供选路 Hint。
本次不引入 CPU KV 缓存、HiCache、Decode offload、Mooncake Store 或 D→P KV 拉取。

## 请求与发布流程

1. 网关启动时探测 Worker，从健康 P 的布局和静态 namespace 构造哈希域。命令行网关使用独立
   `spawn` CPU 进程完成分词和页面哈希，返回 token IDs 与打包的 32 字节摘要，不在 HTTP 异步循环计算长 prompt。
   该进程有排队上限和超时，取消后丢弃结果；运行中的 CPU 任务完成前仍占用排队额度。
   嵌入式 `create_app` 未提供准备进程时使用线程；Token ID 客户端在原有阻塞调用线程计算哈希。
2. Router 为每个健康 P 后台发送 `prefix_sync`，复用现有控制连接，不新增网络端口。
   请求选路只读本地镜像，不再逐请求向所有候选 P 发送 `prefix_match` RPC。
3. Router 对每个候选 P 从第 0 页逐页查询，遇到第一个未命中立即停止；按命中 token 数应用既有门槛和负载公式。
4. 请求仍先向选定 D 执行 `reserve`。D 返回的真实锁定前缀决定 P→D 待传后缀，P 的选路 Hint 不参与裁剪。
5. P 的 PrefillManager 仍以本地真实 Radix 匹配、锁定、检查容量及分配页；Hint 不构成 KV 存在或页租约的证明。
6. P 完成 Prefill，并在 reader 安全释放后由 `CacheManager.cache_req` 把有效 KV 插入真实 Radix，
   完成去重与页所有权更新，才发布整页哈希。排队请求、未完成 Prefill、Chunked Prefill 中间私有页不发布。
   完整有效 KV 因取消或完成而继续留在真实缓存时，元数据也继续保留。
7. 真实 Radix 淘汰叶节点时删除对应页记录；结构拆分只更新页与节点的归属，不改变内容摘要。
   这些事件在 GPU 调度所有者线程产生；RPC 线程只读取 CPU 元数据，不访问 GPU Tensor。

## 哈希与页边界

使用完整 SHA-256 摘要，不使用 Python 内置 `hash()` 或短整型截断。根种子为：

```text
manifest = canonical_json({
    hash_version: 1,
    model: checkpoint_fingerprint,
    version_group: {
        kv_layout, dtype, layers, kv_heads, head_dim,
        tp_rank, tp_size, packing, quantization
    },
    page_size,
    cache_namespace
})
H0 = SHA256(b"minisgl-prefix-root-v1\0" + uint32_le(len(utf8(manifest))) + utf8(manifest))
Hi = SHA256(b"minisgl-prefix-page-v1\0" + H(i-1) + int32_le(tokens_of_page_i))
```

`model` 使用既有 KVLayout 的检查点 SHA-256 指纹。`kv_layout` 固定为
`mha-kv-layer-page-token-head-dim/v1`；manifest 按键排序、无空格，以 UTF-8 编码。
token ID 必须为非负 int32。不同 P 的 `worker_id` 和 `instance_id` 不参与内容哈希；
`instance_id` 独立约束记录的生命周期。namespace 是 Worker 启动配置，P/D 需一致；未实现请求级租户缓存隔离。

设 prompt 长度为 `N`，`B = MHAKVCache.page_size = CacheManager.page_size = KVLayout.page_size`。
启动时检查三者相等，只生成 `N // B` 个满页摘要。尾部残页不增加命中；`N < B` 时命中为 0。
例如 `B=16, N=35` 只探测两页，最长 Hint 为 32 token。若第 0 页缺失，即使后续摘要存在也返回 0。

```text
hbm_hit_tokens = matched_pages * B
reuse_hint = min(hbm_hit_tokens, floor((N - 1) / B) * B)
balanced_work = pending_prefill_tokens + N - reuse_hint
```

命中 token 门槛与命中比例使用 `hbm_hit_tokens`，比例的分母仍为完整 prompt 长度。
`balanced` 使用 `reuse_hint`，对应真实 P 查询 `prompt[:-1]` 后按页对齐的边界。
因此即使 HBM Hint 覆盖全部满页，也不能把 P 前向成本计为零。

## 同步、失效与容量

每个 WorkerService 启动生成新的 UUID `instance_id`。health 发现实例变化、不可用或 retired 时，
Router 清空该 P 全部页记录；延迟到达的旧实例响应不能写回新镜像。
health 的 `prefix_index` 和独立的 `prefix_index_version=1` 能力描述该机制，不改变 handoff v2 或层组票据 v3。

P 为每次页面增删分配单调 `seq`，有界日志记录 `add/remove/reset`。初次同步或发现序号缺口时，
Router 分批获取固定序号的快照，再顺序补齐日志；完成追赶前不会使用该镜像选路。
每批最多 512 页/事件，每轮每个 P 最多 16 批。快照仅包含有界 CPU 元数据，最多保留两份、30 秒过期；
多个 Router 可以独立读取同一冻结快照。日志不足以追赶时重新获取快照。

源端元数据上限取 `min(真实 KV 池页数, pd_prefix_max_pages)`，有界事件日志独立控制。
源端达到上限时只删除元数据叶页，不驱逐或锁定真实 HBM 页；超长链允许少报。
Router 对每个 P 及全体镜像设置条目上限，按 LRU 删除记录并递归清理孤儿后缀，其他 P 的记录不受该删除影响。
索引中不保存 GPU Tensor、物理页索引或 Radix handle，不延长 HBM 页的生命周期。

后台同步失败只清空 Hint，不因此退役推理 Worker；镜像过期、未就绪、哈希域不匹配、旧 Worker 或非 Radix P
均按无命中回退 `power_of_two`。缓存真实淘汰与后台同步之间可能有短暂延迟，因此执行时必须重新校验本地 Radix。

## 配置与源码

默认路由仍为 `power_of_two`。在网关、地址列表 Token ID 客户端或本地多 Worker 启动器上增加：

```text
--routing-policy cache_aware --decode-policy power_of_two
--affinity-mode balanced
```

| 参数 | 默认值 | 配置位置与含义 |
|---|---|---|
| `--prefix-index-max-entries` | 262144 | 路由端全部 P 镜像的总页条目上限 |
| `--prefix-index-max-worker-entries` | 65536 | 路由端单 P 镜像的页条目上限 |
| `--prefix-index-max-age` | 6 秒 | 镜像最近一次成功同步的有效期 |
| `--prefix-sync-interval` | 0.2 秒 | 后台每轮同步结束后的等待时间 |
| `--pd-cache-namespace` | `default` | Worker 哈希域，1～1024 UTF-8 字节；P/D 配置一致 |
| `--pd-prefix-max-pages` | 65536 | Worker 源端元数据页条目上限，仅 Radix P 发布 |
| `--pd-prefix-event-limit` | 32768 | Worker 源端页事件日志上限 |

本地启动器将 `--pd-*` 参数透传给 Worker；远程地址模式需在各 Worker 启动命令配置。
Worker health 的 `prefix_index` 包含实例、域、seq、pages、max_pages、event_count、oldest_seq 和禁用原因。
旧 `prefix_match` 控制接口保留用于诊断，当前 `cache_aware` 选路不调用它。

| 源码 | 职责 |
|---|---|
| [prefix_hash.py](../../python/minisgl/disagg/prefix_hash.py) | 规范哈希域、int32 LE 编码与页面摘要 |
| [prefix_events.py](../../python/minisgl/disagg/prefix_events.py) | P 已提交 KV 的元数据、增删日志和分页快照 |
| [prefix_index.py](../../python/minisgl/disagg/prefix_index.py) | Router 实例隔离、顺序匹配、过期和容量清理 |
| [prompt_prepare.py](../../python/minisgl/disagg/prompt_prepare.py) | 有界 CPU 分词/哈希进程与取消 |
| [router.py](../../python/minisgl/disagg/router.py)、[routed_client.py](../../python/minisgl/disagg/routed_client.py) | 后台同步、health 失效及路由接入 |
| [radix_cache.py](../../python/minisgl/kvcache/radix_cache.py)、[cache.py](../../python/minisgl/scheduler/cache.py) | 真实页发布、拆分及淘汰钩子 |

本轮按要求未新增或运行测试、未开展实验。只做源码审查、Python 编译和 `git diff --check`；
历史路由测试结果不代表本次新索引已完成运行验收。
