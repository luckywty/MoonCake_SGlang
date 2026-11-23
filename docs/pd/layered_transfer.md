# Mooncake 逐层 KV 传输

Mooncake split Worker 现在在层组 KV 就绪后异步传输，让早期层的打包和
RDMA 与后续层 Prefill 计算重叠。TCP 仍使用完整 Prefill 后的整体字节交接。
本实现仍是 TP=1、greedy、每个 Worker 一次处理一个请求，关闭 CUDA Graph。

## 启动与内存预算

Engine 初始化 CUDA、模型和 KV 池后，后端绑定相同 GPU，再创建独立
`transfer_stream`。P 一次性分配并注册 2～4 个固定容量槽位，启动一个常驻发送线程；
D 一次性分配并注册一个连续接收池。正常请求不注册或注销显存，正常关闭才注销。

新增参数（两端保持一致的层分组配置）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--pd-layers-per-chunk` | 1 | 每次发送的层数；尾组可以少于该值 |
| `--pd-staging-slots` | 3 | P 环形槽位数，允许 2～4 |

最大接收 token 数为 `min(engine.max_seq_len, max_transfer_bytes // bytes_per_token)`。
D 池覆盖这些 token 的全部层；P 每槽覆盖这些 token 的一组层。P 的总注册字节为
`staging_slots * max_tokens * bytes_per_token * min(layers_per_chunk, layers) / layers`。
超出 `--pd-max-registered-bytes` 或注册数量上限会在就绪前报错。Engine 仍按总注册预算
预留显存。大层组配合多个槽位时应相应调整预算，不会静默缩小容量或切回 CPU。

## 请求流程

1. 客户端先与 P 协商能力；Mooncake 模式下，在发出 Prefill 前向 D 提交完整请求。
2. D 校验 prompt/输出预算，预留本地私有 KV 页和接收池租约，返回版本 2 的票据。
   票据包含模型布局、EOS、TE session、基地址、逐层偏移和层组大小。
3. P 校验票据。发送线程通过独立后端控制连接取得 D 对本次租约的写入许可。
   旧票据、已撤销租约、模型/分组/EOS 不匹配均不能启动写入。
4. Prefill 的 attention 写完本层 KV 后，计算线程记录 CUDA event；层组边界只入队
   `(group, slots, event)`，随后继续发射后续算子，不等待打包、空闲槽位或 RDMA。
5. 发送线程取空闲槽位，在 `transfer_stream` 上 `wait_event`，用 `index_select(out=...)`
   直接 gather 到预注册槽位，采用 `[layer, K/V, token, head, dim]` 布局。
   发送线程等待打包完成 event 后才提交对应远端偏移的 RDMA Write。
6. 槽位用完时和尾部收敛时，发送线程调用非阻塞 `drain(0)` 并在重试间休眠释放 GIL。
   原生终态与 `ticket.drained` 都确认后才移除 operation、复用原注册槽位。
7. P 先返回首 token，再等待后台完成。后台把本次所有提交和完成的层组位图直接交给 D，
   D 校验覆盖范围及请求/session 身份，执行 RDMA 写入可见性处理并确认完成。
8. 客户端收到 `sent` 后向 D 发送包含 handoff/首 token 的 `commit`。D 确认全部层，
   scatter 到自己的 KV 页，等待安装结束，归还接收池租约，发送 `installed` 并继续 Decode。

客户端只转发控制数据。GPU KV 不经过客户端、CPU bytes、NumPy 或 helper。
两端模型和最终 KV 池仍独立存在，GPU 上的打包/安装拷贝也仍然存在。

## Chunked Prefill 与源页所有权

当前在**最后一个 token chunk** 中逐层发送完整 prompt KV：该层最后一个 chunk 写入后，
此前 token chunk 和命中前缀的 KV 才共同构成完整层。较早 chunk 不重复发送。
因此不会把未完成层或部分 prompt 当作完整 KV；这不是 token chunk 级流式传输。
后续若增加逐 token chunk 发送，需要额外的 token 区间、远端 scatter 描述和覆盖位图。

P 的原请求页一直保留到后台打包/收尾结束，防止队列中的 gather 读取已复用页。
计算流在逐层回调内没有同步等待；原有输出 token 拷贝、请求尾部和 D 的安装收尾
仍保留必要同步。后台 `wait_event` 与打包完成 host fence 不可省略，
`SYNC_MEMOPS` 不能代替跨 stream 的就绪关系。

## EOS、取消与完成未知

- 首 token 为 EOS 或预算只有 1 时，P 停止未提交任务，排空已提交操作，向 D 证明停止写入；
  客户端收到 `retired` 后释放 D 的私有页和接收租约，不启动 Decode。
- 客户端关闭 generator 或断连时，两端撤销请求。D 尚未授予写权限时可以直接归还租约；
  授予写权限后，没有有效的发送端终态证明就必须隔离整个接收池。
- 收到取消后的晚到成功只回收资源，不安装 KV。重复终态消息必须与已记录内容一致，
  旧请求的终态不能影响新租约。
- `COMPLETION_UNKNOWN` 保留 ticket、槽位和注册；`FAILED_DRAINED` 只证明 P 本地排空，
  同样不能批准 D 地址复用。无法证明安全时 P 后端退役，D 保留隔离区直到退役/进程退出。
- 正常的完成证明丢失后，发送端使用独立连接补送一次；仍丢失则 D 继续隔离。
  超时、关闭、注销失败均不构成释放在途地址的依据。

当前 D 主循环仍串行，预留请求可能等待前一个 Decode 结束；本改造不提供多请求组批。
两端需要使用配套的新客户端和 Worker，不能用旧 Mooncake 整体交接客户端绕过前置握手。

## 验证

`tests/integration/test_pd_layered.py` 使用真实发送队列、runtime、调度器与 KV 池，
替换 CUDA event/原生 DMA 为可控 CPU 适配器。所有权测试使用不打开端口的内存控制通道，
端到端测试另使用真实 socket。覆盖提前发送、背压、槽位覆写、
分组尾层、dtype/稀疏页、重复请求复用、EOS、断连、未知/失败排空及资源回收。
这些测试不证明真实 CUDA 并发、GPUDirect 可见性或性能。
`test_pd_layered_cuda.py` 另提供真实 CUDA 的分组打包/安装检查，无 CUDA 时跳过。
就绪日志的 `overlap=false` 仍指原有调度重叠关闭；`kv_transfer_overlap=true` 表示启用逐层路径。

2026-11-23 本地 CPU 环境（PyTorch 2.9.1+cpu）的沙箱内测试子集记录为
**98 passed、10 skipped、5 deselected、113 subtests passed**；随后对收发资源所有权相关
20 项测试复核通过。10 项 GPU 测试跳过，5 项新增真实 socket 用例未执行；
完整网络回归的沙箱外执行未获批准，因此这不是完整 P/D 或 GPU/RDMA 验收结论。
本轮 Ruff、语法编译、shell 语法和 `git diff --check` 均通过。

在具备 GPU/RDMA 和兼容原生扩展的机器上可运行（IP/HCA/模型按实际配置填写）：

```bash
PYTHONPATH=python python tools/pd/verify_phase2_gpu.py \
  --model /models/Qwen3-0.6B --prefill-device 0 --decode-device 1 \
  --transfer-backend mooncake --layers-per-chunk 2 --staging-slots 3 \
  --mooncake-host 10.0.0.11 --mooncake-device mlx5_0 \
  --output-dir /tmp/minisgl-layered-gpu
```

该工具启动同机独立 P/D 进程，比较 Unified token 并检查请求结束后的注册、隔离和在途状态；
它不能代替双服务器验收。双机启动沿用 Mooncake 启动模板并在两端添加相同的
`--pd-layers-per-chunk`。实际重叠比例应以 GPU/network 时间线测量，不以 CPU 测试通过推断。
