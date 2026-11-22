# Worker 内持有 Mooncake TE 的 GPU 直传

`--pd-transfer-backend mooncake` 使用每个 Worker 常驻的 `mooncake.engine.TransferEngine`，
通过 GPUDirect RDMA 传输 CUDA Tensor。旧 helper 子进程入口已经退役。
当前实现保留 TP=1、未量化 MHA KV、greedy、串行请求及 eager 执行。

## 数据路径

```text
P GPU KV pool
  -> GPU index_select 打包副本
  -> P Worker 注册 GPU 地址
  -> Mooncake / GPUDirect RDMA
  -> D Worker 独立 GPU 接收 Tensor
  -> GPU index_copy_ 写入 D 自己分配的 KV slots
  -> Decode
```

正常 Mooncake 数据路径没有 CPU KV staging，也不把完整 KV 转成 bytes、NumPy 或 CPU Tensor。
GPU 内仍有打包和写回。TCP 连接只携带 handoff、接收地址、长度、状态与收尾确认；
CPU 仍处理页索引、请求元数据和输出 token。独立 `tcp` 后端保留原有 CPU bytes 与 SHA-256 路径。

## 运行条件

- GPU 运行时和完整 mini-SGLang CUDA 依赖。
- 支持 GPU 内存注册的 RDMA NIC、驱动和 GPU/NIC 拓扑；支持 peer-memory 或 DMA-BUF 注册。
- 与本地 Mooncake `301261358eef0b13fa55645ccde4a988f1ca7505` 接口兼容的 CUDA 构建，
  包含 `submit_scatter_write`、`ScatterTransferTicket.drain` 及 GPU 内存注册接口。
- `cuda-bindings` 提供 CUDA Driver API，用于地址检查、同步属性及 RDMA 写入可见性处理。

包版本号不代表实际扩展的能力。运行信息记录 Worker PID、TE session、扩展来源和传输资源状态。
不具备 RDMA/CUDA 条件时明确失败，不自动退回 TCP、CPU staging 或 helper。
配置中不能强制 TCP、SHM、NVLink、TENT 等与此模式冲突的路径。

Engine 先初始化 CUDA 和加载模型，后端随后绑定同一 GPU、创建并初始化 TE；
后端构造不会预先初始化 `torch.cuda`。后台原生清理线程也在该 Worker 进程内运行。

## 地址与资源参数

| 参数 | 含义 |
|---|---|
| `--pd-mooncake-host` | 本机可供对端访问的 TE 地址，不能填对端 IP 或通配地址 |
| `--pd-mooncake-device` | 可选的 RDMA HCA 名称，如 `mlx5_0`；不是 CUDA GPU 编号 |
| `--pd-data-port` | D 端固定的控制握手端口，不传输 KV 字节 |
| `--pd-advertise-host` | D 向 P 公布的握手地址 |
| `--pd-max-transfer-bytes` | 单次 KV payload 上限，默认 1 GiB |
| `--pd-max-registered-bytes` | 每个 Worker 活动及隔离 GPU 注册区的总预算，默认单次上限的两倍 |
| `--pd-max-pending-transfers` | 活动及遗留注册区数量上限，默认 32 |
| `--pd-max-quarantine-seconds` | 未确认资源的最长处置时间，默认 300 秒；不是释放内存的证明 |

总传输预算会在 Engine 计算 KV 页数时预留；显式指定 `--num-pages` 也会检查预算。
两台服务器分别加载相同模型权重，dtype、页大小和布局必须匹配。TE 自身的 RPC/RDMA
连接也必须可达，不能只开放上述外层握手端口。

例如在 GPU 运行环境中，两台服务器分别启动：

```bash
# P: 10.0.0.11
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m minisgl \
  --model /models/Qwen3-0.6B --worker-role prefill \
  --host 0.0.0.0 --port 1919 --dtype bfloat16 \
  --pd-transfer-backend mooncake --pd-mooncake-host 10.0.0.11 \
  --pd-mooncake-device mlx5_0 --page-size 1

# D: 10.0.0.12
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m minisgl \
  --model /models/Qwen3-0.6B --worker-role decode \
  --host 0.0.0.0 --port 1929 --pd-data-port 1931 --dtype bfloat16 \
  --pd-advertise-host 10.0.0.12 --pd-transfer-backend mooncake \
  --pd-mooncake-host 10.0.0.12 --pd-mooncake-device mlx5_0 --page-size 1

PYTHONPATH=python python -m minisgl.disagg.client \
  --prefill 10.0.0.11:1919 --decode 10.0.0.12:1929 \
  --input-ids '[1,2,3,4,5]' --max-tokens 8 --ignore-eos
```

模型路径、IP 和 HCA 名称需要按实际环境填写。这些命令不是已完成的运行验收记录。

## 同步、取消与回收

1. P 打包完成后才释放原请求页；TE 使用的独立 GPU Tensor 由 runtime 保持强引用。
2. D 在可信传输完成确认后，执行设备所需的 GPUDirect 写入可见性处理，再进行 GPU 写回。
3. D 等写回 CUDA 工作完成后释放接收 Tensor，随后开始 Decode。
4. 业务取消会撤销交付，不能据此释放仍可能被原生操作访问的 GPU 地址。
5. `COMPLETION_UNKNOWN` 的 ticket、Tensor 和注册继续保留。后台线程继续 `drain`，
   已完成的旧请求通过独立控制连接收尾，绝不再安装到取消后的请求。
6. P 的 `FAILED_DRAINED` 仅证明 P 本地源区排空，不能直接批准 D 的接收区释放。
   无法确认安全的 D 注册区保持隔离，不复用该地址。
7. 资源上限耗尽、隔离超期或引擎失效会使后端停止服务。未知资源不能在普通
   `finally` 中注销、清空或丢弃最后一个 ticket 引用；Worker 必要时以非零状态退出，
   由部署层重启。

原生操作正常完成后的注销失败也会保留对象并报告错误，不把失败解释为已释放。
正常关闭只清理能够证明安全的资源；不能通过 `del engine` 假装所有远端访问已经结束。

## 验证范围

本轮按要求只修改实现和说明，不新增或执行测试，也不安装运行依赖。
静态语法、接口和差异检查不能证明 GPUDirect 路径可用。后续仍需分别执行：
GPU 字节对照、实际 RDMA 路径检查、Unified/P/D token 对照、取消/晚到写入、
显存上限，以及双服务器现场验收。
