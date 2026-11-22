# FluxInfer · Phase 2 移植、启动与验收

## 从 Phase 1 创建隔离工作区

本阶段补丁基于 `267ea95e795b200d86880a31f1d869c875534658`。先确认自己的分支已包含 Phase 1 六个 disagg 文件和角色配置。你的 FP8/EAGLE/Radix 改动应保存在自己的提交中，再从该 HEAD 建 worktree，便于回滚和前后比较。

```bash
cd /path/to/FluxInfer
git status --short
git branch fluxinfer/backup-before-phase2 HEAD
git worktree add -b fluxinfer/pd-phase2 ../FluxInfer-pd-phase2 HEAD
cd ../FluxInfer-pd-phase2
fluxinfer_package=$(realpath /path/to/fluxinfer-phase2)
git apply --check "$fluxinfer_package/phase2.patch"
git apply --index "$fluxinfer_package/phase2.patch"
git diff --cached --check
git diff --cached --stat
```

`git apply --check` 失败时不会应用补丁。对照以下锚点人工插入，在已有函数中保留你的原优化。完整修改文件在包内 `files/`，用于逐行比较。

## 人工移植顺序和锚点

| 顺序 | 文件 | 操作 |
|---|---|---|
| 1 | disagg/control.py、tcp_backend.py | 新增完整文件。依赖 Phase 1 protocol.py、transfer.py |
| 2 | tests/disagg/test_tcp.py | 新增，先验标准库跨进程 TCP、撤销和完整性 |
| 3 | kvcache/mha_pool.py | 保留原池构造和 store_kv，在其后插入 _transfer_indices、export_tokens、import_tokens，新增 sys import |
| 4 | scheduler/cache.py | 合并 allocate_paged 的分配回滚，新增 reserve_import、discard_private。保留自己的 _allocate、_free、prefix cache 策略 |
| 5 | scheduler/scheduler.py、disagg/role.py | 新增内部 split 构造参数。将 _prepare_batch 中分配后的元数据准备提取为 _prepare_batch_metadata |
| 6 | disagg/worker.py | 完整新增，调用上面方法和既有 Prefill/DecodeManager，不改 Engine 和 attention 核心 |
| 7 | disagg/client.py | 完整新增，先运行 RPC token ID 生成 |
| 8 | server/args.py、server/launch.py | 插入新参数及分派，检查放在模型加载前 |
| 9 | disagg/gateway.py | 完整新增，对接现有请求模型和 DetokenizeManager |
| 10 | 所有 Phase 2 测试、tools/pd/test_phase2.sh、verify_phase2_gpu.py、文档 | 完整新增，更新原 test_pd_entrypoints.py 中阶段角色行为 |

表内路径除另有前缀外，均在 `python/minisgl/` 下。完整相对路径和改动类型见 `CHANGES.tsv`。

### 兼容性复核

- FP8 分支应在本次 split 基线关闭 KV 量化，确认实际 pool dtype 是 BF16/FP16/FP32，且 pool 是上游 MHAKVCache 布局。新方法不会传输 scale，不能把 FP8 bytes 标成 BF16。保留你原 Unified 的量化逻辑。
- EAGLE-3 的 draft KV、hidden feature 和多 token 推进尚未纳入 handoff。用你已有的配置关闭投机，核对 _forward 返回单个 next token。不要覆盖你的原 EAGLE 构造和执行方法。
- 自定义 CacheManager 可能已有分配事务与引用计数。将 reserve_import 接入你的真实分配器，失败释放必须按实际已配页边界计算。borrowed prefix 只解锁，不能当私有页归还。
- 自定义 Graph key/attention 路径留在原方法中。_prepare_batch_metadata 应保留你自己的 metadata 构造，_prepare_batch 的 pad→allocate→metadata 顺序要一致。
- PDWorkerScheduler 当前使用现有 PrefillManager 的单请求 batch。若你的管理器改变了 PendingReq、Req 或 ChunkedReq 字段，按实际类型合并。完整函数调用及原始行号在 phase2.md，函数定位在 SOURCE_LOCATIONS.json。

## CPU 验收

仅传输和控制层使用标准库。

```bash
PYTHONPATH=python python3 -S -m unittest discover -s tests/disagg -v
```

完整 CPU 测试使用项目依赖与 pytest。

```bash
FLUXINFER_TEST_PYTHON=$(command -v python) bash tools/pd/test_phase2.sh
```

本次完整结果为 105 passed、1 skipped，实际以包内 TEST_REPORT 和日志为准。跳过项需要 CUDA 及 FLUXINFER_PD_GPU_MODEL。

如要复现本次 CPU QA 环境，在独立 Python 3.12 venv 安装快照。该快照包含 torch==2.9.1+cpu，避免将它安装到现有 GPU 环境中。

```bash
python3.12 -m venv /tmp/fluxinfer-phase2-cpu
/tmp/fluxinfer-phase2-cpu/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$fluxinfer_package/evidence/test-requirements.txt"
FLUXINFER_TEST_PYTHON=/tmp/fluxinfer-phase2-cpu/bin/python \
  bash tools/pd/test_phase2.sh
```

这些测试不要求下载模型。CPU 参考 TinyEngine 驱动真实 scheduler/cache/pool 和 TCP，实际 CUDA Engine 验收另列。

## 单 GPU 的最小手动功能运行

下列三条命令在三个终端分别运行。模型应是你本机已验证可运行的 Qwen3-0.6B 目录。两份模型和小 KV pool 在同一 GPU 共享显存，本次没有在 5060 Ti 上实测。显存允许时，该流程可以验证跨进程交接。

P 只执行 Prefill，RPC 为 1919，TP=1 Gloo 为本机 1920。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m minisgl \
  --model /models/Qwen3-0.6B --worker-role prefill \
  --host 127.0.0.1 --port 1919 --dtype bfloat16 \
  --attention-backend fi --num-pages 512 --page-size 1 \
  --max-running-requests 4 --max-seq-len-override 512 \
  --max-prefill-length 256 --cuda-graph-max-bs 0 --pd-timeout 120
```

D 只执行 Decode，RPC 为 1929，本机 Gloo 为 1930，KV 数据接收为 1931。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m minisgl \
  --model /models/Qwen3-0.6B --worker-role decode \
  --host 127.0.0.1 --port 1929 --pd-data-port 1931 \
  --dtype bfloat16 --attention-backend fi --num-pages 512 --page-size 1 \
  --max-running-requests 4 --max-seq-len-override 512 \
  --max-prefill-length 256 --cuda-graph-max-bs 0 --pd-timeout 120
```

等待两个进程各自输出 pd_worker/status=ready。原始 token ID 客户端可以在普通 CPU Python 进程运行。

```bash
PYTHONPATH=python python -m minisgl.disagg.client \
  --prefill 127.0.0.1:1919 --decode 127.0.0.1:1929 \
  --input-ids '[1,2,3,4,5]' --max-tokens 8 --ignore-eos
```

客户端返回 token_ids 和 completion_tokens。max_tokens=1 或首 token EOS 时，不会联系 D。

## HTTP 网关

另开终端启动网关。这里的 --model 用于 tokenizer 与对外模型标识，不会创建推理 Engine。

```bash
PYTHONPATH=python python -m minisgl.disagg.gateway \
  --model /models/Qwen3-0.6B \
  --prefill 127.0.0.1:1919 --decode 127.0.0.1:1929 \
  --host 127.0.0.1 --port 1940
```

最小非流式请求。

```bash
curl --fail-with-body http://127.0.0.1:1940/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3-0.6B","prompt":"Hello","temperature":0,"max_tokens":8,"ignore_eos":true,"stream":false}'
```

流式请求。

```bash
curl --no-buffer --fail-with-body http://127.0.0.1:1940/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":16,"stream":true}'
```

当前只支持 greedy 单结果请求。新网关会拒绝尚未实现的 stop/penalty、n>1 或非 greedy 配置。HTTP 出错会返回明确错误，SSE 出错包含 error 后结束。断开连接会设置取消 Event，并关闭对应该请求的生成器和控制连接。

关闭终端程序时使用 Ctrl+C。worker 在当前 batch 完成和资源清理后退出。

## 双 GPU 与双节点

同一服务器上，将 D 命令的 CUDA_VISIBLE_DEVICES 改为 1，P 继续使用 0。两进程都使用 TP=1，它们各自看到逻辑 cuda:0。

跨节点手动运行时，P 使用可供客户端连接的 RPC 地址。D 还需公布 P 可连接的数据地址。例如 D 主机为 10.0.0.12。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m minisgl \
  --model /models/Qwen3-0.6B --worker-role decode \
  --host 0.0.0.0 --port 1929 --pd-data-port 1931 \
  --pd-advertise-host 10.0.0.12 --dtype bfloat16 \
  --attention-backend fi --page-size 1 --num-pages 512 \
  --max-seq-len-override 512 --max-prefill-length 256
```

客户端连接 `P_IP:1919` 与 `10.0.0.12:1929`，P 根据 D 的 ticket 连接 `10.0.0.12:1931`。两端的模型内容、有效配置、dtype 和页布局必须匹配。Gloo 的 world_size=1 仅用于各 worker 的原引擎初始化，不在节点间承担 KV 传输。本次跨节点与双 GPU 运行尚未验证。

## 自动 GPU 对照验收

脚本会自行创建并关闭自己的 P/D 进程，并在单独 Unified 进程采集原 Engine 的 greedy token。它不复用手动 worker。

同一张 GPU 的跨进程功能验证。

```bash
PYTHONPATH=python python tools/pd/verify_phase2_gpu.py \
  --model /models/Qwen3-0.6B --prefill-device 0 --decode-device 0 \
  --page-size 1 --cache-type radix --output-dir /tmp/fluxinfer-phase2-single
```

双 GPU。

```bash
PYTHONPATH=python python tools/pd/verify_phase2_gpu.py \
  --model /models/Qwen3-0.6B --prefill-device 0 --decode-device 1 \
  --page-size 1 --cache-type radix --output-dir /tmp/fluxinfer-phase2-dual
```

再以 --page-size 4 验尾页和管理页映射，或 --cache-type naive 验无前缀缓存基线。每次使用自己的 output-dir 保存日志。成功后生成 gpu_results.json，里面含每组 Unified 与 P/D token ID 列表。失败会非零退出并保留 Unified/P/D 日志；每次启动会移除该输出目录内的旧 gpu_results.json，避免失败后误读旧 PASS。

也可 opt-in pytest 的 GPU 项。

```bash
FLUXINFER_PD_GPU_MODEL=/models/Qwen3-0.6B \
FLUXINFER_PD_PREFILL_DEVICE=0 FLUXINFER_PD_DECODE_DEVICE=1 \
PYTHONPATH=python python -m pytest -o addopts= -v tests/integration/test_pd_gpu.py
```

GPU Python 环境需要现有 mini-SGLang 的完整运行依赖及 CUDA 工具链。脚本的 Unified 基线使用原离线入口的 Gloo 地址，失败时查看 unified.log 中的实际原因。本次这组运行没有执行，不能将 CPU 测试 PASS 作为它的验收结果。

## 诊断和回滚

| 现象 | 具体排查 |
|---|---|
| CLI 在启动前退出 | 检查 TP=1、真实权重、合法端口、有限正超时；D wildcard bind 还要设置公布地址 |
| 模型/layout 不兼容 | 比较 P/D health 的 layout。fingerprint 绑定实际 safetensors 内容及有效 ModelConfig，dtype、页大小、head 数也要一致 |
| 数据接收超时 | 检查 D ticket 中的 endpoint 是 P 能访问的地址，控制端口和数据端口分别连通 |
| KV 容量不足 | prompt+最大输出预算必须在 max_seq_len、页表与池容量内，D 会在传输前检查完整预算并预留 prompt 页 |
| 取消后等待清理 | IO 检查按短轮询进行，连接建立受 socket timeout 限制，GPU batch 在完成后释放页 |
| CUDA/依赖错误 | 先检查同配置 Unified 基线。不可恢复清理时 worker 会退出，保留日志后重启进程 |
| greedy token 不一致 | 固定权重/config、关闭自定义 FP8/EAGLE、使用同 dtype/layout，查看逐 token 记录和首 token 边界 |

应用后尚未提交且没有额外编辑时，可反向应用。

```bash
git apply --reverse --check "$fluxinfer_package/phase2.patch"
git apply --reverse --index "$fluxinfer_package/phase2.patch"
git status --short
```

已提交则 `git revert <你的Phase2提交SHA>`。人工编辑后反向 check 失败时，按差异撤回本阶段插入，保留自己的原优化。验证后提交独立阶段。

```bash
git diff --cached --check
git commit -m "feat: add minimal TCP prefill decode disaggregation"
```
