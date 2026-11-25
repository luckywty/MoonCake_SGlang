# 配套 Mooncake host DRAM 接口补丁

`host-dram.patch` 保存本项目远端 DRAM 路径所需的原生接口改动，基线为
[`301261358eef0b13fa55645ccde4a988f1ca7505`](https://github.com/kvcache-ai/Mooncake/commit/301261358eef0b13fa55645ccde4a988f1ca7505)。
本项目没有向 Mooncake 官方仓库推送这些改动；补丁随本仓库版本管理。

补丁包含独立 TE 的显式 RPC 端口、classic RDMA reader fence、Python binding 与两个原生测试。
原有 `initialize` 和 P→D 交接接口保持兼容。

在 mini-sglang 目录中，针对全新 Mooncake 工作副本执行：

```bash
git clone https://github.com/kvcache-ai/Mooncake.git ../Mooncake
git -C ../Mooncake checkout 301261358eef0b13fa55645ccde4a988f1ca7505
git -C ../Mooncake apply --check "$PWD/patches/mooncake/host-dram.patch"
git -C ../Mooncake apply "$PWD/patches/mooncake/host-dram.patch"
```

然后按 Mooncake 自身构建说明安装开发依赖、编译并安装 Python binding。
原生测试位于 `config_test` 的 `FixedRpcPortTest` 和
`rdma_transport_submit_task_test` 的 `ReaderFenceRejectsNewPassiveHandshakes`。
本次环境缺少 `yaml-cpp` 等开发依赖，尚未编译或运行这些原生测试。

当前开发工作区的相邻 Mooncake 已经包含补丁内容，无须重复应用。可用以下命令核对：

```bash
git -C ../Mooncake apply --reverse --check "$PWD/patches/mooncake/host-dram.patch"
```

补丁已在基线的独立文件副本中实际应用，并核对全部 15 个文件与开发工作区逐字节一致。
功能配置、CPU 测试结果与实机验收边界见 [DRAM KV 缓存说明](../../docs/pd/dram_cache.md)。
