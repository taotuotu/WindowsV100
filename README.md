<!-- Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE. -->
# The Ultimate V100 Windows Local Inference Engine for Qwen3.8-27B

在 Windows 上运行 Qwen3.8-27B：常驻本地 API、自带浏览器聊天、真实 token 速率和前缀缓存。

这是 [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) 的 Windows 衍生版本，基线为 `b37d0dd`，主项目许可为 Apache-2.0。模型运行时、Volta 算子和推测解码来自上游；本分支补充 Windows 平台兼容、文本程序入口、便携启动器和浏览器界面。完整来源见 [NOTICE](NOTICE)、[第三方许可](THIRD_PARTY_NOTICES.md) 与 [来源记录](docs/provenance.md)。

## 支持范围

- **显卡：V100 32GB / 同类 SM70 Volta 32GB。** 当前 Windows 构建面向 `sm_70`；不提供 RTX 显卡兼容承诺。
- **模型：固定的 Qwen3.8-27B mixed NVFP4/FP8 `.ninfer` v2**，约 23.72 GB。下载器固定 revision、大小和 SHA256；权重不随源码或程序 ZIP 分发。
- **输入：文本与工具历史。** 常驻服务提供 OpenAI Chat Completions / Responses、Anthropic Messages 和流式输出；Windows 构建关闭图像、视频输入。
- **网页：** 多轮聊天、思考折叠、停止生成、完整模型标识、decode tok/s、首字延迟和缓存命中；全局速度面板同时监控其它平台的 API 生成调用。
- **KV：** BF16 / INT8 / FP8；默认 BF16。MTP 默认 learned window 3、optimized proposal head，前缀复用默认开启。

此程序加载显式注册的 `.ninfer` artifact。GGUF 和任意 Q2/Q3/Q4/Q5/Q6 文件不能直接使用；其它模型的上游注册情况不等于本 Windows 发行版已经测过。

## 快速开始

需要 Windows x64、匹配的 NVIDIA 驱动，以及 [Microsoft x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。预编译 ZIP 会带 CUDA 12 runtime；驱动和模型需自行安装或下载。建议至少 32 GB 系统内存、40 GB 可用磁盘。

在解压后的程序目录，或已经完成构建的源码根目录，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download-model.ps1
.\start-ninfer.bat
```

`start-ninfer.bat` 是一键入口：启动模型，等待健康状态和模型列表就绪，再自动打开 **http://127.0.0.1:8110/**。首次初始化可能需要几分钟，默认等待上限10分钟。已运行的同一服务会被复用；本次新启动的服务由启动窗口持有，Ctrl+C 或关闭窗口停止它。启动失败会保留报错窗口；`stop-ninfer.bat` 可停止匹配的实例。页面聊天只保存在内存里，刷新会清空。

页面的“全局速度”每秒左右刷新，包含网页、ZCode 和其它平台的生成调用：服务总解码/预填充吞吐、运行/排队状态、累计解码和缓存命中、最近请求与速率趋势。实时吞吐是相邻服务快照之间的真实 token 增量/耗时，与单次回答的解码阶段平均速率分开显示；解码计数不含预填充产生的首 token。请求记录只有匿名计时与计数，服务重启后重置。

源码用户先安装 Visual Studio 2022 的 C++ / Windows SDK 组件、CUDA Toolkit **12.9** 和 CMake 3.28+，再构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-v100.ps1 -Step all -Parallel 4
```

CUDA 13 已移除 Volta 编译支持。更详细的安装、配置和问题处理见 [Windows 使用指南](docs/windows-v100.md)。

## 接入其它客户端

| 项目 | 默认值 |
|---|---|
| 协议 | OpenAI 兼容 |
| Base URL | `http://127.0.0.1:8110/v1` |
| Model | `qwen3.8-27b` |
| API key | 默认不鉴权；客户端必填时可填 `local` |
| 上下文容量 | 8192，输入与输出合计 |
| 活动请求 | 1，额外请求排队 |

提高容量或指定设备：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 -Device 1 -Context 153600 -KvDtype bf16
```

153600 是本机 32GB Volta 上成功分配的配置，需要充足空闲显存；它不代表 15 万 token 实际输入的性能或质量已经验证。可把个人设置写入 `.local/windows-server.psd1`，显式命令参数优先；该目录不会进入源码或发行包。

## 性能与继续优化

手动任务反馈：**英文作文约90 tok/s、代码约90 tok/s、中文作文约70 tok/s**。这些是日常任务的近似平均速率，输入/输出长度与重复次数尚未完整归档。

此前记录的 BF16 KV、MTP3、实际 5118-token 输入任务平均约 **63.87 decode tok/s**。真实 HTTP 调用观察到约 **46–138 tok/s**，接受率、输出内容和上下文不同，不能把这个范围当成固定吞吐承诺。

前缀缓存已在真实客户端多轮调用中生效：一次 13894-token 请求命中 13447 token，只计算余下 447 token，服务端首字约 0.82 秒。冷前缀依然需要完整预填充。

下一步有价值的方向是冷长输入的预填充分块和主机提交开销，以及受控比较 MTP 窗口。**本轮发布准备没有新增速度测试或宣称新的提速。** 历史测量、不同配置比较的限制及优化依据见 [Windows 性能记录](docs/windows-performance.md)。上游 Linux 的合成续写约 219 tok/s 不作为此 Windows 版本的普通任务速度。

## 打包与开发（源码目录）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1
```

打包器生成源码包和 Windows 程序包，附许可证及校验信息；不打包权重、驱动、个人配置、运行日志或聊天记录。二进制只面向上述 Windows / Volta 配置。底层架构及原项目文档保留在 [docs](docs/README.md)。
