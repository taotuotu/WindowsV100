<!-- Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE. -->
# The Ultimate V100 Windows Local Inference Engine for Qwen3.8-27B

在 Windows 上运行 Qwen3.8-27B：常驻本地 API、自带浏览器聊天、真实 token 速率和前缀缓存。

这是 [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) 的 Windows 衍生版本，基线为 `b37d0dd`，主项目许可为 Apache-2.0。模型运行时、Volta 算子和推测解码来自上游；本分支补充 Windows 平台兼容、文本程序入口、便携启动器和浏览器界面。完整来源见 [NOTICE](NOTICE)、[第三方许可](THIRD_PARTY_NOTICES.md) 与 [来源记录](docs/provenance.md)。

## 支持范围

- **显卡：V100 32GB / 同类 SM70 Volta 32GB。** 当前 Windows 构建面向 `sm_70`；不提供 RTX 显卡兼容承诺。
- **模型：固定的 Qwen3.8-27B mixed NVFP4/FP8 `.ninfer` v2**，约 23.72 GB。下载器固定 revision、大小和 SHA256；权重不随源码或程序 ZIP 分发。
- **输入：文本、工具历史与可选图像。** 服务提供 OpenAI Chat Completions / Responses、Anthropic Messages 和流式输出。Windows 通过 `-Vision` 接受 JPEG、PNG、BMP 图片；网页提供上传与预览，视频输入不支持。
- **网页：** 多轮聊天、思考折叠、停止生成、完整模型标识、decode tok/s、首字延迟和缓存命中；全局速度面板同时监控其它平台的 API 生成调用。
- **KV：** BF16 / INT8 / FP8；默认 BF16。MTP 默认 learned window 6、optimized proposal head，前缀复用默认开启。

此程序加载显式注册的 `.ninfer` artifact。GGUF 和任意 Q2/Q3/Q4/Q5/Q6 文件不能直接使用；其它模型的上游注册情况不等于本 Windows 发行版已经测过。

## 快速开始

需要 Windows x64、匹配的 NVIDIA 驱动，以及 [Microsoft x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。预编译 ZIP 会带 CUDA 12 runtime；驱动和模型需自行安装或下载。建议至少 32 GB 系统内存、40 GB 可用磁盘用于程序和模型；默认磁盘会话缓存还可使用最多32GiB，可通过降低预算或 `-NoDiskCache` 控制。

在解压后的程序目录，或已经完成构建的源码根目录，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download-model.ps1
.\start-ninfer.bat
```

`start-ninfer.bat` 是一键入口：启动模型，等待健康状态和模型列表就绪，再自动打开 **http://127.0.0.1:8110/**。首次初始化可能需要几分钟，默认等待上限10分钟。已运行的同一服务会被复用；本次新启动的服务由启动窗口持有，Ctrl+C 或关闭窗口停止它。使用 `start-ninfer.bat -Vision` 启用图像识别；也可在 `.local/windows-server.psd1` 设置 `Vision = $true`。启动失败会保留报错窗口。`stop-ninfer.bat` 一键结束本仓库 `build*`、`bin`、`dist` 和 `.local` 下路径匹配的 NInfer server、text/CLI 与 perplexity 进程，包括尚未监听的初始化进程；`stop-ninfer.bat -List` 只列出候选而不停止。它只处理明确的 NInfer 可执行文件名和当前仓库路径内的程序。

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
| 上下文容量 | 143600，输入与输出合计 |
| MTP 草稿长度 | 6，optimized head |
| 预填充分块 | 2048 |
| 活动请求 | 1，同时发来的第二个生成请求返回 HTTP 429 |

指定设备及启动参数：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 -Device 1 -Context 143600 -KvDtype bf16 -DraftTokens 6 -PrefillChunk 2048
```

默认采用本机使用者的运行配置：143600 上下文、BF16 KV、MTP6、prefill2048。使用者报告其任务中 MTP6 最快，且容量需降至143600避免 OOM；这属于本机使用反馈，完整输入长度与受控对比尚未归档。可把个人设置写入 `.local/windows-server.psd1`，显式命令参数优先；该目录不会进入源码或发行包。其它电脑可用 `-Context` 调整显存预算。

**长上下文 MTP 声明：** 本机使用反馈显示，随着实际上下文占用增加，较大 MTP 草稿长度相对于较小草稿长度的速度表现会变差，长上下文下可能更慢。MTP6 是当前默认；长上下文任务应结合实际输出吞吐选择档位。这里指实际输入与历史占用，详见 [性能记录](docs/windows-performance.md#长上下文与-mtp-档位)。

### 会话缓存与关闭服务

Windows 服务的显存/主机内存里只保留当前会话；切换会话时会清掉 Engine 中旧的 inactive private/shared checkpoint，再按需从本地磁盘恢复目标会话。网页每个新对话发送独立 `X-NInfer-Session` 标识；外部客户端可发送同一请求头区分会话。不发送时，Chat/Anthropic 按首个 user turn 的内容摘要识别，Responses 沿用 `previous_response_id` 链的会话标识。模型仍逐项验证真实 token 前缀，标识只决定缓存归属。

默认启用本地磁盘缓存，目录为 `.local/context-cache`，总预算32GiB、最多8个会话；可用 `-NoDiskCache`、`-DiskCacheDirectory`、`-DiskCacheMiB`、`-DiskCacheSessions` 调整，个人配置项为 `DiskCache`、`DiskCacheDirectory`、`DiskCacheMiB`、`DiskCacheSessions`。缓存仅支持此 Windows 注册的27B MTP/None运行路径和 BF16/INT8/FP8 KV，并绑定 Vision 开关及模型 artifact 全文件SHA、模型/权重ID、KV类型和MTP参数等完整身份。首次启动需读取整个artifact计算SHA，可能延长启动。

磁盘内容是 Engine 生成的真实完整 KV、GDN/recurrent、hidden state、shared/private owner与精确前缀快照，不是聊天文本替代。生成正常终结、停止或断线后自动保存当前完整且稳定的状态；切回会话时惰性恢复，取消导致内存中没有可续接的检查点时也会恢复同一会话上一份完整快照。每次保存都写完整快照，可能达数GiB并延后下一轮请求；它不增加模型运行所需GPU容量。写入采用带checksum的完整代文件与原子head切换并保留上一代；无安全checkpoint或不完整写入不会替换已提交代。进程被强制终止等任意崩溃可能丢失最近一次成功提交之后的更新。

停止或断线时，服务端已提交的输出可能多于客户端实际收到的文字；客户端也可能改写历史。下一轮只复用与实际回传历史精确匹配的检查点，必要时退回较早的前缀并重算回答尾部。落盘保证保存完整状态，不保证任意被截断的回答都能全量命中。

网页完整输入计数表示本轮历史总量；“缓存命中”和“需 Prefill”显示哪些部分已经计算、哪些需要重算。启动时页面默认勾选“保留历史思考”，显式发送 `preserve_thinking=true` 以便后续轮次保留并复用实际历史reasoning；关闭后会移除非空的历史reasoning，并可能让整段更早的历史重新prefill。外部平台若要复用含实际思考的历史，应在消息中回传 `reasoning_content` 并发送 `preserve_thinking=true`，或以 `--preserve-thinking` 启动服务；服务通用默认仍为 false。空的 canonical `<think>` prologue 另有定点保留修复，不代表非空推理在 `preserve_thinking=false` 时会留下。

网页自动保存期间“发送”按钮暂时禁用，但草稿仍可编辑；`/ui/metrics` 的磁盘缓存状态会显示保存/恢复运行状态。网页“清理全部缓存”在服务空闲时执行，同时删除内存及磁盘上所有会话快照并保留页面历史；下一轮需要重建。“关闭推理服务”取消当前生成并关闭当前服务。要关闭这个安装目录的全部 NInfer 程序，双击 `stop-ninfer.bat`，包含尚未启动监听的初始化进程。

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
