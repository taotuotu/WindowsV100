# Windows / V100 使用指南

Windows 专用程序面向单张 32GB SM70 Volta、固定 Qwen3.8-27B mixed NVFP4/FP8 `.ninfer` v2，支持文本和可选静态图像输入。原项目的 RTX 5090、Linux、视频及多模型文档保留用于开发参考，不代表此发行版的验证范围。性能证据以 [Windows 性能记录](windows-performance.md) 为准。

## 1. 环境

运行预编译包需要：

- Windows x64，32GB V100 / 同类 SM70 Volta。16GB V100 装不下该约 22.09 GiB 权重。
- 匹配设备的 NVIDIA 驱动；CUDA 12.9 Update 2 对应 Windows 驱动版本为 576.57。具体显卡能使用的驱动需按 NVIDIA 支持范围选择。
- [Microsoft x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
- 32GB 系统内存是基础建议；约 40GB 可用磁盘用于程序和模型。大上下文捕获检查点时，Host 归档峰值还可能额外占用最多32GiB系统内存（单份归档最多16GiB），与默认最多32GiB磁盘会话缓存预算分开；使用大上下文时按峰值增加系统内存，或降低上下文，或用 `-NoDiskCache` 关闭归档。下载失败时 `.partial` 文件会保留以便续传。

程序包只带 CUDA 12 runtime，不带 `nvcuda.dll` 或显卡驱动；源码构建要求另见下节。官方依据：[CUDA 12.9 Windows 指南](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-microsoft-windows/index.html)、[CUDA 12.9 Update 2 发布说明](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html)。

## 2. 下载兼容模型

在程序目录或源码根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download-model.ps1
```

下载器固定 [上游 v2 发布版本](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/tree/52907138a5d23a8f7f868ba7b773e721fd275405)，并检查：

```text
revision: 52907138a5d23a8f7f868ba7b773e721fd275405
upstream file: qwen3_8_27b_nvfp4.ninfer
destination: models/qwen3_8_27b_nvfp4_v2.ninfer
bytes: 23719496192
SHA256: 552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c
identity: qwen3.8-27b / nvfp4, container v2
```

该内容与本机此前固定的 `11dbbbbbc33db198afe2f02c9232c771ff7031be` revision 文件具有相同大小与 SHA256。上游 `main` 已换成 v3，不能以未固定的“最新”文件替代。模型不在发行 ZIP 中；上游许可与来源范围见 [provenance](provenance.md)。

## 3. 启动和配置

双击 `start-ninfer.bat`，一次完成模型启动、就绪等待和打开浏览器。它先找源码构建的 Release 程序，再找程序包的 `bin/ninfer-windows-serve.exe`；等待 `/health`、预期模型列表和 `/ui/model-info` 中的 `disk_cache_enabled` 状态匹配后才打开聊天页。首次初始化可能需要几分钟，默认等待上限600秒，可用 `-ReadyTimeoutSeconds` 调整。旧服务若缺少这个字段，或磁盘缓存开关与本次配置不一致，启动器会要求先停止并重启，不把旧二进制当成已支持恢复。

已经健康运行的同一程序直接复用；同一程序正在初始化时等待，不重新加载模型。其它程序占用端口会显示错误。新服务在当前启动窗口中运行，Ctrl+C 或关闭窗口停止本次启动的服务；复用已有服务时不取得其进程所有权。失败会暂停保留报错，浏览器关联失败则显示可复制的聊天地址。`stop-ninfer.bat` 优先向当前安装目录在本机回环地址监听的服务请求正常关闭；配置了 API key 时会使用该配置认证，最多等待30秒。服务未监听、连接/认证失败或超时会显示警告，再强制结束本仓库 `build*`、`bin`、`dist` 和 `.local` 下路径匹配的 NInfer server、text/CLI 与 perplexity 进程，也会识别尚未监听的初始化进程。`stop-ninfer.bat -List` 只列出候选 PID 与路径，不停止进程。

脚本 `run-ninfer-server.ps1 -OpenChat` 启用上述一键模式；不带该开关时维持纯 API 服务入口。一键模式的模型、地址、端口与密钥应使用脚本的 `-Model/-ListenHost/-Port/-ApiKey` 参数，避免原生额外参数覆盖探测目标。额外的 `--model-id` 会被用于预期别名探测。

启动器默认将会话快照写入源码/程序目录的 `.local/context-cache`，磁盘预算最多32GiB、8个会话。捕获归档另有最多32GiB的 Host 内存峰值预算，单份最多16GiB，与磁盘预算无关；大上下文机器应留出相应系统内存。可使用 `-NoDiskCache` 关闭磁盘缓存，或传 `-DiskCacheDirectory PATH -DiskCacheMiB N -DiskCacheSessions N` 调整；也可在忽略的 `.local/windows-server.psd1` 配置 `DiskCache = $false`、`DiskCacheDirectory`、`DiskCacheMiB` 与 `DiskCacheSessions`。相对命令行目录以当前工作目录解析；默认及个人配置中的相对目录以仓库/程序目录解析。完整缓存文件不是普通聊天记录，打包器会排除这些生成文件。

启用图像识别可在启动时加 `-Vision`：

```powershell
.\start-ninfer.bat -Vision
# 或直接运行 API 服务
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 -Vision
```

也可在 `.local/windows-server.psd1` 中设 `Vision = $true`，让之后的启动默认启用。Windows 使用系统自带 WIC 解码 JPEG、PNG 和 BMP；JPEG EXIF 方向会按图像显示方向校正，解码像素与请求字节仍受处理器预算限制。WebP、TIFF、GIF 和视频不在此 Windows 路径的支持范围内。图像随后由现有 CPU resize、BF16 patch 预处理与模型内置 Vision Encoder 处理；SM70 对 D=72、H=16 的非因果单段图像注意力，在 raw patch 长度1024–16384时使用 Tensor Core FlashMMA 快速路径，其它形状回退 SIMT 路径；不会为了命中快速路径额外缩小图像。此开关不修改 `Context`。启动器请求 Vision 时会核验 `/ui/model-info` 的 `vision_enabled=true`；已有纯文本服务会提示先停止再重启。

2026-10-02 本机正常启动已同时分配 Vision、153600 BF16 KV、MTP3 与上述前缀缓存。
启动快照剩余约262MiB显存；视觉权重约282MiB，workspace相对纯文本增加约160MiB。
这是启动与接口读取结果，尚未主动验证图片生成、识图质量、视觉速度或满上下文稳定性。
这是旧 MTP3 / prefill512 配置的快照。当前默认改为使用者运行反馈中的
MTP6 / prefill2048 / 143600；其它电脑需按实际空闲显存选择容量。

图像可使用 inline `data:image/...;base64,...` 或 HTTP(S) 图片 URL。远程 URL 获取要求 Windows 10 21H1 或更新版本；inline data URI 不增加这项系统版本要求。

| 设置 | 公共默认 |
|---|---|
| Device | `auto`，首个 SM70 且总显存不少于 30 GiB 的 CUDA 设备 |
| Model | `models/qwen3_8_27b_nvfp4_v2.ninfer` |
| Context / KV capacity | 143600，输入和输出合计 |
| KV dtype | `bf16`，K/V 均为 16 位 |
| 推测解码 | learned MTP6、optimized head |
| Prefill chunk | 2048 |
| Vision 图片输入 | 默认关闭；用 `-Vision` 或本机配置 `Vision = $true` 开启 |
| Listen / port | `127.0.0.1:8110` |
| 前缀缓存 | 开启；当前会话独占；额外 Device State 1、Host State 8、Host KV 0 |
| 磁盘会话缓存 | 默认开启；`.local/context-cache`；磁盘预算32GiB、最多8个会话；Host 捕获峰值另限32GiB |
| 活动请求 | 生成/捕获期间1；第二个请求返回429/`inference_busy`；背景磁盘写入期间可发下一轮 |
| 思考与采样 | 默认启用模型原生思考，未指定挡位时采用模板默认 `xhigh`；temperature 0、seed 123；请求可覆盖 |

网页默认勾选“启用思考模式”，不额外指定 `reasoning_effort` 或思考 token 预算，沿用模型模板的原生默认。外部 Agent 未传思考设置时也默认启用；显式传入 `reasoning_effort: "none"` 或 `enable_thinking: false` 可关闭。原生启动参数 `--no-thinking` 可关闭服务默认思考。思考与正文仍共用请求的输出 token 上限。

显式参数示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 `
    -Model .\models\qwen3_8_27b_nvfp4_v2.ninfer -Device auto `
    -Context 143600 -KvDtype bf16 -DraftTokens 6 -PrefillChunk 2048
```

也可以新建 `.local/windows-server.psd1`：

```powershell
@{
    Device = 'auto'
    Context = 143600
    KvDtype = 'bf16'
    DraftTokens = 6
    PrefillChunk = 2048
    DiskCache = $true
    DiskCacheDirectory = '.local/context-cache'
    DiskCacheMiB = 32768
    DiskCacheSessions = 8
    Port = 8110
}
```

脚本显式参数优先于个人配置，个人配置优先于公共默认。个人文件不会打包或纳入 Git。
2026-10-02 使用者反馈其任务中 MTP 草稿长度6最快，prefill2048需要将上下文降至143600
才能避免 OOM；当前公共默认采用这组三项，BF16 KV保持。此为本机反馈，不能当成所有任务、
电脑或满上下文工作负载的性能保证，完整受控对比尚未归档。

**长上下文 MTP 声明：** 使用者反馈实际上下文占用越大，较大 MTP 草稿长度相对较小档位的速度表现越差，长上下文下可能更慢。选择 `DraftTokens` 时结合实际上下文与输出吞吐；MTP6 为当前默认。完整口径见 [性能记录](windows-performance.md#长上下文与-mtp-档位)。

本机成功分配 BF16 / MTP3 / prefix-cache 的 153600 容量：纯文本启动快照剩余约704MiB，开启 Vision 后约262MiB。空闲显存、KV、prefill chunk 和其它进程会改变上限；容量成功不代表已经喂满该长度。增大 `Context` 时先给输入和输出留足总预算。

直接调用 EXE 时，可用 `--device auto` 或数字 ordinal，以及 `--model PATH`。`--help` 不枚举 GPU。默认模型相对当前工作目录；启动器会传入解析后的模型路径。未指定 KV 容量时跟随 `--max-context`；也可显式用 `--kv-capacity N`。精确选项以 EXE 的 `--help` 为准。

## 4. 网页和 API

一键入口 `start-ninfer.bat` 会在就绪后打开 **http://127.0.0.1:8110/**。已有服务也可直接访问此地址。页面从 `GET /ui/model-info` 读取实际模型 ID、artifact basename、KV、MTP 档位、容量和设备；未知信息显示未获取。网页源代码构建时嵌入 EXE，运行不需要 CDN、npm 或额外静态文件。

多轮聊天与 API key 仅在页面内存里；刷新清空。停止生成保留已经收到的文本。输出作为文本呈现。

服务启用 Vision 后，输入框的“添加图片”支持 PNG/JPEG/BMP，本轮最多4张、单张16MiB。
图片在本地读取为 data URI，显示预览并允许移除；发送后按标准 `image_url` 内容项进入聊天历史，
后续轮次保留这些图像。没有文字时使用“请描述图像”。附件读取完成后才能发送；新对话清除图片与
文字历史。页面不会把图片上传给外部服务。4张是网页附件上限，API仍按总体媒体、上下文和显存预算
接收；图片数量合规不代表一定能放进模型上下文。

### 当前会话独占缓存

Windows同一时间只接收一个生成请求，名额覆盖输入准备、生成和稳定检查点捕获，并在HTTP终帧前释放。其它生成请求收到429后重试；查看全局速度与模型信息不占这个名额。准备失败的请求不会切换缓存归属。检查点捕获完成后，请求可以返回，异步磁盘写入不占生成名额；同一会话可在后台写盘期间继续使用 GPU。服务在完整输入准备成功后、提交推理前比较会话标识：同会话保留当前 Engine 活动目录；新会话先清除旧的 inactive private/shared checkpoint，再尝试恢复该会话最近一次安全提交的磁盘快照，找不到兼容快照才冷启动。目标会话若仍有待写入快照，则先等该次写入完成再恢复。

网页每个页面/新对话带稳定的 `X-NInfer-Session: web-UUID`。其它平台支持自定义HTTP头时，为每条对话使用独立标识，并在后续请求保持相同值。标识限1–128个ASCII字母、数字、点、下划线、冒号或连字符。省略时，Chat/Anthropic使用首个user turn内容摘要；相同首句会归为同一缓存会话，需要精确区分时发送请求头。Responses继续使用其响应链已有会话标识；显式请求头优先。前缀身份始终由实际token/position校验，标识不替代内容匹配。

网页仍按Chat协议发送自己的完整历史，并显示“输入 tokens / 缓存命中 / 需 Prefill”。完整输入总量不表示每次都重新计算所有token。每个稳定 Engine 检查点先同步捕获为不可变 Host 归档；捕获期间单请求名额仍忙。首次、没有历史快照、早期历史/模板改变或清缓存之后需要重建；磁盘切回会话时为惰性恢复，不会让多个会话同时驻留 GPU/Engine 活动目录。磁盘快照保存完整 Engine checkpoint 目录中的 private continuation、shared prefix 与精确 frontier，包括 Main/backend KV、GDN/recurrent 及 hidden StateImage，不是只存 SessionEndpoint 或聊天文本。

磁盘缓存默认只处理此Windows注册27B的 MTP/None 路径、BF16/INT8/FP8 KV 与 Vision 开关对应的完整身份。store 会在启动时对实际模型 artifact 做一次 whole-file SHA256，并将模型/权重 ID、KV dtype、spec backend、draft window、proposal head 等与快照身份比较；Engine/Program 还会校验运行 archive revision 与 StateImage/KV pool geometry，因此更改 `Context` 或 KV容量后旧快照不兼容，会冷启动重新 prefill。artifact 的完整读取可能延长启动。

每份 Host 归档最多16GiB；活动写入、最新待写入及正在捕获的归档，以实际 `allocated_bytes` 计合计不超过32GiB。该 Host 内存峰值与默认32GiB磁盘预算分开。磁盘写入有界异步进行，捕获完成后请求返回，不会因背景保存禁用下一轮发送；同会话可在后台保存期间复用 GPU，切回有待写入的会话则等待该会话的保存完成。队列最多一个活动写入和一个最新待写入归档。收到 `[DONE]` 时最新一代可能仍在写盘。持久化使用完整代文件与原子 head 切换并保留上一份完整提交；checksum 或兼容性校验失败时可回退上一代，不完整写入不会替换已提交代。清理缓存会使旧写入失效，避免清理后被重新创建。正常关闭会排空待写队列；强制杀进程或断电时只能恢复最近一份完整成功提交。

2026-10-02的宽输入NVFP4 SwiGLU修复将gate/up中间投影保留为FP32，最后才舍入输出。
旧数学版本计算出的KV/状态不用于新版本续接：Windows缓存命名空间采用math revision1，
Program owner archive采用revision4，文件外层格式仍为v2。升级后每个旧会话首次回传完整历史会
重新Prefill；之后保存和复用新版本状态。旧快照文件不在升级时批量删除，仍受原磁盘预算和淘汰策略管理；
浏览器及外部客户端的聊天文本不受影响。

网页设置默认勾选“保留历史思考”，请求显式发送 `preserve_thinking=true`。关闭后会移除非空历史reasoning，可能从较早的assistant回答前重新Prefill。外部平台要复用含实际思考的历史，需回传 `reasoning_content` 和 `preserve_thinking=true`，或以 `--preserve-thinking` 启动服务；通用服务默认仍是false。单独的空 canonical `<think>` prologue 已有保留修复。

停止或断线时，服务端可能已经提交了客户端尚未收到的输出；客户端回传的半截回答与该 endpoint 不同。缓存仍须精确匹配实际历史，不能把超前状态直接当作半截回答的状态；此时可能退回较早的 response-replay/shared checkpoint，再计算回答尾部。磁盘保存与恢复不保证任意截断位置全量命中。

网页显示准备缓存、准备上限、准备耗时，以及保存队列和内存占用；准备期间“发送”仍因单请求名额占用而不可用，捕获完成并交付结果后，后台保存期间可以发送下一轮。“清理全部缓存”在没有请求交付时执行，同时清除 Engine 内存与磁盘上的所有会话快照并使旧写入失效，保留网页历史与模型权重；清理会取消已排队的旧保存，不能被旧 writer 复活，下一轮需要重建。“关闭推理服务”请求当前服务正常退出，正常关闭会先排空待写入队列，模型显存在进程退出后释放。正常终结、停止或断线后会尝试保存当前完整且稳定的状态；若取消路径未能安全 freeze，则继续保留之前的完整提交。`stop-ninfer.bat` 优先向当前安装目录在本机回环地址监听的服务发送正常关闭请求；配置了 API key 时会使用该配置认证，最多等待30秒。服务未监听、连接/认证失败或等待超时会显示警告，再强制结束此安装目录匹配的 NInfer 进程，包括初始化阶段。`run-ninfer-server.ps1 -Stop` 也转到这个入口。

控制接口为本机 `POST /ui/cache/clear` 与 `POST /ui/shutdown`，需 `X-NInfer-Control: 1`，配置了API key时仍需相同鉴权。清理成功返回200/`cleared`，有请求交付中返回409/`context_busy`；关闭返回202/`shutdown_requested`。关闭请求被接受不等于进程已经退出。控制接口不支持跨域浏览器控制；网页在同源地址调用。

| 客户端设置 | 值 |
|---|---|
| 类型 | OpenAI 兼容 |
| Base URL | `http://127.0.0.1:8110/v1` |
| 模型 | `qwen3.8-27b` |
| Key | 默认不鉴权；若客户端必填可填 `local` |

设置接口 key：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 -ApiKey 'your-local-key'
```

随后网页和客户端填写相同 key。`/health` 提供健康状态，`/v1/models` 提供模型列表。Vision 开启时可通过已支持的 OpenAI/Anthropic 图像字段提交 JPEG、PNG 或 BMP；视频仍返回不支持。默认仅本机可访问；另一台设备或云服务不能访问此电脑的 `127.0.0.1`。HTTP 协议和拒绝项见 [serving](serving.md)。

### 速率如何计算

- 网页 tok/s 来自服务端 SSE `timings.predicted_per_second`：已接受输出 token 数减 1，除以生成墙钟耗时。
- 输出 token 数、缓存命中、prefill 进度与速率取服务端字段，不按字符猜 token。
- 网页首字是发请求至首个正文/思考 delta 的时间，包括网络和排队；与服务端单独阶段时间不同。
- 计时字段缺失显示 `—`。

### 全局速度面板

顶部“全局速度”覆盖所有进入生成流程的 OpenAI Chat Completions、Responses、Anthropic Messages 请求，包括流式与非流式调用。它通过 `GET /ui/metrics` 每秒左右获取全局快照；配置 API key 时沿用页面当前密钥。无需开启请求 JSONL，也无需让其它客户端修改发送参数。

- 实时解码与 prefill：相邻快照的引擎已提交解码/实际计算输入 token 增量，除以服务端单调时间增量；包含所有请求，空闲时为0。解码计数不含 prefill 产生的首输出 token。
- 调度器：运行中、排队、预填充、解码就绪及准备上下文数量。
- 累计解码与缓存命中：当前进程内统计，启动预热不计入解码/prefill累计量；缓存命中累计自已完成生成请求。
- 最近请求：后端最多32条匿名记录，页面展示最近10条；包含协议、状态、输入/缓存/输出计数，以及完成后的解码平均速率。运行中尚未确定的字段显示 `—`。
- 最新完成请求：平均解码沿用 SSE `(completion_tokens-1)/generation_wall_seconds`；TTFT为服务端请求阶段口径，与网页网络首字时间不同。
- 图表：最多90个采样点。首次采样、重新连接、服务重启或页面恢复后先建立基准；缺失统计显示 `—`。页面隐藏时暂停，API key改变时重新连接。

计数对象是进入生成 prepare 的请求尝试，包含 prepare 拒绝；鉴权/JSON解析错误和面板轮询不是生成任务。表格不包含提示、输出正文、密钥、客户端地址。统计仅保存在服务内存中，重启重置；协议名不等于某个客户端应用名。

### 客户端每轮发送完整历史

完整历史依然要传输和分词，前缀缓存可减少重复 GPU prefill。命中依赖精确 token / 模板前缀和保存的状态；重启、改写早期 system、工具表、消息顺序或 thinking 设置会影响命中。

需要记录时显式传入 `-RequestLog .\logs\requests.requests.jsonl`。观察 `prefix_cache_hit_tokens` 和 `computed_prefill_tokens`；不以客户端“正在读取上下文”的显示判断是否全量计算。运行日志被 Git 和打包器排除。

## 5. 从源码构建

安装 Visual Studio 2022 C++ / Windows SDK 组件、CUDA Toolkit **12.9**、CMake 3.28+、Git。CUDA 13 不再生成 Volta device code。编译的参考环境为 MSVC 19.44 / CUDA 12.9 / Windows x64；首次配置联网拉取固定 CUTLASS revision。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-v100.ps1 -Step all -Parallel 4
```

仅构建服务：`-Step server`。默认产物：

```text
build-win-v100/apps/windows-serve/Release/ninfer-windows-serve.exe
build-win-v100/apps/windows-text/Release/ninfer-windows-text.exe
```

单次 CLI 示例（选择你自己的 GPU ordinal）：

```powershell
& .\build-win-v100\apps\windows-text\Release\ninfer-windows-text.exe `
    --model .\models\qwen3_8_27b_nvfp4_v2.ninfer --device 0 --max-context 8192 `
    --max-new 256 --kv-dtype bf16 --draft-tokens 3 --lm-head-draft `
    --greedy --seed 123 --no-thinking --prompt '解释模型量化与显存的关系。'
```

`--no-mtp` 关闭推测解码；learned window 支持 1..7。上游 context lookup 在精确复制命中且 learned proposal 同意时可延长验证至 15 个候选；这不构成普通任务 200 tok/s 保证。

## 6. 打包（源码目录）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1
```

源码与程序 ZIP 写入 `dist/`，使用允许列表组包；不收集个人配置、模型、运行日志或 Git 凭据。程序包附 `cudart64_12.dll` 及相关许可说明，MSVC runtime 作为官方安装先决条件。包内 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md` 和 `licenses/` 必须保留。
