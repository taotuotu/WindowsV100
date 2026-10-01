# Windows / V100 使用指南

Windows 专用程序面向单张 32GB SM70 Volta、固定 Qwen3.8-27B mixed NVFP4/FP8 `.ninfer` v2、文本输入。原项目的 RTX 5090、Linux、多媒体及多模型文档保留用于开发参考，不代表此发行版的验证范围。性能证据以 [Windows 性能记录](windows-performance.md) 为准。

## 1. 环境

运行预编译包需要：

- Windows x64，32GB V100 / 同类 SM70 Volta。16GB V100 装不下该约 22.09 GiB 权重。
- 匹配设备的 NVIDIA 驱动；CUDA 12.9 Update 2 对应 Windows 驱动版本为 576.57。具体显卡能使用的驱动需按 NVIDIA 支持范围选择。
- [Microsoft x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
- 至少 32GB 系统内存、约 40GB 可用磁盘；下载失败时 `.partial` 文件会保留以便续传。

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

双击 `start-ninfer.bat`。启动器先找程序包的 `bin/ninfer-windows-serve.exe`，再找源码构建的 Release 程序。终端显示加载与预热进度，随后监听 `127.0.0.1:8110`。Ctrl+C 退出；`stop-ninfer.bat` 只停止端口和程序路径均匹配的实例。

| 设置 | 公共默认 |
|---|---|
| Device | `auto`，首个 SM70 且总显存不少于 30 GiB 的 CUDA 设备 |
| Model | `models/qwen3_8_27b_nvfp4_v2.ninfer` |
| Context / KV capacity | 8192，输入和输出合计 |
| KV dtype | `bf16`，K/V 均为 16 位 |
| 推测解码 | learned MTP3、optimized head |
| Prefill chunk | 512 |
| Listen / port | `127.0.0.1:8110` |
| 前缀缓存 | 开启；额外 Device State 1、Host State 2、Host KV 0 |
| 活动请求 | 1 |
| 思考与采样 | 默认思考关闭、temperature 0、seed 123；请求可覆盖 |

显式参数示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-ninfer-server.ps1 `
    -Model .\models\qwen3_8_27b_nvfp4_v2.ninfer -Device auto `
    -Context 32768 -KvDtype bf16 -DraftTokens 3 -PrefillChunk 512
```

也可以新建 `.local/windows-server.psd1`：

```powershell
@{
    Device = 'auto'
    Context = 32768
    KvDtype = 'bf16'
    Port = 8110
}
```

脚本显式参数优先于个人配置，个人配置优先于公共默认。个人文件不会打包或纳入 Git。已有电脑上保留的大上下文配置不改变公共默认。

本机曾成功分配 BF16 / MTP3 / prefix-cache 的 153600 容量，剩余显存约 704MiB。空闲显存、KV、prefill chunk 和其它进程会改变上限；容量成功不代表已经喂满该长度。增大 `Context` 时先给输入和输出留足总预算。

直接调用 EXE 时，可用 `--device auto` 或数字 ordinal，以及 `--model PATH`。`--help` 不枚举 GPU。默认模型相对当前工作目录；启动器会传入解析后的模型路径。未指定 KV 容量时跟随 `--max-context`；也可显式用 `--kv-capacity N`。精确选项以 EXE 的 `--help` 为准。

## 4. 网页和 API

双击 `open-chat.bat`，默认打开 **http://127.0.0.1:8110/**。页面从 `GET /ui/model-info` 读取实际模型 ID、artifact basename、KV、MTP 档位、容量和设备；未知信息显示未获取。网页源代码构建时嵌入 EXE，运行不需要 CDN、npm 或额外静态文件。

多轮聊天与 API key 仅在页面内存里；刷新清空。停止生成保留已经收到的文本。输出作为文本呈现。

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

随后网页和客户端填写相同 key。`/health` 提供健康状态，`/v1/models` 提供模型列表。默认仅本机可访问；另一台设备或云服务不能访问此电脑的 `127.0.0.1`。HTTP 协议和拒绝项见 [serving](serving.md)，Windows 多媒体输入被关闭。

### 速率如何计算

- 网页 tok/s 来自服务端 SSE `timings.predicted_per_second`：已接受输出 token 数减 1，除以生成墙钟耗时。
- 输出 token 数、缓存命中、prefill 进度与速率取服务端字段，不按字符猜 token。
- 网页首字是发请求至首个正文/思考 delta 的时间，包括网络和排队；与服务端单独阶段时间不同。
- 计时字段缺失显示 `—`。

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
