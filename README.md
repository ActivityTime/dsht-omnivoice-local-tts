# OmniVoice 本地 TTS 服务（DSHT 配音 OpenAI 兼容适配目标）

> **DSHT 适配目标**：为 DeepSeek Harness / Tavern（DSHT）提供一个**本地、离线、OpenAI 兼容**的 TTS 配音后端，
> 通过 DSHT 原生 `openai` 配音 provider 直接调用，无需任何插件。

OmniVoice（k2-fsa / 小米 Next-gen Kaldi 开源 TTS，~0.6B 参数，646 语种，支持**声音克隆**与
**文字造声 Voice Design**，权重 CC-BY-NC 4.0 非商业）的本地部署封装。对外暴露
`POST /v1/audio/speech`，与 OpenAI TTS API 完全兼容。

- ✅ 本机运行时完全自包含：内置独立 Python 运行时（`python/`），不依赖系统或外部 Python；
  模型权重本地（`models/`），无任何网络调用（首次下载模型除外）。
- ✅ 已验证：RTX 3080 CUDA、bfloat16、`/health` 返回 `loaded:true`，`/v1/audio/speech` 实测产出合法 MP3。
- ⚠️ 模型权重与 Python 运行时体积较大（合计约 11–12 GB），**不纳入 git**，由 `setup.ps1` 与
  `download_model.py` 在本机重建/拉取（见下方「自举」）。

## 快速开始（本机已有完整环境时）

```bat
# 双击或用命令行启动（路径全部以脚本自身位置派生，可整体搬目录）
start.cmd
```

DSHT 配音配置：

| 字段 | 值 |
|---|---|
| 方案 | `openai`（OpenAI 兼容） |
| 接口地址（Provider Endpoint） | `http://127.0.0.1:8890/v1` ⚠️ 只写到 `/v1`，DSHT 会自己补 `/audio/speech` |
| 模型 | `gpt-4o-mini-tts`（服务端忽略此值；DSHT 仅当名字含 `gpt-4o` 时才发送情绪指令） |
| API Key | 必填（DSHT 端校验非空），内容随意如 `local`；服务端默认不校验 |
| 声音列表 | `alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse` |
| 默认声音 | `alloy` 走映射；或填 `female, elderly, 四川话` 触发文字造声 |

## 管理控制台

服务启动后打开 **http://127.0.0.1:8890/admin** —— 一个自包含的单文件页面（无 CDN、离线可用）：

- **自定义参数发起请求**：文本、声音/属性描述、附加属性、语速 `speed`、扩散步数 `num_step`、输出格式
  `mp3`/`wav`、是否附加属性；内置**属性速选标签**（英文 23 项 / 中文 25 项），点选自动按正确分隔符拼装，
  避免中英混用被模型拒绝。
- **请求与产出全量归档**：每次请求（含失败）写入 `history/<id>/`，含 `meta.json`（完整参数、耗时、字节数、
  错误信息）与实际音频文件，页面内可直接试听、下载、复用参数、单条删除或清空。
- 顶部实时显示模型加载状态、设备、dtype、CUDA。

> OpenAI 端点（`/v1/audio/speech`，即 DSHT 走的那条）默认**不写归档**，避免聊天逐条刷盘；
> 需要时也记录可设环境变量 `OMNI_TTS_LOG_API=1`。

## 声音克隆（Voice Clone）

属性造声（Voice Design）只能从**固定属性白名单**里组合（EN 23 / ZH 25 项，无情绪轴、不可中英混用），
**无法**做出「某个具体角色 + 某种情绪」。要突破这个限制得走**声音克隆**：给一段参考音频，模型复刻其中的
音色与情绪，用来朗读任意新文本。

在管理台的「声音克隆 · 音色库」卡片上传即可：

| 项 | 要求 |
|---|---|
| 参考音频 | 3–10 秒（>20 秒官方警告：更慢、更占内存、质量下降），wav/mp3/flac/ogg |
| `ref_text` | **必填**，参考音频的逐字稿。留空会触发 Whisper ASR 自动转写（多占约 3 GB，本机不划算） |
| 音色名称 | 创建后，`voice` 字段填这个名字即可使用（DSHT 里一样） |

每个音色在 `voices/` 下存三个文件：`<name>.pt`（官方 `VoiceClonePrompt.save()` 产物，约 7 KB）、
`<name>.json`（元信息）、`<name>.ref<ext>`（原始音频，供回听）。生成时**内存优先**加载，重启不丢。

> 两种模式互斥：命中克隆音色时，属性串不再参与（音色已由参考音频决定）。

详细使用、排障、各资源位置见 [`USAGE.md`](./USAGE.md)。

## 自举（全新机器 / 从 git 克隆后）

```powershell
# 管理员或非管理员均可；会下载 embed Python、安装依赖、拉取模型权重
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` 会：
1. 下载官方 embeddable CPython 3.13 并解压到 `python/`（自带运行时，便携）；
2. 配置 `python313._pth` 启用 `site-packages`；
3. `pip install -r requirements.txt`（torch/torchaudio 走 PyTorch CUDA 12.8 源）；
4. 运行 `download_model.py` 从 hf-mirror 拉取 OmniVoice 权重到 `models/`。

> 注：本仓库最初在某台 System32 被安全软件锁死的机器上部署，曾对 `libtorchaudio.pyd` 做过
> `api-ms-win-crt → ucrtbase` 的 pefile 改写并补 `vcomp140.dll`。这些补丁已嵌入 `python/Lib/site-packages`，
> 在干净的 Windows 上无害（ucrtbase 是真正的 CRT 实现），通常无需在目标机重做。详见 `USAGE.md` 故障速查。

## 目录结构

| 路径 | 说明 | 入 git |
|---|---|---|
| `server.py` | FastAPI OpenAI 兼容 TTS 服务 | ✅ |
| `admin.html` | 管理控制台页面（自包含，无外链） | ✅ |
| `start.cmd` | Windows 启动器 | ✅ |
| `voices.json` | 音色 → 文字造声属性映射 | ✅ |
| `download_model.py` | 从 hf-mirror 拉取模型 | ✅ |
| `requirements.txt` | Python 依赖 | ✅ |
| `setup.ps1` | 全新机器自举脚本 | ✅ |
| `USAGE.md` | 完整使用与部署文档 | ✅ |
| `python/` | 内置 embeddable Python 运行时 | ❌（体积大） |
| `models/` | OmniVoice 模型权重（3.1 GB） | ❌（体积大） |
| `history/` | 管理台请求归档（参数 + 音频产出，运行时生成） | ❌ |
| `voices/` | 声音克隆音色库（`.pt` prompt + 原始参考音频，运行时生成） | ❌ |
| `scratch/` | 调试/遗留归档 | ❌ |

## 架构与归属：哪些是上游，哪些是我们的实现

**本项目不是 TTS 模型的实现者。** 模型、推理库、运行时全部来自第三方；我们写的是"让它在 Windows 上跑起来、
并让 DSHT 能直接调用"的那一层。

| 层 | 归属 | 具体 |
|---|---|---|
| 模型架构与权重 | **上游**：k2-fsa（小米 Next-gen Kaldi 团队） | `k2-fsa/OmniVoice`，CC-BY-NC 4.0 非商业 |
| 推理库 | **上游**：同上，作者 Han Zhu | PyPI `omnivoice` 0.2.1：提供 `from_pretrained` / `generate` / `create_voice_clone_prompt` |
| 深度学习运行时 | **上游**：PyTorch 官方 | torch / torchaudio 2.8.0+cu128 |
| Python / Web 框架 | **上游**：python.org、FastAPI 等 | embeddable CPython 3.13.2、fastapi、uvicorn、lameenc、soundfile |
| OpenAI 兼容服务层 | **本项目** | `server.py`：端点、参数解析、鉴权、克隆缓存、归档 |
| 管理控制台 | **本项目** | `admin.html`：自包含单文件页面 |
| 音色与参数映射 | **本项目** | `voices.json`（OpenAI 音色名 → 造声属性） |
| 启动与自举 | **本项目** | `start.cmd`、`setup.ps1`、`download_model.py` |
| 文档 | **本项目** | `README.md`、`USAGE.md` |

我们实际新增的东西（上游没有的）：

1. **OpenAI 兼容适配层**——让 DSHT 原生 `openai` 配音 provider 无需插件直连。
2. **`voice` 字段双语义**——命中 `voices.json` 走映射，未命中整串当造声属性透传。
3. **声音克隆接入**——上传参考音频建 prompt，落盘 `.pt` + 内存缓存，与属性造声互斥共存。
4. **请求归档与回放**——每次请求（含失败）落 `history/<id>/`，可试听、下载、复用参数。
5. **管理控制台**——自包含、无 CDN、离线可用。
6. **提交内存优化**——计算线程钉 1 + `low_cpu_mem_usage=True`，解决本机 `os error 1455`。
7. **Windows DLL 加载修复链**——`api-ms-win-crt → ucrtbase` 改写、`vcomp140` 副本、`add_dll_directory` 补全。
8. **torio stub 与 dist-info 补齐**——torchaudio 2.8 拆分 io 后端后的兼容处理。
9. **自包含打包**——官方 embeddable Python 内嵌进项目，整目录可搬、不依赖外部 Python。

> 仓库里的 `README_upstream.md` 与 `voice-design_upstream.md` 是**留存的上游文档**，不是本项目作品。

## 许可

- 代码：MIT（如适用）。
- OmniVoice 模型权重：**CC-BY-NC 4.0（非商业）**，使用需遵守其许可；本仓库不含权重，由用户自行下载。
