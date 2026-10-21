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
| 接口地址（Provider Endpoint） | `http://127.0.0.1:8890/v1/audio/speech` ⚠️ 含完整路径 |
| 模型 | `omnivoice`（任意非空） |
| API Key | 留空 |
| 声音列表 | `alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse` |
| 默认声音 | `alloy` 走映射；或填 `female, elderly, 四川话` 触发文字造声 |

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
| `start.cmd` | Windows 启动器 | ✅ |
| `voices.json` | 音色 → 文字造声属性映射 | ✅ |
| `download_model.py` | 从 hf-mirror 拉取模型 | ✅ |
| `requirements.txt` | Python 依赖 | ✅ |
| `setup.ps1` | 全新机器自举脚本 | ✅ |
| `USAGE.md` | 完整使用与部署文档 | ✅ |
| `python/` | 内置 embeddable Python 运行时 | ❌（体积大） |
| `models/` | OmniVoice 模型权重（3.1 GB） | ❌（体积大） |
| `scratch/` | 调试/遗留归档 | ❌ |

## 许可

- 代码：MIT（如适用）。
- OmniVoice 模型权重：**CC-BY-NC 4.0（非商业）**，使用需遵守其许可；本仓库不含权重，由用户自行下载。
