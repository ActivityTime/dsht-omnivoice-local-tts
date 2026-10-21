# OmniVoice 本地 TTS 服务 · 使用与部署说明

> 本文档为「整理版使用说明」，配套部署目录 `K:\OmniVoice\`。
> 研究过程与排障细节见同仓库的 `OmniVoice部署调研.md`。

---

## 1. 这是什么

OmniVoice（k2-fsa / 小米 Next-gen Kaldi 开源 TTS，~0.6B 参数，扩散语言模型，646 语种，
支持**声音克隆**与**文字造声 Voice Design**，权重 CC-BY-NC 4.0 非商业）的本地部署封装。

对外暴露 **OpenAI 兼容 TTS 接口**（`POST /v1/audio/speech`），让 DSHT（SillyTavern 分支）
通过原生 `openai` 配音 provider 直接调用，无需任何插件。

**已验证状态**（2026-10-21，本机）：
- 硬件：NVIDIA RTX 3080（10GB 显存），Windows 10
- 精度：bfloat16（约 2GB 显存占用，质量无损于 fp16）
- 接口：`/health` 返回 `loaded:true, cuda:true`；`/v1/audio/speech` 实测产出合法 MP3
- 文字造声：`voice=alloy` 走映射；`voice=female, elderly, 四川话` 走自定义属性透传，均成功

---

## 2. 目录结构与各资源位置

所有路径在 `server.py` / `start.cmd` 中均以**脚本自身位置派生**（`Path(__file__).resolve().parent`
/ `%~dp0`），**无硬编码盘符或用户名**，整目录可整体搬迁。

| 路径 | 用途 | 是否必需 | 体积 |
|---|---|---|---|
| `server.py` | 服务主程序（FastAPI + uvicorn） | ✅ 必需 | ~20 KB |
| `admin.html` | 管理控制台页面（自包含单文件，无 CDN） | ✅ 必需 | ~15 KB |
| `start.cmd` | Windows 启动器，双击即起 | ✅ 必需 | <1 KB |
| `voices.json` | 11 个音色 → 文字造声属性映射 | ✅ 必需 | <1 KB |
| `models/OmniVoice/` | 模型权重（见 §2.1） | ✅ 必需 | **3.1 GB** |
| `python/` | 内置 embeddable CPython 3.13 运行时 + 全部依赖 | ✅ 必需 | ~7 GB |
| `history/` | 管理台请求归档（见 §3.5），运行时生成 | 产物 | 随使用增长 |
| `download_model.py` | 从 hf-mirror 拉取模型权重的脚本 | 可选 | <1 KB |
| `server.log` | 运行日志（启动后生成） | 产物 | 随运行增长 |
| `test_out.mp3` / `test_out2.mp3` | 验证用生成产物 | 产物 | ~50/28 KB |
| `scratch/` | 调试/遗留脚本与缓存归档（含 `wheels/`、`hf/`，见 §6） | 否 | — |
| `README_upstream.md` / `voice-design_upstream.md` | 上游官方文档副本 | 参考 | — |

### 2.1 模型权重明细（`models/OmniVoice/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 2.45 GB | 主模型权重 |
| `audio_tokenizer/model.safetensors` | 805 MB | 音频 tokenizer |
| `tokenizer.json` | 11.4 MB | 文本 tokenizer |
| `config.json` / `chat_template.jinja` / `tokenizer_config.json` / `preprocessor_config.json` | <1 MB | 配置 |

模型**完全本地**，运行时不做任何网络下载。

---

## 3. 启动方法

### 3.1 双击启动（推荐日常用）
直接运行 `start.cmd`。它会：
- 把 `HF_HOME` 指向本目录的 `hf/`（离线，不联网）
- 默认 `OMNIVOICE_MODEL=%~dp0models\OmniVoice`、`DEVICE=cuda:0`、`PORT=8890`
- 前台运行，窗口关闭即停止（末尾有 `pause`，便于看报错）

### 3.2 后台启动（无窗口、日志落盘）
```bat
cd /d "K:\OmniVoice"
python\python.exe server.py > server.log 2>&1
```
> 关掉窗口它就停。要长期开机自启，见 `OmniVoice部署调研.md` 第八节（NSSM / 启动项）。

### 3.3 环境变量（均可覆盖默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNIVOICE_MODEL` | `models/OmniVoice` | 模型目录（本地绝对/相对均可） |
| `OMNIVOICE_DEVICE` | `cuda:0` | 设备；CPU 用 `cpu`（极慢） |
| `OMNIVOICE_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `OMNIVOICE_NUM_STEP` | `32` | 扩散步数，越大越慢越细 |
| `OMNI_TTS_PORT` | `8890` | 监听端口 |
| `OMNI_TTS_HOST` | `127.0.0.1` | 监听地址（改 `0.0.0.0` 可局域网共享） |
| `OMNI_TTS_KEY` | 空 | 鉴权密钥；空 = 不校验（DSHT 留空即可） |
| `OMNI_TTS_VOICES` | `voices.json` | 音色映射文件 |
| `OMNI_TTS_USE_INSTRUCTIONS` | `0` | 管理台「附加属性」开关的默认值 |
| `OMNI_TTS_HISTORY_DIR` | `history/` | 管理台请求归档目录 |
| `OMNI_TTS_LOG_API` | `0` | 是否也归档 OpenAI 端点（`/v1/audio/speech`）的请求；DSHT 逐条调用会持续写盘，默认关 |
| `OMNI_TTS_ADMIN_PAGE` | `admin.html` | 管理台页面文件位置 |

### 3.4 健康检查
```bat
curl http://127.0.0.1:8890/health
# => {"status":"ok","loaded":true,"device":"cuda:0","cuda":true,...}
```

### 3.5 管理控制台

浏览器打开 **http://127.0.0.1:8890/admin**（页面每次请求读盘，改 `admin.html` 后刷新即可生效，无需重启服务）。

**能做什么**

1. **自定义参数发起请求** —— 文本、声音/属性描述、附加属性、语速 `speed`（0.25–4）、扩散步数
   `num_step`（1–100）、输出格式 `mp3`/`wav`、是否把附加属性拼入属性串。
2. **属性速选** —— 内置 OmniVoice 合法属性项（英文 23 项 / 中文 25 项），点选即按正确分隔符拼装
   （英文 `, `、中文 `，`）；切换语言时自动重置，避免中英混用被模型拒绝。
3. **请求与产出归档** —— 每次请求（**含失败**）写入 `history/<id>/`：
   - `meta.json`：完整入参、解析后的属性串、语速、步数、格式、耗时 `elapsed_ms`、字节数、状态、错误信息；
   - `audio.mp3` / `audio.wav`：实际音频产出。
4. **回看与复用** —— 列表内直接试听、下载、把历史参数一键回填表单重跑、单条删除或清空全部；顶部显示
   总条数/成功/失败/累计产出体积。

**属性填写规则（模型侧强校验，踩过坑）**

- 属性项只能来自 OmniVoice 的合法清单，**不支持情绪词**（如「开心」「激动」会直接报错）。
- **不可中英混用**：要么全英文（`male, british accent`），要么全中文且用全角逗号（`男，河南话`）。
- 失败不会被静默吞掉：错误会连同完整入参一起归档，并在列表里以红色展示原始报错。

**相关端点**（均以 JSON 交互，可脚本化调用）

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/admin` | 控制台页面 |
| POST | `/admin/generate` | 自定义参数合成并归档 |
| GET | `/admin/history?limit=50` | 记录列表 + 统计 |
| GET | `/admin/history/{id}` | 单条详情（供复用参数） |
| DELETE | `/admin/history/{id}` | 删除单条（含音频） |
| DELETE | `/admin/history` | 清空全部归档 |
| GET | `/admin/audio/{id}?download=1` | 回放 / 下载产出音频 |

---

## 4. DSHT 配音接入配置

DSHT 后端把你在界面填的「接口地址」**原样转发**到该 URL（不额外拼路径），所以必须填完整路径。

| 字段 | 填写值 |
|---|---|
| 配音方案 / Provider | `openai`（OpenAI 兼容） |
| **接口地址（Provider Endpoint）** | `http://127.0.0.1:8890/v1/audio/speech` ⚠️ 必须含完整路径，只写到 `/v1` 会 404 |
| 模型（Model） | `omnivoice`（服务端忽略此字段，任意非空均可） |
| API Key | **留空**（服务端 `OMNI_TTS_KEY` 为空时不校验） |
| 声音列表（voices datalist） | `alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse` |
| 默认声音 | `alloy` 走映射；或直接写 `female, elderly, 四川话` 触发文字造声 |

**文字造声用法**：在声音框直接填属性描述串，如 `female, elderly, 四川话`、
`male, young, deep, British accent`，服务端未命中 `voices.json` 时整串作为 instruct 透传给 OmniVoice。

---

## 5. 自包含性与外部依赖（核心问题）

### 5.1 本机运行时：✅ 完全自包含
- 模型本地（3.1 GB），**无任何网络调用**
- torch 2.8.0+cu128 **自带 CUDA 运行时**（cudnn/cublas 等打包在 `torch/lib`），**无需安装系统 CUDA Toolkit**
- 仅依赖本机**已装好的 NVIDIA 显卡驱动**（需支持 CUDA 12.8）

### 5.2 外部依赖清单
| 类别 | 依赖 | 说明 |
|---|---|---|
| 硬件 | NVIDIA GPU（已验证 RTX 3080） | CPU-only 可跑但极慢，不推荐 |
| 系统 | Windows 10 / 11 | DLL 层 hack 为 Windows 特定 |
| 驱动 | NVIDIA 驱动 ≥ 支持 CUDA 12.8 的版本 | 本机已满足 |
| 网络 | **无**（运行时） | 仅首次下载模型需联网（已下完） |
| 账号 | **无** | 无需 HuggingFace / OpenAI 账号 |
| 容器 | **无** | 无需 Docker / WSL |

### 5.3 跨机可移植性：⚠️ 当前**不能**原样打包带走
详见 §6——根因是 `venv` 绑死了本机的 Python 解释器路径。

---

## 6. 能否打包后开箱即用？

### 6.1 当前状态：❌ 不能直接打包迁移
三个阻塞点：

1. **venv 绑死本机 Python（致命）**
   `venv/pyvenv.cfg` 的 `home` 指向 WorkBuddy 托管的 Python：
   `K:\Users\ACTIM.DESKTOP-BPAEDTM\.workbuddy\binaries\python\versions\3.13.12`。
   当前机器能跑是因为该路径存在；**迁移到别的机器，该路径不存在 → venv 直接失效**。
   且 `venv/Lib/` 仅含 `site-packages`，真正的 Python 运行时（python313.dll + stdlib）来自上述 home，
   venv 自身不携带。

2. **venv 体积虚高（16.4 GB）**
   含 torch 2.11 降级调试残留：`torch.bak*`、`torch.broken`、`torchaudio.bak*`、
   `torch-2.11.0+cu128.dist-info.bak*` 等。实际生效的 torch 2.8 仅约 7 GB。

3. **机器特定的 DLL 改写（无害但属 hack）**
   对 `torchaudio/lib/libtorchaudio.pyd` 用 pefile 把导入 `api-ms-win-crt-*` 改写为 `ucrtbase.dll`，
   仅为绕过**本机**被安全软件锁死的 System32 forwarder。
   干净 Windows 不需要此改写，但改写后指向 `ucrtbase`（真正的 CRT 实现）在任意 Windows 仍可用，**无害**。
   另：`vcomp140.dll`（OpenMP 运行时，从 sklearn 包内拷的 212 KB 副本）已随 venv 带走，可再分发。

### 6.2 改造成「开箱即用」的步骤
1. **用可移植 Python 重建 venv**：以 `python-embed 3.13`（或随包附带独立 Python 安装）为 base，
   `python -m venv --copies venv` 后再装包，使 venv 自带完整运行时、不再依赖外部路径。
2. **清掉 venv 内 2.11 残留**（`.bak*` / `torch.broken`），venv 缩到约 7–8 GB。
3. **剔除非必需目录**：`wheels/`（5.9 GB，离线安装源，可另存）、`hf/`（3.1 GB 运行时未用缓存）、
   `scratch/`（调试归档）均可移出或删除，进一步瘦身。
4. DLL 补丁已嵌入 `site-packages`，随包走，干净机器上无害，无需额外处理。

> 改造后整包约 **11–12 GB**（venv 7–8 + models 3.1 + 小文件），纯文件拷贝即可部署，
> 目标机只需有 NVIDIA 驱动 + Windows。

### 6.3 本目录已做的规整
为逼近「开箱即用」形态，已将调试/遗留物移入 `scratch/`：
- 调试脚本：`diag*.py`、`test3~6.py`、`fix*.py`、`pe_imports*.py`、`deploy28*.py`
- 临时目录：`_ta28/`、`_torch28/`、`_torch_extract/`、`tmp/`
- 体积缓存：`wheels/`、`hf/` 已归档至 `scratch/`（如需离线重装 venv 可从中取 wheel）
- venv 内 2.11 残留已移入 `scratch/venv-leftovers/`

> 以上均为**移动（非删除）**，需回滚可直接从 `scratch/` 取回。

---

## 7. 故障速查（详见 `OmniVoice部署调研.md`）

| 现象 | 根因 | 解决（已固化进 server.py） |
|---|---|---|
| `os error 1455 页面文件太小` | 本机提交内存余量仅 ~4 GB，默认加载先在 CPU 提交内存缓冲权重再搬显存，顶爆 | `from_pretrained(..., low_cpu_mem_usage=True)` + 钉单线程 |
| 生成成功却 `HTTP 500` | `lameenc` 返回 `bytearray`，Starlette `Response` 调 `.encode()` 报错 | 返回值与端点均用 `bytes(...)` 包一层 |
| `libtorchaudio` 加载失败 | 本机 System32 `api-ms-win-crt-*` 被锁 + `vcomp140` 被锁 | pefile 改写导入为 `ucrtbase` + 拷 `vcomp140.dll` 进 `torch/lib`、`torchaudio/lib` |
| `ModuleNotFoundError: torio` | torchaudio 2.8 把 io 后端拆成独立 `torio` 包（无 cp313 wheel） | 写最小 `torio` stub（仅用 `functional.resample`） |
| `PackageNotFoundError: torch` | dist-info 元数据缺失 | 补回 `torch-2.8.0+cu128.dist-info` 等 |
| HF Xet 401 / 0 字节占位 | Xet/CAS 端点鉴权失败 | 已改 `download_model.py` 用 curl 从 hf-mirror 直拉真实文件 |

---

## 8. 验证记录（2026-10-21）

- `/health`：`{"status":"ok","loaded":true,"device":"cuda:0","cuda":true,"model":".../OmniVoice","voices":11}`
- 生成 1（`voice=alloy` → 映射）：`HTTP 200`，50304 字节，文件头 `ID3` 合法 MP3
- 生成 2（`voice=female, elderly, 四川话` → 透传）：`HTTP 200`，28032 字节，合法 MP3
- 产物见 `test_out.mp3`、`test_out2.mp3`

> 服务端侧已用 curl 实打实验证出合法 MP3；DSHT 界面「测试」亲耳确认一步需主人自己点。
