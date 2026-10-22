# OmniVoice 本地 TTS 服务 · 使用与部署说明

> 本文档为「整理版使用说明」，配套部署目录即本文所在的项目根目录（`<项目根>`）。
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
| `settings.py` | 可配置环境变量的**声明式清单**（46 项，含默认值/说明/生效方式），配置页的数据源（见 §3.10） | ✅ 必需 | ~25 KB |
| `emotion_map.py` | 情绪标签映射（DSHT 适配层）的抽取与解析逻辑（见 §3.11） | ✅ 必需 | ~15 KB |
| `admin.html` | 管理控制台页面（自包含单文件，无 CDN） | ✅ 必需 | ~15 KB |
| `start.cmd` | Windows 启动器，双击即起 | ✅ 必需 | <1 KB |
| `voices.json` | 11 个音色 → 文字造声属性映射 | ✅ 必需 | <1 KB |
| `models/OmniVoice/` | 模型权重（见 §2.1） | ✅ 必需 | **3.1 GB** |
| `python/` | 内置 embeddable CPython 3.13 运行时 + 全部依赖 | ✅ 必需 | ~7 GB |
| `history/` | 管理台请求归档（见 §3.5），运行时生成 | 产物 | 随使用增长 |
| `voices/` | 声音克隆音色库（见 §3.7），运行时生成 | 产物 | 每音色约 7 KB + 原始音频 |
| `model_config.json` | 模型注册表 + 当前生效模型（见 §3.9），运行时生成，不入库 | 产物 | <1 KB |
| `runtime_config.json` | 控制台「配置」页写入的变量覆盖（见 §3.10），运行时生成，不入库 | 产物 | <1 KB |
| `emotion_map.json` | 情绪标签映射表（当前生效，见 §3.11），运行时生成，不入库 | 产物 | <8 KB |
| `emotion_presets.json` | 自定义映射模板库（见 §3.11），运行时生成，不入库 | 产物 | <8 KB |
| `finetunes/` | 外部权重目录的**目录联接**（Windows junction，可选），用于把别盘的模型挂进项目 | 可选 | 0（不复制数据） |
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
cd /d "<项目根>"
python\python.exe server.py > server.log 2>&1
```
> 关掉窗口它就停。要长期开机自启，见 `OmniVoice部署调研.md` 第八节（NSSM / 启动项）。

### 3.3 环境变量（均可覆盖默认值）

> **也可以在控制台里改**：管理台「配置」页能查看并修改下面**全部 46 个变量**
> （默认值、当前值、取值范围、说明、生效方式），改动存到 `runtime_config.json`，
> **重启后自动生效**。完整说明见 §3.10。
>
> 下表是同一份清单的手写副本，**真相源是 `settings.py`**；
> `scratch/check_settings.py` 会双向校验两者不漂移。

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNIVOICE_MODEL` | `models/OmniVoice` | 模型目录（本地绝对/相对均可） |
| `OMNIVOICE_DEVICE` | `cuda:0` | 设备；CPU 用 `cpu`（极慢） |
| `OMNIVOICE_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `OMNIVOICE_NUM_STEP` | `32` | 扩散步数，越大越慢越细 |
| `OMNIVOICE_ASR_MODEL` | `models/whisper-large-v3-turbo`（项目内） | 逐字稿留空时 Whisper ASR 的模型位置；本地目录或 HF repo id 均可 |
| `OMNIVOICE_ASR_DEVICE` | 空（跟随主模型） | 逐字稿留空时 Whisper ASR 的加载设备；显存不够时设 `cpu`（更慢但不占显存） |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | 生成默认：引导强度（0–20） |
| `OMNIVOICE_T_SHIFT` | `0.1` | 生成默认：时间步偏移（0–1，小=强调低 SNR） |
| `OMNIVOICE_LAYER_PENALTY_FACTOR` | `5.0` | 生成默认：层惩罚因子（0–50，促使靠前 codebook 层先解码） |
| `OMNIVOICE_POSITION_TEMPERATURE` | `5.0` | 生成默认：位置选择温度（0.01–50） |
| `OMNIVOICE_CLASS_TEMPERATURE` | `0.0` | 生成默认：token 采样温度（0–10，0=贪心确定性） |
| `OMNIVOICE_DENOISE` | `1` | 生成默认：是否加 `<|denoise|>` 令牌 |
| `OMNIVOICE_POSTPROCESS_OUTPUT` | `0` | 生成默认：后处理（剪 >0.5s 句中静音 / 淡入淡出 / 边距静音）；默认关闭，设 `1` 强制开启 |
| `OMNIVOICE_PAD_DURATION` | `0.1` | 生成默认：边距静音秒数（0–2，0=不加） |
| `OMNIVOICE_FADE_DURATION` | `0.1` | 生成默认：淡入淡出秒数（0–2，0=不淡入淡出） |
| `OMNIVOICE_AUDIO_CHUNK_DURATION` | `15` | 生成默认：长文本分块时长（秒，1–60，块间交叉淡化） |
| `OMNIVOICE_AUDIO_CHUNK_THRESHOLD` | `30` | 生成默认：预估音频超此秒数才分块（1–300） |
| `OMNI_TTS_AUTO_DURATION` | `1` | **短句自动补时长**（见 §3.8）。默认开：模型会提前约 1.4 s 收尾，短句被压得只剩一点声音；开启后按文本估算长度并显式传给模型。请求自带 `duration` 时不介入。**即时生效** |
| `OMNI_TTS_AUTO_DURATION_RATE` | `65` | 目标语速：每单位「字符权重」的毫秒数（汉字权重 3.0 → 约 195 ms/字 ≈ 中文 5 字/秒）。**越大越慢** |
| `OMNI_TTS_AUTO_DURATION_LEAD` | `1.4` | 提前收尾的补偿秒数，仅对**克隆音色**生效（文字造声路径不提前收尾）。**即时生效** |
| `OMNI_TTS_PORT` | `8890` | 监听端口 |
| `OMNI_TTS_HOST` | `127.0.0.1` | 监听地址（改 `0.0.0.0` 可局域网共享） |
| `OMNI_TTS_KEY` | 空 | 鉴权密钥；空 = 不校验（DSHT 留空即可） |
| `OMNI_TTS_VOICES` | `voices.json` | 音色映射文件 |
| `OMNI_TTS_EMOTION_MAP_OFFICIAL` | `1` | 情绪映射是否允许指向**官方非语言标签**（laughter/sigh/surprise-ah/question-ah…）和**声音属性**。默认 `1`：允许，取向由所选映射模板决定（见 §3.11）。设 `0` 则屏蔽这两类目标，整张表只剩微调的语气/情绪标签。**即时生效** |
| `OMNI_TTS_HISTORY_DIR` | `history/` | 管理台请求归档目录 |
| `OMNI_TTS_CLONE_DIR` | `voices/` | 声音克隆音色库目录 |
| `OMNI_TTS_LOG_API` | `1` | 兼容开关：`0` 可关闭 OpenAI 端点（`/v1/audio/speech`）归档（即使 `OMNI_API_RECORD=1`）；DSHT 逐条调用也不再写盘 |
| `OMNI_API_RECORD` | `1` | 是否归档 OpenAI 端点请求（含失败与**缓存命中**），落 `records/` 并按三阈值轮转 |
| `OMNI_TTS_RECORD_DIR` | `records/` | OpenAI 端点记录归档目录 |
| `OMNI_RECORD_MAX_ENTRIES` | `500` | 记录轮转：最多保留条数（≤0 不限） |
| `OMNI_RECORD_MAX_MB` | `2048` | 记录轮转：最大体积 MB（≤0 不限） |
| `OMNI_RECORD_MAX_AGE_DAYS` | `30` | 记录轮转：超龄天数即删（≤0 不限） |
| `OMNI_CACHE` | `1` | 响应缓存：同参数请求直接返回已生成音频（见下） |
| `OMNI_TTS_CACHE_DIR` | `cache/` | 响应缓存目录（按 key 分桶） |
| `OMNI_CACHE_MAX_ENTRIES` | `2000` | 缓存轮转：最多条数 |
| `OMNI_CACHE_MAX_MB` | `2048` | 缓存轮转：最大体积 MB |
| `OMNI_CACHE_MAX_AGE_DAYS` | `30` | 缓存轮转：超龄天数 |
| `OMNI_API_LOG_MAX` | `300` | 管理台「请求日志」环形缓冲条数（内存，不落盘） |
| `OMNI_HISTORY_MAX_ENTRIES` | `500` | 管理台历史归档轮转：最多条数 |
| `OMNI_HISTORY_MAX_MB` | `2048` | 管理台历史归档轮转：最大体积 MB |
| `OMNI_HISTORY_MAX_AGE_DAYS` | `30` | 管理台历史归档轮转：超龄天数 |
| `OMNI_LOG_FILE` | `server.log` | 运行日志文件（由 `RotatingFileHandler` 托管，按大小轮转） |
| `OMNI_LOG_MAX_MB` | `10` | 运行日志轮转：单文件最大 MB |
| `OMNI_LOG_BACKUPS` | `5` | 运行日志轮转：保留备份数 |
| `OMNI_TTS_ADMIN_PAGE` | `admin.html` | 管理台页面文件位置 |
| `OMNI_TTS_TRAY` | `1` | 是否显示系统托盘图标（仅 Windows 有效）；`0` 关闭 |
| `OMNI_MODEL_CONFIG` | `model_config.json` | 模型注册表文件位置（运行时状态，不入库） |
| `OMNI_TTS_MODEL_DIRS` | 空 | 追加的模型扫描目录（`os.pathsep` 分隔，如 `<目录1>;<目录2>`） |

### 3.4 健康检查
```bat
curl http://127.0.0.1:8890/health
# => {"status":"ok","loaded":true,"device":"cuda:0","cuda":true,...}
```

### 3.5 管理控制台

浏览器打开 **http://127.0.0.1:8890/admin**（页面每次请求读盘，改 `admin.html` 后刷新即可生效，无需重启服务）。

**能做什么**

1. **自定义参数发起请求** —— 文本、声音/属性描述、附加属性、语速 `speed`（0.25–4）、扩散步数
   `num_step`（1–100）、输出格式 `mp3`/`wav`/`flac`/`opus`/`pcm`。
2. **高级参数折叠区** —— `OmniVoiceGenerationConfig` 全字段，按**语言与文本 / 采样与引导 / 后处理 /
   长文本分块**四组呈现，每项标注「默认值 · 建议值 · 说明」：语言 `language`（带 646 项自动补全）、
   固定时长 `duration`、`guidance_scale`、`t_shift`、`layer_penalty_factor`、`position_temperature`、
   `class_temperature`、`pad_duration`、`fade_duration`、`audio_chunk_duration`、`audio_chunk_threshold`、
   `denoise`、`postprocess_output`、`normalize_text`。默认值由 `/health` 的 `gen_defaults` 实时下发
   （环境变量改了页面跟着变），附「重置为默认」按钮一键还原全部字段（含语速与步数）。
2. **属性速选** —— 内置 OmniVoice 合法属性项（英文 23 项 / 中文 25 项），点选即按正确分隔符拼装
   （英文 `, `、中文 `，`）；切换语言时自动重置，避免中英混用被模型拒绝。
3. **请求与产出归档** —— 每次请求（**含失败**）写入 `history/<id>/`：
   - `meta.json`：完整入参、解析后的属性串、语速、步数、格式、`gen_params`（偏离默认值的高级参数）、耗时 `elapsed_ms`、字节数、状态、错误信息；
   - `audio.mp3` / `audio.wav`：实际音频产出。
4. **回看与复用** —— 列表内直接试听、下载、把历史参数一键回填表单重跑、单条删除或清空全部；顶部显示
   总条数/成功/失败/累计产出体积。
5. **API 记录子页面** —— 控制台底部「API 记录」区块分三块：①「请求日志」表（时间 / 状态 / 缓存命中 / 音色 / 模式 / 格式 / 耗时 / 字节 / 文本，环形缓冲最近 300 条）；②「音频记录与轮转」列表（与 history 同款可试听 / 下载 / 删除，并显示当前用量 vs 三阈值）；③ 顶部统计当前缓存命中 / 未命中 / 写入数与磁盘占用。DSHT 每次调用都会在此留档（默认开启，关闭见 `OMNI_API_RECORD` / `OMNI_TTS_LOG_API`）。
6. **响应缓存** —— 相同参数（`text` + `voice` + `instruct` + `format` + 全部生成参数；克隆音色还含其 `.pt` 文件签名；**当前模型 id**）的请求直接返回已生成音频，响应头 `X-Cache: hit`，跳过模型推理（实测加速数百倍）。缓存落 `cache/` 并按三阈值轮转；**命中条目仍会写入 API 记录**以保持审计完整。
   > 缓存 key **包含当前模型 id**：切换模型（§3.9）后同一句文本会重新生成，不会返回上一个模型的音频；
   > 切回原模型则又能命中它自己的缓存。
7. **模型管理面板** —— 顶部导航「模型管理」：显示当前生效模型（名称/类型/路径/设备/dtype/ASR）、
   已注册模型下拉选择、「加载并设为当前」「扫描目录」「删除所选」，
   以及手工注册新模型（id / 名称 / 路径 / 类型 checkpoint|lora）。切换即时生效并写入
   `model_config.json`，**重启后自动恢复**。详见 §3.9。

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
| GET | `/admin/api-records?limit=50` | OpenAI 端点音频记录列表 + 轮转统计（含播放 / 下载） |
| GET | `/admin/api-records/{id}` | 单条详情 |
| DELETE | `/admin/api-records/{id}` | 删除单条（含音频） |
| DELETE | `/admin/api-records` | 清空全部端点记录 |
| GET | `/admin/api-audio/{id}?ext=mp3&download=1` | 回放 / 下载端点产出音频 |
| GET | `/admin/api-log?limit=100` | 最近请求日志（环形缓冲，含缓存命中 / 失败、缓存磁盘统计） |
| GET | `/admin/languages` | 模型 646 种语言表（`ids` + `names`），供 `language` 自动补全 |
| GET | `/admin/models` | 模型注册表 + 当前生效模型 + 扫描目录（§3.9） |
| POST | `/admin/models/load` | 运行时切换模型并持久化（§3.9） |
| POST | `/admin/models/register` | 注册新模型（§3.9） |
| DELETE | `/admin/models/{id}` | 从注册表移除模型（§3.9） |
| POST | `/admin/models/scan` | 扫描可加载的模型目录（§3.9） |

---

### 3.6 系统托盘图标（通知区域）

服务启动后自动在**系统托盘**（任务栏右下角通知区域，时钟旁边）放置一个常驻图标，不必开浏览器也能找到服务：

| 操作 | 行为 |
|---|---|
| 左键单击 | 打开管理控制台（`http://127.0.0.1:8890/admin`） |
| 右键菜单 → 打开管理台 | 同上 |
| 右键菜单 → 复制完整端点（curl 等） | 复制 `http://127.0.0.1:8890/v1/audio/speech`（给 curl / 通用 OpenAI 客户端） |
| 右键菜单 → 复制 DSHT 根地址 | 复制 `http://127.0.0.1:8890/v1`（**填 DSHT 接口地址用**；DSHT 会自己补 `/audio/speech`） |
| 右键菜单 → 退出服务 | 立即停止服务（`os._exit(0)`，无优雅停机，勿在有合成任务时点） |

- **零依赖**：直接 ctypes 调 `Shell_NotifyIconW` + 自绘 32×32 ICO（圆角蓝底 + 白色音量条），不引入 pystray / Pillow。图标首次启动时写到 `tray.ico`（与 `server.py` 同目录），之后直接复用。
- **开关**：环境变量 `OMNI_TTS_TRAY=0` 关闭；非 Windows 平台自动跳过。
- 图标找不到时先看任务栏的「^」隐藏图标区——新图标可能被收在那里；explorer 重启后托盘图标会消失，需重启本服务才会重新注册（`NIM_ADD` 不跨 explorer 生命周期持久化）。

---

### 3.7 声音克隆（Voice Clone）

属性造声（Voice Design）受**固定属性白名单**限制（EN 23 / ZH 25 项，无情绪轴、不可中英混用），
做不出「某个具体角色 + 某种情绪」。声音克隆用一段参考音频绕开这个限制：音色与情绪都来自那段音频，
模型据此朗读任意新文本。

| 项 | 要求 |
|---|---|
| 参考音频 | 3–10 秒（>20 秒官方会警告：更慢、更占内存、克隆质量下降），wav/mp3/flac/ogg |
| `ref_text` | 参考音频的逐字稿，**可留空**——留空时先找同目录同名文本，找不到就用 Whisper 自动转写（见下） |
| 音色名称 | 可留空 = 用音频文件名（去扩展名）；已存在则自动追加 `_1`、`_2`… 不覆盖。手填的名称已存在则报 409 |

**逐字稿留空时发生什么（按优先级）**

管理台的克隆卡片走**本地音频路径**（音频文件的绝对路径，不要求在某个"库"里，任意位置都行）。
`ref_text` 留空时按顺序尝试：

1. **同目录同名文本**（免费、最准）：`<stem>.txt` → `<stem>.lab` → `<stem>.json`
   （json 取 `transcription` / `text` / `transcript`）。不做任何搜索/全库匹配。
2. **Whisper 自动转写**：同目录也没有时，模型用自带的 Whisper
   （`openai/whisper-large-v3-turbo`）转写。模型走**本地副本** `models/whisper-large-v3-turbo`
   （与主模型同一约定，`OMNIVOICE_ASR_MODEL` 可覆盖指向别处），首次加载约 1.6GB 常驻 ASR 设备
   （默认跟着主模型走 cuda:0，可用 `OMNIVOICE_ASR_DEVICE=cpu` 挪到内存）；每次转写几秒。
   转写结果会写进音色元信息（`ref_text_source: "asr"`），控制台列表里带 `ASR` 标记。
3. 手填永远优先级最高，也永远最准——逐字稿与音频不一致会让对齐漂移，克隆音质下降。

路径填目录 + 点「批量注册」则扫描目录下音频（按文件名排序，默认最多 20 条），
逐个取各自同目录的逐字稿，音色名 = 「音色名称」前缀 + 文件名。没有逐字稿的默认**跳过**；
勾选「无逐字稿时 ASR 转写」（或 API 传 `asr=1`）才逐个自动转写。

- 嫌打字麻烦就点路径框右侧的 **「选文件…」/「选目录…」**：由服务端进程弹出 Windows 原生
  选择框（PowerShell + WinForms，**无需额外依赖**），选中后自动回填绝对路径。取消则返回空。

> 为什么界面不做浏览器上传/文件选择器：浏览器只能拿到文件名，拿不到绝对路径，
> 也就读不到音频旁边的同名文本——那条路对"同目录取逐字稿"是假的。
> `POST /admin/voices` 仍接受 `audio` 表单字段（`ref_text` 现在也可留空，走 ASR），
> 供脚本/curl 使用（素材库的 `scripts/register.py` 就走这条）。

**存储与缓存**（`voices/<name>.*`）：

| 文件 | 内容 |
|---|---|
| `<name>.pt` | `VoiceClonePrompt.save()` 产物，约 7 KB（音频已 token 化，极小） |
| `<name>.json` | 元信息：`ref_text`（ASR 路径下为转写结果）、`ref_text_source`（`provided`/`asr`）、时长、采样率、来源文件名、`preprocess_prompt`、创建时间 |
| `<name>.ref<ext>` | 原始参考音频，供页面回听 |

进程内另有 `dict` 缓存已加载的 prompt，命中即不再读盘；**服务重启不丢失**（落在磁盘上）。

**`preprocess_prompt`（默认开）**

OmniVoice 建克隆 prompt 时会做三步预处理：裁掉参考音频里的静音、超 20s 的按最大静音间隙切短、
逐字稿末尾补一个句末标点。管理台克隆卡片有勾选框；关掉（`preprocess_prompt=0`）= 原样嵌入，
仅用于逐字稿已带标点且静音本身有意义的场景。生成端点不暴露此字段——我们恒走预建好的
`voice_clone_prompt`，该字段只在建 prompt 时起作用。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/admin/voices` | 音色库列表（只读元信息，不加载 prompt） |
| POST | `/admin/voices` | `name`（可空）+ `local_path`（服务端读盘）+ `ref_text`（可空：同目录同名文本 → 找不到走 ASR）+ `preprocess_prompt`（`"1"`/`"0"`，默认 `"1"`）；`audio` 上传字段仍兼容（`ref_text` 可空，直走 ASR） |
| POST | `/admin/import-local` | `path`（目录或文件）+ `prefix` + `limit` + `preprocess_prompt` + `asr`（`"1"`=无逐字稿的也自动转写，默认跳过）：批量注册 |
| POST | `/admin/pick-path` | `{"mode":"file"|"dir","title","initial_dir"}`：服务端弹原生选择框，返回选中路径（空=取消） |
| DELETE | `/admin/voices/{name}` | 删除音色及其参考音频 |
| GET | `/admin/voices/{name}/audio` | 回听原始参考音频 |

> **「声音 / 属性描述」字段允许混用**（2026-10-22 起）。一个字段里可以同时写：
>
> | 写法 | 例子 | 处理方式 |
> |---|---|---|
> | 官方预设名 | `alloy`、`fable` | 展开成属性串（`voices.json`） |
> | 文字属性 | `female, elderly, 四川话` | 直接作为属性项 |
> | 克隆音色名 | `ja_Ganyu_0001` | **走克隆**，并从属性串中剔除 |
> | 行内标签 | `[happy]`、`[laughter]` | **移到文本开头**，并从属性串中剔除 |
> | 认不出的项 | `温柔地`、`[nope]` | **丢弃**，并在请求日志里标记 |
>
> - **不报错**：任何无法使用的项都被丢弃而不是让请求失败，音频照常产出。
> - **请求日志标记**：`/admin/api-records` 与 `history/` 的记录里带
>   `dropped`（丢弃了什么）、`matched_clone`（命中了哪个音色）、
>   `inline_tags`（用了哪些行内标签）、`presets`（命中了哪些预设）、
>   `instruct`（最终属性串），`voice_mode` 为 `clone` / `design` / `clone+design`。
> - **行内标签按当前模型判定**：`emotion-en` 认 8 个情绪标签
>   （`[singing] [happy] [sad] [angry] [nervous] [whisper] [calm] [excited]`）+ 官方 13 个符号；
>   官方基座只认 13 个符号，多的会被丢弃并标记。切换模型后无需改请求。
> - **冲突裁决**（同类属性以**先出现者**为准，其余丢弃并标记）：
>   ① 多个克隆 → 第一个胜出；② `voice` 与 `instructions` 各有克隆 → `voice` 优先；
>   ③ 预设与手写属性同类冲突 → 先出现者胜出；
>   ④ `british accent` 与 `四川话` 不能共存（前者英语专用、后者中文专用）→ 丢弃 accent；
>   ⑤ 中英混写 → 自动统一到单一语种。
> - **附加属性（`instructions`）同逻辑处理**：可写预设名 / 克隆名 / 行内标签，会追加合并并按同样规则去重。
>
> **克隆音色与文字属性可以同时使用**。
>
> - **怎么传**：`voice` 填克隆音色名，属性写在 `instructions` 字段
>   （管理台：「声音」填克隆名 + 「附加属性」填属性串即可，恒生效）。
> - **效果**：属性会**叠加在克隆音色之上**，实测把克隆的音高从 308 Hz 压到 118 Hz
>   （`male, low pitch`），或抬到 338 Hz（`female, high pitch`）。
> - **情绪短语先走映射层**：DSHT 的情绪短语（如「温柔地」）虽然不在 voice-design 白名单里，
>   但会**先经 §3.11 的映射层**转成模型认得的标签（基座上是耳语属性、情绪微调上是 `[whisper]`
>   行内标签）；**映射不到的**才丢弃而非报错，请求照常成功。
>
> ℹ️ `instructions` **恒被解析**，没有开关。早期版本有 `OMNI_TTS_USE_INSTRUCTIONS` 环境变量与
> 请求参数 `use_instructions`，已移除——关掉它会**静默丢掉 DSHT 的全部情绪**
> （DSHT 只在 `instructions` 里送情绪，且从不带该参数）。现在仍带 `use_instructions` 的请求
> 会被接受但该字段忽略，不影响旧调用方。
>
> ⚠️ 属性是**相对参考音色的偏移**，不是绝对设定：参考音色已满足某属性时位移趋近于零
> （给女声克隆加 `female, high pitch` 几乎不动）。**想改克隆的性别/音高，
> 换参考音频比加属性更可靠。**
>
> ⚠️ **行内标签在克隆模式下效果被削弱**：`[whisper]` 在克隆下仍明显（RMS −48%），
> 但 `[happy]` 在克隆下几乎无变化（RMS +0.2%），而纯文字造声下是 +14.6%——
> 克隆音色本身已经占满了「表现力」。**要情绪请用纯文字造声，要音色请用克隆。**

> ⚠️ **克隆 + 短句会被截尾**（见 §3.7.1「短句长度补偿」）：模型会提前约 1.4 秒收尾，
> 10 字台词实测只剩 0.22 秒有声（约 31 字/秒）。**默认已自动补偿**，别再手动调 `speed` 去救。

### 3.7.1 短句长度补偿（为什么短台词会被截尾）

**现象**：克隆音色念短台词时尾部消失。实测（`ja_Kurumi_0011_恋慕_恋爱情话`，10 字）：

| 用例 | 有声时长 | 语速 |
|---|---:|---:|
| 克隆 · 10 字（补偿前） | **0.22 s** | 31 字/秒 ← 根本听不清 |
| 设计音色 · 10 字 | 1.58 s | 6.3 字/秒（正常） |
| 克隆 · 10 字（补偿后） | **1.11 s** | 6.3 字/秒 |
| 克隆 · 19 字（补偿前 → 后） | 2.60 s → 3.16 s | 6.5 → 5.4 |

**根因**：模型会**提前约 1.4 秒发结束符**，且这个提前量是**固定的**、与句子长短无关：

| 文本 | 上游预算（目标 token / 帧率） | 实际输出 | 差额 |
|---|---:|---:|---:|
| 10 字 | 1.97 s | 0.60 s | −1.37 |
| 19 字 | 4.56 s | 2.94 s | −1.62 |
| 31 字 | 7.99 s | 6.59 s | −1.40 |

长句丢 1.4 s 只是收尾紧一点；短句丢 1.4 s 就是**整句没了**。文字造声路径不这样（输出≈预算），
所以补偿只对**克隆音色**生效。

**做法**：按文本的字符权重估算目标时长（`weight × 65 ms`，汉字权重 3.0 → 约 195 ms/字），
加上 1.4 s 提前量，作为 `duration` 显式传给模型；**只有估算值超过上游自身预算时才介入**，
所以长句、以及文字造声路径**行为完全不变**。请求里自带 `duration` 时不介入。

**三个旋钮**（配置页「生成」组，**即时生效**）：`OMNI_TTS_AUTO_DURATION`（默认开）、
`OMNI_TTS_AUTO_DURATION_RATE`（65，越大越慢）、`OMNI_TTS_AUTO_DURATION_LEAD`（1.4，仍截尾就调大）。

### 3.8 OpenAI 端点请求参数全表（`POST /v1/audio/speech`）

除 OpenAI 标准字段外，OmniVoice 的 `OmniVoiceGenerationConfig` 全字段 + `language` / `duration` /
`normalize_text` 均可按请求覆盖。缺省值即 §3.3 环境变量（`/health` 的 `gen_defaults` 可查实时值）。

| 字段 | 类型 | 默认 | 范围 / 取值 | 建议 | 说明 |
|---|---|---|---|---|---|
| `input`（或 `text`） | str | — | 必填，≤4000 字 | — | 要合成的文本 |
| `voice` | str | `alloy` | **可混用**：预设名 / 属性串 / 克隆音色名 / 行内标签 | 按需 | 见 §3.7 与 §4；无法识别的项被丢弃并在请求日志标记 |
| `instructions` | str | 空 | 与 `voice` 同逻辑（属性项 / 预设名 / 克隆名 / 行内标签） | 留空 | **恒拼入**（无开关；旧参数 `use_instructions` 已废弃，传了也忽略） |
| `response_format` | str | `mp3` | `mp3` `wav` `pcm` `flac` `opus` | `mp3` | 未知值返回 **400**（列出合法值），不静默回落 |
| `speed` | num | `1.0` | 0.25–4 | 0.8–1.2 | 语速倍率 |
| `num_step` | int | `32` | 1–100 | 16–48 | 扩散步数，越大越稳越慢，48+ 收益递减 |
| `language` | str | 空 | ISO 代码 / 英文名称 / `auto` / 空 | 按主要语言填 | 空=语言无关模式（中英混说用它）；`auto` 按文本猜（kana→ja、hangul→ko、汉字→zh、西里尔→ru、其余→en）；无法识别返回 400 |
| `duration` | num | 空 | 0.05–600 | 留空 | 固定输出秒数，**覆盖 speed**；空=按文本估计（克隆短句会自动补偿，见 §3.7.1） |
| `normalize_text` | bool | `false` | — | 按需 | 数字/日期/货币转读法。**本机未装 WeTextProcessing/num2words，勾选返回 400**，不会静默跳过 |
| `guidance_scale` | num | `2.0` | 0–20 | 1.5–3.0 | 分类器自由引导强度，越大越严格贴合 instruct/文本，过高发紧不自然 |
| `t_shift` | num | `0.1` | 0–1 | 0.05–0.2 | 时间步偏移，越小越强调低 SNR 段（先定结构后细节） |
| `layer_penalty_factor` | num | `5.0` | 0–50 | 保持默认 | 层惩罚因子，促使靠前 codebook 层（语义）先解码；调小声学瑕疵 |
| `position_temperature` | num | `5.0` | 0.01–50 | 保持默认 | 位置选择温度 |
| `class_temperature` | num | `0.0` | 0–10 | 0 | token 采样温度（0=贪心确定性）；想多样可试 0.3–0.8，可能读错字 |
| `denoise` | bool | `true` | — | 保持开启 | 是否加 `<|denoise|>` 令牌 |
| `postprocess_output` | bool | `false` | — | 默认关闭；`OMNIVOICE_POSTPROCESS_OUTPUT=1` 强制开启 | 剪 >0.5s 句中静音 / 淡入淡出 / 边距静音 |
| `pad_duration` | num | `0.1` | 0–2 | 0.05–0.2 | 输出两端静音秒数；0=不加，句首辅音可能被切 |
| `fade_duration` | num | `0.1` | 0–2 | 0.05–0.2 | 头尾淡入淡出秒数，防爆音 |
| `audio_chunk_duration` | num | `15` | 1–60 | 10–20 | 长文本分块时长（秒），块间交叉淡化 |
| `audio_chunk_threshold` | num | `30` | 1–300 | 20–60 | 预估音频超此秒数才分块 |

> 管理台高级参数区按 **语言与文本 / 采样与引导 / 后处理 / 长文本分块** 四组呈现，每个字段旁标注
> 「默认 · 建议 · 说明」，默认值由 `/health` 的 `gen_defaults` 实时下发（env 改了页面跟着变），
> 并有「重置为默认」按钮一键还原（含主表单的语速与步数）。

**输出格式说明**

| 值 | 容器 / 编码 | 备注 |
|---|---|---|
| `mp3` | MPEG Layer III（lameenc，64 kbps） | 体积最小，DSHT 默认 |
| `wav` | WAV / PCM 16-bit | 无损，兼容性最好 |
| `flac` | FLAC / PCM 16-bit | 无损压缩 |
| `opus` | **OGG 容器** + Opus | libsndfile 无裸 Opus 复用器，媒体类型是 `audio/ogg` |
| `pcm` | **裸 PCM** 16-bit 小端单声道 24 kHz | 无容器无文件头，OpenAI `pcm` 语义 |

> 不支持 AAC——libsndfile 不写 AAC。（旧版把 flac/opus/aac 全落 WAV 是假支持，已修。）

**文本处理要点**（标点不会丢，但会被规范化）

- 全角括号转半角、压缩空白、去汉字两侧空格；克隆模式 `ref_text + " " + text` 拼接。
- 中文标点走 byte-level fallback（模型侧保留），ASCII 标点有专属 token；标点影响时长估计（权重 0.5）。
- 支持 14 个非语言标签（`[laughter]` 等）与大写拼音+声调数字的发音控制语法（`[B EY1 S]`）。
- 预估 >30s 的长文本按标点分块（缩写感知），约 15s/块交叉淡化拼接。

### 3.9 模型管理（运行时切换 · 持久化生效）

服务可加载**多个模型**并在运行时热切换，切换结果写入 `model_config.json`，
**重启后自动恢复**上次的选择。管理台「模型管理」面板或下面的 API 都能操作。

**支持的模型类型**

| type | 含义 | 加载方式 |
|---|---|---|
| `checkpoint` | 完整权重（官方基座 / 全参微调） | 直接 `from_pretrained(path)` |
| `lora` | PEFT 适配器 | 先加载其 `base` 指定的基座，再把 LoRA 增量**手动合并**进 `model.llm` |

> **为什么 LoRA 走手动合并而不是 PEFT 自动合并**：PEFT 的 `PeftModel.from_pretrained()`
> 会把 Windows 盘符路径当成 HF repo id 校验而拒绝；且 OmniVoice 的 `model.llm` 是裸
> `Qwen3Model`（无 generation mixin），PEFT 初始化时取 `prepare_inputs_for_generation`
> 直接 AttributeError。手动合并直接读 `lora_A`/`lora_B`，按 `adapter_config.json` 的
> `r`/`lora_alpha`/`use_rslora` 算缩放（RSLoRA 为 `α/√r`）后累加进基座权重。

**模型注册表 `model_config.json`**

- 位于项目根目录，**已加入 `.gitignore`**（本机运行时状态，不入库）。
- 路径以**相对项目根**存储（如 `models/OmniVoice`、`finetunes/xxx`），整体搬走项目仍可用。
- `active` 字段即当前生效的模型；`/health` 会回报 `model` / `model_name` / `model_type` /
  `loaded_id`。
- 删掉该文件即回到默认（只注册官方基座 `base`）。

**扫描目录**（管理台「扫描目录」按钮 / `POST /admin/models/scan`）

按运行时装配位置推导，**不含任何写死的盘符**：

1. `<项目根>/models`、`<项目根>/finetunes`
2. 项目根**同级**目录中名字含 `finetune` / `model` 的目录
3. 环境变量 `OMNI_TTS_MODEL_DIRS`（`os.pathsep` 分隔）追加的任意目录

把外部权重目录接进项目的方式（Windows 目录联接，不复制数据）：

```powershell
New-Item -ItemType Junction -Path "<项目根>\finetunes" -Target "<权重盘>:\OmniVoice_finetunes"
```

扫描会按目录内容自动判定类型：有 `adapter_config.json` → `lora`；
`config.json` 的 `architectures` 含 `OmniVoice` → `checkpoint`。

**API**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/admin/models` | 列出注册表、当前生效模型、设备/dtype/ASR、扫描目录 |
| POST | `/admin/models/load` | `{"id":"..."}`：运行时切换（重建模型，约数十秒）并持久化 |
| POST | `/admin/models/register` | `{"id","name","path","type","base"}`：`type` ∈ `checkpoint`/`lora`；`base` 省略时取第一个 checkpoint |
| DELETE | `/admin/models/{id}` | 从注册表移除（**不删磁盘文件**）；`base` 与当前生效模型受保护 |
| POST | `/admin/models/scan` | 扫描可加载目录，返回每个目录的 `type` 与是否已注册 |

**切换语义**

- 新模型在**锁外**加载（重 I/O + 显存），完成后在锁内**原子替换**；旧模型在替换后才释放，
  因此在途请求不会撞到 `None`。
- 切换失败（如路径错误、显存不足）**保留原模型继续服务**，并返回 500 + 原因。
- 释放旧模型后会 `gc.collect()` + `torch.cuda.empty_cache()` 归还显存。
- 切换需要重建整个模型（约 1–2 分钟，取决于权重），期间新合成请求会等待。

### 3.10 配置页面（环境变量与参数默认值）

管理台顶部导航「配置」：**查看并修改全部 46 个可配置变量**。

**每个变量显示**

| 元素 | 含义 |
|---|---|
| 变量名 | 如 `OMNIVOICE_GUIDANCE_SCALE` |
| **即时生效** / **需重启** | 绿色标签＝改完立刻应用；橙色标签＝下次启动服务时生效 |
| **默认** / **已配置** / **环境变量覆盖** | 当前值来自哪里 |
| 输入控件 | 文本框 / 数字框（带 min–max）/ 下拉框 / 开关 |
| 说明 + 默认值 + 范围 | 逐项的中文说明 |

**分组**：服务与网络（6）· 模型与推理（8）· 生成参数默认值（11）· 目录与轮转（4）·
保留条数与配额（14）· 日志（3）· 指令与属性（1）。

**生效方式**

- **12 项即时生效**：11 个生成参数（`guidance_scale` / `t_shift` / `layer_penalty_factor` /
  `position_temperature` / `class_temperature` / `denoise` / `postprocess_output` /
  `pad_duration` / `fade_duration` / `audio_chunk_duration` / `audio_chunk_threshold`）
  直接改内存里的 `GEN_DEFAULTS`，保存即生效，管理台的高级参数区默认值也跟着变；
  以及 `OMNI_TTS_MODEL_DIRS`（每次扫描都读环境）。
- **34 项需重启**：路径、端口、设备、精度、轮转上限等启动时读取的项。
  保存后启动日志会列出注入的变量：``applied configured env: NAME=value, ...``。

**优先级**：**真实环境变量 > 配置文件 > 内置默认**。运维在启动前设的环境变量优先，
控制台里存的同名值不生效，页面会标为「环境变量覆盖」并在保存时提示
「被环境变量覆盖（改动暂不生效）」。

**存储**：`runtime_config.json`（项目根，**已 gitignore**）。删掉即回到全部默认。

**校验**：按类型校验（整数 / 浮点 / 布尔 / 枚举）+ 取值范围；**任一项非法则整个请求被拒
（400），且不会写入一半**。未知变量名同样被拒。

**API**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/admin/settings` | 变量清单、当前值、来源、说明、分组、配置文件路径 |
| POST | `/admin/settings` | `{"values":{"NAME":"value"}}`：保存（live 的立刻应用） |
| POST | `/admin/settings/reset` | `{}` 全清；`{"names":["X"]}` 清指定项（live 的恢复默认） |

> **清单的真相源是 `settings.py`**（声明式，含默认值/说明/生效方式）。
> `scratch/check_settings.py` 双向校验「代码读了但清单缺」「清单有但代码没读」
> 「默认值与代码不一致」，防止两边漂移。

### 3.11 情绪标签映射（DSHT 适配层）

DSHT 发在 `instructions` 字段里的是**情绪短语**（如「温柔地」「生气」）。
OmniVoice 的语音设计属性**没有情绪轴**，这些短语原本会被直接丢弃。
本节就是把它们接到模型真正认得的标签上。

**映射表**：控制台顶部导航「**情绪标签映射**」直达（独立面板，也可直接访问
`/admin#emotionmap`），存于 `emotion_map.json`（已 gitignore），
首次运行写入 **75 条内置默认**（微调组 62 条 + 事件型 16 条）。可增删改，**保存后立即生效**。

**映射模板（预设组）**

当前**生效表**只有一份，但可以存成/导入任意多个**模板**。模板库在 `emotion_presets.json`（已 gitignore）。

| 模板 | 来源 | 目标取向 | 条数 |
|---|---|---|---|
| `微调映射模板` | 内置（只读） | 只指向微调的语气/情绪标签：`whisper` `happy` `sad` `angry` `nervous` `calm` `excited` `singing` | 62 |
| `官方映射模板` | 内置（只读） | 只指向官方符号与声音属性：`laughter` `sigh` `surprise-ah` `question-ah` `dissatisfaction-hnn` `whisper` … | 75 |
| `自定义映射模板#名字N` | 你自己存 | 任意 | 任意 |

四个动作（面板顶部一排按钮）：

| 动作 | 说明 |
|---|---|
| **替换导入** | 当前生效表整张换成模板内容（未保存的改动会丢） |
| **追加导入** | 模板条目并入当前表：同名覆盖、其余保留（可叠加多套） |
| **保存 / 覆盖为模板** | 把当前生效表存成模板；内置模板名不可占用。想从零建组：先「清空」再取名保存 |
| **删除模板** | 只删模板，当前生效表不动 |

生效表会记住自己来自哪个模板（面板右上方显示）；一旦手改条目就变成「自定义（未保存为模板）」。
**恢复内置默认** = 两条内置模板叠加后的那张 75 条表（事件型短语按官方符号读）。

**匹配规则**

- **最长优先 + 子串匹配**：「用温柔的语气说话」会命中「温柔」；重叠片段只取一个，最长的优先。
- 匹配到的短语会**从文本里摘除**，剩余部分再按普通属性解析。

**目标按当前模型能力自动解析**（同一张表在两套模型上都成立）

| 目标指向 | 落到哪 | 例子 |
|---|---|---|
| 当前模型的**行内标签** | 加到**文本开头** | emotion-en 下「温柔地」→ `[whisper]` |
| **语音设计属性** | 进**属性串** | 官方基座下「温柔地」→ 属性 `whisper` |
| 两者都不是 | 丢弃并在请求日志标出 | 官方基座下的「生气」（基座无情绪轴） |

> 所以「温柔地」在官方基座上会变成耳语属性、在情绪微调上会变成 `[whisper]` 行内标签——
> **同一份配置自动适配，不用为两套模型各配一次**。

**两类目标：是"听起来像"还是"真的发声"**

| 类别 | 目标 | 内置默认怎么读 | 例子 |
|---|---|---|---|
| 语气 / 情绪（微调标签） | `whisper` `happy` `sad` `angry` `nervous` `calm` `excited` `singing` | 温柔、轻声、害羞、开心、生气、困惑、释然 | 让这句**听起来**是这个情绪 |
| 非语言事件 / 属性 | 13 个官方符号（laughter/sigh/surprise-ah/question-ah…）与语音设计属性 | 惊讶、叹气、笑、疑问、确认、不满 | 让这句**真的**笑一声 / 叹一口气 / "啊？" |

> 官方那 13 个符号是**发声事件**，套在台词上会盖住角色音色——不过实测后发现，
> 之前听到"音色变了"的真正原因是**音色名被短语匹配啃断**（见下）导致克隆没生效，
> 不是这些符号本身。所以**默认不再屏蔽**它们，取向交给所选模板决定。
> `OMNI_TTS_EMOTION_MAP_OFFICIAL=0` 可重新屏蔽官方符号与属性，把整张表压回只剩微调标签
> （配置页「指令与属性」分组，**即时生效**）；此时面板会把这类条目标注「策略禁用」。
> `whisper` 是唯一的例外豁免：它在基座上是属性、在微调上是行内标签，始终可用。

**名字优先，短语其次（防误伤）**

- **先认领名字，再匹配短语**：音色名、预设名、属性、`[标签]` 一律先被结构化识别，
  剩下的自由文本才做短语匹配；匹配时还会把已识别的名字**挖空屏蔽**。
  所以 `ja_Kotori_0015_欢笑_疑问询问` 不会因为内含「疑问」被啃坏，
  `ja_Kotori_0007_平静_疑问询问 害羞` 也能同时认出音色和情绪。

**实测**（情绪微调模型，文本「今天真是太好了。」相对无标签）：

| 短语 | RMS | 频谱质心 |
|---|---:|---:|
| 温柔地 | **−46.2%** | +55.9% |
| 叹气 | −26.3% | +64.6% |
| 生气 | −21.2% | +4.4% |
| 唱歌 | −1.8% | +17.0% |

**API**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/admin/emotion-map` | 映射表、每个目标的类型与当前是否可用、当前模型支持的行内标签、全部可选目标、模板列表 `presets`、生效表来源 `source` |
| POST | `/admin/emotion-map` | `{"mappings":{...}}` 整表替换，或 `{"phrase":"温柔地","targets":["whisper"]}` 单条增/覆盖，`{"phrase":"x","remove":true}` 删除 |
| POST | `/admin/emotion-map/reset` | 恢复内置默认表 |
| POST | `/admin/emotion-map/clear` | 清空生效表（模板不受影响） |
| POST | `/admin/emotion-map/preset/apply` | `{"name":"官方映射模板","mode":"replace"\|"merge"}` 导入模板 |
| POST | `/admin/emotion-map/preset/save` | `{"name":"自定义映射模板#1"}` 存当前表；也可带 `{"mappings":{...}}` 直接存指定内容 |
| POST | `/admin/emotion-map/preset/delete` | `{"name":"..."}` 删除自定义模板（内置模板拒绝） |

> **目标标签名**用 OmniVoice 的英文名（`whisper` / `happy` / `angry` / `sigh` / `laughter` …），
> 控制台会标出每个目标是「行内标签」「属性」还是「当前模型不可用」。
> 情绪微调模型加载后，「行内标签」集合会从 13 个官方符号扩展到 21 个——**切模型后映射表不用改**。

---

## 4. DSHT 配音接入配置

DSHT 的 `openai` 渠道会**自己拼路径**：最终请求地址 = 你填的地址去掉末尾斜杠 + `/audio/speech`
（见 tavern 插件 `lib/domain/speech.js`）。所以这里填**根地址**，不能填完整端点。

| 字段 | 填写值 |
|---|---|
| 配音方案 / Provider | `openai`（OpenAI 兼容） |
| **接口地址（Provider Endpoint）** | `http://127.0.0.1:8890/v1` ⚠️ 只写到 `/v1`；DSHT 会补 `/audio/speech`。填成 `.../v1/audio/speech` 会变成 `/v1/audio/speech/audio/speech` 而 404 |
| 模型（Model） | **服务端忽略此字段**，但 DSHT 用它决定要不要带情绪：模型名匹配 `/gpt-4o/` 时才发送 `instructions`。想要情绪就填 `gpt-4o-mini-tts`（默认值），填别的则整条链路不带情绪 |
| API Key | **必须填**（DSHT 端校验非空），内容随意，例如 `local`；服务端 `OMNI_TTS_KEY` 为空时不校验 |
| 声音列表（voices datalist） | `alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse` |
| 默认声音 | `alloy` 走映射；或直接写 `female, elderly, 四川话` 触发文字造声；也可直接写克隆音色名 |
| 旁白声音 | 留空 = 旁白不念；填了则旁白段用它单独请求 |

**文字造声用法**：在声音框直接填属性描述串，如 `female, elderly, 四川话`、
`male, young, deep, British accent`，服务端未命中 `voices.json` 时整串作为 instruct 透传给 OmniVoice。

### 4.1 DSHT 实际会发送什么（协议真相）

按上表配置后，DSHT 每念一句就发**一次** `POST /v1/audio/speech`，请求体固定 5 个字段、最多 6 个：

```json
{
  "model": "gpt-4o-mini-tts",
  "input": "等、等等，那里不行！我刚吃饱，肚子鼓出来很丑的……",
  "voice": "ja_Kurumi_0011_恋慕_恋爱情话",
  "response_format": "mp3",
  "instructions": "用害羞的语气说这句话。"
}
```

| 字段 | 来源 | 说明 |
|---|---|---|
| `model` | 配置里的模型名 | 服务端忽略；仅 DSHT 用 `/gpt-4o/` 判断是否发 `instructions` |
| `input` | 台词文本 | 经 `speechText()` 清洗：删 `<...>` 标签与 `* _ ~ \` # > \|`，压空白，截 1000 字。**方括号不清洗**，所以行内标签可存活 |
| `voice` | 角色卡 `voices.openai` → 默认声音 → 旁白声音（旁白段） | 三选一；旁白留空则整段跳过、不请求 |
| `response_format` | 恒为 `mp3` | 服务端按此返回 |
| `instructions` | `用<中文情绪>的语气说这句话。` | **仅**当模型名含 `gpt-4o` 且情绪非 `neutral` 才出现。情绪由后台 `cast_submit` 标注，取值是下表 28 个英文 key |

**不会**发送：`speed`、`language`、`duration`、`num_step`、`use_instructions`、`normalize_text`。
这些参数面只对**手写调用**（curl / 管理台）有意义，DSHT 通道一律走服务端默认值（`speed=1.0`、`num_step=32`、`language=auto`）。

情绪词对照（DSHT 侧英文 key → 它发出的中文词 → 服务端映射层目标）：

| key | 中文 | key | 中文 | key | 中文 |
|---|---|---|---|---|---|
| admiration | 钦佩 | disgust | 嫌弃 | optimism | 期待 |
| amusement | 逗乐 | embarrassment | 害羞 | pride | 得意 |
| anger | 生气 | excitement | 兴奋 | realization | 恍然 |
| annoyance | 烦躁 | fear | 害怕 | relief | 释然 |
| approval | 赞同 | gratitude | 感激 | remorse | 懊悔 |
| caring | 关心 | grief | 悲痛 | sadness | 难过 |
| confusion | 困惑 | joy | 开心 | surprise | 惊讶 |
| curiosity | 好奇 | love | 爱慕 | nervousness | 紧张 |
| desire | 渴望 | neutral | 平静（不发指令） | disappointment | 失望 |

> 映射层收到的是**整句**「用X的语气说这句话。」，会先剥掉句式骨架再按子串匹配 X，
> 因此上表 28 个词全部覆盖；映射不到的词会记进归档的 `dropped`/`unmatched` 而不是静默吞掉。

**DSHT 侧也有缓存**：它按 `sha256(渠道 + 配置 + voice + emotion + text)` 把音频存进
`<data>/speech-cache/`。同一个文本改了服务端配置（换模型、改映射表）后，DSHT 不会重新请求——
排障时若怀疑"改了没生效"，先清那个目录再看。

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
   `<用户目录>/.workbuddy/binaries/python/versions/<版本号>`。
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
