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
| `voices/` | 声音克隆音色库（见 §3.7），运行时生成 | 产物 | 每音色约 7 KB + 原始音频 |
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
| `OMNIVOICE_ASR_MODEL` | `models/whisper-large-v3-turbo`（项目内） | 逐字稿留空时 Whisper ASR 的模型位置；本地目录或 HF repo id 均可 |
| `OMNIVOICE_ASR_DEVICE` | 空（跟随主模型） | 逐字稿留空时 Whisper ASR 的加载设备；显存不够时设 `cpu`（更慢但不占显存） |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | 生成默认：引导强度（0–20） |
| `OMNIVOICE_T_SHIFT` | `0.1` | 生成默认：时间步偏移（0–1，小=强调低 SNR） |
| `OMNIVOICE_LAYER_PENALTY_FACTOR` | `5.0` | 生成默认：层惩罚因子（0–50，促使靠前 codebook 层先解码） |
| `OMNIVOICE_POSITION_TEMPERATURE` | `5.0` | 生成默认：位置选择温度（0.01–50） |
| `OMNIVOICE_CLASS_TEMPERATURE` | `0.0` | 生成默认：token 采样温度（0–10，0=贪心确定性） |
| `OMNIVOICE_DENOISE` | `1` | 生成默认：是否加 `<|denoise|>` 令牌 |
| `OMNIVOICE_POSTPROCESS_OUTPUT` | `1` | 生成默认：后处理（剪 >0.5s 句中静音 / 淡入淡出 / 边距静音） |
| `OMNIVOICE_PAD_DURATION` | `0.1` | 生成默认：边距静音秒数（0–2，0=不加） |
| `OMNIVOICE_FADE_DURATION` | `0.1` | 生成默认：淡入淡出秒数（0–2，0=不淡入淡出） |
| `OMNIVOICE_AUDIO_CHUNK_DURATION` | `15` | 生成默认：长文本分块时长（秒，1–60，块间交叉淡化） |
| `OMNIVOICE_AUDIO_CHUNK_THRESHOLD` | `30` | 生成默认：预估音频超此秒数才分块（1–300） |
| `OMNI_TTS_PORT` | `8890` | 监听端口 |
| `OMNI_TTS_HOST` | `127.0.0.1` | 监听地址（改 `0.0.0.0` 可局域网共享） |
| `OMNI_TTS_KEY` | 空 | 鉴权密钥；空 = 不校验（DSHT 留空即可） |
| `OMNI_TTS_VOICES` | `voices.json` | 音色映射文件 |
| `OMNI_TTS_USE_INSTRUCTIONS` | `0` | 管理台「附加属性」开关的默认值 |
| `OMNI_TTS_HISTORY_DIR` | `history/` | 管理台请求归档目录 |
| `OMNI_TTS_CLONE_DIR` | `voices/` | 声音克隆音色库目录 |
| `OMNI_TTS_LOG_API` | `0` | 是否也归档 OpenAI 端点（`/v1/audio/speech`）的请求；DSHT 逐条调用会持续写盘，默认关 |
| `OMNI_TTS_ADMIN_PAGE` | `admin.html` | 管理台页面文件位置 |
| `OMNI_TTS_TRAY` | `1` | 是否显示系统托盘图标（仅 Windows 有效）；`0` 关闭 |

### 3.4 健康检查
```bat
curl http://127.0.0.1:8890/health
# => {"status":"ok","loaded":true,"device":"cuda:0","cuda":true,...}
```

### 3.5 管理控制台

浏览器打开 **http://127.0.0.1:8890/admin**（页面每次请求读盘，改 `admin.html` 后刷新即可生效，无需重启服务）。

**能做什么**

1. **自定义参数发起请求** —— 文本、声音/属性描述、附加属性、语速 `speed`（0.25–4）、扩散步数
   `num_step`（1–100）、输出格式 `mp3`/`wav`/`flac`/`opus`/`pcm`、是否把附加属性拼入属性串。
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
| GET | `/admin/languages` | 模型 646 种语言表（`ids` + `names`），供 `language` 自动补全 |

---

### 3.6 系统托盘图标（通知区域）

服务启动后自动在**系统托盘**（任务栏右下角通知区域，时钟旁边）放置一个常驻图标，不必开浏览器也能找到服务：

| 操作 | 行为 |
|---|---|
| 左键单击 | 打开管理控制台（`http://127.0.0.1:8890/admin`） |
| 右键菜单 → 打开管理台 | 同上 |
| 右键菜单 → 复制 OpenAI 端点 | 复制 `http://127.0.0.1:8890/v1/audio/speech` 到剪贴板（填 DSHT Provider Endpoint 用） |
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

> 两种模式**互斥**：`voice` 命中克隆音色时，属性串不再参与生成（音色已由参考音频决定）。
> DSHT 侧无感——voice 字段填克隆音色名即可，走的还是同一个 OpenAI 端点。

### 3.8 OpenAI 端点请求参数全表（`POST /v1/audio/speech`）

除 OpenAI 标准字段外，OmniVoice 的 `OmniVoiceGenerationConfig` 全字段 + `language` / `duration` /
`normalize_text` 均可按请求覆盖。缺省值即 §3.3 环境变量（`/health` 的 `gen_defaults` 可查实时值）。

| 字段 | 类型 | 默认 | 范围 / 取值 | 建议 | 说明 |
|---|---|---|---|---|---|
| `input`（或 `text`） | str | — | 必填，≤4000 字 | — | 要合成的文本 |
| `voice` | str | `alloy` | 预设名 / 属性串 / 克隆音色名 | 按需 | 见 §3.7 与 §4 |
| `instructions` | str | 空 | OmniVoice 合法属性项 | 留空 | 需 `use_instructions` 或模型名含 gpt-4o 才拼入 |
| `response_format` | str | `mp3` | `mp3` `wav` `pcm` `flac` `opus` | `mp3` | 未知值返回 **400**（列出合法值），不静默回落 |
| `speed` | num | `1.0` | 0.25–4 | 0.8–1.2 | 语速倍率 |
| `num_step` | int | `32` | 1–100 | 16–48 | 扩散步数，越大越稳越慢，48+ 收益递减 |
| `language` | str | 空 | ISO 代码 / 英文名称 / `auto` / 空 | 按主要语言填 | 空=语言无关模式（中英混说用它）；`auto` 按文本猜（kana→ja、hangul→ko、汉字→zh、西里尔→ru、其余→en）；无法识别返回 400 |
| `duration` | num | 空 | 0.05–600 | 留空 | 固定输出秒数，**覆盖 speed**；空=按文本估计 |
| `normalize_text` | bool | `false` | — | 按需 | 数字/日期/货币转读法。**本机未装 WeTextProcessing/num2words，勾选返回 400**，不会静默跳过 |
| `guidance_scale` | num | `2.0` | 0–20 | 1.5–3.0 | 分类器自由引导强度，越大越严格贴合 instruct/文本，过高发紧不自然 |
| `t_shift` | num | `0.1` | 0–1 | 0.05–0.2 | 时间步偏移，越小越强调低 SNR 段（先定结构后细节） |
| `layer_penalty_factor` | num | `5.0` | 0–50 | 保持默认 | 层惩罚因子，促使靠前 codebook 层（语义）先解码；调小声学瑕疵 |
| `position_temperature` | num | `5.0` | 0.01–50 | 保持默认 | 位置选择温度 |
| `class_temperature` | num | `0.0` | 0–10 | 0 | token 采样温度（0=贪心确定性）；想多样可试 0.3–0.8，可能读错字 |
| `denoise` | bool | `true` | — | 保持开启 | 是否加 `<|denoise|>` 令牌 |
| `postprocess_output` | bool | `true` | — | 保持开启 | 剪 >0.5s 句中静音 / 淡入淡出 / 边距静音 |
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
