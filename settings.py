"""Declarative registry of every configurable environment variable.

This is the single source of truth for what the service exposes: the
variable name, its default, what it does, and whether changing it takes
effect immediately or needs a restart.

Nothing here is read at import time by the callers -- the point is that
`server.py` keeps using plain `os.environ.get(...)` exactly as before.
Instead the values configured in `runtime_config.json` are injected into
the environment at startup (see `apply_configured_env`), and the
environment in turn keeps priority over the file, so an operator can
always override a stored value for a single run.

The defaults are the same ones `server.py` falls back to; `check_settings`
verifies the two stay in step.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Groups, in the order they are shown in the console.
GROUPS: list[tuple[str, str]] = [
    ("service", "服务与网络"),
    ("model", "模型与推理"),
    ("generation", "生成参数默认值"),
    ("storage", "目录与轮转"),
    ("limits", "保留条数与配额"),
    ("logging", "日志"),
    ("instruct", "指令与属性"),
]

# name -> spec
#
#   default    the fallback used when nothing is configured
#   desc       what the variable does, shown in the console
#   group      one of the group keys above
#   kind       "str" | "int" | "float" | "bool" | "path"
#   live       True when a change takes effect without a restart
#   env        the environment variable that finally decides the value
#   min/max    accepted range for int/float
#   choices    the accepted values for a bool
SETTINGS: dict[str, dict[str, Any]] = {
    # --- service -----------------------------------------------------------
    "OMNI_TTS_PORT": {
        "default": "8890", "kind": "int", "group": "service", "live": False,
        "min": 1, "max": 65535,
        "desc": "监听端口。改后需重启。",
    },
    "OMNI_TTS_HOST": {
        "default": "127.0.0.1", "kind": "str", "group": "service", "live": False,
        "desc": "监听地址。改成 0.0.0.0 可供局域网访问（注意：接口无鉴权时别暴露到公网）。",
    },
    "OMNI_TTS_KEY": {
        "default": "", "kind": "str", "group": "service", "live": False,
        "desc": "OpenAI 端点鉴权密钥；留空 = 不校验（DSHT 侧留空即可）。",
    },
    "OMNI_TTS_TRAY": {
        "default": "1", "kind": "bool", "group": "service", "live": False,
        "desc": "是否显示系统托盘图标（仅 Windows 有效）。0 = 关闭。",
    },
    "OMNI_TTS_ADMIN_PAGE": {
        "default": "admin.html", "kind": "path", "group": "service", "live": False,
        "desc": "管理台页面文件位置。",
    },
    "OMNI_TTS_VOICES": {
        "default": "voices.json", "kind": "path", "group": "service", "live": False,
        "desc": "预设音色 → 文字造声属性的映射表。",
    },
    # --- model -------------------------------------------------------------
    "OMNIVOICE_MODEL": {
        "default": "models/OmniVoice", "kind": "path", "group": "model", "live": False,
        "desc": "模型目录。已由「模型管理」接管，运行时切换更方便。",
    },
    "OMNIVOICE_DEVICE": {
        "default": "cuda:0", "kind": "str", "group": "model", "live": False,
        "desc": "推理设备。cpu 极慢，仅用于排障。",
    },
    "OMNIVOICE_DTYPE": {
        "default": "bfloat16", "kind": "str", "group": "model", "live": False,
        "choices": ["bfloat16", "float16", "float32"],
        "desc": "权重精度。30 系显卡用 bfloat16。",
    },
    "OMNIVOICE_NUM_STEP": {
        "default": "32", "kind": "int", "group": "model", "live": False,
        "min": 1, "max": 100,
        "desc": "默认扩散步数。越大越稳越慢，48 以上收益递减。",
    },
    "OMNIVOICE_ASR_MODEL": {
        "default": "models/whisper-large-v3-turbo", "kind": "path", "group": "model",
        "live": False,
        "desc": "逐字稿留空时 Whisper ASR 的模型位置。本地目录或 HF repo id 均可。",
    },
    "OMNIVOICE_ASR_DEVICE": {
        "default": "", "kind": "str", "group": "model", "live": False,
        "desc": "ASR 的加载设备；留空 = 跟随主模型。显存不够时填 cpu（更慢但不占显存）。",
    },
    "OMNI_MODEL_CONFIG": {
        "default": "model_config.json", "kind": "path", "group": "model", "live": False,
        "desc": "模型注册表文件（运行时状态，不入库）。",
    },
    "OMNI_TTS_EMOTION_MAP_OFFICIAL": {
        "default": "1", "kind": "bool", "group": "instruct", "live": True,
        "desc": "情绪映射是否允许指向官方非语言标签（笑/叹息/惊呼/疑问…）与声音属性。"
                "默认 1：允许，映射目标由所选映射模板决定（官方模板/微调模板）。"
                "设 0 则屏蔽官方标签与声音属性，整张表只剩微调的语气/情绪标签"
                "（happy/sad/angry/whisper…）。",
    },
    "OMNI_TTS_MODEL_DIRS": {
        "default": "", "kind": "str", "group": "model", "live": True,
        "desc": "追加的模型扫描目录，多个用 ; 分隔。改后立即生效（下次扫描时）。",
    },
    # --- generation (hot) --------------------------------------------------
    "OMNIVOICE_GUIDANCE_SCALE": {
        "default": "2.0", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 20.0,
        "desc": "分类器自由引导强度。越大越贴合 instruct/文本，过高发紧不自然。",
    },
    "OMNIVOICE_T_SHIFT": {
        "default": "0.1", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 1.0,
        "desc": "时间步偏移。越小越强调低 SNR 段（先定结构后细节）。",
    },
    "OMNIVOICE_LAYER_PENALTY_FACTOR": {
        "default": "5.0", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 50.0,
        "desc": "层惩罚因子，促使靠前 codebook 层（语义）先解码。",
    },
    "OMNIVOICE_POSITION_TEMPERATURE": {
        "default": "5.0", "kind": "float", "group": "generation", "live": True,
        "min": 0.01, "max": 50.0,
        "desc": "位置选择温度。",
    },
    "OMNIVOICE_CLASS_TEMPERATURE": {
        "default": "0.0", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 10.0,
        "desc": "token 采样温度。0 = 贪心确定性；想多样可试 0.3–0.8，可能读错字。",
    },
    "OMNIVOICE_DENOISE": {
        "default": "1", "kind": "bool", "group": "generation", "live": True,
        "desc": "是否加 <|denoise|> 令牌。",
    },
    "OMNIVOICE_POSTPROCESS_OUTPUT": {
        "default": "0", "kind": "bool", "group": "generation", "live": True,
        "desc": "后处理：剪句中静音 / 淡入淡出 / 边距静音。默认关闭。",
    },
    "OMNIVOICE_PAD_DURATION": {
        "default": "0.1", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 2.0,
        "desc": "输出两端静音秒数。0 = 不加（句首辅音可能被切）。",
    },
    "OMNIVOICE_FADE_DURATION": {
        "default": "0.1", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 2.0,
        "desc": "头尾淡入淡出秒数，防爆音。",
    },
    "OMNIVOICE_AUDIO_CHUNK_DURATION": {
        "default": "15.0", "kind": "float", "group": "generation", "live": True,
        "min": 1.0, "max": 60.0,
        "desc": "长文本分块时长（秒），块间交叉淡化。",
    },
    "OMNIVOICE_AUDIO_CHUNK_THRESHOLD": {
        "default": "30.0", "kind": "float", "group": "generation", "live": True,
        "min": 1.0, "max": 300.0,
        "desc": "预估音频超此秒数才分块。",
    },
    # --- 短句长度补偿（见 server._auto_duration）----------------------------
    "OMNI_TTS_AUTO_DURATION": {
        "default": "1", "kind": "bool", "group": "generation", "live": True,
        "desc": "短句自动补时长。**默认开**：模型会提前约 1.4 秒收尾（实测），"
                "让短句只剩一点点声音；开启后按文本估算长度并显式传给模型，"
                "长句（自身预算已够）不受影响。请求里显式给了 duration 时不介入。",
    },
    "OMNI_TTS_AUTO_DURATION_RATE": {
        "default": "65", "kind": "float", "group": "generation", "live": True,
        "min": 30.0, "max": 150.0,
        "desc": "目标语速：每个「字符权重」折算的毫秒数（汉字权重 3.0 → 约 195 ms/字）。"
                "**越大越慢**：65 ≈ 中文 5 字/秒；调小变快、调大变慢。",
    },
    "OMNI_TTS_AUTO_DURATION_LEAD": {
        "default": "1.4", "kind": "float", "group": "generation", "live": True,
        "min": 0.0, "max": 5.0,
        "desc": "模型提前收尾的秒数补偿，仅对克隆音色生效（文字造声路径不提前收尾）。"
                "觉得补得过头就调小，觉得仍被截断就调大。",
    },
    # --- storage -----------------------------------------------------------
    "OMNI_TTS_HISTORY_DIR": {
        "default": "history", "kind": "path", "group": "storage", "live": False,
        "desc": "管理台请求归档目录。",
    },
    "OMNI_TTS_CLONE_DIR": {
        "default": "voices", "kind": "path", "group": "storage", "live": False,
        "desc": "声音克隆音色库目录（.pt prompt + 原始参考音频）。",
    },
    "OMNI_TTS_RECORD_DIR": {
        "default": "records", "kind": "path", "group": "storage", "live": False,
        "desc": "OpenAI 端点记录归档目录。",
    },
    "OMNI_TTS_CACHE_DIR": {
        "default": "cache", "kind": "path", "group": "storage", "live": False,
        "desc": "响应缓存目录（按 key 分桶）。",
    },
    # --- limits ------------------------------------------------------------
    "OMNI_API_RECORD": {
        "default": "1", "kind": "bool", "group": "limits", "live": False,
        "desc": "是否归档 OpenAI 端点请求（含失败与缓存命中）。",
    },
    "OMNI_TTS_LOG_API": {
        "default": "1", "kind": "bool", "group": "limits", "live": False,
        "desc": "兼容开关：0 可关闭 OpenAI 端点归档。",
    },
    "OMNI_RECORD_MAX_ENTRIES": {
        "default": "500", "kind": "int", "group": "limits", "live": False,
        "min": 0, "max": 100000,
        "desc": "端点记录轮转：最多保留条数（0 = 不限）。",
    },
    "OMNI_RECORD_MAX_MB": {
        "default": "2048", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 1000000,
        "desc": "端点记录轮转：最大体积 MB（0 = 不限）。",
    },
    "OMNI_RECORD_MAX_AGE_DAYS": {
        "default": "30", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 3650,
        "desc": "端点记录轮转：超龄天数即删（0 = 不限）。",
    },
    "OMNI_CACHE": {
        "default": "1", "kind": "bool", "group": "limits", "live": False,
        "desc": "响应缓存开关：同参数请求直接返回已生成音频。",
    },
    "OMNI_CACHE_MAX_ENTRIES": {
        "default": "2000", "kind": "int", "group": "limits", "live": False,
        "min": 0, "max": 1000000,
        "desc": "缓存轮转：最多条数（0 = 不限）。",
    },
    "OMNI_CACHE_MAX_MB": {
        "default": "2048", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 1000000,
        "desc": "缓存轮转：最大体积 MB（0 = 不限）。",
    },
    "OMNI_CACHE_MAX_AGE_DAYS": {
        "default": "30", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 3650,
        "desc": "缓存轮转：超龄天数即删（0 = 不限）。",
    },
    "OMNI_HISTORY_MAX_ENTRIES": {
        "default": "500", "kind": "int", "group": "limits", "live": False,
        "min": 0, "max": 100000,
        "desc": "管理台历史归档轮转：最多条数（0 = 不限）。",
    },
    "OMNI_HISTORY_MAX_MB": {
        "default": "2048", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 1000000,
        "desc": "管理台历史归档轮转：最大体积 MB（0 = 不限）。",
    },
    "OMNI_HISTORY_MAX_AGE_DAYS": {
        "default": "30", "kind": "float", "group": "limits", "live": False,
        "min": 0, "max": 3650,
        "desc": "管理台历史归档轮转：超龄天数即删（0 = 不限）。",
    },
    "OMNI_API_LOG_MAX": {
        "default": "300", "kind": "int", "group": "limits", "live": False,
        "min": 0, "max": 100000,
        "desc": "请求日志环形缓冲条数（内存，不落盘）。",
    },
    # --- logging -----------------------------------------------------------
    "OMNI_LOG_FILE": {
        "default": "server.log", "kind": "path", "group": "logging", "live": False,
        "desc": "运行日志文件位置（由 RotatingFileHandler 托管）。",
    },
    "OMNI_LOG_MAX_MB": {
        "default": "10", "kind": "float", "group": "logging", "live": False,
        "min": 0, "max": 10000,
        "desc": "日志轮转：单文件最大 MB。",
    },
    "OMNI_LOG_BACKUPS": {
        "default": "5", "kind": "int", "group": "logging", "live": False,
        "min": 0, "max": 1000,
        "desc": "日志轮转：保留备份数。",
    },
}

# Variables the service sets for itself (thread limits, the HF cache). They are
# implementation details rather than knobs, so they are not offered for
# configuration.
INTERNAL_ENV = {
    "HF_ENDPOINT", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}

CONFIG_FILENAME = "runtime_config.json"

# The variables this process actually took from the stored configuration.
# A variable that is in the environment but not in here was set by whoever
# launched the process, so it wins and the stored value is inert. Keeping the
# distinction matters: the console shows where each value came from, and a
# stored value that is being overridden must say so.
_APPLIED: set[str] = set()


def mark_applied(name: str) -> None:
    """Record that a stored value is the one in effect for this process."""
    _APPLIED.add(name)


def unmark_applied(name: str) -> None:
    """Forget a variable: its stored value is gone, so the default applies."""
    _APPLIED.discard(name)


def applied_names() -> set[str]:
    return set(_APPLIED)


def config_path() -> Path:
    return Path(os.environ.get("OMNI_RUNTIME_CONFIG", HERE / CONFIG_FILENAME))


def load_stored() -> dict[str, str]:
    """The values stored in runtime_config.json (empty when absent)."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float, bool))}


def save_stored(values: dict[str, str]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def apply_configured_env() -> dict[str, str]:
    """Inject the stored values into the environment at startup.

    A value already present in the real environment wins: an operator
    setting a variable for a single run must not be overridden by a stored
    value. Returns the values that were actually injected.
    """
    stored = load_stored()
    injected: dict[str, str] = {}
    for name, value in stored.items():
        if name not in SETTINGS:
            continue
        if name in os.environ:
            continue
        os.environ[name] = value
        mark_applied(name)
        injected[name] = value
    return injected


def value_source(name: str) -> str:
    """Where the current value comes from: env, file, or the built-in default.

    "file" means the stored configuration is the value in effect. "env" means
    the process was started with the variable set, so a stored value for it
    would be inert.
    """
    if name in _APPLIED:
        return "file"
    if name in os.environ:
        return "env"
    if name in load_stored():
        return "file"
    return "default"


def current_value(name: str) -> str:
    spec = SETTINGS[name]
    raw = os.environ.get(name)
    if raw is None:
        raw = load_stored().get(name, spec["default"])
    return raw


def validate(name: str, raw: str) -> str:
    """Check a value against the spec and return it normalised, or raise ValueError."""
    spec = SETTINGS[name]
    kind = spec.get("kind", "str")
    text = (raw or "").strip()
    if kind == "bool":
        low = text.lower()
        if low in ("1", "true", "yes", "on"):
            return "1"
        if low in ("0", "false", "no", "off", ""):
            return "0"
        raise ValueError(f"{name}: 布尔值只能是 1/0（true/false 也可）")
    if kind == "int":
        try:
            number = int(float(text))
        except ValueError:
            raise ValueError(f"{name}: 需要一个整数，收到 {raw!r}") from None
        low, high = spec.get("min"), spec.get("max")
        if low is not None and number < low:
            raise ValueError(f"{name}: 不能小于 {low}")
        if high is not None and number > high:
            raise ValueError(f"{name}: 不能大于 {high}")
        return str(number)
    if kind == "float":
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"{name}: 需要一个数字，收到 {raw!r}") from None
        low, high = spec.get("min"), spec.get("max")
        if low is not None and number < low:
            raise ValueError(f"{name}: 不能小于 {low}")
        if high is not None and number > high:
            raise ValueError(f"{name}: 不能大于 {high}")
        return repr(number) if number != int(number) else str(int(number))
    choices = spec.get("choices")
    if choices and text not in choices:
        raise ValueError(f"{name}: 只能是 {' / '.join(choices)}")
    return text


def describe() -> list[dict[str, Any]]:
    """The full picture for the console: every setting and where it stands."""
    stored = load_stored()
    out: list[dict[str, Any]] = []
    for name, spec in SETTINGS.items():
        out.append({
            "name": name,
            "default": spec["default"],
            "value": current_value(name),
            "source": value_source(name),
            "stored": stored.get(name),
            "group": spec.get("group", "service"),
            "kind": spec.get("kind", "str"),
            "desc": spec.get("desc", ""),
            "live": bool(spec.get("live")),
            "min": spec.get("min"),
            "max": spec.get("max"),
            "choices": spec.get("choices"),
        })
    return out


def check_against_source(source: str) -> list[str]:
    """Find variables the code reads but the registry does not list.

    Used by the consistency check: the registry is only trustworthy if it
    covers every knob the service actually reads.
    """
    import re

    found = set(re.findall(r'os\.environ\.get\(\s*["\'](OMNI[A-Z_0-9]*)["\']', source))
    found |= set(re.findall(r'os\.environ\.get\(\s*["\'](OMNIVOICE_[A-Z_0-9]*)["\']', source))
    return sorted(found - set(SETTINGS) - INTERNAL_ENV)
