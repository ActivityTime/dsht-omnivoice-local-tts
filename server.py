#!/usr/bin/env python
"""OmniVoice OpenAI-compatible TTS server.

Exposes `POST /v1/audio/speech` in the shape DSH-Tavern's "OpenAI compatible"
speech provider expects:

    POST {baseUrl}/audio/speech
    Authorization: Bearer <any>
    {"model": "...", "input": "...", "voice": "alloy", "response_format": "mp3"}

Running OmniVoice in "voice design" mode by default: each incoming `voice`
name is mapped to a set of speaker attributes (gender / age / pitch / style /
accent / dialect) via voices.json. Anything not present in that map is passed
through verbatim as the design instruction, so custom voice labels work too.

Every path is resolved relative to this file or from the environment; nothing
is hard-coded to a specific machine.
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# --- Windows DLL resolution fix ---------------------------------------------
# torch.ops.load_library() uses loader flags that skip the executable dir, and a
# tamper-protected System32 locks the api-ms/OpenMP forwarders. Register the
# package DLL directories explicitly so torch/torchaudio load on import.
_VENV = Path(sys.executable).resolve().parent.parent
for _d in [
    _VENV / "Lib" / "site-packages" / "torchaudio" / "lib",
    _VENV / "Lib" / "site-packages" / "torch" / "lib",
    _VENV / "Scripts",
]:
    try:
        os.add_dll_directory(str(_d))
    except Exception:
        pass

# Default to a HF mirror so model weights download in this region.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Disable the Xet/CAS storage backend: some mirrors reject its reconstruction
# endpoint (cas-server.xethub.hf.co) with 401, and plain HTTPS download works.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_XET_DISABLED"] = "1"

# Memory footprint: the host has limited commit memory, so pin compute threads
# to 1 and let from_pretrained stream weights straight to the device instead of
# buffering them in CPU RAM. Mirrors the configuration that successfully loads
# the model under tight paging-file headroom (this is what loadtest.py proved).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------------------------------------------------------------------

import numpy as np

HERE = Path(__file__).resolve().parent

# The bundled interpreter puts its own zip on sys.path[0] rather than the
# script's directory, so a sibling module is not importable by default.
# Put the project root on the path explicitly.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Configured values are injected into the environment before anything reads
# it, so the code below keeps using plain os.environ.get(...) unchanged. A
# variable already present in the real environment wins: an operator can
# always override a stored value for a single run.
try:
    import settings as _settings  # noqa: E402
    import emotion_map as _emotion_map  # noqa: E402

    _INJECTED = _settings.apply_configured_env()
    _EMOTION_TABLE = _emotion_map.ensure_map()
    if _INJECTED:
        # logging is not configured yet this early, so this line goes to
        # stderr only; the same list is repeated in the startup log below.
        print(
            f"[omnivoice] applied {len(_INJECTED)} configured env values: "
            + ", ".join(sorted(_INJECTED)),
            flush=True,
        )
except Exception as _error:  # noqa: BLE001
    _settings = None
    _emotion_map = None
    _INJECTED = {}
    _EMOTION_TABLE = {}
    print(f"[omnivoice] settings registry unavailable: {_error}", flush=True)

# Hugging Face cache lives next to this file unless the caller overrides it.
os.environ.setdefault("HF_HOME", str(HERE / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HERE / "hf" / "hub"))

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402
import uvicorn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("omnivoice-openai")

# --- log rotation --------------------------------------------------------------
# server.log used to depend on a shell redirect (>> server.log 2>&1): it grew
# without bound and vanished entirely when the server was started any other way
# (start.cmd, double click). The file is now owned by a RotatingFileHandler, so
# every launch path writes to it and rotation is enforced by size.
LOG_FILE = Path(os.environ.get("OMNI_LOG_FILE", HERE / "server.log"))
LOG_MAX_BYTES = int(float(os.environ.get("OMNI_LOG_MAX_MB", "10")) * 1024 * 1024)
LOG_BACKUPS = int(os.environ.get("OMNI_LOG_BACKUPS", "5"))


def setup_logging() -> None:
    """Install console + size-rotated file logging on the root logger.

    Called from main() right before uvicorn.run(..., log_config=None): with
    uvicorn's own logging config skipped, its "uvicorn"/"uvicorn.access"
    records propagate to the root logger configured here, so the access log
    (method/path/status per request) lands in the rotating file as well.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    try:
        rotating = RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)
    except OSError as error:
        sys.stderr.write(f"[logging] file handler disabled ({LOG_FILE}): {error}\n")

# Default to the locally-snapshotted copy (downloaded by download_model.py) so we
# never depend on the HF Xet/CAS backend at inference time. Override with the
# OMNIVOICE_MODEL env var if you ever want to pull from the hub again.
MODEL_ID = os.environ.get("OMNIVOICE_MODEL", str(HERE / "models" / "OmniVoice"))
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE_NAME = os.environ.get("OMNIVOICE_DTYPE", "bfloat16").lower()
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))
# Where the optional Whisper ASR (auto-transcription when a clone is created
# without ref_text) lives. Defaults to the locally-snapshotted copy next to
# this file (same convention as MODEL_ID) so the first ASR request never needs
# the network; override with OMNIVOICE_ASR_MODEL to pull from the hub again.
# Its device defaults to the main model's device; set OMNIVOICE_ASR_DEVICE=cpu
# when the GPU has no headroom for the extra ~1.6 GB.
ASR_MODEL_ID = os.environ.get(
    "OMNIVOICE_ASR_MODEL", str(HERE / "models" / "whisper-large-v3-turbo")
)
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE", "").strip() or None
API_KEY = os.environ.get("OMNI_TTS_KEY", "")  # empty = accept any key
VOICES_FILE = Path(os.environ.get("OMNI_TTS_VOICES", HERE / "voices.json"))
# The `instructions` field is always parsed, the same way as the voice field.
# It used to sit behind a switch, left over from a time when an emotion phrase
# here had to be ignored wholesale; the parser now keeps only whitelisted
# attribute items and routes emotion phrases through the mapping layer, so
# there is nothing left for a switch to protect and nothing to turn off --
# turning it off silently dropped every emotion DSH-Tavern sent.

# --- request history (admin console) ----------------------------------------
# Each request made through the admin console is persisted as a directory
# holding meta.json plus the decoded audio, so runs can be replayed later.
HISTORY_DIR = Path(os.environ.get("OMNI_TTS_HISTORY_DIR", HERE / "history"))
ADMIN_FILE = Path(os.environ.get("OMNI_TTS_ADMIN_PAGE", HERE / "admin.html"))
# Rotation for the admin-console archive, same three-threshold scheme as the
# API records below. Disabled per axis when the limit is <= 0.
HISTORY_MAX_ENTRIES = int(os.environ.get("OMNI_HISTORY_MAX_ENTRIES", "500"))
HISTORY_MAX_BYTES = int(float(os.environ.get("OMNI_HISTORY_MAX_MB", "2048")) * 1024 * 1024)
HISTORY_MAX_AGE_DAYS = float(os.environ.get("OMNI_HISTORY_MAX_AGE_DAYS", "30"))

# --- cloned voices (voice clone prompts) ------------------------------------
# create_voice_clone_prompt() is expensive, so its result is written to disk
# with VoiceClonePrompt.save() and memoised in memory. Each voice is three
# files: <name>.pt (official prompt), <name>.json (our metadata), and
# <name>.ref<ext> (the original upload, kept for playback).
CLONE_DIR = Path(os.environ.get("OMNI_TTS_CLONE_DIR", HERE / "voices"))
_CLONE_SUFFIX = ".pt"
_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
# Rejects path separators and Windows-reserved characters.
_NAME_RE = re.compile(r"^[^\\/:*?\"<>|\r\n\t]{1,64}$")
_clone_cache: dict[str, Any] = {}
# OpenAI-endpoint traffic is archived by default. This supersedes the old
# OMNI_TTS_LOG_API=0 default-off behaviour: rotation (below) bounds the disk
# usage, so the "DSH Tavern fires one request per chat bubble" worry is gone.
# Set OMNI_API_RECORD=0 (or the legacy OMNI_TTS_LOG_API=0) to switch it off.
API_RECORD = (
    os.environ.get("OMNI_API_RECORD", "1") == "1"
    and os.environ.get("OMNI_TTS_LOG_API", "1") == "1"
)
RECORD_DIR = Path(os.environ.get("OMNI_TTS_RECORD_DIR", HERE / "records"))
# Rotation for the endpoint archive: oldest entries are pruned once any limit
# trips. A limit <= 0 disables that particular axis.
RECORD_MAX_ENTRIES = int(os.environ.get("OMNI_RECORD_MAX_ENTRIES", "500"))
RECORD_MAX_BYTES = int(float(os.environ.get("OMNI_RECORD_MAX_MB", "2048")) * 1024 * 1024)
RECORD_MAX_AGE_DAYS = float(os.environ.get("OMNI_RECORD_MAX_AGE_DAYS", "30"))
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# --- response cache ------------------------------------------------------------
# Requests whose audio-affecting surface is byte-identical (text + voice +
# format + the fully resolved generation config) are served from disk without
# touching the model: DSH Tavern re-sends the same bubble on re-renders, and
# the console re-runs while tuning knobs. Cached audio is copied into the
# record archive on a hit, so the audit trail stays complete.
CACHE_DIR = Path(os.environ.get("OMNI_TTS_CACHE_DIR", HERE / "cache"))
CACHE_ENABLED = os.environ.get("OMNI_CACHE", "1") == "1"
CACHE_MAX_ENTRIES = int(os.environ.get("OMNI_CACHE_MAX_ENTRIES", "2000"))
CACHE_MAX_BYTES = int(float(os.environ.get("OMNI_CACHE_MAX_MB", "2048")) * 1024 * 1024)
CACHE_MAX_AGE_DAYS = float(os.environ.get("OMNI_CACHE_MAX_AGE_DAYS", "30"))


def _model_scan_roots() -> list[Path]:
    """Directories that may hold loadable model folders.

    Derived at runtime from the install location — never a literal drive
    letter — so the project stays portable across machines. Override with
    OMNI_TTS_MODEL_DIRS (os.pathsep-separated) to add your own roots.
    """
    roots: list[Path] = [HERE / "models", HERE / "finetunes"]
    # sibling directories of the install dir that look like model collections
    try:
        for sib in sorted(HERE.parent.iterdir()):
            if sib.is_dir() and ("finetune" in sib.name.lower() or "model" in sib.name.lower()):
                roots.append(sib)
    except OSError:
        pass
    for extra in os.environ.get("OMNI_TTS_MODEL_DIRS", "").split(os.pathsep):
        extra = extra.strip()
        if extra:
            roots.append(Path(extra))
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

# In-memory ring of the most recent endpoint requests (failures and cache hits
# included). RAM only — a live view for the console, not an audit trail.
API_LOG_MAX = int(os.environ.get("OMNI_API_LOG_MAX", "300"))
_api_log_ring: deque[dict[str, Any]] = deque(maxlen=max(1, API_LOG_MAX))
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0, "puts": 0, "pruned": 0}

SAMPLE_RATE = 24000
MAX_TEXT = 4000

# --- generation knobs (OmniVoiceGenerationConfig) ----------------------------
# The model exposes a full generation-config surface; every field is tunable
# per request and these are the process-wide defaults (env-overridable).
# `language` / `duration` / `normalize_text` are opt-in only (no default) —
# see _parse_gen_params.
GEN_DEFAULTS: dict[str, Any] = {
    "guidance_scale": float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0")),
    "t_shift": float(os.environ.get("OMNIVOICE_T_SHIFT", "0.1")),
    "layer_penalty_factor": float(os.environ.get("OMNIVOICE_LAYER_PENALTY_FACTOR", "5.0")),
    "position_temperature": float(os.environ.get("OMNIVOICE_POSITION_TEMPERATURE", "5.0")),
    "class_temperature": float(os.environ.get("OMNIVOICE_CLASS_TEMPERATURE", "0.0")),
    "denoise": os.environ.get("OMNIVOICE_DENOISE", "1") == "1",
    "postprocess_output": os.environ.get("OMNIVOICE_POSTPROCESS_OUTPUT", "0") == "1",
    "pad_duration": float(os.environ.get("OMNIVOICE_PAD_DURATION", "0.1")),
    "fade_duration": float(os.environ.get("OMNIVOICE_FADE_DURATION", "0.1")),
    "audio_chunk_duration": float(os.environ.get("OMNIVOICE_AUDIO_CHUNK_DURATION", "15.0")),
    "audio_chunk_threshold": float(os.environ.get("OMNIVOICE_AUDIO_CHUNK_THRESHOLD", "30.0")),
}

# Per-request numeric knobs: name -> (min, max). Out-of-range is a 400, never a
# silent clamp — the caller should learn the boundary, not get surprised.
_GEN_NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "speed": (0.25, 4.0),
    "num_step": (1, 100),
    "guidance_scale": (0.0, 20.0),
    "t_shift": (0.0, 1.0),
    "layer_penalty_factor": (0.0, 50.0),
    "position_temperature": (0.01, 50.0),
    "class_temperature": (0.0, 10.0),
    "pad_duration": (0.0, 2.0),
    "fade_duration": (0.0, 2.0),
    "audio_chunk_duration": (1.0, 60.0),
    "audio_chunk_threshold": (1.0, 300.0),
    # Fixed output seconds; None (default) lets the model estimate from text.
    "duration": (0.05, 600.0),
}
_GEN_INT_KEYS = ("num_step",)
_GEN_BOOL_KEYS = ("denoise", "postprocess_output")
_TN_OK: bool | None = None  # cached probe for the optional normalisation stack


def _normalize_text_available() -> bool:
    """Whether number→words normalisation can actually run on this host.

    WeTextProcessing covers zh/en; num2words alone still covers bare integers
    in other languages. Neither is installed here, so `normalize_text` would
    be a silent no-op — the API refuses instead of pretending.
    """
    global _TN_OK
    if _TN_OK is None:
        try:
            import tn.chinese.normalizer  # noqa: F401, PLC0415

            _TN_OK = True
        except Exception:  # noqa: BLE001
            try:
                import num2words  # noqa: F401, PLC0415

                _TN_OK = True
            except Exception:  # noqa: BLE001
                _TN_OK = False
    return _TN_OK


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _detect_language(text: str) -> str:
    """Coarse single-language guess for `language=auto` (ISO 639-3 codes).

    Mixed-script text is a guess by nature — the model's language-agnostic
    mode (empty language) handles mixed text without one.
    """
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"  # kana
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"  # hangul
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    return "en"


def _resolve_language_param(value: Any, text: str) -> str | None:
    """Validate the `language` field against the model's own 646-entry table."""
    raw = str(value or "").strip()
    if not raw or raw.lower() == "none":
        return None
    if raw.lower() == "auto":
        return _detect_language(text) if text else None
    try:
        from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return raw  # table unavailable: let the model judge it
    if raw in LANG_IDS:
        return raw
    mapped = LANG_NAME_TO_ID.get(raw.lower())
    if mapped:
        return mapped
    raise HTTPException(
        status_code=400,
        detail=f"无法识别的语言「{raw}」：填 ISO 代码（如 zh / en / ja）、完整名称"
        "（如 Chinese / Japanese / Korean）、auto（按文本猜测），或留空（语言无关模式）",
    )


def _parse_gen_params(body: dict[str, Any], text: str = "") -> dict[str, Any]:
    """Resolve the complete generation parameter set from a request body.

    Covers every field of OmniVoiceGenerationConfig plus `language`,
    `duration` and `normalize_text`. Missing keys fall back to GEN_DEFAULTS
    (env-overridable); invalid values raise 400 with the exact boundary.
    """
    gen: dict[str, Any] = {}
    for key, (low, high) in _GEN_NUMERIC_LIMITS.items():
        raw = body.get(key)
        if raw is None or raw == "":
            if key in GEN_DEFAULTS:
                gen[key] = GEN_DEFAULTS[key]
            elif key == "num_step":
                gen[key] = NUM_STEP
            elif key == "speed":
                gen[key] = 1.0
            else:
                gen[key] = None  # duration: model-estimated unless asked
            continue
        try:
            value = int(raw) if key in _GEN_INT_KEYS else float(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"参数 {key} 必须是数字，收到：{raw!r}")
        if not low <= value <= high:
            raise HTTPException(
                status_code=400, detail=f"参数 {key} 超出允许范围 [{low}, {high}]：{value}"
            )
        gen[key] = value
    for key in _GEN_BOOL_KEYS:
        gen[key] = _as_bool(body[key]) if key in body else GEN_DEFAULTS[key]
    if _as_bool(body.get("normalize_text")):
        if not _normalize_text_available():
            raise HTTPException(
                status_code=400,
                detail="normalize_text 需要 WeTextProcessing（或 num2words），当前环境未安装；"
                "数字朗读请自行在文本里写好读法（如「二零二六年」）",
            )
        gen["normalize_text"] = True
    else:
        gen["normalize_text"] = False
    gen["language"] = _resolve_language_param(body.get("language"), text)
    return gen


def _gen_overrides(gen: dict[str, Any]) -> dict[str, Any]:
    """Subset of a resolved `gen` dict that deviates from the defaults.

    History records store this instead of the full resolved set: the console
    then renders only what the caller actually changed, and "复用参数" can
    restore exactly those fields.
    """
    out: dict[str, Any] = {}
    for key, value in gen.items():
        if key == "speed":
            if value != 1.0:
                out[key] = value
        elif key == "num_step":
            if value != NUM_STEP:
                out[key] = value
        elif key in GEN_DEFAULTS:
            if value != GEN_DEFAULTS[key]:
                out[key] = value
        elif value not in (None, False):  # language / duration / normalize_text
            out[key] = value
    return out

# OpenAI's published voice names, used as the fallback map when voices.json is
# missing. Values are OmniVoice design attributes (see docs/voice-design.md).
DEFAULT_VOICES: dict[str, str] = {
    "alloy": "female, young adult, moderate pitch",
    "ash": "male, young adult, low pitch",
    "ballad": "male, middle-aged, low pitch",
    "coral": "female, young adult, high pitch",
    "echo": "male, young adult, moderate pitch",
    "fable": "female, middle-aged, moderate pitch, british accent",
    "nova": "female, young adult, high pitch",
    "onyx": "male, elderly, very low pitch",
    "sage": "female, middle-aged, moderate pitch",
    "shimmer": "female, young adult, high pitch, whisper",
    "verse": "male, middle-aged, moderate pitch",
}

_state: dict[str, Any] = {"model": None, "lock": threading.Lock(), "model_id": None}


# --- model registry + persistent selection ----------------------------------
# Models (base checkpoints, full finetunes, or LoRA adapters) are registered in
# model_config.json. The `active` entry is loaded at startup and can be swapped
# at runtime from the admin console; the choice is persisted so it survives
# restarts. A "lora" entry is loaded by instantiating its `base` checkpoint and
# then merging the adapter weights into model.llm in-place (PEFT's auto-merge is
# skipped — it rejects Windows paths and OmniVoice's bare Qwen3Model lacks the
# generation mixin; the manual merge is the path proven in scratch/test_lora_zh.py).

CONFIG_PATH = Path(os.environ.get("OMNI_MODEL_CONFIG", HERE / "model_config.json"))


def _store_path(path: Path) -> str:
    """Serialise a model path for the registry, relative to the install root.

    Keeps model_config.json portable: anything under the project directory is
    stored as a relative path, so the registry survives moving the whole
    install. Paths outside it (e.g. a weights dir on another volume) stay
    absolute because there is nothing to be relative to.

    Note: `absolute()` is used rather than `resolve()` on purpose — resolve()
    follows directory junctions, which would turn a project-local junction
    (e.g. finetunes/ -> another volume) back into an absolute foreign path and
    defeat the relativisation.
    """
    p = Path(os.path.abspath(path))
    try:
        return str(p.relative_to(HERE))
    except Exception:  # noqa: BLE001
        return str(p)


def _resolve_model_ref(ref: str | None) -> str:
    """Resolve a stored model reference to something the library accepts.

    A reference may be a local directory (absolute, or relative to the install
    root) or a Hugging Face repo id ("org/name"). A repo id must be passed
    through untouched, so the two are told apart by shape:
      - absolute path  -> path
      - relative path that exists under the install root -> path
      - anything else containing a separator -> repo id (org/name)
      - plain name -> repo id
    """
    raw = str(ref or "").strip()
    if not raw:
        return str(HERE / "models" / "whisper-large-v3-turbo")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded)
    if p.is_absolute():
        return str(p)
    if not os.sep in raw and "/" not in raw:
        return raw  # plain name -> repo id
    candidate = HERE / p
    if candidate.exists():
        return str(candidate)
    return raw  # "org/name" style -> repo id


def _default_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "base",
            "name": "OmniVoice 官方基座",
            "path": "models/OmniVoice",  # relative to the install root
            "type": "checkpoint",
        }
    ]


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("models"):
                data.setdefault("active", data["models"][0]["id"])
                data.setdefault("device", DEVICE)
                data.setdefault("dtype", DTYPE_NAME)
                data.setdefault("num_step", NUM_STEP)
                data.setdefault("asr_model", _store_path(Path(ASR_MODEL_ID)))
                data.setdefault("asr_device", ASR_DEVICE)
                return data
        except Exception as error:  # noqa: BLE001
            log.warning("model_config.json 不可读 (%s)，回退默认", error)
    cfg = {
        "active": "base",
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "num_step": NUM_STEP,
        "asr_model": _store_path(Path(ASR_MODEL_ID)),
        "asr_device": ASR_DEVICE,
        "models": _default_registry(),
    }
    save_config(cfg)
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)
    except OSError as error:
        log.warning("保存 model_config.json 失败: %s", error)


MODEL_CONFIG = load_config()


def _find_model(cfg: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for entry in cfg.get("models", []):
        if entry.get("id") == model_id:
            return entry
    return None


def _first_checkpoint_id(cfg: dict[str, Any]) -> str | None:
    for entry in cfg.get("models", []):
        if entry.get("type") != "lora":
            return entry["id"]
    return cfg.get("models", [{}])[0].get("id")


def _instantiate(path: str, device: str, dtype, asr_model: str, asr_device: str | None):
    import torch
    from omnivoice import OmniVoice

    # set_num_interop_threads may only be called once per process, before any
    # parallel work has started. A runtime model swap calls this again, so the
    # second call raises and aborts the load — hence the one-shot guard.
    torch.set_num_threads(1)
    if not _state.get("interop_set"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _state["interop_set"] = True
    extra = {"asr_model_name": asr_model}
    if asr_device:
        extra["asr_device"] = asr_device
    return OmniVoice.from_pretrained(
        path, device_map=device, dtype=dtype, low_cpu_mem_usage=True, **extra
    )


def _merge_lora(model, lora_dir: str) -> None:
    """Manually merge a PEFT LoRA into `model.llm` weights in-place.

    PEFT's `PeftModel.from_pretrained(...).merge_and_unload()` is skipped: it
    treats a Windows drive-letter path as an HF repo id (validate_repo_id
    rejects it), and OmniVoice's `model.llm` is a bare Qwen3Model with no
    generation mixin, so PEFT init also fails with AttributeError. Instead we
    read lora_A/lora_B straight from the safetensors and add the delta in-place.
    r/alpha/use_rslora come from adapter_config.json so the scaling is exact.
    """
    import torch  # noqa: F401
    import safetensors.torch as st  # type: ignore

    cfg_path = Path(lora_dir) / "adapter_config.json"
    r, alpha, rslora = 16, 32, False
    if cfg_path.exists():
        try:
            ac = json.loads(cfg_path.read_text(encoding="utf-8"))
            r = int(ac.get("r", r))
            alpha = int(ac.get("lora_alpha", alpha))
            rslora = bool(ac.get("use_rslora", rslora))
        except Exception:  # noqa: BLE001
            pass
    scaling = (alpha / r) if not rslora else (alpha / (r ** 0.5))

    st_path = None
    for cand in ("model.safetensors", "adapter_model.safetensors"):
        p = Path(lora_dir) / cand
        if p.exists():
            st_path = p
            break
    if st_path is None:
        raise RuntimeError(f"LoRA 权重未找到：{lora_dir}")
    tensors = st.load_file(str(st_path))
    device = next(model.llm.parameters()).device
    count = 0
    with torch.no_grad():
        for key in list(tensors.keys()):
            if not key.endswith("lora_A.default.weight"):
                continue
            prefix = key[: -len("lora_A.default.weight")]
            bkey = prefix + "lora_B.default.weight"
            if bkey not in tensors:
                continue
            A = tensors[key].float()
            B = tensors[bkey].float()
            delta = (B @ A) * scaling
            modpath = prefix
            if modpath.startswith("llm."):
                modpath = modpath[len("llm."):]
            # the captured prefix still carries the trailing "." that separated
            # it from "lora_A.default.weight" — drop it before resolving.
            modpath = modpath.rstrip(".")
            # PEFT saves adapter keys as "<lora_root>.base_model.model.<module>".
            # OmniVoice's model.llm is a bare Qwen3Model, whose direct children
            # are e.g. "layers.0.mlp.down_proj" — so the whole
            # "base_model.model." hop has to go, not just "base_model.".
            for hop in ("base_model.model.", "base_model.", "model."):
                if modpath.startswith(hop):
                    modpath = modpath[len(hop):]
                    break
            mod = model.llm.get_submodule(modpath)
            mod.weight.data += delta.to(mod.weight.device, mod.weight.dtype)
            count += 1
    log.info("merged %d LoRA modules (scaling=%.4f) from %s", count, scaling, lora_dir)


def _load_entry(entry: dict[str, Any], cfg: dict[str, Any]):
    if entry.get("type") == "lora":
        base_id = entry.get("base") or _first_checkpoint_id(cfg)
        base_entry = _find_model(cfg, base_id) or _default_registry()[0]
        model = _instantiate(
            str(_entry_path(base_entry)), cfg["device"], resolve_dtype(),
            _resolve_model_ref(cfg.get("asr_model")), cfg["asr_device"],
        )
        _merge_lora(model, str(_entry_path(entry)))
        return model
    return _instantiate(
        str(_entry_path(entry)), cfg["device"], resolve_dtype(),
        _resolve_model_ref(cfg.get("asr_model")), cfg["asr_device"],
    )


def reload_model(model_id: str) -> dict[str, Any]:
    """Swap the live model to `model_id` and persist it as active.

    The new model is materialised outside the lock (heavy I/O + GPU), then
    swapped atomically under the lock; the previous model is released only after
    the swap so in-flight requests never see a None. Returns a status dict.
    """
    cfg = MODEL_CONFIG
    entry = _find_model(cfg, model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"未找到模型「{model_id}」")
    new_model = _load_entry(entry, cfg)  # heavy, no lock held
    old = None
    with _state["lock"]:
        old = _state["model"]
        _state["model"] = new_model
        _state["model_id"] = model_id
        cfg["active"] = model_id
    if old is not None and old is not new_model:
        try:
            del old
        except Exception:  # noqa: BLE001
            pass
        import gc  # noqa: E402

        gc.collect()
        try:
            import torch  # noqa: F401

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    save_config(cfg)
    return {
        "ok": True,
        "active": model_id,
        "model": {
            "id": entry["id"],
            "name": entry.get("name"),
            "type": entry.get("type"),
            "path": entry.get("path"),
        },
    }


def load_voices() -> dict[str, str]:
    if VOICES_FILE.exists():
        try:
            data = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as error:  # noqa: BLE001
            log.warning("voices.json unreadable (%s); using defaults", error)
    return dict(DEFAULT_VOICES)


def resolve_dtype():
    import torch

    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if DTYPE_NAME not in table:
        raise SystemExit(f"unknown OMNIVOICE_DTYPE: {DTYPE_NAME}")
    return table[DTYPE_NAME]


def get_model():
    """Load the active model once (at startup or first request), per config.

    Tries the configured `active` entry first; on failure (e.g. a misconfigured
    path) it falls back to the local base checkpoint so the service still boots.
    """
    if _state["model"] is not None:
        return _state["model"]
    with _state["lock"]:
        if _state["model"] is not None:
            return _state["model"]
        cfg = MODEL_CONFIG
        try:
            entry = _find_model(cfg, cfg["active"]) or _default_registry()[0]
            model = _load_entry(entry, cfg)
            _state["model"] = model
            _state["model_id"] = entry["id"]
            log.info("model ready: %s (%s)", entry["id"], entry.get("name"))
        except Exception as error:  # noqa: BLE001
            log.error("active model load failed (%s); falling back to base checkpoint", error)
            model = _instantiate(
                str(HERE / "models" / "OmniVoice"),
                cfg["device"],
                resolve_dtype(),
                _resolve_model_ref(cfg.get("asr_model")),
                cfg["asr_device"],
            )
            _state["model"] = model
            _state["model_id"] = "base"
        return _state["model"]


def to_mp3(samples: np.ndarray, bit_rate: int = 64) -> bytes:
    import lameenc

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bit_rate)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm.tobytes()) + encoder.flush())


def to_wav(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def to_flac(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def to_opus(samples: np.ndarray) -> bytes:
    """Opus inside an Ogg container — libsndfile has no bare-opus muxer."""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="OGG", subtype="OPUS")
    return buffer.getvalue()


def to_pcm(samples: np.ndarray) -> bytes:
    """Raw little-endian 16-bit mono at the model rate (OpenAI `pcm` semantics)."""
    pcm = np.clip(samples, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


AUDIO_FORMATS = {
    "mp3": ("audio/mpeg", to_mp3),
    "wav": ("audio/wav", to_wav),
    "pcm": ("audio/pcm", to_pcm),
    "flac": ("audio/flac", to_flac),
    "opus": ("audio/ogg", to_opus),
}
AUDIO_FORMAT_NAMES = tuple(sorted(AUDIO_FORMATS))


def _load_design_vocab() -> tuple[
    frozenset[str], dict[str, str], dict[str, str], list[frozenset[str]]
]:
    """The upstream voice-design vocabulary.

    Returns the valid items, the EN/ZH mappings and the mutually exclusive
    category groups (gender, age, pitch, style, accent, dialect) -- the
    groups are needed to drop a conflicting item instead of letting the
    upstream validator raise on it.

    Imported lazily because the library lives in the bundled interpreter; a
    missing import must not take the whole service down, so the caller gets
    empty vocabularies and the request falls back to plain cloning.
    """
    try:
        from omnivoice.utils.voice_design import (  # noqa: PLC0415
            _INSTRUCT_ALL_VALID,
            _INSTRUCT_EN_TO_ZH,
            _INSTRUCT_MUTUALLY_EXCLUSIVE,
            _INSTRUCT_ZH_TO_EN,
        )
    except Exception:  # noqa: BLE001
        return frozenset(), {}, {}, []
    return (
        frozenset(_INSTRUCT_ALL_VALID),
        dict(_INSTRUCT_EN_TO_ZH),
        dict(_INSTRUCT_ZH_TO_EN),
        [frozenset(group) for group in _INSTRUCT_MUTUALLY_EXCLUSIVE],
    )


def _dedupe_by_category(
    items: list[str], categories: list[frozenset[str]]
) -> tuple[list[str], list[str]]:
    """Keep at most one item per mutually exclusive category.

    The upstream validator raises when a category appears twice ("male" and
    "female" together). Here the first item of a category wins and the
    later ones are dropped, so mixing sources -- a preset plus a hand
    written attribute, or the voice field plus the extra instructions --
    cannot make a request fail. Returns `(kept, dropped)`.
    """
    kept: list[str] = []
    dropped: list[str] = []
    taken: set[str] = set()
    for item in items:
        group = next((cat for cat in categories if item in cat), None)
        if group is None:
            # Not categorised (should not happen for a whitelisted item).
            kept.append(item)
            continue
        marker = next(iter(sorted(group)))
        if marker in taken:
            # The category already has an item; the first one wins.
            dropped.append(item)
            continue
        taken.add(marker)
        kept.append(item)
    return kept, dropped


def _normalize_design_items(items: list[str]) -> tuple[str, list[str]]:
    """Turn raw attribute items into one valid instruct string.

    Three things have to happen, in this order:
      1. an accent (English-only) and a dialect (Chinese-only) cannot be
         combined -- the upstream validator raises on that pair, so the
         accent is dropped;
      2. at most one item per category may survive;
      3. the rest is unified into a single language, because a mixed
         Chinese/English instruct is rejected too.

    Returns `(instruct, dropped)`.
    """
    valid, en_to_zh, zh_to_en, categories = _load_design_vocab()
    dropped: list[str] = []

    has_dialect = any(item.endswith("话") for item in items)
    has_accent = any(" accent" in item for item in items)
    if has_dialect and has_accent:
        kept = []
        for item in items:
            if " accent" in item:
                dropped.append(item)
            else:
                kept.append(item)
        items = kept

    items, conflicts = _dedupe_by_category(items, categories)
    dropped.extend(conflicts)
    if not items:
        return "", dropped

    has_zh = any(any("一" <= c <= "鿿" for c in item) for item in items)
    if has_zh:
        items = [en_to_zh.get(item, item) for item in items]
    else:
        items = [zh_to_en.get(item, item) for item in items]
    return ("，" if has_zh else ", ").join(items), dropped


def _merge_instructs(first: str, second: str) -> tuple[str, list[str]]:
    """Merge two normalised attribute strings, one item per category.

    Both sides are already normalised, so they may disagree on language
    ("female" against "女") and may repeat a category (a preset's "female"
    against an explicit "male"). Splitting both into items and
    normalising once gives a string the upstream validator accepts.
    Returns `(merged, dropped)`.
    """
    valid, _, _, _ = _load_design_vocab()
    items: list[str] = []
    for chunk in (first, second):
        for raw in re.split(r"\s*[,，]\s*", chunk or ""):
            item = raw.strip().lower()
            if item and item in valid:
                items.append(item)
    if not items:
        return "", []
    return _normalize_design_items(items)


def _voice_mode(clone: Any, instruct: str | None) -> str:
    """Which generation path a request took: clone, design, or both."""
    if clone is not None and instruct:
        return "clone+design"
    return "clone" if clone is not None else "design"


# --- short-line duration compensation -------------------------------------
#
# The model stops before it has used the length it was given. Measured over
# the clone path it emits roughly
#
#     (target tokens / frame rate) - 1.4 s
#
# -- the lead is a fixed ~1.4 s, whatever the line length. On a long line
# that is a rounding error; on a short one it is most of the line. A
# ten-character line measured 0.22 s of speech, about 45 characters a
# second: the tail is simply gone. The designed-voice path does not do
# this, so the lead is only added when a clone is in play.
#
# Passing an explicit duration puts the length back under our control: the
# model fills what it is given, so asking for the wanted length plus the
# lead lands on the wanted length. Lines whose own budget is already big
# enough are left alone, so long lines keep the pacing they have today.
_FALLBACK_REF_TEXT = "Nice to meet you."
_FALLBACK_REF_TOKENS = 25


def _duration_estimator():
    """The estimator the loaded model uses, or a fresh one of the same kind."""
    est = getattr(_state.get("model"), "duration_estimator", None)
    if est is not None:
        return est
    try:
        from omnivoice.utils.duration import RuleDurationEstimator  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return RuleDurationEstimator()


def _frame_rate() -> float:
    try:
        return float(_state["model"].audio_tokenizer.config.frame_rate)
    except Exception:  # noqa: BLE001
        return 24.0


def _auto_duration(text: str, clone: Any, speed: float | None) -> float | None:
    """A `duration` that leaves the model room to finish a short line."""
    if not _as_bool(os.environ.get("OMNI_TTS_AUTO_DURATION", "1")):
        return None
    est_obj = _duration_estimator()
    if est_obj is None or not text:
        return None

    ref_text = str(getattr(clone, "ref_text", "") or "")
    ref_tokens = None
    tokens = getattr(clone, "ref_audio_tokens", None)
    if tokens is not None:
        try:
            ref_tokens = int(tokens.size(-1))
        except Exception:  # noqa: BLE001
            ref_tokens = None
    if not ref_text or not ref_tokens:
        ref_text, ref_tokens = _FALLBACK_REF_TEXT, _FALLBACK_REF_TOKENS

    try:
        est = est_obj.estimate_duration(text, ref_text, ref_tokens)
        weight = est_obj.calculate_total_weight(text)
    except Exception:  # noqa: BLE001
        return None
    if speed and speed != 1.0:
        est = est / speed
    try:
        rate = float(os.environ.get("OMNI_TTS_AUTO_DURATION_RATE", "65"))
        lead = float(os.environ.get("OMNI_TTS_AUTO_DURATION_LEAD", "1.4"))
    except ValueError:
        return None

    budget = est / _frame_rate()
    wanted = weight * rate / 1000.0 + (lead if clone is not None else 0.0)
    if wanted <= budget:
        return None
    return round(min(max(wanted, 0.05), 600.0), 2)


# --- voice field parsing -----------------------------------------------------
# The voice field is a free-form string and may mix several kinds of item:
#
#   "alloy"                     an official preset (voices.json) -> attributes
#   "female, 四川话"             voice-design attributes
#   "ja_Ganyu_0001"             a stored clone -> the clone is used, the item
#                               itself is stripped from the attribute string
#   "[happy]"                   an inline tag -> stripped from the attribute
#                               string and prepended to the text
#   "whatever"                  not recognised -> dropped, and reported
#
# Nothing here raises: an unusable item is dropped and recorded so the
# request still produces audio. The report is attached to the archived
# request so the console can show what happened to each item.

# The base checkpoint understands the 13 upstream non-verbal symbols.
# Both vocabularies are owned by `emotion_map`, which needs them to build
# its default table; the literals are only a fallback for when that import
# failed and the adapter is unavailable anyway.
OFFICIAL_INLINE_TAGS = tuple(
    getattr(_emotion_map, "OFFICIAL_INLINE_TAGS", None)
    or (
        "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
        "question-oh", "question-ei", "question-yi", "surprise-ah",
        "surprise-oh", "surprise-wa", "surprise-yo", "dissatisfaction-hnn",
    )
)

# The ModelsLab singing/emotion finetune adds eight more inline tags.
EMOTION_INLINE_TAGS = tuple(
    getattr(_emotion_map, "FINETUNE_INLINE_TAGS", None)
    or ("singing", "happy", "sad", "angry", "nervous", "whisper", "calm", "excited")
)

# Which inline tags a given checkpoint understands. Keyed by a marker that
# appears in the model's directory or id, so a renamed folder still matches.
_MODEL_INLINE_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("omnivoice-singing", EMOTION_INLINE_TAGS),
    ("omnivoice_singing", EMOTION_INLINE_TAGS),
)


def _active_model_entry() -> dict[str, Any]:
    entry = _find_model(MODEL_CONFIG, MODEL_CONFIG.get("active") or "")
    return entry or {}


def _active_inline_tags() -> tuple[str, ...]:
    """Inline tags the currently loaded checkpoint understands.

    A tag the model does not know is not an error -- it is dropped and
    reported, because a mistyped tag should not cost the user their audio.
    """
    entry = _active_model_entry()
    haystack = f"{entry.get('id', '')} {entry.get('name', '')} {entry.get('path', '')}"
    for marker, tags in _MODEL_INLINE_TAGS:
        if marker.lower() in haystack.lower():
            return OFFICIAL_INLINE_TAGS + tags
    return OFFICIAL_INLINE_TAGS


@dataclass
class VoiceParts:
    """A parsed voice field: what each kind of item resolved to."""

    instruct: str = ""          # voice-design attribute string
    # The clone prompt object. Typed loosely on purpose: `from __future__
    # import annotations` turns annotations into strings, and a dataclass
    # resolving a quoted annotation trips over it.
    clone: object | None = None
    clone_name: str = ""        # which clone was matched
    inline_tags: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    presets: list[str] = field(default_factory=list)
    # Emotion phrases that the adapter turned into a tag, as
    # "phrase -> target" pairs.
    emotion_mapped: list[dict[str, str]] = field(default_factory=list)

    @property
    def inline_prefix(self) -> str:
        """The inline tags, formatted to be prepended to the text."""
        return " ".join(f"[{tag}]" for tag in self.inline_tags)


def _split_voice_items(text: str) -> list[str]:
    """Split a voice field into items, keeping inline tags intact.

    Inline tags arrive in their own bracket form and are not comma
    separated, so the brackets are pulled out first and the remainder is
    split on commas. The tags keep their brackets so the caller can tell
    a tag from an attribute.
    """
    tags: list[str] = []
    rest = re.sub(
        r"\[([^\[\]]+)\]",
        lambda m: tags.append(f"[{m.group(1).strip()}]") and " ",
        text or "",
    )
    items = [part.strip() for part in re.split(r"\s*[,，]\s*", rest) if part.strip()]
    return items + tags


def _claim_item(
    item: str,
    parts: VoiceParts,
    design_items: list[str],
    voices: dict[str, str],
    clones: dict[str, Any],
    inline_tags: tuple[str, ...],
    valid: frozenset[str],
) -> bool:
    """Take one item as a tag, a clone, a preset or an attribute.

    Returns True when the item was recognised. Names win over emotion
    phrases: a clone called `..._欢笑_疑问询问` contains 疑问, which the
    emotion table knows, so names have to be claimed before any phrase
    matching runs or the name would be eaten.
    """
    low = item.lower()

    if item.startswith("[") and item.endswith("]"):
        tag = item[1:-1].strip().lower()
        if tag in inline_tags:
            if tag not in parts.inline_tags:
                parts.inline_tags.append(tag)
        else:
            return False
        return True

    if item in clones:
        if parts.clone is None:
            parts.clone = clones[item]
            parts.clone_name = item
            return True
        return False

    preset = voices.get(item) or voices.get(low)
    if preset:
        parts.presets.append(item)
        for sub in _split_voice_items(preset):
            sub_low = sub.strip().lower()
            if sub_low in valid and sub_low not in design_items:
                design_items.append(sub_low)
        return True

    if low in valid:
        if low not in design_items:
            design_items.append(low)
        return True

    return False


def parse_voice_field(
    voice: str,
    voices: dict[str, str],
    clones: dict[str, Any],
    inline_tags: tuple[str, ...],
    emotion_table: dict[str, list[str]] | None = None,
    blocked: frozenset[str] = frozenset(),
) -> VoiceParts:
    """Resolve a free-form voice field into attributes, a clone and tags.

    A stored clone wins over the attribute map for the same name; the
    attributes are still collected and applied on top of the clone, since
    the model takes both signals at once.

    Emotion phrases (the adapter for DSH-Tavern's `instructions`) are read
    out of what is *left* once every name has been claimed, never out of
    the names themselves.
    """
    valid, _, _, _ = _load_design_vocab()
    parts = VoiceParts()
    design_items: list[str] = []

    unknown: list[str] = []
    for item in _split_voice_items(voice or ""):
        if not _claim_item(item, parts, design_items, voices, clones, inline_tags, valid):
            unknown.append(item)

    # The leftovers are where the emotion phrases live: "害羞" on its own,
    # or DSH-Tavern's whole "用害羞的语气说这句话。".
    leftover: list[str] = []
    if emotion_table and unknown:
        # Anything already recognised is off limits to phrase matching, so a
        # phrase cannot bite into a clone name that shares its characters.
        protected = tuple(clones) + tuple(voices) + tuple(valid) + inline_tags
        for fragment in unknown:
            rest, mapped_inline, mapped_design, unmatched, mapped = _emotion_map.extract(
                fragment, emotion_table, inline_tags, valid, protected, blocked
            )
            for tag in mapped_inline:
                if tag not in parts.inline_tags:
                    parts.inline_tags.append(tag)
            design_items.extend(mapped_design)
            parts.dropped.extend(unmatched)
            parts.emotion_mapped.extend(mapped)
            if rest.strip():
                leftover.append(rest.strip())
        unknown = leftover

    # A phrase may have been sitting next to a name ("<clone> 害羞"), so
    # what survived the phrase pass gets one more look.
    still_unknown: list[str] = []
    for fragment in unknown:
        for item in _split_voice_items(fragment):
            if not _claim_item(item, parts, design_items, voices, clones, inline_tags, valid):
                still_unknown.append(item)
    unknown = still_unknown

    # If nothing at all was understood, report the remainder as one item.
    # Splitting a sentence like "用温柔的语气说话" into "用" "的" "语气"
    # "说话" would fill the log with fragments that say nothing useful.
    if unknown and not design_items and not parts.inline_tags and parts.clone is None:
        parts.dropped.append((voice or "").strip() or unknown[0])
    elif unknown:
        parts.dropped.extend(unknown)

    # Unify the attribute items into a single language, the way the upstream
    # validator would: a Chinese item forces Chinese, otherwise English.
    if design_items:
        # One item per category, a single language, no accent/dialect clash.
        parts.instruct, conflicts = _normalize_design_items(design_items)
        parts.dropped.extend(conflicts)

    return parts


def _clone_path(name: str) -> Path:
    """`<name>.pt` inside the clone directory; the name is validated first."""
    if not _NAME_RE.match(name or ""):
        raise HTTPException(status_code=400, detail="invalid voice name")
    return CLONE_DIR / f"{name}{_CLONE_SUFFIX}"


def clone_names() -> list[str]:
    """Names of every voice prompt stored on disk."""
    if not CLONE_DIR.exists():
        return []
    return sorted(p.name[: -len(_CLONE_SUFFIX)] for p in CLONE_DIR.glob(f"*{_CLONE_SUFFIX}"))


def get_clone(name: str):
    """VoiceClonePrompt for `name`: memory first, otherwise load from disk."""
    if name in _clone_cache:
        return _clone_cache[name]
    path = _clone_path(name)
    if not path.exists():
        return None
    from omnivoice import VoiceClonePrompt

    prompt = VoiceClonePrompt.load(str(path))
    _clone_cache[name] = prompt
    return prompt


def resolve_voice(
    voice: str,
    instructions: str,
    voices: dict[str, str],
) -> tuple[str | None, Any, VoiceParts]:
    """Resolve the voice field and the extra instructions.

    Returns `(instruct, clone, parts)`. `parts` carries the full report of
    what each item in the voice field resolved to -- the clone that was
    matched, the inline tags that were recognised, and the items that were
    dropped -- so the request can be archived with that detail.

    The voice field may mix presets, design attributes, clone names and
    inline tags. The extra instructions are appended to the attribute
    string and parsed the same way.

    A clone and a design attribute are not exclusive: the model takes the
    reference audio and the attributes as separate signals, so both apply.
    """
    clones = {name: get_clone(name) for name in clone_names()}
    clones = {name: prompt for name, prompt in clones.items() if prompt is not None}
    inline_tags = _active_inline_tags()
    table = _EMOTION_TABLE if _emotion_map is not None else {}
    # Attributes and (unless enabled) the official event symbols are not
    # something a mapped phrase may turn into: both change the voice itself.
    valid, _, _, _ = _load_design_vocab()
    blocked = _emotion_map.blocked_targets(valid) if _emotion_map is not None else frozenset()

    parts = parse_voice_field(voice or "", voices, clones, inline_tags, table, blocked)

    # Always parsed: an emotion phrase here is the only channel DSH-Tavern
    # has for expressing one, and the mapping layer turns it into a tag.
    extra = instructions or ""
    if extra:
        extra_parts = parse_voice_field(extra, voices, clones, inline_tags, table)
        # A clone named in the extra instructions cannot replace the one
        # from the voice field: the model takes a single clone.
        if extra_parts.clone is not None and parts.clone is None:
            parts.clone = extra_parts.clone
            parts.clone_name = extra_parts.clone_name
        elif extra_parts.clone is not None:
            extra_parts.dropped.append(extra_parts.clone_name)
        for tag in extra_parts.inline_tags:
            if tag not in parts.inline_tags:
                parts.inline_tags.append(tag)
        parts.dropped.extend(extra_parts.dropped)
        parts.presets.extend(extra_parts.presets)
        parts.emotion_mapped.extend(extra_parts.emotion_mapped)
        if extra_parts.instruct:
            # The two attribute strings were normalised separately, so they
            # can disagree on language and repeat a category (a preset's
            # "female" against an explicit "male"). Merge the items and
            # normalise once, so one item per category survives.
            merged, dropped = _merge_instructs(parts.instruct, extra_parts.instruct)
            parts.instruct = merged
            parts.dropped.extend(dropped)

    return (parts.instruct or None), parts.clone, parts


def run_generate(
    *,
    text: str,
    instruct: str | None,
    speed: float,
    num_step: int,
    fmt: str,
    clone=None,
    gen: dict[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Shared decode path for the OpenAI endpoint and the admin console."""
    model = get_model()
    params = dict(gen or {})
    with _state["lock"]:
        kwargs: dict[str, Any] = {"text": text, "speed": speed, "num_step": num_step}
        # OmniVoiceGenerationConfig surface: only send what differs from the
        # config defaults so `generate()` picks model defaults for the rest.
        for key, value in params.items():
            if key in ("speed", "num_step", "language", "duration", "normalize_text"):
                continue
            if key in GEN_DEFAULTS and value == GEN_DEFAULTS[key]:
                continue
            kwargs[key] = value
        if params.get("normalize_text"):
            kwargs["normalize_text"] = True
        if params.get("language"):
            kwargs["language"] = params["language"]
        if params.get("duration") is not None:
            kwargs["duration"] = params["duration"]
        if clone is not None:
            kwargs["voice_clone_prompt"] = clone
        # The instruct is sent even when a clone is present: the model takes
        # the reference audio and the design attributes as separate signals,
        # so a clone can be shifted by a design attribute (e.g. pitch).
        if instruct:
            kwargs["instruct"] = instruct
        chunks = model.generate(**kwargs)
    if not chunks:
        raise RuntimeError("model returned no audio")
    samples = chunks[0] if isinstance(chunks, (list, tuple)) else chunks
    media_type, encoder = AUDIO_FORMATS[fmt]
    return bytes(encoder(np.asarray(samples, dtype=np.float32))), media_type


def _record_dir(root: Path, rid: str) -> Path:
    if not _ID_RE.match(rid or ""):
        raise HTTPException(status_code=400, detail="bad record id")
    return root / rid


def _history_dir(rid: str) -> Path:
    return _record_dir(HISTORY_DIR, rid)


def _read_meta(directory: Path) -> dict[str, Any] | None:
    meta = directory / "meta.json"
    if not meta.exists():
        return None
    try:
        rec = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    rec.setdefault("id", directory.name)
    return rec


def _all_records(root: Path = HISTORY_DIR, url_prefix: str = "/admin/audio") -> list[dict[str, Any]]:
    """Every stored record, newest first (by directory mtime)."""
    if not root.exists():
        return []
    stamped: list[tuple[float, dict[str, Any]]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        rec = _read_meta(directory)
        if rec is None:
            continue
        if rec.get("audio_file") and (directory / rec["audio_file"]).exists():
            ext = Path(rec["audio_file"]).suffix.lstrip(".") or "mp3"
            rec["audio_url"] = f"{url_prefix}/{rec['id']}?ext={ext}"
        stamped.append((directory.stat().st_mtime, rec))
    stamped.sort(key=lambda pair: pair[0], reverse=True)
    return [rec for _, rec in stamped]


def save_record(
    fields: dict[str, Any],
    payload: bytes | None,
    ext: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Persist one request + its audio output under `<root>/<id>/`."""
    target = root or HISTORY_DIR
    target.mkdir(parents=True, exist_ok=True)
    rid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    entry = dict(fields)
    entry["id"] = rid
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    entry["status"] = "ok" if payload else "error"
    directory = target / rid
    directory.mkdir(parents=True, exist_ok=True)
    if payload:
        (directory / f"audio.{ext}").write_bytes(payload)
        entry["audio_file"] = f"audio.{ext}"
        entry["bytes"] = len(payload)
    (directory / "meta.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The admin-console archive rotates on the same scheme as the API records.
    # (API records are pruned inside save_api_record instead, to avoid double
    # work — both paths stay within their own limits.)
    if target == HISTORY_DIR:
        prune_directory(
            HISTORY_DIR,
            max_entries=HISTORY_MAX_ENTRIES,
            max_bytes=HISTORY_MAX_BYTES,
            max_age_days=HISTORY_MAX_AGE_DAYS,
        )
    return entry


def prune_directory(
    root: Path,
    *,
    max_entries: int,
    max_bytes: int,
    max_age_days: float,
) -> dict[str, Any]:
    """Prune `<root>/<id>/` entries beyond the limits; return a stats snapshot.

    Age-based pruning is unconditional (every entry older than the limit goes),
    then count/size are enforced oldest-first. A limit <= 0 disables that axis,
    which makes this doubles as a disk-usage stats call. Never raises: rotation
    is housekeeping, not a request-path dependency.
    """
    stats: dict[str, Any] = {"entries": 0, "bytes": 0, "pruned": 0, "oldest_age_days": None}
    if not root.exists():
        return stats
    try:
        entries: list[tuple[float, int, Path]] = []
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            try:
                size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                entries.append((directory.stat().st_mtime, size, directory))
            except OSError:
                continue
    except OSError:
        return stats
    entries.sort()  # oldest first
    now = time.time()
    keep: list[tuple[float, int, Path]] = []
    for mtime, size, directory in entries:
        if max_age_days > 0 and (now - mtime) / 86400.0 > max_age_days:
            shutil.rmtree(directory, ignore_errors=True)
            stats["pruned"] += 1
        else:
            keep.append((mtime, size, directory))
    total_bytes = sum(size for _, size, _ in keep)
    while keep and (
        (max_entries > 0 and len(keep) > max_entries)
        or (max_bytes > 0 and total_bytes > max_bytes)
    ):
        _, size, directory = keep.pop(0)
        shutil.rmtree(directory, ignore_errors=True)
        total_bytes -= size
        stats["pruned"] += 1
    stats["entries"] = len(keep)
    stats["bytes"] = total_bytes
    if keep:
        stats["oldest_age_days"] = round((now - keep[0][0]) / 86400.0, 2)
    return stats


# --- response cache helpers -----------------------------------------------------


def _clone_identity(name: str) -> str:
    """Stable identity of a clone prompt: `<name>@<mtime_ns>:<size>`.

    The name alone is not enough — re-cloning a name replaces the .pt file and
    must invalidate the audio cached under the old prompt. Falls back to the
    bare name if the file cannot be stat'ed (the model call will fail loudly).
    """
    try:
        stat = _clone_path(name).stat()
        return f"{name}@{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return name


def _cache_key(
    *,
    text: str,
    voice: str,
    instruct: str | None,
    fmt: str,
    speed: float,
    num_step: int,
    gen: dict[str, Any],
    clone_identity: str | None = None,
    model_id: str | None = None,
) -> str:
    """Hash of everything that can change the output bytes.

    `gen` must be the fully resolved set (defaults applied), so an omitted knob
    and an explicitly-defaulted one hash identically. Clones contribute their
    file signature; design voices contribute the resolved instruct string.

    `model_id` must be part of the hash: the active model can be swapped at
    runtime (see /admin/models/load), and the same request against a different
    checkpoint yields different audio. Without it a cache hit would replay the
    previous model's voice/emotion — audible and wrong.
    """
    payload = {
        "text": text,
        "voice": (voice or "").strip(),
        "instruct": instruct or "",
        "clone": clone_identity or "",
        "format": fmt,
        "speed": speed,
        "num_step": num_step,
        "gen": {key: gen.get(key) for key in sorted(gen)},
        "model": model_id or "",
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_dir(key: str) -> Path:
    return CACHE_DIR / key[:2] / key


def cache_get(key: str) -> tuple[bytes, str, str] | None:
    """Cached `(payload, media_type, fmt)` for `key`, or None on any miss."""
    directory = _cache_dir(key)
    try:
        candidates = sorted(p for p in directory.glob("audio.*") if p.is_file())
        if not candidates:
            return None
        payload = candidates[0].read_bytes()
        fmt = candidates[0].suffix.lstrip(".").lower()
    except OSError:
        return None
    if fmt not in AUDIO_FORMATS:
        return None
    _cache_stats["hits"] += 1
    return payload, AUDIO_FORMATS[fmt][0], fmt


def cache_put(key: str, payload: bytes, fmt: str, echo: dict[str, Any]) -> None:
    """Store one response; every failure is logged and swallowed."""
    try:
        directory = _cache_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"audio.{fmt}"
        tmp = directory / f"audio.{fmt}.{uuid.uuid4().hex[:6]}.tmp"
        tmp.write_bytes(payload)
        os.replace(tmp, final)  # atomic: readers never see a partial file
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "key": key,
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "bytes": len(payload),
                    "format": fmt,
                    "hits": 0,
                    "params": echo,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _cache_stats["puts"] += 1
    except OSError as error:
        log.warning("cache write failed for %s…: %s", key[:12], error)
        return
    stats = prune_directory(
        CACHE_DIR,
        max_entries=CACHE_MAX_ENTRIES,
        max_bytes=CACHE_MAX_BYTES,
        max_age_days=CACHE_MAX_AGE_DAYS,
    )
    _cache_stats["pruned"] += stats["pruned"]


def cache_stats() -> dict[str, Any]:
    """Live counters plus the on-disk footprint and its limits."""
    disk = prune_directory(CACHE_DIR, max_entries=0, max_bytes=0, max_age_days=0)
    return {
        "enabled": CACHE_ENABLED,
        "dir": str(CACHE_DIR),
        "entries": disk["entries"],
        "bytes": disk["bytes"],
        "oldest_age_days": disk["oldest_age_days"],
        "limits": {
            "entries": CACHE_MAX_ENTRIES,
            "bytes": CACHE_MAX_BYTES,
            "age_days": CACHE_MAX_AGE_DAYS,
        },
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "puts": _cache_stats["puts"],
        "pruned": _cache_stats["pruned"],
    }


def api_log_append(entry: dict[str, Any]) -> None:
    """Newest-first ring of recent endpoint requests (RAM only)."""
    _api_log_ring.appendleft(entry)


def save_api_record(
    *,
    text: str,
    voice: str,
    voice_mode: str,
    instruct: str | None,
    speed: float,
    num_step: int,
    fmt: str,
    gen_overrides: dict[str, Any],
    elapsed_ms: int,
    payload: bytes | None,
    cache: str,
    client: str,
    error: str | None = None,
    voice_report: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Archive one /v1/audio/speech request under records/ (failures included).

    Rotation runs after every write, so the archive stays within its limits
    even under sustained traffic. Never raises: the response is already on its
    way out, archiving must not be able to fail it.
    """
    if not API_RECORD:
        return None
    try:
        entry = save_record(
            {
                "source": "api",
                "text": text,
                "voice": voice,
                "voice_mode": voice_mode,
                "instruct": instruct,
                "speed": speed,
                "num_step": num_step,
                "format": fmt,
                "gen_params": gen_overrides,
                "elapsed_ms": elapsed_ms,
                "cache": cache,
                "client": client,
                **(voice_report or {}),
                **({"error": error} if error else {}),
            },
            payload,
            fmt,
            root=RECORD_DIR,
        )
        prune_directory(
            RECORD_DIR,
            max_entries=RECORD_MAX_ENTRIES,
            max_bytes=RECORD_MAX_BYTES,
            max_age_days=RECORD_MAX_AGE_DAYS,
        )
        return entry
    except Exception:  # noqa: BLE001
        log.warning("failed to persist api record")
        return None


app = FastAPI(title="OmniVoice OpenAI-compatible TTS", version="1.0.0")
# Permissive CORS: the console is also usable by opening admin.html straight
# from disk (file://), and everything here is a loopback-only local service.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health() -> dict[str, Any]:
    import torch

    active = _find_model(MODEL_CONFIG, MODEL_CONFIG.get("active") or "")
    return {
        "status": "ok",
        "model": (active or {}).get("id") or MODEL_CONFIG.get("active"),
        "model_name": (active or {}).get("name"),
        "model_type": (active or {}).get("type"),
        "loaded": _state["model"] is not None,
        "loaded_id": _state.get("model_id"),
        "device": MODEL_CONFIG.get("device", DEVICE),
        "dtype": MODEL_CONFIG.get("dtype", DTYPE_NAME),
        "cuda": torch.cuda.is_available(),
        "voices": sorted(load_voices().keys()),
        "voice_attrs": load_voices(),
        "clones": clone_names(),
        # So the console can show/restore the real per-field defaults instead
        # of a hard-coded copy that drifts from the env-overridable values.
        "gen_defaults": GEN_DEFAULTS,
        "num_step_default": NUM_STEP,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    active = _find_model(MODEL_CONFIG, MODEL_CONFIG.get("active") or "")
    mid = (active or {}).get("id") or MODEL_CONFIG.get("active") or MODEL_ID
    return {"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "local"}]}


@app.get("/v1/voices")
def voices() -> dict[str, Any]:
    return {"voices": load_voices()}


@app.post("/v1/audio/speech")
async def speech(request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": {"message": "body must be JSON"}}, status_code=400)

    text = str(body.get("input") or body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": {"message": "input is empty"}}, status_code=400)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT]

    voice = str(body.get("voice") or "")
    instructions = str(body.get("instructions") or "")
    fmt = str(body.get("response_format") or "mp3").lower()
    if fmt not in AUDIO_FORMATS:
        return JSONResponse(
            {
                "error": {
                    "message": f"unsupported response_format {fmt!r}; "
                    f"supported: {', '.join(AUDIO_FORMAT_NAMES)}"
                }
            },
            status_code=400,
        )

    try:
        gen = _parse_gen_params(body, text)
    except HTTPException as error:
        return JSONResponse({"error": {"message": str(error.detail)}}, status_code=400)
    gen_overrides = _gen_overrides(gen)
    speed = gen.pop("speed")
    num_step = gen.pop("num_step")

    voices_map = load_voices()
    # `use_instructions` is gone: the field is always parsed. A request that
    # still carries it is accepted and the flag ignored, so older callers --
    # and anything that copied the old docs -- keep working rather than
    # silently losing their emotion.
    instruct, clone, voice_parts = resolve_voice(voice, instructions, voices_map)
    # Inline tags travel in the text, so they are prepended to it. The
    # cache key uses the resulting text, so a tagged request never hits
    # an untagged entry.
    if voice_parts.inline_prefix:
        text = f"{voice_parts.inline_prefix} {text}"
    # Only when the caller left the length to the model: an explicit
    # `duration` is exactly that, a decision, and is left alone.
    if gen.get("duration") is None:
        auto = _auto_duration(text, clone, speed)
        if auto is not None:
            gen["duration"] = auto
            gen_overrides["duration"] = auto
    voice_mode = _voice_mode(clone, instruct)
    # What each item in the voice field resolved to. Kept in the archive so
    # the console can show dropped items, the matched clone and the inline
    # tags instead of leaving the user guessing.
    voice_report = {
        "dropped": voice_parts.dropped,
        "matched_clone": voice_parts.clone_name,
        "inline_tags": voice_parts.inline_tags,
        "presets": voice_parts.presets,
        # What the caller actually sent, so the archive can be read back.
        "instructions": instructions,
        "emotion_mapped": voice_parts.emotion_mapped,
    }
    client = request.client.host if request.client else ""
    started = time.perf_counter()

    # --- cache lookup: byte-identical request, skip the model entirely -------
    key = None
    if CACHE_ENABLED:
        key = _cache_key(
            text=text,
            voice=voice,
            instruct=instruct,
            fmt=fmt,
            speed=speed,
            num_step=num_step,
            gen=gen,
            clone_identity=_clone_identity(voice) if clone is not None else None,
            model_id=_state.get("model_id") or MODEL_CONFIG.get("active") or "",
        )
        cached = cache_get(key)
        if cached is not None:
            payload, media_type, _ = cached
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info(
                "tts cache hit: %d chars, voice=%r, %d bytes %s, %d ms",
                len(text), voice, len(payload), fmt, elapsed_ms,
            )
            record = save_api_record(
                text=text, voice=voice, voice_mode=voice_mode, instruct=instruct,
                speed=speed, num_step=num_step, fmt=fmt, gen_overrides=gen_overrides,
                elapsed_ms=elapsed_ms, payload=payload, cache="hit", client=client,
                voice_report=voice_report,
            )
            api_log_append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "status": "ok", "cache": "hit", "text": text, "voice": voice,
                "voice_mode": voice_mode, "format": fmt, "elapsed_ms": elapsed_ms,
                "bytes": len(payload), "client": client, "id": (record or {}).get("id"),
                **voice_report,
            })
            return Response(content=payload, media_type=media_type, headers={"X-Cache": "hit"})

    try:
        payload, media_type = run_generate(
            text=text,
            instruct=instruct,
            speed=speed,
            num_step=num_step,
            fmt=fmt,
            clone=clone,
            gen=gen,
        )
    except Exception as error:  # noqa: BLE001
        log.exception("generation failed")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        record = save_api_record(
            text=text, voice=voice, voice_mode=voice_mode, instruct=instruct,
            speed=speed, num_step=num_step, fmt=fmt, gen_overrides=gen_overrides,
            elapsed_ms=elapsed_ms, payload=None, cache="miss", client=client,
            error=str(error), voice_report=voice_report,
        )
        api_log_append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "status": "error", "cache": "miss" if key else "off", "text": text,
            "voice": voice, "voice_mode": voice_mode, "format": fmt, **voice_report,
            "elapsed_ms": elapsed_ms, "bytes": 0, "client": client,
            "error": str(error), "id": (record or {}).get("id"),
        })
        return JSONResponse({"error": {"message": f"generation failed: {error}"}}, status_code=500)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "tts ok: %d chars, voice=%r, instruct=%r, %d bytes %s",
        len(text), voice, instruct, len(payload), fmt,
    )
    if key is not None:
        _cache_stats["misses"] += 1
        cache_put(key, payload, fmt, echo=gen_overrides)
    record = save_api_record(
        text=text, voice=voice, voice_mode=voice_mode, instruct=instruct,
        speed=speed, num_step=num_step, fmt=fmt, gen_overrides=gen_overrides,
        elapsed_ms=elapsed_ms, payload=payload, cache="miss" if key else "off",
        client=client, voice_report=voice_report,
    )
    api_log_append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "status": "ok", "cache": "miss" if key else "off", "text": text,
        "voice": voice, "voice_mode": voice_mode, "format": fmt, **voice_report,
        "elapsed_ms": elapsed_ms, "bytes": len(payload), "client": client,
        "id": (record or {}).get("id"),
    })
    headers = {"X-Cache": "miss"} if key is not None else {}
    return Response(content=payload, media_type=media_type, headers=headers)


# --- admin console ----------------------------------------------------------
# A single self-contained page (no CDN, no build step) plus the JSON API it
# talks to. Every request made here is archived under history/<id>/ with the
# decoded audio so runs can be replayed, downloaded or audited afterwards.


@app.get("/admin", include_in_schema=False)
def admin_page():
    if not ADMIN_FILE.exists():
        return HTMLResponse("<h1>admin.html not found next to server.py</h1>", status_code=500)
    return HTMLResponse(ADMIN_FILE.read_text(encoding="utf-8"))


@app.get("/admin/history", include_in_schema=False)
def admin_history(limit: int = 50) -> dict[str, Any]:
    records = _all_records()
    rotation = prune_directory(
        HISTORY_DIR,
        max_entries=HISTORY_MAX_ENTRIES,
        max_bytes=HISTORY_MAX_BYTES,
        max_age_days=HISTORY_MAX_AGE_DAYS,
    )
    rotation["limits"] = {
        "entries": HISTORY_MAX_ENTRIES,
        "bytes": HISTORY_MAX_BYTES,
        "age_days": HISTORY_MAX_AGE_DAYS,
    }
    rotation["enabled"] = True
    return {
        "items": records[: max(1, min(limit, 500))],
        "total": len(records),
        "ok": sum(1 for r in records if r.get("status") == "ok"),
        "failed": sum(1 for r in records if r.get("status") != "ok"),
        "bytes_total": sum(int(r.get("bytes") or 0) for r in records),
        "rotation": rotation,
    }


@app.get("/admin/history/{rid}", include_in_schema=False)
def admin_get_record(rid: str) -> dict[str, Any]:
    rec = _read_meta(_history_dir(rid))
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    return rec


@app.delete("/admin/history/{rid}", include_in_schema=False)
def admin_delete_record(rid: str) -> dict[str, Any]:
    directory = _history_dir(rid)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="record not found")
    shutil.rmtree(directory, ignore_errors=True)
    return {"ok": True}


@app.delete("/admin/history", include_in_schema=False)
def admin_clear_history() -> dict[str, Any]:
    if HISTORY_DIR.exists():
        shutil.rmtree(HISTORY_DIR, ignore_errors=True)
    return {"ok": True}


@app.get("/admin/audio/{rid}", include_in_schema=False)
def admin_audio(rid: str, download: int = 0):
    directory = _history_dir(rid)
    rec = _read_meta(directory)
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    name = rec.get("audio_file")
    if not name or not (directory / name).exists():
        raise HTTPException(status_code=404, detail="record has no audio")
    fmt = Path(name).suffix.lstrip(".").lower() or "mp3"
    media_type = AUDIO_FORMATS.get(fmt, AUDIO_FORMATS["mp3"])[0]
    headers = {"Content-Disposition": f'attachment; filename="{rid}.{fmt}"'} if download else {}
    return Response(content=(directory / name).read_bytes(), media_type=media_type, headers=headers)


# --- endpoint archive (records/) + live request log -----------------------------
# Separate from history/ (console-only, manually managed): every
# /v1/audio/speech request lands here with its audio, and rotation prunes the
# oldest once any configured limit trips. GETs also prune, so age limits self-
# enforce even when no new traffic arrives.


@app.get("/admin/api-records", include_in_schema=False)
def admin_api_records(limit: int = 50) -> dict[str, Any]:
    records = _all_records(RECORD_DIR, url_prefix="/admin/api-audio")
    rotation = prune_directory(
        RECORD_DIR,
        max_entries=RECORD_MAX_ENTRIES,
        max_bytes=RECORD_MAX_BYTES,
        max_age_days=RECORD_MAX_AGE_DAYS,
    )
    rotation["limits"] = {
        "entries": RECORD_MAX_ENTRIES,
        "bytes": RECORD_MAX_BYTES,
        "age_days": RECORD_MAX_AGE_DAYS,
    }
    rotation["enabled"] = API_RECORD
    return {
        "items": records[: max(1, min(limit, 500))],
        "total": len(records),
        "ok": sum(1 for r in records if r.get("status") == "ok"),
        "failed": sum(1 for r in records if r.get("status") != "ok"),
        "bytes_total": sum(int(r.get("bytes") or 0) for r in records),
        "rotation": rotation,
    }


@app.get("/admin/api-records/{rid}", include_in_schema=False)
def admin_api_get_record(rid: str) -> dict[str, Any]:
    rec = _read_meta(_record_dir(RECORD_DIR, rid))
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    return rec


@app.delete("/admin/api-records/{rid}", include_in_schema=False)
def admin_api_delete_record(rid: str) -> dict[str, Any]:
    directory = _record_dir(RECORD_DIR, rid)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="record not found")
    shutil.rmtree(directory, ignore_errors=True)
    return {"ok": True}


@app.delete("/admin/api-records", include_in_schema=False)
def admin_api_clear_records() -> dict[str, Any]:
    if RECORD_DIR.exists():
        shutil.rmtree(RECORD_DIR, ignore_errors=True)
    return {"ok": True}


@app.get("/admin/api-audio/{rid}", include_in_schema=False)
def admin_api_audio(rid: str, download: int = 0):
    directory = _record_dir(RECORD_DIR, rid)
    rec = _read_meta(directory)
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")
    name = rec.get("audio_file")
    if not name or not (directory / name).exists():
        raise HTTPException(status_code=404, detail="record has no audio")
    fmt = Path(name).suffix.lstrip(".").lower() or "mp3"
    media_type = AUDIO_FORMATS.get(fmt, AUDIO_FORMATS["mp3"])[0]
    headers = {"Content-Disposition": f'attachment; filename="{rid}.{fmt}"'} if download else {}
    return Response(content=(directory / name).read_bytes(), media_type=media_type, headers=headers)


@app.get("/admin/api-log", include_in_schema=False)
def admin_api_log(limit: int = 100) -> dict[str, Any]:
    """In-memory ring of recent endpoint requests, newest first (RAM only)."""
    items = list(_api_log_ring)
    return {
        "items": items[: max(1, min(limit, 500))],
        "total": len(items),
        "cache": cache_stats(),
    }


_REF_SUFFIXES = (".ref.wav", ".ref.mp3", ".ref.flac", ".ref.ogg", ".ref.m4a", ".ref.aac")
_REF_MEDIA = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


@app.get("/admin/voices", include_in_schema=False)
def admin_voices() -> dict[str, Any]:
    """Stored clones: metadata only, no prompt is decoded or loaded."""
    items: list[dict[str, Any]] = []
    for name in clone_names():
        entry: dict[str, Any] = {"name": name, "cached": name in _clone_cache}
        meta = CLONE_DIR / f"{name}.json"
        if meta.exists():
            try:
                entry.update(json.loads(meta.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        items.append(entry)
    return {"items": items, "total": len(items)}


@app.get("/admin/languages", include_in_schema=False)
def admin_languages() -> dict[str, Any]:
    """The model's language table, for `language=` autocomplete.

    `ids` are the ISO codes the model accepts; `names` the English names it
    also understands (both directions resolve in `_resolve_language_param`).
    """
    try:
        from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {"available": False, "ids": [], "names": []}
    return {
        "available": True,
        "ids": sorted(LANG_IDS),
        "names": sorted(LANG_NAMES),
    }


# --- model management (console) ---------------------------------------------
# List / switch / register / delete / scan the models in model_config.json.
# `load` swaps the live model at runtime and persists the choice; it runs off the
# event loop (run_in_threadpool) so the 60–90 s load never blocks requests.

@app.get("/admin/models", include_in_schema=False)
def admin_models() -> dict[str, Any]:
    cfg = MODEL_CONFIG
    active = cfg.get("active")
    active_entry = _find_model(cfg, active) or {}
    return {
        "active": active,
        "device": cfg.get("device"),
        "dtype": cfg.get("dtype"),
        "num_step": cfg.get("num_step"),
        "asr_model": _resolve_model_ref(cfg.get("asr_model")),
        "loaded": _state["model"] is not None,
        "loaded_id": _state.get("model_id"),
        "current": {
            "id": active_entry.get("id"),
            "name": active_entry.get("name"),
            "type": active_entry.get("type"),
            "path": str(_entry_path(active_entry)) if active_entry else None,
        },
        "models": [
            {**e, "path": str(_entry_path(e)), "active": e.get("id") == active}
            for e in cfg.get("models", [])
        ],
        "scan_dirs": [str(p) for p in _model_scan_roots()],
    }


@app.post("/admin/models/load", include_in_schema=False)
async def admin_models_load(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    model_id = str(body.get("id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="缺少 id")
    try:
        return await run_in_threadpool(reload_model, model_id)
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"加载失败：{error}")


@app.post("/admin/models/register", include_in_schema=False)
async def admin_models_register(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    model_id = str(body.get("id") or "").strip()
    name = str(body.get("name") or "").strip()
    path = str(body.get("path") or "").strip()
    mtype = str(body.get("type") or "checkpoint").strip().lower()
    base = str(body.get("base") or "").strip() or None
    if not model_id or not name or not path:
        raise HTTPException(status_code=400, detail="id / name / path 均为必填")
    if mtype not in ("checkpoint", "lora"):
        raise HTTPException(status_code=400, detail="type 只能是 checkpoint 或 lora")
    p = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path.strip()))))
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在：{p}")
    if mtype == "lora" and base is None:
        base = _first_checkpoint_id(MODEL_CONFIG)
    cfg = MODEL_CONFIG
    if _find_model(cfg, model_id) is not None:
        raise HTTPException(status_code=409, detail=f"模型 id「{model_id}」已存在")
    cfg["models"].append(
        {
            "id": model_id,
            "name": name,
            "path": _store_path(p),
            "type": mtype,
            **({"base": base} if mtype == "lora" else {}),
        }
    )
    save_config(cfg)
    return {"ok": True, "models": cfg["models"]}


@app.delete("/admin/models/{model_id}", include_in_schema=False)
def admin_models_delete(model_id: str) -> dict[str, Any]:
    if model_id == "base":
        raise HTTPException(status_code=400, detail="base 是默认模型，不可删除")
    if model_id == MODEL_CONFIG.get("active"):
        raise HTTPException(status_code=400, detail="当前正在使用的模型不可删除（先切换到其它模型）")
    cfg = MODEL_CONFIG
    before = len(cfg["models"])
    cfg["models"] = [e for e in cfg["models"] if e.get("id") != model_id]
    if len(cfg["models"]) == before:
        raise HTTPException(status_code=404, detail="未找到模型")
    save_config(cfg)
    return {"ok": True, "models": cfg["models"]}


@app.post("/admin/models/scan", include_in_schema=False)
def admin_models_scan() -> dict[str, Any]:
    """Scan known directories for loadable model folders not yet registered."""
    dirs = _model_scan_roots()
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    registered_paths = {
        str(_entry_path(e)) for e in MODEL_CONFIG.get("models", [])
    }
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            mtype = _detect_model_type(sub)
            if mtype is None:
                continue
            key = str(sub)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "path": str(sub),
                    "type": mtype,
                    "name": sub.name,
                    "registered": key in registered_paths
                    or str(sub.resolve()) in registered_paths,
                }
            )
    return {"ok": True, "dirs": [str(d) for d in dirs], "found": found}


# --- settings ----------------------------------------------------------------
# Every configurable environment variable, its default, and where the current
# value comes from. The registry lives in settings.py; the values the user
# sets are stored in runtime_config.json and injected into the environment at
# startup, so a change to a start-up-time variable needs a restart while a
# "live" one takes effect at once.

# The generation defaults whose environment variable maps onto a field of
# GEN_DEFAULTS. Changing one of these can be applied to the running process.
_LIVE_GEN_KEYS = {
    "OMNIVOICE_GUIDANCE_SCALE": "guidance_scale",
    "OMNIVOICE_T_SHIFT": "t_shift",
    "OMNIVOICE_LAYER_PENALTY_FACTOR": "layer_penalty_factor",
    "OMNIVOICE_POSITION_TEMPERATURE": "position_temperature",
    "OMNIVOICE_CLASS_TEMPERATURE": "class_temperature",
    "OMNIVOICE_DENOISE": "denoise",
    "OMNIVOICE_POSTPROCESS_OUTPUT": "postprocess_output",
    "OMNIVOICE_PAD_DURATION": "pad_duration",
    "OMNIVOICE_FADE_DURATION": "fade_duration",
    "OMNIVOICE_AUDIO_CHUNK_DURATION": "audio_chunk_duration",
    "OMNIVOICE_AUDIO_CHUNK_THRESHOLD": "audio_chunk_threshold",
}

# Settings that are not generation-config keys but are still read at the
# point of use, so writing them into the environment is all it takes for a
# change to land on the next request.
_LIVE_ENV_KEYS = frozenset({
    "OMNI_TTS_MODEL_DIRS",
    "OMNI_TTS_EMOTION_MAP_OFFICIAL",
    "OMNI_TTS_AUTO_DURATION",
    "OMNI_TTS_AUTO_DURATION_RATE",
    "OMNI_TTS_AUTO_DURATION_LEAD",
})


def _apply_live_settings(
    values: dict[str, str], to_environment: bool = True
) -> list[str]:
    """Push the values that can change without a restart into the process.

    `to_environment` writes the value back into the environment so the
    console shows what is in effect; the reset path leaves the environment
    alone, because there the built-in default is what applies.

    Returns the names that were applied live.
    """
    applied: list[str] = []
    for name, raw in values.items():
        if name in _LIVE_GEN_KEYS:
            key = _LIVE_GEN_KEYS[name]
            spec = _settings.SETTINGS[name]
            try:
                if spec.get("kind") == "bool":
                    GEN_DEFAULTS[key] = raw == "1"
                elif spec.get("kind") == "int":
                    GEN_DEFAULTS[key] = int(raw)
                else:
                    GEN_DEFAULTS[key] = float(raw)
            except (TypeError, ValueError):
                continue
            if to_environment:
                # Keep the environment in step with the live value, so the
                # console shows what is actually in effect.
                os.environ[name] = raw
                _settings.mark_applied(name)
            applied.append(name)
        elif name in _LIVE_ENV_KEYS:
            # Read at the point of use -- on every scan, on every request,
            # or when a line is about to be voiced -- so setting it here is
            # all it takes for the next call to pick it up.
            os.environ[name] = raw
            if to_environment:
                _settings.mark_applied(name)
            applied.append(name)
    return applied


@app.get("/admin/settings", include_in_schema=False)
def admin_settings() -> dict[str, Any]:
    """Every configurable variable, with its default, current value and source."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="settings registry unavailable")
    stored = _settings.load_stored()
    return {
        "ok": True,
        "groups": [{"key": key, "name": name} for key, name in _settings.GROUPS],
        "settings": _settings.describe(),
        "config_file": str(_settings.config_path()),
        "config_exists": _settings.config_path().exists(),
        "stored_count": len(stored),
        "restart_required": [
            item["name"] for item in _settings.describe() if not item["live"]
        ],
    }


@app.post("/admin/settings", include_in_schema=False)
async def admin_settings_save(request: Request) -> dict[str, Any]:
    """Store configuration values. Live ones apply at once, the rest on restart."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="settings registry unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    raw_values = body.get("values")
    if not isinstance(raw_values, dict):
        raise HTTPException(status_code=400, detail="缺少 values 对象")

    stored = _settings.load_stored()
    changed: dict[str, str] = {}
    errors: list[str] = []
    for name, raw in raw_values.items():
        if name not in _settings.SETTINGS:
            errors.append(f"未知变量：{name}")
            continue
        try:
            normalised = _settings.validate(name, str(raw))
        except ValueError as error:
            errors.append(str(error))
            continue
        stored[name] = normalised
        changed[name] = normalised

    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))

    _settings.save_stored(stored)
    # A value the real environment already holds is not overridden, so warn
    # rather than pretend the change took effect. Variables this process
    # injected from the file are ours, not an operator's override.
    applied = _settings.applied_names()
    shadowed = [
        name for name in changed
        if name in os.environ and name not in applied
        and os.environ[name] != changed[name]
    ]
    live = _apply_live_settings(changed)
    return {
        "ok": True,
        "changed": changed,
        "live_applied": live,
        "needs_restart": [name for name in changed if name not in live],
        "shadowed_by_env": shadowed,
        "settings": _settings.describe(),
    }


@app.post("/admin/settings/reset", include_in_schema=False)
async def admin_settings_reset(request: Request) -> dict[str, Any]:
    """Clear the stored configuration (the environment keeps its own values)."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="settings registry unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    names = body.get("names") if isinstance(body, dict) else None
    stored = _settings.load_stored()
    cleared: list[str] = []
    if isinstance(names, list) and names:
        for name in names:
            if stored.pop(str(name), None) is not None:
                cleared.append(str(name))
    else:
        cleared = list(stored)
        stored = {}
    _settings.save_stored(stored)
    # Only the values this process applied from the file are ours to remove;
    # a variable the operator set in the environment stays untouched.
    for name in _settings.applied_names():
        if name not in stored and name in os.environ:
            os.environ.pop(name, None)
    # Put the live values back to their defaults so the running process
    # matches the cleared configuration. The default is applied to the
    # in-process state only -- the environment is left clean, so the value
    # reads as "default" rather than as an environment override.
    defaults = {
        name: _settings.SETTINGS[name]["default"] for name in cleared
        if name in _settings.SETTINGS
    }
    restored = _apply_live_settings(defaults, to_environment=False)
    # The stored value is gone, so these are no longer "configured".
    for name in restored:
        _settings.unmark_applied(name)
    return {
        "ok": True,
        "cleared": cleared,
        "restored_live": restored,
        "settings": _settings.describe(),
    }


# --- emotion map (adapter for DSH-Tavern's emotion phrases) ---------------

def _emotion_payload(table: dict[str, list[str]], source: str | None = None) -> dict[str, Any]:
    """One response shape for everything that changes the mapping table."""
    valid, _, _, _ = _load_design_vocab()
    blocked = _emotion_map.blocked_targets(valid)
    inline_tags = _active_inline_tags()
    return {
        "ok": True,
        "map_file": str(_emotion_map.map_path()),
        "preset_file": str(_emotion_map.presets_path()),
        "active_model": _state.get("model_id") or MODEL_CONFIG.get("active"),
        "inline_tags": list(inline_tags),
        "targets": _emotion_map.known_targets(inline_tags, valid),
        "mappings": _emotion_map.describe(table, inline_tags, valid, blocked),
        "count": len(table),
        "blocked": sorted(blocked),
        "official_enabled": _emotion_map.official_enabled(),
        "presets": _emotion_map.list_presets(),
        "source": source,
    }


def _emotion_commit(table: dict[str, list[str]], source: str | None = None) -> dict[str, Any]:
    """Store a table and refresh the copy the request path reads."""
    _emotion_map.save_map(table, source)
    # Update in place: a plain assignment here would create a local name
    # and leave the module-level table -- the one the parser reads -- stale.
    _EMOTION_TABLE.clear()
    _EMOTION_TABLE.update(table)
    return _emotion_payload(table, source)


@app.get("/admin/emotion-map", include_in_schema=False)
def admin_emotion_map() -> dict[str, Any]:
    """The mapping table, plus which targets the loaded checkpoint can use."""
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    table, source = _emotion_map.load_meta()
    return _emotion_payload(table, source)


@app.post("/admin/emotion-map", include_in_schema=False)
async def admin_emotion_map_save(request: Request) -> dict[str, Any]:
    """Store the mapping table.

    Accepts either a full table or a single entry, so the console can add
    one row without resending everything.
    """
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}

    table = _emotion_map.load_map()
    added: list[str] = []
    removed: list[str] = []
    errors: list[str] = []

    raw = body.get("mappings")
    if isinstance(raw, dict):
        # A full replacement: every phrase in the payload is the table.
        table = {}
        for key, value in raw.items():
            phrase = str(key).strip()
            targets = _emotion_map._normalise(value)
            if not phrase:
                continue
            if not targets:
                errors.append(f"「{phrase}」没有指定目标标签")
                continue
            table[phrase] = targets
            added.append(phrase)
    else:
        phrase = str(body.get("phrase") or "").strip()
        targets = _emotion_map._normalise(body.get("targets"))
        remove = bool(body.get("remove"))
        if not phrase:
            raise HTTPException(status_code=400, detail="缺少 phrase")
        if remove:
            if table.pop(phrase, None) is not None:
                removed.append(phrase)
        else:
            if not targets:
                raise HTTPException(status_code=400, detail="缺少 targets")
            table[phrase] = targets
            added.append(phrase)

    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))

    # An edit means the table is no longer whichever preset it came from.
    payload = _emotion_commit(table, None)
    payload["added"] = added
    payload["removed"] = removed
    return payload


@app.post("/admin/emotion-map/reset", include_in_schema=False)
def admin_emotion_map_reset() -> dict[str, Any]:
    """Restore the built-in default table."""
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    table = _emotion_map.default_table()
    return _emotion_commit(table, _emotion_map.DEFAULT_SOURCE)


@app.post("/admin/emotion-map/clear", include_in_schema=False)
def admin_emotion_map_clear() -> dict[str, Any]:
    """Empty the active table, so a preset can be built up from nothing."""
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    return _emotion_commit({}, None)


@app.post("/admin/emotion-map/preset/apply", include_in_schema=False)
async def admin_emotion_map_preset_apply(request: Request) -> dict[str, Any]:
    """Import a preset: `replace` swaps the table, `merge` layers it on."""
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺少模板名")
    mode = str(body.get("mode") or "replace").strip().lower()
    if mode not in ("replace", "merge"):
        raise HTTPException(status_code=400, detail="mode 只能是 replace 或 merge")
    current, _ = _emotion_map.load_meta()
    try:
        table = _emotion_map.apply_preset(current, name, mode)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    # A merge is its own thing rather than the preset, so only a clean
    # replacement can claim the preset's name.
    return _emotion_commit(table, name if mode == "replace" else None)


@app.post("/admin/emotion-map/preset/save", include_in_schema=False)
async def admin_emotion_map_preset_save(request: Request) -> dict[str, Any]:
    """Store a table as a user preset, creating it or overwriting it.

    With no `mappings` in the body the active table is what gets saved,
    which is the "save as" the console does.
    """
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺少模板名")
    raw = body.get("mappings")
    if isinstance(raw, dict):
        table = _emotion_map._normalise_table(raw)
    elif raw is None:
        table, _ = _emotion_map.load_meta()
    else:
        raise HTTPException(status_code=400, detail="mappings 必须是对象")
    try:
        count = _emotion_map.save_preset(name, table)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    payload = _emotion_payload(table)
    payload["saved"] = {"name": name, "count": count}
    return payload


@app.post("/admin/emotion-map/preset/delete", include_in_schema=False)
async def admin_emotion_map_preset_delete(request: Request) -> dict[str, Any]:
    """Remove a user preset. The active table is left alone."""
    if _emotion_map is None:
        raise HTTPException(status_code=503, detail="emotion map unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺少模板名")
    try:
        removed = _emotion_map.delete_preset(name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    table, source = _emotion_map.load_meta()
    payload = _emotion_payload(table, source)
    payload["removed"] = bool(removed)
    return payload


def _detect_model_type(folder: Path) -> str | None:
    """Classify a directory as 'lora' or 'checkpoint' (or None if not a model)."""
    if (folder / "adapter_config.json").exists():
        return "lora"
    cfg_file = folder / "config.json"
    if cfg_file.is_file():
        try:
            obj = json.loads(cfg_file.read_text(encoding="utf-8"))
            arch = obj.get("architectures") or []
            if any("OmniVoice" in str(a) for a in arch):
                return "checkpoint"
        except Exception:  # noqa: BLE001
            pass
    return None


def _decode_audio(raw: bytes) -> tuple[Any, int]:
    """Decode uploaded bytes into a mono float32 waveform + sample rate."""
    import soundfile as sf  # noqa: PLC0415

    if not raw:
        raise HTTPException(status_code=400, detail="audio is empty")
    try:
        data, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解码音频：{error}")
    wave = np.asarray(data, dtype=np.float32)
    if wave.ndim == 2:  # soundfile returns (frames, channels)
        wave = wave.mean(axis=1)
    if wave.size == 0:
        raise HTTPException(status_code=400, detail="audio is empty after decode")
    return wave, int(sr)


def _store_clone(
    name: str,
    wave: Any,
    sr: int,
    ref_text: str | None,
    source: str,
    raw: bytes | None = None,
    preprocess_prompt: bool = True,
) -> dict[str, Any]:
    """Encode a voice-clone prompt and persist prompt + reference audio + meta.

    `ref_text` may be empty: the model then auto-transcribes the reference
    audio with its bundled Whisper ASR (local copy under
    `models/whisper-large-v3-turbo`; override with OMNIVOICE_ASR_MODEL). The
    transcript lands in the returned prompt, so it is recorded in meta.json
    even though the caller supplied nothing.

    `preprocess_prompt` (OmniVoice default True) trims silence in the reference
    audio and appends a sentence-final punctuation to `ref_text` when missing.
    Turn it off only when the raw clip must be embedded verbatim — e.g. the
    transcript is already punctuated and the silence carries meaning.
    """
    import torch  # noqa: PLC0415

    model = get_model()
    auto_asr = not (ref_text or "").strip()
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=(torch.from_numpy(wave).unsqueeze(0), int(sr)),
            ref_text=None if auto_asr else ref_text,
            preprocess_prompt=preprocess_prompt,
        )
    except Exception as error:  # noqa: BLE001
        log.exception("voice clone prompt failed")
        if auto_asr:
            raise HTTPException(
                status_code=500,
                detail=f"克隆失败（ASR 自动转写路径）：{error}。"
                "ASR 模型默认位于项目 models/whisper-large-v3-turbo；"
                "若移动过位置，设 OMNIVOICE_ASR_MODEL 指向它；"
                "若显存不足，设 OMNIVOICE_ASR_DEVICE=cpu 可将 ASR 放到内存（更慢）。",
            )
        raise HTTPException(status_code=500, detail=f"克隆失败：{error}")

    # With ASR the transcript only exists inside the prompt; surface it.
    resolved_text = str(prompt.ref_text or "")
    duration = float(np.asarray(wave).size) / float(sr)
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    prompt.save(str(CLONE_DIR / f"{name}{_CLONE_SUFFIX}"))
    suffix = Path(source).suffix.lower()
    ref_suffix = f".ref{suffix}" if f".ref{suffix}" in _REF_SUFFIXES else ".ref.wav"
    if raw:
        (CLONE_DIR / f"{name}{ref_suffix}").write_bytes(raw)
    (CLONE_DIR / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "ref_text": resolved_text,
                "ref_text_source": "asr" if auto_asr else "provided",
                "duration": round(duration, 2),
                "sample_rate": int(sr),
                "source": source,
                "preprocess_prompt": bool(preprocess_prompt),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _clone_cache.pop(name, None)
    log.info(
        "voice cloned: %s (%.1fs, %d chars ref_text%s, preprocess_prompt=%s)",
        name,
        duration,
        len(resolved_text),
        ", auto-transcribed" if auto_asr else "",
        preprocess_prompt,
    )
    return {
        "ok": True,
        "name": name,
        "duration": round(duration, 2),
        "ref_text": resolved_text,
        "ref_text_source": "asr" if auto_asr else "provided",
    }


@app.post("/admin/voices", include_in_schema=False)
async def admin_add_voice(
    name: str = Form(""),
    ref_text: str = Form(""),
    local_path: str = Form(""),
    preprocess_prompt: str = Form("1"),
    audio: UploadFile | None = File(None),
):
    """Build a clone prompt from one reference clip and store it.

    Two ways to hand over the clip:

    * `local_path` — a path on the server host. The audio is read straight off
      disk, and when `ref_text` is empty the transcript is taken from the
      **same directory** (`<stem>.txt` / `.lab` / `.json`). No searching.
    * `audio` — a browser upload. Browsers cannot see sibling files, so an
      empty `ref_text` goes straight to ASR here.

    `ref_text` is optional in every mode. When it ends up empty the model
    auto-transcribes the clip with its bundled Whisper ASR (local copy under
    `models/whisper-large-v3-turbo`; override with OMNIVOICE_ASR_MODEL). The
    transcript is stored in the voice metadata, and returned as
    `ref_text_source: "asr"`. Hand-writing the transcript stays the most
    accurate option — a wrong transcript misaligns the clone.

    An empty `name` is derived from the audio file name (extension stripped);
    on collision the first free `<stem>_1`, `<stem>_2`, … is taken instead of
    overwriting. A *typed* name that already exists is still a hard error.
    """
    typed_name = (name or "").strip()
    ref_text = (ref_text or "").strip()
    source_path = (local_path or "").strip()

    if source_path:
        resolved = _resolve_local(source_path)
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail=f"本地路径不是文件：{resolved}")
        raw = resolved.read_bytes()
        filename = resolved.name
    else:
        if audio is None:
            raise HTTPException(
                status_code=400,
                detail="请选择参考音频文件，或填写本地音频路径（服务端同目录取逐字稿）",
            )
        raw = await audio.read()
        filename = audio.filename or ""

    if typed_name:
        if not _NAME_RE.match(typed_name):
            raise HTTPException(status_code=400, detail="invalid voice name")
        if _clone_path(typed_name).exists():
            raise HTTPException(
                status_code=409, detail=f"音色「{typed_name}」已存在，换个名字或先删除"
            )
        name = typed_name
    else:
        stem = _safe_name(Path(filename).stem)
        if not stem:
            raise HTTPException(
                status_code=400,
                detail="无法从文件名推导音色名（文件名为空或全是非法字符），请手填音色名称",
            )
        name = _unique_clone_name(stem)

    if not ref_text and source_path:
        # 留空 + 本地路径：先免费拿同目录逐字稿；没有再让 _store_clone 走 ASR
        ref_text = _sidecar_text(resolved)

    wave, sr = _decode_audio(raw)
    return _store_clone(
        name,
        wave,
        sr,
        ref_text,
        filename,
        raw=raw,
        preprocess_prompt=_as_bool(preprocess_prompt),
    )


@app.post("/admin/import-local", include_in_schema=False)
def admin_import_local(
    path: str = Form(...),
    prefix: str = Form(""),
    limit: int = Form(20),
    overwrite: int = Form(0),
    preprocess_prompt: str = Form("1"),
    asr: int = Form(0),
) -> dict[str, Any]:
    """Register every clip in a local directory.

    Reads the audio off disk and takes each transcript from the **same
    directory** (`<stem>.txt` / `.lab` / `.json`) — nothing is searched for.
    By default clips without a sibling transcript are skipped.

    `path`       file or directory on the server host
    `prefix`     prepended to each stem to build the voice name
    `limit`      max clips to import from a directory (sorted by name)
    `preprocess_prompt`  "0" keeps reference audio verbatim (no silence trim)
    `asr`        "1" = clips without a sibling transcript are auto-transcribed
                 by the bundled Whisper ASR instead of skipped (first use
                 downloads ~1.6 GB; each clip takes a few seconds)
    """
    resolved = _resolve_local(path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{resolved}")

    if resolved.is_file():
        files = [resolved]
    else:
        files = sorted(
            f
            for f in resolved.iterdir()
            if f.is_file() and f.suffix.lower() in _AUDIO_SUFFIXES
        )[: max(1, min(200, int(limit or 20)))]

    prefix_clean = _safe_name(prefix, trim=False) if (prefix or "").strip() else ""
    keep_raw = not _as_bool(preprocess_prompt)
    use_asr = _as_bool(asr)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for f in files:
        text = _sidecar_text(f)
        if not text:
            if not use_asr:
                skipped.append({"file": f.name, "reason": "同目录没有同名 .txt/.lab/.json"})
                continue
            # asr=1：逐字稿留空，由 _store_clone 走 Whisper 自动转写
        name = _safe_name(f"{prefix_clean}{f.stem}") if prefix_clean else _safe_name(f.stem)
        if not name:
            skipped.append({"file": f.name, "reason": f"非法音色名：{f.stem}"})
            continue
        if _clone_path(name).exists():
            if not overwrite:
                # 名称来自文件名：按用户规则自动加序号，而不是跳过
                name = _unique_clone_name(name)
        try:
            raw = f.read_bytes()
            wave, sr = _decode_audio(raw)
        except HTTPException as error:
            skipped.append({"file": f.name, "reason": str(error.detail)})
            continue
        except Exception as error:  # noqa: BLE001
            skipped.append({"file": f.name, "reason": f"读取失败：{error}"})
            continue
        try:
            created.append(
                _store_clone(name, wave, sr, text, f.name, raw=raw, preprocess_prompt=not keep_raw)
            )
        except HTTPException as error:
            skipped.append({"file": f.name, "reason": str(error.detail)})

    return {
        "ok": True,
        "path": str(resolved),
        "created": created,
        "skipped": skipped,
        "total": len(files),
    }


_TEXT_SUFFIXES = (".txt", ".lab")
_JSON_SUFFIX = ".json"
_TRANSCRIPT_KEYS = ("transcription", "text", "transcript")


def _sidecar_text(audio_path: Path) -> str:
    """Read the transcript sitting **next to** `audio_path` (same stem).

    Looks for `<stem>.txt` / `<stem>.lab` first, then `<stem>.json` with a
    `transcription` / `text` / `transcript` field. Returns "" when absent.
    """
    for suffix in _TEXT_SUFFIXES:
        candidate = audio_path.with_name(audio_path.stem + suffix)
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001
                pass
    js = audio_path.with_name(audio_path.stem + ".json")
    if js.is_file():
        try:
            obj = json.loads(js.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            return ""
        for key in _TRANSCRIPT_KEYS:
            if obj.get(key):
                return str(obj[key]).strip()
    return ""


def _resolve_local(raw_path: str) -> Path:
    """Expand + resolve a path typed on the admin page."""
    target = Path(os.path.expandvars(os.path.expanduser((raw_path or "").strip())))
    try:
        return target.resolve()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"路径不可解析：{raw_path}")


def _entry_path(entry: dict[str, Any]) -> Path:
    """Resolve a registry entry's stored path back to an absolute path."""
    raw = str(entry.get("path") or "").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = HERE / p
    return p


def _safe_name(raw_name: str, trim: bool = True) -> str:
    """Coerce an arbitrary stem into something `_NAME_RE` accepts.

    `trim=False` keeps a trailing separator (e.g. the `_` in a user-supplied
    name prefix), so `prefix + stem` still reads as two parts.
    """
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", (raw_name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    if trim:
        cleaned = cleaned.strip("._ ")
    return cleaned[:64]


def _unique_clone_name(base: str) -> str:
    """First free clone name: `base`, then `base_1`, `base_2`, ...

    Used when the name is derived from a file name, so re-registering the same
    clip never overwrites an existing voice. Explicitly typed names are *not*
    routed through here — a collision there is an error, not a silent rename.
    """
    if not _clone_path(base).exists():
        return base
    for index in range(1, 1000):
        suffix = f"_{index}"
        candidate = base[: 64 - len(suffix)] + suffix
        if not _clone_path(candidate).exists():
            return candidate
    raise HTTPException(status_code=409, detail=f"名称 {base} 的可用序号已用尽")


# --- native file picker -----------------------------------------------------
# The browser only ever hands us a file name, so the admin page takes an
# absolute path instead. To spare typing it, the server can raise a native
# Windows dialog (PowerShell + WinForms — no extra Python dependency) and hand
# the chosen path back for the page to fill in.
_PICK_TIMEOUT = 600  # seconds; a dialog left open must not hang a worker forever
_PICK_FILTER = (
    "音频文件 (*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac)|*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac"
    "|逐字稿 (*.txt;*.lab;*.json)|*.txt;*.lab;*.json"
    "|所有文件 (*.*)|*.*"
)


def _ps_quote(value: str) -> str:
    """Escape a value for embedding in a single-quoted PowerShell string."""
    return (value or "").replace("'", "''")


def _win32() -> tuple[Any, Any, Any]:
    """(user32, kernel32, shell32) with 64-bit-safe prototypes.

    ctypes defaults to 32-bit `int` for arguments without argtypes, which
    silently truncates HWND/HMENU/HICON handles and LPARAM on x64 — the tray
    icon's DefWindowProcW blew up with "int too long to convert" before these
    were declared. Idempotent: prototypes are installed once per process.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    if getattr(user32, "_omnivoice_prototypes", False):
        return user32, kernel32, shell32
    user32._omnivoice_prototypes = True

    user32.CreateWindowExW.restype = ctypes.c_void_p
    user32.CreateWindowExW.argtypes = [
        ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.DefWindowProcW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t
    ]
    user32.RegisterClassW.restype = ctypes.c_uint
    user32.RegisterClassW.argtypes = [ctypes.c_void_p]
    user32.CreatePopupMenu.restype = ctypes.c_void_p
    user32.AppendMenuW.restype = ctypes.c_int
    user32.AppendMenuW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_wchar_p
    ]
    user32.GetCursorPos.restype = ctypes.c_int
    user32.GetCursorPos.argtypes = [ctypes.POINTER(_MSG)]
    user32.SetForegroundWindow.restype = ctypes.c_int
    user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    user32.TrackPopupMenu.restype = ctypes.c_int
    user32.TrackPopupMenu.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    user32.DestroyMenu.restype = ctypes.c_int
    user32.DestroyMenu.argtypes = [ctypes.c_void_p]
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.LoadImageW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    user32.LoadIconW.restype = ctypes.c_void_p
    user32.LoadIconW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    user32.GetMessageW.restype = ctypes.c_int
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(_MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint
    ]
    user32.TranslateMessage.restype = ctypes.c_int
    user32.TranslateMessage.argtypes = [ctypes.POINTER(_MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(_MSG)]
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.keybd_event.argtypes = [
        ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ctypes.c_void_p
    ]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.CloseClipboard.restype = ctypes.c_int

    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

    shell32.Shell_NotifyIconW.restype = ctypes.c_int
    shell32.Shell_NotifyIconW.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    return user32, kernel32, shell32


def _nudge_foreground() -> None:
    """Let a window raised by this (non-foreground) process reach the front.

    Windows only lets the *foreground* process steal focus. A server process
    popping a dialog from a worker thread therefore shows it behind whatever
    the user is looking at — exactly the "silently sitting on the desktop"
    symptom. Tapping and releasing ALT is the long-standing way to lift that
    lock; a bare down/up pair has no visible side effect.
    """
    if os.name != "nt":
        return
    try:
        user32, _, _ = _win32()
        user32.keybd_event(0xA4, 0, 0, None)  # VK_MENU down
        user32.keybd_event(0xA4, 0, 2, None)  # KEYEVENTF_KEYUP
    except Exception:  # noqa: BLE001
        pass


def _native_pick(mode: str, title: str, initial_dir: str) -> str:
    """Raise a native Open-file / folder dialog and return the chosen path.

    Runs synchronously on a worker thread (the route is a plain `def`), so the
    asyncio loop keeps serving while the dialog is open. The path is returned
    base64-encoded to survive any console code page.
    """
    if os.name != "nt":
        raise HTTPException(status_code=400, detail="原生选择框仅支持 Windows")
    start = (initial_dir or "").strip()
    if not start or not Path(start).is_dir():
        start = str(HERE)

    if mode == "dir":
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.StartPosition = 'CenterScreen'
$owner.Show()
$owner.Activate()
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = '{_ps_quote(title)}'
$d.SelectedPath = '{_ps_quote(start)}'
$d.ShowNewFolderButton = $true
$r = $d.ShowDialog($owner)
$owner.Close()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($d.SelectedPath)))
}}
"""
    else:
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.StartPosition = 'CenterScreen'
$owner.Show()
$owner.Activate()
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = '{_ps_quote(title)}'
$d.InitialDirectory = '{_ps_quote(start)}'
$d.Filter = '{_ps_quote(_PICK_FILTER)}'
$d.Multiselect = $false
$r = $d.ShowDialog($owner)
$owner.Close()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($d.FileName)))
}}
"""

    # Lift Windows' foreground lock so the dialog lands in front of the browser.
    _nudge_foreground()
    try:
        proc = subprocess.run(  # noqa: S603
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=_PICK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="选择框超时未操作，已关闭")
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"无法启动选择框：{error}")

    out = (proc.stdout or "").strip().splitlines()
    encoded = out[-1].strip() if out else ""
    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded, validate=False).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="选择框返回内容无法解析")


@app.post("/admin/pick-path", include_in_schema=False)
async def admin_pick_path(request: Request) -> dict[str, Any]:
    """Open a native dialog on the server host and return the chosen path.

    Body: `{"mode": "file" | "dir", "title": "...", "initial_dir": "..."}`.
    `initial_dir` is usually the directory of whatever is already in the path
    box, so the dialog opens where the user left off. An empty `path` means the
    dialog was cancelled.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    mode = str(body.get("mode") or "file").strip().lower()
    if mode not in ("file", "dir"):
        raise HTTPException(status_code=400, detail="mode 只能是 file 或 dir")
    title = str(body.get("title") or "选择参考音频").strip()
    initial_dir = str(body.get("initial_dir") or "").strip()

    # Dialog owner is this process, not the browser session; run it on a worker
    # thread so the event loop stays responsive.
    path = await run_in_threadpool(_native_pick, mode, title, initial_dir)
    return {"ok": True, "mode": mode, "path": path}


@app.delete("/admin/voices/{name}", include_in_schema=False)
def admin_delete_voice(name: str) -> dict[str, Any]:
    path = _clone_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="voice not found")
    path.unlink(missing_ok=True)
    (CLONE_DIR / f"{name}.json").unlink(missing_ok=True)
    for suffix in _REF_SUFFIXES:
        (CLONE_DIR / f"{name}{suffix}").unlink(missing_ok=True)
    _clone_cache.pop(name, None)
    return {"ok": True}


@app.get("/admin/voices/{name}/audio", include_in_schema=False)
def admin_voice_audio(name: str):
    """Play back the reference clip a clone was built from."""
    if not _NAME_RE.match(name or ""):
        raise HTTPException(status_code=400, detail="invalid voice name")
    for suffix in _REF_SUFFIXES:
        candidate = CLONE_DIR / f"{name}{suffix}"
        if candidate.exists():
            ext = suffix.replace(".ref", "")
            return Response(
                content=candidate.read_bytes(),
                media_type=_REF_MEDIA.get(ext, "application/octet-stream"),
            )
    raise HTTPException(status_code=404, detail="reference audio not found")


@app.post("/admin/generate", include_in_schema=False)
async def admin_generate(request: Request):
    """Console-initiated synthesis: arbitrary params, always archived."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}

    text = str(body.get("input") or body.get("text") or "").strip()
    if not text:
        return JSONResponse({"status": "error", "error": "input is empty"}, status_code=400)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT]

    voice = str(body.get("voice") or "alloy")
    instructions = str(body.get("instructions") or "")
    fmt = str(body.get("response_format") or "mp3").lower()
    if fmt not in AUDIO_FORMATS:
        return JSONResponse(
            {
                "status": "error",
                "error": f"unsupported response_format {fmt!r}; "
                f"supported: {', '.join(AUDIO_FORMAT_NAMES)}",
            },
            status_code=400,
        )
    try:
        gen = _parse_gen_params(body, text)
    except HTTPException as error:
        return JSONResponse({"status": "error", "error": str(error.detail)}, status_code=400)
    gen_overrides = _gen_overrides(gen)
    speed = gen.pop("speed")
    num_step = gen.pop("num_step")

    instruct, clone, voice_parts = resolve_voice(voice, instructions, load_voices())
    if voice_parts.inline_prefix:
        text = f"{voice_parts.inline_prefix} {text}"
    if gen.get("duration") is None:
        auto = _auto_duration(text, clone, speed)
        if auto is not None:
            gen["duration"] = auto
            gen_overrides["duration"] = auto
    fields: dict[str, Any] = {
        "source": "admin",
        "text": text,
        "voice": voice,
        "voice_mode": _voice_mode(clone, instruct),
        "instructions": instructions,
        "instruct": instruct,
        "speed": speed,
        "num_step": num_step,
        "format": fmt,
        "gen_params": gen_overrides,
        # What each item in the voice field resolved to.
        "dropped": voice_parts.dropped,
        "matched_clone": voice_parts.clone_name,
        "inline_tags": voice_parts.inline_tags,
        "presets": voice_parts.presets,
    }

    started = time.perf_counter()
    try:
        payload, _ = run_generate(
            text=text,
            instruct=instruct,
            speed=speed,
            num_step=num_step,
            fmt=fmt,
            clone=clone,
            gen=gen,
        )
    except Exception as error:  # noqa: BLE001
        log.exception("admin generation failed")
        entry = save_record(
            dict(fields, elapsed_ms=int((time.perf_counter() - started) * 1000), error=str(error)),
            None,
            fmt,
        )
        return JSONResponse({"status": "error", "id": entry["id"], "error": str(error)}, status_code=500)

    entry = save_record(dict(fields, elapsed_ms=int((time.perf_counter() - started) * 1000)), payload, fmt)
    entry["audio_url"] = f"/admin/audio/{entry['id']}?ext={fmt}"
    return entry


# --- system tray icon -------------------------------------------------------
# A resident icon in the notification area (bottom-right of the screen), so the
# server is reachable without the console: left click opens the admin console,
# right click offers "open / copy endpoint / quit". Implemented straight against
# Shell_NotifyIcon via ctypes — pystray + Pillow would pull a wheel into the
# self-contained runtime for something this small.
TRAY_ENABLED = os.environ.get("OMNI_TTS_TRAY", "1") == "1"
_TRAY_ICON_FILE = HERE / "tray.ico"

_WM_TRAY = 0x0400  # WM_USER
_WM_LBUTTONUP = 0x0202
_WM_RBUTTONUP = 0x0205
_NIM_ADD = 0x00000000
_NIM_MODIFY = 0x00000001
_NIM_DELETE = 0x00000002
_NIF_MESSAGE = 0x00000001
_NIF_ICON = 0x00000002
_NIF_TIP = 0x00000004
_TPM_RIGHTBUTTON = 0x0002
_MF_STRING = 0x0000
_MF_SEPARATOR = 0x0800
_CF_UNICODETEXT = 13
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x0010
_LR_DEFAULTSIZE = 0x0040
_IDI_APPLICATION = 32512


def _tray_icon_bytes() -> bytes:
    """Build a 32x32 .ico (BGRA BMP) with rounded tile + white volume bars.

    Written to `tray.ico` next to this file on first run because LoadImageW
    wants a path; the file is a runtime artifact — delete it and the icon
    simply regenerates, so a stale copy can never outlive its generator.
    """
    size = 32
    pixels = bytearray(size * size * 4)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        idx = ((size - 1 - y) * size + x) * 4  # BMP rows are bottom-up
        pixels[idx : idx + 4] = bytes((color[2], color[1], color[0], 255))

    radius = 7
    for y in range(size):
        for x in range(size):
            corner = (radius, radius, size - 1 - radius, size - 1 - radius)
            inside = True
            if not (radius <= x < size - radius) and not (radius <= y < size - radius):
                dx = max(corner[0] - x, 0, x - corner[2])
                dy = max(corner[1] - y, 0, y - corner[3])
                if dx * dx + dy * dy > radius * radius:
                    inside = False
            if not inside:
                continue
            t = (x + y) / (2 * size - 2)
            put(x, y, (int(0x1B + 0x3D * t), int(0x6E + 0x3B * t), int(0xF3 + 0x0C * t)))
    # three white bars (a little volume-meter glyph), 4px wide
    for cx, height in ((10, 9), (15, 15), (20, 20)):
        for x in range(cx, cx + 3):
            for y in range(size - 6 - height, size - 5):
                put(x, y, (255, 255, 255))

    header = 40
    pixels_offset = 22 + header
    and_mask = b"\x00" * (size // 8 * size)  # 1bpp, all-zero = fully visible
    payload = pixels + and_mask
    bmp = (
        b"".join(
            [
                (40).to_bytes(4, "little"),  # biSize
                size.to_bytes(4, "little"),
                (size * 2).to_bytes(4, "little"),  # XOR + AND rows
                (1).to_bytes(2, "little"),  # planes
                (32).to_bytes(2, "little"),  # bitCount
                (0).to_bytes(4, "little"),  # compression
                len(payload).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
            ]
        )
        + payload
    )
    ico = bytearray()
    ico += (0).to_bytes(2, "little")  # reserved
    ico += (1).to_bytes(2, "little")  # type: icon
    ico += (1).to_bytes(2, "little")  # count
    ico += bytes((size, size, 0, 0, 1, 32, len(bmp).to_bytes(4, "little")[0], 0, 0, 0, 0, 0))
    ico += (pixels_offset).to_bytes(4, "little")
    return bytes(ico) + bmp


def _tray_icon_path() -> str:
    try:
        if not _TRAY_ICON_FILE.exists():
            _TRAY_ICON_FILE.write_bytes(_tray_icon_bytes())
        if _TRAY_ICON_FILE.is_file():
            return str(_TRAY_ICON_FILE)
    except Exception as error:  # noqa: BLE001
        log.warning("tray icon write failed: %s", error)
    return ""


class _NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("hWnd", ctypes.c_void_p),
        ("uID", ctypes.c_uint32),
        ("uFlags", ctypes.c_uint32),
        ("uCallbackMessage", ctypes.c_uint32),
        ("hIcon", ctypes.c_void_p),
        ("szTip", ctypes.c_wchar * 128),
        ("dwState", ctypes.c_uint32),
        ("dwStateMask", ctypes.c_uint32),
        ("szInfo", ctypes.c_wchar * 256),
        ("uVersion", ctypes.c_uint32),
        ("szInfoTitle", ctypes.c_wchar * 64),
        ("dwInfoFlags", ctypes.c_uint32),
        ("guidItem", ctypes.c_byte * 16),
        ("hBalloonIcon", ctypes.c_void_p),
    ]


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_uint),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.c_void_p),
        ("hIcon", ctypes.c_void_p),
        ("hCursor", ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p),
        ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),
    ]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t),
        ("time", ctypes.c_uint),
        ("pt_x", ctypes.c_long),
        ("pt_y", ctypes.c_long),
    ]


_WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t
)
_WM_DESTROY = 0x0002
_WM_COMMAND = 0x0111
_ID_OPEN = 1
_ID_COPY = 2
_ID_EXIT = 3
_ID_COPY_ROOT = 4


def _tray_open_admin() -> None:
    try:
        webbrowser.open(f"http://127.0.0.1:{int(os.environ.get('OMNI_TTS_PORT', '8890'))}/admin")
    except Exception as error:  # noqa: BLE001
        log.warning("open admin failed: %s", error)


def _tray_copy_endpoint() -> None:
    url = f"http://127.0.0.1:{int(os.environ.get('OMNI_TTS_PORT', '8890'))}/v1/audio/speech"
    _put_clipboard_text(url)
    log.info("endpoint copied to clipboard: %s", url)


def _tray_copy_root() -> None:
    """The base URL DSH-Tavern wants: it appends `/audio/speech` itself.

    Copying the full endpoint here would be wrong for that one caller --
    pasted into its Endpoint field it becomes
    `/v1/audio/speech/audio/speech` and every line 404s.
    """
    url = f"http://127.0.0.1:{int(os.environ.get('OMNI_TTS_PORT', '8890'))}/v1"
    _put_clipboard_text(url)
    log.info("base url copied to clipboard: %s", url)


def _put_clipboard_text(url: str) -> None:
    try:
        user32, kernel32, _ = _win32()
        if not user32.OpenClipboard(None):
            return
        try:
            user32.EmptyClipboard()
            buf = ctypes.create_unicode_buffer(url)
            handle = kernel32.GlobalAlloc(0x0042, len(buf) * 2)  # GMEM_MOVEABLE|GHND
            locked = kernel32.GlobalLock(handle)
            ctypes.memmove(locked, buf, len(buf) * 2)
            kernel32.GlobalUnlock(handle)
            user32.SetClipboardData(_CF_UNICODETEXT, handle)
        finally:
            user32.CloseClipboard()
    except Exception as error:  # noqa: BLE001
        log.warning("clipboard copy failed: %s", error)


def _tray_thread() -> None:
    """Own a hidden window + notification-area icon; blocks in a message loop."""
    try:
        user32, kernel32, shell32 = _win32()

        tip = f"OmniVoice TTS · :{os.environ.get('OMNI_TTS_PORT', '8890')}（左键管理台 / 右键菜单）"
        icon_path = _tray_icon_path()
        hicon = None
        if icon_path:
            hicon = user32.LoadImageW(
                None,
                icon_path,
                _IMAGE_ICON,
                32,
                32,
                _LR_LOADFROMFILE | _LR_DEFAULTSIZE,
            )
        if not hicon:
            hicon = user32.LoadIconW(None, _IDI_APPLICATION)

        state: dict[str, Any] = {"nid": None}

        def wnd_proc(hwnd, message, wparam, lparam):
            if message == _WM_TRAY:
                if lparam == _WM_LBUTTONUP:
                    _tray_open_admin()
                elif lparam == _WM_RBUTTONUP:
                    menu = user32.CreatePopupMenu()
                    user32.AppendMenuW(menu, _MF_STRING, _ID_OPEN, "打开管理台")
                    user32.AppendMenuW(menu, _MF_STRING, _ID_COPY, "复制完整端点（curl 等）")
                    user32.AppendMenuW(menu, _MF_STRING, _ID_COPY_ROOT, "复制 DSHT 根地址")
                    user32.AppendMenuW(menu, _MF_SEPARATOR, 0, None)
                    user32.AppendMenuW(menu, _MF_STRING, _ID_EXIT, "退出服务")
                    pt = _MSG()
                    user32.GetCursorPos(ctypes.byref(pt))
                    user32.SetForegroundWindow(hwnd)
                    user32.TrackPopupMenu(
                        menu, _TPM_RIGHTBUTTON, pt.pt_x, pt.pt_y, 0, hwnd, None
                    )
                    user32.DestroyMenu(menu)
                return 0
            if message == _WM_COMMAND:
                command = int(wparam) & 0xFFFF
                if command == _ID_OPEN:
                    _tray_open_admin()
                elif command == _ID_COPY:
                    _tray_copy_endpoint()
                elif command == _ID_COPY_ROOT:
                    _tray_copy_root()
                elif command == _ID_EXIT:
                    log.info("quit requested from tray icon")
                    os._exit(0)
                return 0
            if message == _WM_DESTROY:
                nid = state["nid"]
                if nid:
                    shell32.Shell_NotifyIconW(_NIM_DELETE, ctypes.byref(nid))
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        proc = _WNDPROC(wnd_proc)  # keep a strong reference for the lifetime
        class_name = "OmniVoiceTrayWindow"
        wndclass = _WNDCLASSW()
        wndclass.lpfnWndProc = ctypes.cast(proc, ctypes.c_void_p)
        wndclass.lpszClassName = class_name
        wndclass.hInstance = kernel32.GetModuleHandleW(None)
        if not user32.RegisterClassW(ctypes.byref(wndclass)):
            raise OSError(f"RegisterClassW failed: {ctypes.get_last_error()}")

        hwnd = user32.CreateWindowExW(
            0, class_name, "OmniVoice TTS", 0x80000000, 0, 0, 0, 0, None, None,
            wndclass.hInstance, None,
        )
        if not hwnd:
            raise OSError(f"CreateWindowExW failed: {ctypes.get_last_error()}")

        nid = _NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(_NOTIFYICONDATAW)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = _NIF_MESSAGE | _NIF_ICON | _NIF_TIP
        nid.uCallbackMessage = _WM_TRAY
        nid.hIcon = hicon
        nid.szTip = tip
        state["nid"] = nid
        if not shell32.Shell_NotifyIconW(_NIM_ADD, ctypes.byref(nid)):
            raise OSError(f"Shell_NotifyIconW failed: {ctypes.get_last_error()}")
        log.info("tray icon ready (notification area)")

        msg = _MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except Exception as error:  # noqa: BLE001
        # Never let the tray take the server down: log and move on.
        log.warning("tray icon disabled: %s", error)


def start_tray() -> None:
    """Start the notification-area icon on a daemon thread (Windows only)."""
    if not TRAY_ENABLED or os.name != "nt":
        return
    thread = threading.Thread(target=_tray_thread, name="omnivoice-tray", daemon=True)
    thread.start()


def main() -> None:
    port = int(os.environ.get("OMNI_TTS_PORT", "8890"))
    host = os.environ.get("OMNI_TTS_HOST", "127.0.0.1")
    setup_logging()
    try:
        get_model()
    except Exception as error:  # noqa: BLE001
        log.error("model load failed: %s", error)
        sys.exit(1)
    log.info("listening on http://%s:%d", host, port)
    log.info("admin console: http://%s:%d/admin", host, port)
    if _INJECTED:
        log.info(
            "applied configured env: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(_INJECTED.items())),
        )
    log.info("log file: %s (<=%d MB x %d backups)", LOG_FILE, LOG_MAX_BYTES // 1048576, LOG_BACKUPS)
    log.info("history archive: %s", HISTORY_DIR)
    if API_RECORD:
        log.info(
            "api records: %s (<=%d entries / <=%d MB / <=%g days)",
            RECORD_DIR, RECORD_MAX_ENTRIES, RECORD_MAX_BYTES // 1048576, RECORD_MAX_AGE_DAYS,
        )
    else:
        log.info("api records: disabled (OMNI_API_RECORD=0)")
    if CACHE_ENABLED:
        log.info("response cache: %s", CACHE_DIR)
    else:
        log.info("response cache: disabled (OMNI_CACHE=0)")
    log.info("cloned voices: %s (%d stored)", CLONE_DIR, len(clone_names()))
    start_tray()
    # log_config=None: uvicorn skips its own logging setup, so its records
    # (including the access log) propagate to the root handlers installed by
    # setup_logging() and land in the rotating server.log.
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


if __name__ == "__main__":
    main()
