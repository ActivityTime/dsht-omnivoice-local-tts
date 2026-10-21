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
from datetime import datetime
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

# Default to the locally-snapshotted copy (downloaded by download_model.py) so we
# never depend on the HF Xet/CAS backend at inference time. Override with the
# OMNIVOICE_MODEL env var if you ever want to pull from the hub again.
MODEL_ID = os.environ.get("OMNIVOICE_MODEL", str(HERE / "models" / "OmniVoice"))
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE_NAME = os.environ.get("OMNIVOICE_DTYPE", "bfloat16").lower()
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))
API_KEY = os.environ.get("OMNI_TTS_KEY", "")  # empty = accept any key
VOICES_FILE = Path(os.environ.get("OMNI_TTS_VOICES", HERE / "voices.json"))
# DSH-Tavern sends `instructions` (an emotion phrase) only when the model name
# contains "gpt-4o". OmniVoice's design attributes have no emotion axis, so by
# default the phrase is ignored; set this to 1 to append it to the instruct.
USE_INSTRUCTIONS = os.environ.get("OMNI_TTS_USE_INSTRUCTIONS", "0") == "1"

# --- request history (admin console) ----------------------------------------
# Each request made through the admin console is persisted as a directory
# holding meta.json plus the decoded audio, so runs can be replayed later.
HISTORY_DIR = Path(os.environ.get("OMNI_TTS_HISTORY_DIR", HERE / "history"))
ADMIN_FILE = Path(os.environ.get("OMNI_TTS_ADMIN_PAGE", HERE / "admin.html"))

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
# Requests arriving on the OpenAI endpoint are only logged when enabled: DSH
# Tavern can fire one per message bubble and would fill the disk.
LOG_API = os.environ.get("OMNI_TTS_LOG_API", "0") == "1"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

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
    "postprocess_output": os.environ.get("OMNIVOICE_POSTPROCESS_OUTPUT", "1") == "1",
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

_state: dict[str, Any] = {"model": None, "lock": threading.Lock()}


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
    """Load the model once, at startup, so no request pays the load cost."""
    if _state["model"] is not None:
        return _state["model"]
    with _state["lock"]:
        if _state["model"] is not None:
            return _state["model"]
        import torch
        from omnivoice import OmniVoice

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        log.info("loading %s on %s as %s", MODEL_ID, DEVICE, DTYPE_NAME)
        model = OmniVoice.from_pretrained(
            MODEL_ID, device_map=DEVICE, dtype=resolve_dtype(), low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            log.info("CUDA ready: %s", torch.cuda.get_device_name(0))
        _state["model"] = model
        log.info("model ready")
        return model


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


def pick_instruct(
    voice: str,
    instructions: str,
    voices: dict[str, str],
    use_instructions: bool | None = None,
) -> str:
    key = (voice or "").strip()
    instruct = voices.get(key, voices.get(key.lower(), ""))
    if not instruct:
        # Not one of the known names: treat the value itself as a design string,
        # so "female, elderly, 四川话" works straight from the voice field.
        instruct = key
    merge = USE_INSTRUCTIONS if use_instructions is None else use_instructions
    if merge and instructions:
        instruct = f"{instruct}, {instructions}" if instruct else instructions
    return instruct.strip(" ,，")


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
    use_instructions: bool | None = None,
) -> tuple[str | None, Any]:
    """Return `(instruct, clone)` for a voice field.

    A stored clone wins over the attribute map, and the two modes are mutually
    exclusive: cloning carries its own timbre, so no instruct is sent.
    """
    key = (voice or "").strip()
    if key in clone_names():
        return None, get_clone(key)
    return pick_instruct(key, instructions, voices, use_instructions), None


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
        else:
            kwargs["instruct"] = instruct or None
        chunks = model.generate(**kwargs)
    if not chunks:
        raise RuntimeError("model returned no audio")
    samples = chunks[0] if isinstance(chunks, (list, tuple)) else chunks
    media_type, encoder = AUDIO_FORMATS[fmt]
    return bytes(encoder(np.asarray(samples, dtype=np.float32))), media_type


def _history_dir(rid: str) -> Path:
    if not _ID_RE.match(rid or ""):
        raise HTTPException(status_code=400, detail="bad record id")
    return HISTORY_DIR / rid


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


def _all_records() -> list[dict[str, Any]]:
    """Every stored record, newest first (by directory mtime)."""
    if not HISTORY_DIR.exists():
        return []
    stamped: list[tuple[float, dict[str, Any]]] = []
    for directory in HISTORY_DIR.iterdir():
        if not directory.is_dir():
            continue
        rec = _read_meta(directory)
        if rec is None:
            continue
        if rec.get("audio_file") and (directory / rec["audio_file"]).exists():
            ext = Path(rec["audio_file"]).suffix.lstrip(".") or "mp3"
            rec["audio_url"] = f"/admin/audio/{rec['id']}?ext={ext}"
        stamped.append((directory.stat().st_mtime, rec))
    stamped.sort(key=lambda pair: pair[0], reverse=True)
    return [rec for _, rec in stamped]


def save_record(fields: dict[str, Any], payload: bytes | None, ext: str) -> dict[str, Any]:
    """Persist one request + its audio output under history/<id>/."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    rid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    entry = dict(fields)
    entry["id"] = rid
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    entry["status"] = "ok" if payload else "error"
    directory = HISTORY_DIR / rid
    directory.mkdir(parents=True, exist_ok=True)
    if payload:
        (directory / f"audio.{ext}").write_bytes(payload)
        entry["audio_file"] = f"audio.{ext}"
        entry["bytes"] = len(payload)
    (directory / "meta.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


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

    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": _state["model"] is not None,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "cuda": torch.cuda.is_available(),
        "voices": sorted(load_voices().keys()),
        "clones": clone_names(),
        # So the console can show/restore the real per-field defaults instead
        # of a hard-coded copy that drifts from the env-overridable values.
        "gen_defaults": GEN_DEFAULTS,
        "num_step_default": NUM_STEP,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


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
    instruct, clone = resolve_voice(voice, instructions, voices_map)
    started = time.perf_counter()

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
        if LOG_API:
            try:
                save_record(
                    {
                        "source": "api",
                        "text": text,
                        "voice": voice,
                        "voice_mode": "clone" if clone is not None else "design",
                        "instruct": instruct,
                        "speed": speed,
                        "num_step": num_step,
                        "format": fmt,
                        "gen_params": gen_overrides,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        "error": str(error),
                    },
                    None,
                    fmt,
                )
            except Exception:  # noqa: BLE001
                log.warning("failed to persist error record")
        return JSONResponse({"error": {"message": f"generation failed: {error}"}}, status_code=500)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info("tts ok: %d chars, voice=%r, instruct=%r, %d bytes %s", len(text), voice, instruct, len(payload), fmt)

    if LOG_API:
        try:
            save_record(
                {
                    "source": "api",
                    "text": text,
                    "voice": voice,
                    "voice_mode": "clone" if clone is not None else "design",
                    "instruct": instruct,
                    "speed": speed,
                    "num_step": num_step,
                    "format": fmt,
                    "gen_params": gen_overrides,
                    "elapsed_ms": elapsed_ms,
                },
                payload,
                fmt,
            )
        except Exception:  # noqa: BLE001
            log.warning("failed to persist api record")
    return Response(content=payload, media_type=media_type)


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
    return {
        "items": records[: max(1, min(limit, 500))],
        "total": len(records),
        "ok": sum(1 for r in records if r.get("status") == "ok"),
        "failed": sum(1 for r in records if r.get("status") != "ok"),
        "bytes_total": sum(int(r.get("bytes") or 0) for r in records),
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
    ref_text: str,
    source: str,
    raw: bytes | None = None,
    preprocess_prompt: bool = True,
) -> dict[str, Any]:
    """Encode a voice-clone prompt and persist prompt + reference audio + meta.

    `preprocess_prompt` (OmniVoice default True) trims silence in the reference
    audio and appends a sentence-final punctuation to `ref_text` when missing.
    Turn it off only when the raw clip must be embedded verbatim — e.g. the
    transcript is already punctuated and the silence carries meaning.
    """
    import torch  # noqa: PLC0415

    model = get_model()
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=(torch.from_numpy(wave).unsqueeze(0), int(sr)),
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )
    except Exception as error:  # noqa: BLE001
        log.exception("voice clone prompt failed")
        raise HTTPException(status_code=500, detail=f"克隆失败：{error}")

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
                "ref_text": ref_text,
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
        "voice cloned: %s (%.1fs, %d chars ref_text, preprocess_prompt=%s)",
        name,
        duration,
        len(ref_text),
        preprocess_prompt,
    )
    return {"ok": True, "name": name, "duration": round(duration, 2)}


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
      empty `ref_text` is an error here unless the page already filled it in
      from a same-named file the user selected alongside the audio.

    `ref_text` stays mandatory in effect: leaving it truly empty makes the
    model load its Whisper ASR (~3 GB) to transcribe, and this host has no
    commit memory to spare.

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

    if not ref_text:
        if not source_path:
            raise HTTPException(
                status_code=400,
                detail="ref_text 必填：浏览器上传拿不到同目录的同名文本，请手填逐字稿，"
                "或改用「本地音频路径」（服务端直接读音频同目录的 .txt/.lab/.json）。"
                "留空会触发 ASR 自动转写，多占约 3GB 内存。",
            )
        ref_text = _sidecar_text(resolved)
        if not ref_text:
            raise HTTPException(
                status_code=400,
                detail=f"音频同目录没有与「{resolved.stem}」同名的 "
                f".txt/.lab/.json：{resolved.parent}",
            )

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
) -> dict[str, Any]:
    """Register every clip in a local directory.

    Reads the audio off disk and takes each transcript from the **same
    directory** (`<stem>.txt` / `.lab` / `.json`) — nothing is searched for.
    Clips without a sibling transcript are skipped, never auto-transcribed.

    `path`       file or directory on the server host
    `prefix`     prepended to each stem to build the voice name
    `limit`      max clips to import from a directory (sorted by name)
    `preprocess_prompt`  "0" keeps reference audio verbatim (no silence trim)
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
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for f in files:
        text = _sidecar_text(f)
        if not text:
            skipped.append({"file": f.name, "reason": "同目录没有同名 .txt/.lab/.json"})
            continue
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
    use_instr = bool(body.get("use_instructions", USE_INSTRUCTIONS))

    try:
        gen = _parse_gen_params(body, text)
    except HTTPException as error:
        return JSONResponse({"status": "error", "error": str(error.detail)}, status_code=400)
    gen_overrides = _gen_overrides(gen)
    speed = gen.pop("speed")
    num_step = gen.pop("num_step")

    instruct, clone = resolve_voice(voice, instructions, load_voices(), use_instr)
    fields: dict[str, Any] = {
        "source": "admin",
        "text": text,
        "voice": voice,
        "voice_mode": "clone" if clone is not None else "design",
        "instructions": instructions,
        "instruct": instruct,
        "use_instructions": use_instr,
        "speed": speed,
        "num_step": num_step,
        "format": fmt,
        "gen_params": gen_overrides,
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


def _tray_open_admin() -> None:
    try:
        webbrowser.open(f"http://127.0.0.1:{int(os.environ.get('OMNI_TTS_PORT', '8890'))}/admin")
    except Exception as error:  # noqa: BLE001
        log.warning("open admin failed: %s", error)


def _tray_copy_endpoint() -> None:
    url = f"http://127.0.0.1:{int(os.environ.get('OMNI_TTS_PORT', '8890'))}/v1/audio/speech"
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
        log.info("endpoint copied to clipboard: %s", url)
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
                    user32.AppendMenuW(menu, _MF_STRING, _ID_COPY, "复制 OpenAI 端点")
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
    try:
        get_model()
    except Exception as error:  # noqa: BLE001
        log.error("model load failed: %s", error)
        sys.exit(1)
    log.info("listening on http://%s:%d", host, port)
    log.info("admin console: http://%s:%d/admin", host, port)
    log.info("history archive: %s", HISTORY_DIR)
    log.info("cloned voices: %s (%d stored)", CLONE_DIR, len(clone_names()))
    start_tray()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
