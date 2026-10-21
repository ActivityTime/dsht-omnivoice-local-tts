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

import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
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

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
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
# Requests arriving on the OpenAI endpoint are only logged when enabled: DSH
# Tavern can fire one per message bubble and would fill the disk.
LOG_API = os.environ.get("OMNI_TTS_LOG_API", "0") == "1"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SAMPLE_RATE = 24000
MAX_TEXT = 4000

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


AUDIO_FORMATS = {
    "mp3": ("audio/mpeg", to_mp3),
    "wav": ("audio/wav", to_wav),
    "pcm": ("audio/wav", to_wav),
    "flac": ("audio/wav", to_wav),
    "opus": ("audio/wav", to_wav),
    "aac": ("audio/wav", to_wav),
}


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


def run_generate(
    *, text: str, instruct: str, speed: float, num_step: int, fmt: str
) -> tuple[bytes, str]:
    """Shared decode path for the OpenAI endpoint and the admin console."""
    model = get_model()
    with _state["lock"]:
        chunks = model.generate(
            text=text, instruct=instruct or None, speed=speed, num_step=num_step
        )
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
        fmt = "mp3"

    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0

    instruct = pick_instruct(voice, instructions, load_voices())
    started = time.perf_counter()

    try:
        payload, media_type = run_generate(
            text=text, instruct=instruct, speed=speed, num_step=NUM_STEP, fmt=fmt
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
                        "instruct": instruct,
                        "speed": speed,
                        "num_step": NUM_STEP,
                        "format": fmt,
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
                    "instruct": instruct,
                    "speed": speed,
                    "num_step": NUM_STEP,
                    "format": fmt,
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
        fmt = "mp3"
    use_instr = bool(body.get("use_instructions", USE_INSTRUCTIONS))

    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.25, min(4.0, speed))
    try:
        num_step = int(body.get("num_step") or NUM_STEP)
    except (TypeError, ValueError):
        num_step = NUM_STEP
    num_step = max(1, min(100, num_step))

    instruct = pick_instruct(voice, instructions, load_voices(), use_instr)
    fields: dict[str, Any] = {
        "source": "admin",
        "text": text,
        "voice": voice,
        "instructions": instructions,
        "instruct": instruct,
        "use_instructions": use_instr,
        "speed": speed,
        "num_step": num_step,
        "format": fmt,
    }

    started = time.perf_counter()
    try:
        payload, _ = run_generate(
            text=text, instruct=instruct, speed=speed, num_step=num_step, fmt=fmt
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
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
