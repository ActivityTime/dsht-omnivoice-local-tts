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
import sys
import threading
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
from fastapi.responses import JSONResponse, Response  # noqa: E402
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


def pick_instruct(voice: str, instructions: str, voices: dict[str, str]) -> str:
    key = (voice or "").strip()
    instruct = voices.get(key, voices.get(key.lower(), ""))
    if not instruct:
        # Not one of the known names: treat the value itself as a design string,
        # so "female, elderly, 四川话" works straight from the voice field.
        instruct = key
    if USE_INSTRUCTIONS and instructions:
        instruct = f"{instruct}, {instructions}" if instruct else instructions
    return instruct.strip(" ,，")


app = FastAPI(title="OmniVoice OpenAI-compatible TTS", version="1.0.0")


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
    model = get_model()

    with _state["lock"]:
        try:
            chunks = model.generate(
                text=text,
                instruct=instruct or None,
                speed=speed,
                num_step=NUM_STEP,
            )
        except Exception as error:  # noqa: BLE001
            log.exception("generation failed")
            return JSONResponse({"error": {"message": f"generation failed: {error}"}}, status_code=500)

    if not chunks:
        return JSONResponse({"error": {"message": "model returned no audio"}}, status_code=500)

    samples = chunks[0] if isinstance(chunks, (list, tuple)) else chunks
    media_type, encoder = AUDIO_FORMATS[fmt]
    payload = bytes(encoder(np.asarray(samples, dtype=np.float32)))
    log.info("tts ok: %d chars, voice=%r, instruct=%r, %d bytes %s", len(text), voice, instruct, len(payload), fmt)
    return Response(content=payload, media_type=media_type)


def main() -> None:
    port = int(os.environ.get("OMNI_TTS_PORT", "8890"))
    host = os.environ.get("OMNI_TTS_HOST", "127.0.0.1")
    try:
        get_model()
    except Exception as error:  # noqa: BLE001
        log.error("model load failed: %s", error)
        sys.exit(1)
    log.info("listening on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
