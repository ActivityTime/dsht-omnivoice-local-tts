@echo off
rem OmniVoice OpenAI-compatible TTS launcher.
rem Every path is derived from this script's own location; nothing is hard-coded.
setlocal
cd /d "%~dp0"

set "HF_HOME=%~dp0hf"
set "HUGGINGFACE_HUB_CACHE=%~dp0hf\hub"
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"

rem bfloat16 = lossless w.r.t. fp16 quality-wise for this model, ~2GB VRAM.
if not defined OMNIVOICE_DTYPE set "OMNIVOICE_DTYPE=bfloat16"
if not defined OMNIVOICE_MODEL set "OMNIVOICE_MODEL=%~dp0models\OmniVoice"
if not defined OMNIVOICE_DEVICE set "OMNIVOICE_DEVICE=cuda:0"
if not defined OMNI_TTS_PORT set "OMNI_TTS_PORT=8890"
if not defined OMNI_TTS_HOST set "OMNI_TTS_HOST=127.0.0.1"

"%~dp0python\python.exe" "%~dp0server.py"
pause
