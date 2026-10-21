#!/usr/bin/env python3
# Download k2-fsa/OmniVoice real files from hf-mirror resolve API into a local dir,
# so OmniVoice.from_pretrained can load from disk without the Xet/CAS 401 problem.
#
# Uses `curl` via subprocess: curl honours the environment HTTPS_PROXY (the active
# one on this box is 127.0.0.1:65173) and returns the real blob (not an Xet
# pointer), whereas urllib's default UA gets 403 from the mirror.
import os
import subprocess
import sys

REPO = "k2-fsa/OmniVoice"
MIRROR = "https://hf-mirror.com"
OUT = r"K:/OmniVoice/models/OmniVoice"

FILES = [
    ".gitattributes",
    "README.md",
    "audio_tokenizer/.gitattributes",
    "audio_tokenizer/LICENSE",
    "audio_tokenizer/README.md",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def run_curl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # -C - : resume if partial; --retry for transient errors; -f fail on HTTP errors
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "3", "-C", "-",
        "--connect-timeout", "30", "--max-time", "1200",
        "-o", dest, url,
    ]
    return subprocess.run(cmd, env=os.environ)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Downloading {REPO} -> {OUT}", flush=True)
    ok = True
    for f in FILES:
        url = f"{MIRROR}/{REPO}/resolve/main/{f}"
        dest = os.path.join(OUT, f)
        print(f"GET {f}", flush=True)
        r = run_curl(url, dest)
        if r.returncode != 0:
            print(f"  FAILED {f} (curl rc={r.returncode})", flush=True)
            ok = False
        else:
            sz = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  OK {f} ({sz} bytes)", flush=True)
    if ok:
        print("ALL FILES DOWNLOADED OK", flush=True)
    else:
        print("SOME FILES FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
