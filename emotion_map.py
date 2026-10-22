"""Adapter from DSH-Tavern's emotion phrases to OmniVoice's own tags.

DSH-Tavern sends an emotion phrase in the `instructions` field -- things
like "温柔地" or "生气". OmniVoice's voice-design vocabulary has no emotion
axis at all, so those phrases used to be dropped: they are not attributes,
and the upstream validator raises on anything it does not recognise.

This module is the missing layer. It maps a phrase onto a tag the loaded
checkpoint actually understands:

  - a phrase that maps to one of the inline tags becomes an inline tag
    and is prepended to the text;
  - a phrase that maps to a voice-design item becomes an attribute;
  - a phrase that maps to neither is dropped and reported.

Which of the two happens is decided by the loaded checkpoint, so the same
mapping works on both: `whisper` is an inline tag on the emotion finetune
and a design attribute on the base checkpoint, and the adapter picks
whichever the current model understands.

The mapping table lives in `emotion_map.json` and is editable from the
console. Matching is by longest substring, so "用温柔的语气说话" hits the
"温柔" entry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

MAP_FILENAME = "emotion_map.json"


def map_path() -> Path:
    return Path(os.environ.get("OMNI_EMOTION_MAP", HERE / MAP_FILENAME))


# The default table, used when no file exists yet. These are the emotion
# words a role-play front end tends to send, mapped to the closest thing
# OmniVoice can express.
#
# Note which entries only work on the emotion finetune: the base checkpoint
# has no emotion axis, so "happy" and friends can only be honoured when
# that model is loaded. "whisper" and the 13 upstream symbols work on both.
DEFAULT_MAPPING: dict[str, list[str]] = {
    # --- voice quality (available on every checkpoint) ---
    "耳语": ["whisper"],
    "耳語": ["whisper"],
    "轻声": ["whisper"],
    "轻声道": ["whisper"],
    "温柔": ["whisper"],
    "温柔地": ["whisper"],
    "溫柔": ["whisper"],
    "柔和": ["whisper"],
    "轻柔": ["whisper"],
    "小声": ["whisper"],
    "慵懒": ["whisper"],
    "懒洋洋": ["whisper"],
    # --- non-verbal sounds (available on every checkpoint) ---
    "笑": ["laughter"],
    "轻笑": ["laughter"],
    "偷笑": ["laughter"],
    "笑声": ["laughter"],
    "叹气": ["sigh"],
    "叹息": ["sigh"],
    "叹息": ["sigh"],
    "惊呼": ["surprise-ah"],
    "惊讶": ["surprise-ah"],
    "驚訝": ["surprise-ah"],
    "疑问": ["question-ah"],
    "疑惑": ["question-ah"],
    "不解": ["question-ah"],
    "确认": ["confirmation-en"],
    "肯定": ["confirmation-en"],
    "不屑": ["dissatisfaction-hnn"],
    "不满": ["dissatisfaction-hnn"],
    # --- emotions (need the emotion finetune) ---
    "开心": ["happy"],
    "高兴": ["happy"],
    "愉快": ["happy"],
    "喜悦": ["happy"],
    "兴奋": ["excited"],
    "激动": ["excited"],
    "雀跃": ["excited"],
    "难过": ["sad"],
    "伤心": ["sad"],
    "悲伤": ["sad"],
    "低落": ["sad"],
    "沮丧": ["sad"],
    "生气": ["angry"],
    "愤怒": ["angry"],
    "恼怒": ["angry"],
    "发火": ["angry"],
    "火大": ["angry"],
    "紧张": ["nervous"],
    "不安": ["nervous"],
    "焦虑": ["nervous"],
    "忐忑": ["nervous"],
    "平静": ["calm"],
    "冷静": ["calm"],
    "淡然": ["calm"],
    "从容": ["calm"],
    "唱歌": ["singing"],
    "哼唱": ["singing"],
    "歌唱": ["singing"],
}


def _normalise(entry: Any) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, (list, tuple)):
        return [str(item).strip() for item in entry if str(item).strip()]
    return []


def load_map() -> dict[str, list[str]]:
    """The stored mapping, falling back to the default table."""
    path = map_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict):
            table = data.get("mappings") if isinstance(data.get("mappings"), dict) else data
            out: dict[str, list[str]] = {}
            for key, value in table.items():
                targets = _normalise(value)
                key_text = str(key).strip()
                if key_text and targets:
                    out[key_text] = targets
            if out:
                return out
    return {key: list(value) for key, value in DEFAULT_MAPPING.items()}


def save_map(table: dict[str, list[str]]) -> None:
    path = map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mappings": {
            key: _normalise(value) for key, value in sorted(table.items())
            if str(key).strip() and _normalise(value)
        }
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def ensure_map() -> dict[str, list[str]]:
    """Write the default table out on first run so it can be edited."""
    if not map_path().exists():
        save_map({key: list(value) for key, value in DEFAULT_MAPPING.items()})
    return load_map()


def extract(
    text: str,
    table: dict[str, list[str]],
    inline_tags: tuple[str, ...],
    valid_design: frozenset[str],
) -> tuple[str, list[str], list[str], list[str]]:
    """Pull the mapped emotion phrases out of a free-form string.

    Returns `(residual_text, inline_tags, design_items, unmatched, mapped)`.
    The residual is what is left after the matched phrases are taken out,
    and it is what should go through normal attribute parsing. `unmatched`
    holds phrases the table knows but the loaded checkpoint cannot express;
    `mapped` records what was translated and into what.
    """
    if not text or not table:
        return text or "", [], [], []

    # Longest phrase first, so "温柔地" wins over "温柔" when both are
    # present and they map to different things.
    keys = sorted(table, key=len, reverse=True)
    residual = text
    inline: list[str] = []
    design: list[str] = []
    unmatched: list[str] = []
    mapped: list[dict[str, str]] = []
    claimed: list[tuple[str, int, int]] = []

    for key in keys:
        if not key or key not in residual:
            continue
        start = residual.find(key)
        claimed.append((key, start, start + len(key)))

    # Keep the earliest longest matches and drop overlaps, so a phrase is
    # not counted twice.
    claimed.sort(key=lambda item: (item[1], -(item[2] - item[1])))
    kept: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for key, start, end in claimed:
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        occupied.append((start, end))
        kept.append((key, start, end))

    for key, start, end in sorted(kept, key=lambda item: item[1]):
        targets = table.get(key) or []
        resolved: list[str] = []
        for target in targets:
            name = str(target).strip().lower()
            if not name:
                continue
            if name in inline_tags:
                if name not in inline:
                    inline.append(name)
                resolved.append(name)
            elif name in valid_design:
                if name not in design:
                    design.append(name)
                resolved.append(name)
        if resolved:
            mapped.append({"phrase": key, "targets": ",".join(resolved)})
        else:
            # The phrase is understood but the loaded checkpoint cannot
            # express it. Report it instead of failing the request.
            unmatched.append(key)

    # Remove the matched phrases from the text.
    for key, start, end in sorted(kept, key=lambda item: item[1], reverse=True):
        residual = residual[:start] + " " + residual[end:]
    residual = residual.strip(" ,，;；、")
    return residual, inline, design, unmatched, mapped


def describe(
    table: dict[str, list[str]],
    inline_tags: tuple[str, ...],
    valid_design: frozenset[str],
) -> list[dict[str, Any]]:
    """The table as the console shows it, with per-target availability."""
    out: list[dict[str, Any]] = []
    for key in sorted(table, key=lambda k: (-len(k), k)):
        targets = table[key]
        rows = []
        for target in targets:
            name = str(target).strip().lower()
            if name in inline_tags:
                kind, available = "inline", True
            elif name in valid_design:
                kind, available = "design", True
            else:
                kind, available = "unknown", False
            rows.append({"name": name, "kind": kind, "available": available})
        out.append({
            "phrase": key,
            "targets": rows,
            "available": all(r["available"] for r in rows) if rows else False,
        })
    return out


def known_targets(
    inline_tags: tuple[str, ...], valid_design: frozenset[str]
) -> list[dict[str, str]]:
    """Every tag a mapping may point at, for the console's picker."""
    out = [{"name": tag, "kind": "inline"} for tag in inline_tags]
    out += [{"name": tag, "kind": "design"} for tag in sorted(valid_design)]
    return out
