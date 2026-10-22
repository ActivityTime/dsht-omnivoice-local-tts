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
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

MAP_FILENAME = "emotion_map.json"

# The two families of inline tags, and why they are treated differently.
#
# The 13 official symbols are *non-verbal events* -- laughing, sighing, an
# "ah?" of surprise. They are loud: tagging a line with them makes the
# model produce the sound itself, which swamps the speaker's voice. The
# finetune's eight are delivery and emotion (happy, sad, whisper, calm…)
# and leave the timbre alone.
#
# So the default policy maps onto the finetune tags only, and treats the
# official symbols -- and voice-design attributes, which change the voice
# itself -- as blocked. `OMNI_TTS_EMOTION_MAP_OFFICIAL=1` lifts that.
OFFICIAL_INLINE_TAGS: tuple[str, ...] = (
    "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
    "question-oh", "question-ei", "question-yi", "surprise-ah",
    "surprise-oh", "surprise-wa", "surprise-yo", "dissatisfaction-hnn",
)
FINETUNE_INLINE_TAGS: tuple[str, ...] = (
    "singing", "happy", "sad", "angry", "nervous", "whisper", "calm", "excited",
)


def official_enabled() -> bool:
    """Whether mappings may point at official symbols and attributes."""
    raw = str(os.environ.get("OMNI_TTS_EMOTION_MAP_OFFICIAL", "0")).strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def blocked_targets(valid_design: frozenset[str]) -> frozenset[str]:
    """Targets the current policy refuses: attributes, plus official tags.

    The finetune's own tags are never blocked, even when a name is also a
    voice-design attribute: `whisper` is an attribute on the base
    checkpoint *and* one of the finetune's delivery tags, and blocking it
    would take away the quiet half of the table for no good reason.
    """
    blocked = set(valid_design) - set(FINETUNE_INLINE_TAGS)
    if not official_enabled():
        blocked |= set(OFFICIAL_INLINE_TAGS)
    return frozenset(blocked)

# DSH-Tavern does not send a bare emotion word: its "OpenAI compatible"
# channel wraps the annotated emotion into one fixed sentence, built by
# `emotionStyle()` in the tavern plugin as
#   "用" + <Chinese emotion> + "的语气说这句话。"
# Without stripping that scaffolding the emotion is still matched (matching
# is by substring) but the leftover shell -- "用 的语气说这句话。" -- goes on
# to attribute parsing, is not an attribute, and is archived as dropped
# noise on every single voiced line.
_DSH_STYLE_SENTENCE = re.compile(
    r"^\s*用\s*(?P<phrase>[^，,。；;！!？?\s]{1,12}?)\s*的?\s*语气\s*"
    r"(?:说|讲|念|朗读|读)?\s*(?:这个?这句话?|这句话|这句|话)?\s*[。.！!？?]?\s*$"
)


def strip_style_sentence(text: str) -> str:
    """Reduce DSH-Tavern's "用X的语气说这句话。" to just the emotion word.

    Anything that is not that exact shape is returned untouched, so a
    caller's own wording is never mangled.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = _DSH_STYLE_SENTENCE.match(raw)
    return match.group("phrase") if match else raw


def map_path() -> Path:
    return Path(os.environ.get("OMNI_EMOTION_MAP", HERE / MAP_FILENAME))


# The default table, used when no file exists yet.
#
# Everything a role-play front end sends -- DSH-Tavern's 28 emotions above
# all -- is mapped onto the finetune's delivery tags, which colour the line
# without replacing the speaker's voice. Entries are one target per phrase:
# a phrase cannot carry both a quiet tag and a loud one.
DEFAULT_MAPPING: dict[str, list[str]] = {
    # --- delivery: quiet / intimate (finetune tag `whisper`) ---
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
    # DSHT: 关心 / 渴望 / 害羞 / 爱慕
    "关心": ["whisper"],
    "渴望": ["whisper"],
    "害羞": ["whisper"],
    "爱慕": ["whisper"],
    # --- emotion (finetune only) ---
    "开心": ["happy"],
    "高兴": ["happy"],
    "愉快": ["happy"],
    "喜悦": ["happy"],
    # DSHT: 钦佩 / 逗乐 / 赞同 / 感激 / 得意
    "钦佩": ["happy"],
    "逗乐": ["happy"],
    "赞同": ["happy"],
    "感激": ["happy"],
    "得意": ["happy"],
    "兴奋": ["excited"],
    "激动": ["excited"],
    "雀跃": ["excited"],
    # DSHT: 好奇 / 期待
    "好奇": ["excited"],
    "期待": ["excited"],
    "难过": ["sad"],
    "伤心": ["sad"],
    "悲伤": ["sad"],
    "低落": ["sad"],
    "沮丧": ["sad"],
    # DSHT: 失望 / 悲痛 / 懊悔
    "失望": ["sad"],
    "悲痛": ["sad"],
    "懊悔": ["sad"],
    "生气": ["angry"],
    "愤怒": ["angry"],
    "恼怒": ["angry"],
    "发火": ["angry"],
    "火大": ["angry"],
    # DSHT: 烦躁 / 嫌弃
    "烦躁": ["angry"],
    "嫌弃": ["angry"],
    "紧张": ["nervous"],
    "不安": ["nervous"],
    "焦虑": ["nervous"],
    "忐忑": ["nervous"],
    # DSHT: 害怕 / 困惑
    "害怕": ["nervous"],
    "困惑": ["nervous"],
    "平静": ["calm"],
    "冷静": ["calm"],
    "淡然": ["calm"],
    "从容": ["calm"],
    # DSHT: 释然
    "释然": ["calm"],
    # --- singing ---
    "唱歌": ["singing"],
    "哼唱": ["singing"],
    "歌唱": ["singing"],
}

# The loud ones: non-verbal events that take the voice over, kept out of
# the default table on purpose and only merged in when
# OMNI_TTS_EMOTION_MAP_OFFICIAL is on. Several name an event rather than a
# delivery ("惊讶" is an "ah!", not a mood), which is exactly why they are
# not what a voiced line wants by default.
#
# Single-character entries are avoided: "笑" would also bite into clone
# names like "..._欢笑_疑问询问".
OFFICIAL_MAPPING: dict[str, list[str]] = {
    "轻笑": ["laughter"],
    "偷笑": ["laughter"],
    "笑声": ["laughter"],
    "叹气": ["sigh"],
    "叹息": ["sigh"],
    "惊呼": ["surprise-ah"],
    "惊讶": ["surprise-ah"],
    "驚訝": ["surprise-ah"],
    "恍然": ["surprise-oh"],
    "疑问": ["question-ah"],
    "疑惑": ["question-ah"],
    "不解": ["question-ah"],
    "确认": ["confirmation-en"],
    "肯定": ["confirmation-en"],
    "不屑": ["dissatisfaction-hnn"],
    "不满": ["dissatisfaction-hnn"],
}


def default_table() -> dict[str, list[str]]:
    """The built-in table: the quiet tags and the loud ones.

    The loud entries are written out even though the default policy blocks
    them, so that turning `OMNI_TTS_EMOTION_MAP_OFFICIAL` on has something
    to enable without a reset; the console marks them as blocked.
    """
    table = {key: list(value) for key, value in DEFAULT_MAPPING.items()}
    table.update({key: list(value) for key, value in OFFICIAL_MAPPING.items()})
    return table


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
    return default_table()


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
        save_map(default_table())
    return load_map()


def _mask_protected(text: str, protected: tuple[str, ...]) -> str:
    """Blank out the spans that belong to names, keeping every index.

    A clone called `ja_Kotori_0015_欢笑_疑问询问` contains 欢笑 and 疑问, and
    the table knows 疑问. Matching inside a name would eat the name, so
    anything the caller has already identified is punched out first. The
    replacement is the same length, so the offsets still line up with the
    original string.
    """
    masked = text
    for name in sorted({str(item) for item in protected if str(item)}, key=len, reverse=True):
        needle = name.lower()
        start = masked.lower().find(needle)
        while start != -1:
            masked = masked[:start] + "\u0000" * len(name) + masked[start + len(name) :]
            start = masked.lower().find(needle)
    return masked


def extract(
    text: str,
    table: dict[str, list[str]],
    inline_tags: tuple[str, ...],
    valid_design: frozenset[str],
    protected: tuple[str, ...] = (),
    blocked: frozenset[str] = frozenset(),
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str]]]:
    """Pull the mapped emotion phrases out of a free-form string.

    Returns `(residual_text, inline_tags, design_items, unmatched, mapped)`.
    The residual is what is left after the matched phrases are taken out,
    and it is what should go through normal attribute parsing. `unmatched`
    holds phrases the table knows but the loaded checkpoint cannot express
    or the current policy blocks; `mapped` records what was translated and
    into what.

    `protected` names spans that must not be matched into -- clone names,
    presets, tags and attributes already recognised elsewhere. `blocked`
    names targets the policy refuses (attributes, and the official symbols
    unless enabled).
    """
    if not text or not table:
        return text or "", [], [], [], []

    # DSH-Tavern's "用X的语气说这句话。" carries one emotion and a lot of
    # scaffolding; take the scaffolding off before matching so it does not
    # end up in the dropped report.
    residual = strip_style_sentence(text)
    if not residual:
        return "", [], [], [], []

    # Longest phrase first, so "温柔地" wins over "温柔" when both are
    # present and they map to different things.
    keys = sorted(table, key=len, reverse=True)
    # Match against the masked copy so a phrase can never land inside a
    # name; the offsets are the same, so the cuts below hit the real text.
    masked = _mask_protected(residual, protected)
    inline: list[str] = []
    design: list[str] = []
    unmatched: list[str] = []
    mapped: list[dict[str, str]] = []
    claimed: list[tuple[str, int, int]] = []

    for key in keys:
        if not key:
            continue
        start = masked.find(key)
        if start < 0:
            continue
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
            if not name or name in blocked:
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
    blocked: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """The table as the console shows it, with per-target availability."""
    out: list[dict[str, Any]] = []
    for key in sorted(table, key=lambda k: (-len(k), k)):
        targets = table[key]
        rows = []
        for target in targets:
            name = str(target).strip().lower()
            if name in blocked:
                kind, available = "blocked", False
            elif name in inline_tags:
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
