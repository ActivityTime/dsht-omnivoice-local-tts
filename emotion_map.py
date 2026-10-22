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

A table can also be kept as a named preset. Two are built in -- one that
aims everything at the finetune's delivery tags, one that aims at the
official symbols and voice attributes -- and any number of user presets can
be saved next to them in `emotion_presets.json`, then imported into the
active table wholesale or on top of it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

MAP_FILENAME = "emotion_map.json"
PRESET_FILENAME = "emotion_presets.json"

# The two families of inline tags, and why they are treated differently.
#
# The 13 official symbols are *non-verbal events* -- laughing, sighing, an
# "ah?" of surprise. They are loud: tagging a line with them makes the
# model produce the sound itself, which swamps the speaker's voice. The
# finetune's eight are delivery and emotion (happy, sad, whisper, calm…)
# and leave the timbre alone.
#
# Both families are therefore allowed by default: which one a phrase lands
# on is the author's call, made by choosing a preset. Setting
# `OMNI_TTS_EMOTION_MAP_OFFICIAL=0` keeps the official symbols and the
# voice-design attributes out of play, leaving only the finetune tags --
# useful when the phrasings really are meant as moods rather than sounds.
OFFICIAL_INLINE_TAGS: tuple[str, ...] = (
    "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
    "question-oh", "question-ei", "question-yi", "surprise-ah",
    "surprise-oh", "surprise-wa", "surprise-yo", "dissatisfaction-hnn",
)
FINETUNE_INLINE_TAGS: tuple[str, ...] = (
    "singing", "happy", "sad", "angry", "nervous", "whisper", "calm", "excited",
)


def official_enabled() -> bool:
    """Whether mappings may point at official symbols and attributes.

    On by default -- the timbre trouble that made the official symbols look
    harmful turned out to be a clone name being eaten by phrase matching,
    not the symbols themselves. Turning it off restricts a table to the
    finetune's delivery tags.
    """
    raw = str(os.environ.get("OMNI_TTS_EMOTION_MAP_OFFICIAL", "1")).strip().lower()
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


# The finetune-side table: everything -- DSH-Tavern's 28 emotions above all
# -- aimed at the finetune's delivery tags, which colour the line without
# replacing the speaker's voice. Entries are one target per phrase: a phrase
# cannot carry both a quiet tag and a loud one.
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
    # --- 事件型短语在微调侧的近似 ---
    # These three are events rather than moods, and the finetune has no
    # event tag, so the preset aims them at the nearest delivery. The
    # built-in default table still reads them as events -- OFFICIAL_MAPPING
    # wins there -- which is why they live here rather than there.
    "不满": ["angry"],
    "惊讶": ["excited"],
    "恍然": ["calm"],
    # --- singing ---
    "唱歌": ["singing"],
    "哼唱": ["singing"],
    "歌唱": ["singing"],
}

# The event-shaped phrases: these name a sound rather than a mood ("惊讶"
# is an "ah!", not a feeling), so they belong on the official symbols and
# are kept in a group of their own. Single-character entries are avoided:
# "笑" would also bite into clone names like "..._欢笑_疑问询问".
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


# The same phrases aimed at the other family: the official symbols, plus
# `whisper` (a voice-design attribute). The official symbols carry no
# emotion axis at all -- they are events -- so every mapping here is an
# approximation, "the sound this mood most often makes". That is the point
# of keeping it as a separate preset: pick it when you want the line to
# *sound*, and the finetune preset when you want it to *feel*.
#
# `singing` is the one finetune tag kept here: nothing among the official
# symbols can stand in for it.
OFFICIAL_PRESET: dict[str, list[str]] = {
    # --- quiet / intimate: `whisper` is a voice-design attribute ---
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
    "关心": ["whisper"],
    "渴望": ["whisper"],
    "害羞": ["whisper"],
    "爱慕": ["whisper"],
    # --- 正面 → 笑声 ---
    "开心": ["laughter"],
    "高兴": ["laughter"],
    "愉快": ["laughter"],
    "喜悦": ["laughter"],
    "钦佩": ["laughter"],
    "逗乐": ["laughter"],
    "赞同": ["laughter"],
    "感激": ["laughter"],
    "得意": ["laughter"],
    "兴奋": ["laughter"],
    "激动": ["laughter"],
    "雀跃": ["laughter"],
    "轻笑": ["laughter"],
    "偷笑": ["laughter"],
    "笑声": ["laughter"],
    # --- 低落 / 紧绷 → 叹气 ---
    "难过": ["sigh"],
    "伤心": ["sigh"],
    "悲伤": ["sigh"],
    "低落": ["sigh"],
    "沮丧": ["sigh"],
    "失望": ["sigh"],
    "悲痛": ["sigh"],
    "懊悔": ["sigh"],
    "紧张": ["sigh"],
    "不安": ["sigh"],
    "焦虑": ["sigh"],
    "忐忑": ["sigh"],
    "叹气": ["sigh"],
    "叹息": ["sigh"],
    # --- 负面强 → 不满的哼声 ---
    "生气": ["dissatisfaction-hnn"],
    "愤怒": ["dissatisfaction-hnn"],
    "恼怒": ["dissatisfaction-hnn"],
    "发火": ["dissatisfaction-hnn"],
    "火大": ["dissatisfaction-hnn"],
    "烦躁": ["dissatisfaction-hnn"],
    "嫌弃": ["dissatisfaction-hnn"],
    "不满": ["dissatisfaction-hnn"],
    "不屑": ["dissatisfaction-hnn"],
    # --- 惊 / 疑 ---
    "惊讶": ["surprise-ah"],
    "驚訝": ["surprise-ah"],
    "惊呼": ["surprise-ah"],
    "害怕": ["surprise-ah"],
    "恍然": ["surprise-oh"],
    "好奇": ["surprise-oh"],
    "期待": ["surprise-oh"],
    "疑问": ["question-ah"],
    "疑惑": ["question-ah"],
    "不解": ["question-ah"],
    "困惑": ["question-ah"],
    "确认": ["confirmation-en"],
    "肯定": ["confirmation-en"],
    # --- 平缓 → 低缓柔和 ---
    "平静": ["whisper"],
    "冷静": ["whisper"],
    "淡然": ["whisper"],
    "从容": ["whisper"],
    "释然": ["whisper"],
    # --- 唱 ---
    "唱歌": ["singing"],
    "哼唱": ["singing"],
    "歌唱": ["singing"],
}


def default_table() -> dict[str, list[str]]:
    """The built-in table: both families, finetune tags first.

    A phrase that appears in both is kept at its finetune target, which is
    the delivery reading; import the official preset to flip a whole table
    over to the event reading.
    """
    table = {key: list(value) for key, value in OFFICIAL_PRESET.items()}
    table.update({key: list(value) for key, value in DEFAULT_MAPPING.items()})
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


def save_map(table: dict[str, list[str]], source: str | None = None) -> None:
    """Store the active table, optionally naming the preset it came from.

    Any later edit drops the name: `source` records provenance, and a table
    that has been edited by hand is no longer that preset.
    """
    path = map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": (str(source).strip() or None) if source else None,
        "mappings": {
            key: _normalise(value) for key, value in sorted(table.items())
            if str(key).strip() and _normalise(value)
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_meta() -> tuple[dict[str, list[str]], str | None]:
    """The active table together with the preset it was imported from."""
    path = map_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict):
            source = data.get("source")
            source = str(source).strip() if isinstance(source, str) and source.strip() else None
            return load_map(), source
    return load_map(), None


def ensure_map() -> dict[str, list[str]]:
    """Write the default table out on first run so it can be edited."""
    path = map_path()
    if not path.exists():
        save_map(default_table(), DEFAULT_SOURCE)
        return load_map()
    # Files written before presets existed carry no "source". If one still
    # holds the built-in default untouched, label it, so the console can say
    # where the table came from instead of calling everything custom.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return load_map()
    if isinstance(data, dict) and "source" not in data:
        table = load_map()
        if table == default_table():
            save_map(table, DEFAULT_SOURCE)
    return load_map()


# --- presets ---------------------------------------------------------------
#
# A preset is a named table. Two ship with the service and are read-only --
# their names belong to the code, so a user preset cannot shadow or delete
# one. Everything else is stored in `emotion_presets.json`, next to the
# active table, and can be created, overwritten and removed freely.
DEFAULT_SOURCE = "内置默认"

BUILTIN_PRESETS: dict[str, dict[str, list[str]]] = {
    "微调映射模板": DEFAULT_MAPPING,
    "官方映射模板": OFFICIAL_PRESET,
}


def presets_path() -> Path:
    return Path(os.environ.get("OMNI_EMOTION_PRESETS", HERE / PRESET_FILENAME))


def _normalise_table(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        phrase = str(key).strip()
        targets = _normalise(value)
        if phrase and targets:
            out[phrase] = targets
    return out


def load_preset_file() -> dict[str, dict[str, list[str]]]:
    """The user presets only, as stored on disk."""
    path = presets_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get("presets")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, list[str]]] = {}
    for name, entry in raw.items():
        key = str(name).strip()
        if not key or key in BUILTIN_PRESETS:
            continue
        body = entry.get("mappings") if isinstance(entry, dict) else entry
        out[key] = _normalise_table(body)
    return out


def _write_preset_file(store: dict[str, dict[str, list[str]]]) -> None:
    path = presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "presets": {
            name: {"mappings": {k: _normalise(v) for k, v in sorted(table.items())}}
            for name, table in sorted(store.items())
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_presets() -> dict[str, dict[str, list[str]]]:
    """Every preset, built-in and user, by name."""
    merged: dict[str, dict[str, list[str]]] = {
        name: {key: list(value) for key, value in table.items()}
        for name, table in BUILTIN_PRESETS.items()
    }
    merged.update(load_preset_file())
    return merged


def list_presets() -> list[dict[str, Any]]:
    """Preset names for the console, built-ins first."""
    presets = load_presets()
    return [
        {"name": name, "builtin": name in BUILTIN_PRESETS, "count": len(table)}
        for name, table in sorted(
            presets.items(), key=lambda kv: (kv[0] not in BUILTIN_PRESETS, kv[0])
        )
    ]


def get_preset(name: str) -> dict[str, list[str]] | None:
    table = load_presets().get(str(name or "").strip())
    return {key: list(value) for key, value in table.items()} if table is not None else None


def save_preset(name: str, table: dict[str, list[str]]) -> int:
    """Store a table as a user preset. Built-in names are off limits."""
    key = str(name or "").strip()
    if not key:
        raise ValueError("模板名不能为空")
    if key in BUILTIN_PRESETS:
        raise ValueError(f"「{key}」是内置模板，不能覆盖；请换一个名字")
    store = load_preset_file()
    store[key] = _normalise_table(table)
    _write_preset_file(store)
    return len(store[key])


def delete_preset(name: str) -> bool:
    key = str(name or "").strip()
    if key in BUILTIN_PRESETS:
        raise ValueError(f"「{key}」是内置模板，不能删除")
    store = load_preset_file()
    if key not in store:
        return False
    del store[key]
    _write_preset_file(store)
    return True


def apply_preset(
    current: dict[str, list[str]], name: str, mode: str = "replace"
) -> dict[str, list[str]]:
    """Import a preset into a table: `replace` swaps it, `merge` layers it.

    Merging overwrites the phrases the preset knows and leaves everything
    else alone, so several presets can be combined.
    """
    table = get_preset(name)
    if table is None:
        raise ValueError(f"没有名为「{name}」的模板")
    if str(mode).strip().lower() == "merge":
        merged = {key: list(value) for key, value in current.items()}
        merged.update({key: list(value) for key, value in table.items()})
        return merged
    return {key: list(value) for key, value in table.items()}


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
