"""Content-addressed cache for world-model calls (the replay backend's store).

Layout (under ``data/cache/worldmodels`` or ``COSMOS_CACHE_DIR``)::

    <fn>/<key>.json            JSON results (assess_clip, chat, explain_signals)
    <fn>/<key>.mp4             media results (what_if, sim2real, ...)
    <fn>/<key>.meta.json       media sidecar: source, latency_ms, prompt, params, ...
    <fn>/alias/<alias>.key     alias -> key (e.g. "same video, any context")
    placeholders/<key>.mp4     replay placeholders (never treated as real hits)

Key = sha256 of a canonical JSON of ``{"fn": fn, "inputs": {...}}`` where every
file input is replaced by the sha256 of its bytes, so moving the data
directory does not invalidate the cache but changing a clip does.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import get_settings

CACHE_VERSION = 1

_FILE_HASH_MEMO: dict[tuple[str, int, int], str] = {}


def cache_root() -> Path:
    return get_settings().cache_dir


def file_sha256(path: str | os.PathLike) -> str:
    p = Path(path)
    st = p.stat()
    memo_key = (str(p.resolve()), st.st_size, int(st.st_mtime_ns))
    hit = _FILE_HASH_MEMO.get(memo_key)
    if hit:
        return hit
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASH_MEMO[memo_key] = digest
    return digest


class FileRef(str):
    """Marks a path whose *content hash* (not its name) goes into the key."""


def _canon(obj: Any) -> Any:
    if isinstance(obj, FileRef):
        return {"__file_sha256__": file_sha256(obj)} if Path(obj).exists() else {"__missing_file__": Path(obj).name}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def make_key(fn: str, inputs: dict) -> str:
    payload = canonical_json({"v": CACHE_VERSION, "fn": fn, "inputs": inputs})
    return hashlib.sha256(payload.encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fn_dir(fn: str) -> Path:
    return cache_root() / re.sub(r"[^A-Za-z0-9_.-]", "_", fn)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def reading_enabled() -> bool:
    return get_settings().cache_mode in ("readwrite", "read")


def writing_enabled() -> bool:
    return get_settings().cache_mode in ("readwrite", "write")


# ------------------------------------------------------------------ JSON results
def get_json(fn: str, key: str) -> Optional[dict]:
    p = _fn_dir(fn) / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def put_json(fn: str, key: str, value: dict, aliases: Iterable[str] = ()) -> Path:
    p = _fn_dir(fn) / f"{key}.json"
    rec = dict(value)
    rec.setdefault("cached_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["cache_key"] = key
    _atomic_write_text(p, json.dumps(rec, indent=2, default=str))
    for a in aliases:
        put_alias(fn, a, key)
    return p


# ------------------------------------------------------------------ media results
def media_path(fn: str, key: str, suffix: str = ".mp4") -> Path:
    return _fn_dir(fn) / f"{key}{suffix}"


def meta_path(fn: str, key: str) -> Path:
    return _fn_dir(fn) / f"{key}.meta.json"


def get_media(fn: str, key: str, suffix: str = ".mp4") -> Optional[tuple[Path, dict]]:
    mp = media_path(fn, key, suffix)
    if not mp.exists() or mp.stat().st_size == 0:
        return None
    meta = {}
    sp = meta_path(fn, key)
    if sp.exists():
        try:
            meta = json.loads(sp.read_text())
        except Exception:
            meta = {}
    return mp, meta


def put_media(fn: str, key: str, src: str | os.PathLike, meta: dict,
              aliases: Iterable[str] = (), suffix: str = ".mp4", move: bool = False) -> Path:
    dst = media_path(fn, key, suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dst.resolve():
        tmp = dst.with_name(f".tmp-{dst.name}")
        (shutil.move if move else shutil.copyfile)(str(src), str(tmp))
        os.replace(tmp, dst)
    rec = dict(meta)
    rec.setdefault("cached_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["cache_key"] = key
    rec["path"] = str(dst)
    _atomic_write_text(meta_path(fn, key), json.dumps(rec, indent=2, default=str))
    for a in aliases:
        put_alias(fn, a, key)
    return dst


# ------------------------------------------------------------------ aliases
def alias_key(*parts: Any) -> str:
    return make_key("alias", {"parts": list(parts)})


def put_alias(fn: str, alias: str, key: str) -> None:
    _atomic_write_text(_fn_dir(fn) / "alias" / f"{alias}.key", key)


def resolve_alias(fn: str, alias: str) -> Optional[str]:
    p = _fn_dir(fn) / "alias" / f"{alias}.key"
    return p.read_text().strip() if p.exists() else None


# ------------------------------------------------------------------ nearest text match
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "to", "in", "on", "with", "is", "it", "what", "if", "show", "me",
         "for", "at", "by", "be", "this", "that", "video", "weld", "welding"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def nearest_media(fn: str, query: str, min_coverage: float = 0.5) -> Optional[tuple[Path, dict, float]]:
    """Best cached media of ``fn`` whose sidecar ``query_text``/``match_text`` covers ``query``.

    Used by replay mode so a live-typed "what if" question that is close to a
    pre-generated showcase prompt still gets a real Cosmos clip (labelled as a
    nearest match). Score = 0.7 * (share of the query's content words found)
    + 0.3 * Jaccard; needs >= 2 shared words and ``min_coverage``.
    """
    q = _tokens(query)
    if not q:
        return None
    best: Optional[tuple[Path, dict, float]] = None
    d = _fn_dir(fn)
    if not d.exists():
        return None
    for sp in d.glob("*.meta.json"):
        try:
            meta = json.loads(sp.read_text())
        except Exception:
            continue
        t = _tokens(f"{meta.get('query_text', '')} {meta.get('match_text', '')}")
        shared = q & t
        if len(shared) < min(2, len(q)) or not t:
            continue
        coverage = len(shared) / len(q)
        if coverage < min_coverage:
            continue
        score = 0.7 * coverage + 0.3 * len(shared) / len(q | t)
        mp = sp.with_name(sp.name.replace(".meta.json", ".mp4"))
        if mp.exists() and (best is None or score > best[2]):
            best = (mp, meta, score)
    return best


# ------------------------------------------------------------------ placeholders / stats
def placeholder_path(key: str, suffix: str = ".mp4") -> Path:
    return cache_root() / "placeholders" / f"{key}{suffix}"


def stats() -> dict:
    root = cache_root()
    out: dict[str, Any] = {"root": str(root), "exists": root.exists(), "functions": {}}
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name == "placeholders":
            out["placeholders"] = len(list(d.glob("*")))
            continue
        n_json = len([p for p in d.glob("*.json") if not p.name.endswith(".meta.json")])
        n_media = len([p for p in d.iterdir() if p.suffix in (".mp4", ".png", ".jpg")])
        out["functions"][d.name] = {"json": n_json, "media": n_media}
    return out
