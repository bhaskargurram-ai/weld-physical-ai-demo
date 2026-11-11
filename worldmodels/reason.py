"""Cosmos 3 Reasoner client: weld QA, WPS compliance, free chat.

Talks OpenAI chat-completions to
  * ``local``: vLLM serving nvidia/Cosmos3-Nano (see worldmodels/serve/reasoner.sh),
    video sent as ``file://`` URL (server runs with ``--allowed-local-media-path /``)
    or as a base64 data URL,
  * ``nim``:   integrate.api.nvidia.com/v1 (model nvidia/cosmos3-nano-reasoner) or a
    self-hosted Cosmos3 NIM, video always as a base64 data URL,
  * ``replay``: cache, then heuristic fallback.

Video sampling follows the Cosmos3 docs: ``media_io_kwargs={"video": {"num_frames": -1, "fps": 4}}``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from . import cache, heuristics
from .config import (SOURCE_CACHE, SOURCE_HEURISTIC, Settings, backend_source, get_settings, is_local_url,
                     resolve_backend)
from .http import BackendError, client, raise_for
from .parsing import AssessmentError, parse_assessment, split_think
from .prompts import (ASSESS_JSON_SCHEMA, CWI_SYSTEM_PROMPT, PROMPT_VERSION, SAMPLING_DIRECT, SAMPLING_REASONING,
                      build_assess_text, build_explain_text, normalize_context)
from .video import data_url

FN_ASSESS = "assess_clip"
FN_CHAT = "chat"
FN_EXPLAIN = "explain_signals"

_MODEL_MEMO: dict[str, str] = {}
_EXTENSION_FIELDS = ("top_k", "repetition_penalty", "media_io_kwargs")


# ------------------------------------------------------------------ endpoint plumbing
def _endpoint(backend: str, s: Settings) -> tuple[str, str, dict]:
    if backend == "nim":
        if not s.nvidia_api_key and not is_local_url(s.nim_reasoner_url):
            raise BackendError("nim backend needs NVIDIA_API_KEY")
        headers = {"Authorization": f"Bearer {s.nvidia_api_key}"} if s.nvidia_api_key else {}
        return s.nim_reasoner_url.rstrip("/"), s.nim_reasoner_model, headers
    return s.reasoner_api, s.reasoner_model, {}


def _resolve_model(api: str, model: str, headers: dict, s: Settings) -> str:
    if model and model != "auto":
        return model
    if api in _MODEL_MEMO:
        return _MODEL_MEMO[api]
    with client(30, headers) as c:
        r = c.get(f"{api}/models")
        raise_for(r, "list models")
        mid = r.json()["data"][0]["id"]
    _MODEL_MEMO[api] = mid
    return mid


def _video_url(video_path: str, backend: str, s: Settings) -> str:
    mode = s.reasoner_video_mode
    if mode == "auto":
        mode = "file" if backend == "local" and is_local_url(s.reasoner_url) else "base64"
    if mode == "file":
        return Path(video_path).resolve().as_uri()
    return data_url(video_path, "video/mp4")


def _user_content(text: str, video_url: Optional[str]) -> Any:
    if not video_url:
        return text
    # Cosmos3 prompt guide: media BEFORE the text prompt.
    return [{"type": "video_url", "video_url": {"url": video_url}}, {"type": "text", "text": text}]


def _post_chat(backend: str, s: Settings, messages: list, *, sampling: dict, has_video: bool,
               response_format: Optional[dict] = None, max_tokens: Optional[int] = None) -> tuple[str, str, dict]:
    api, model, headers = _endpoint(backend, s)
    model = _resolve_model(api, model, headers, s)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or s.reasoner_max_tokens,
        "seed": s.reasoner_seed,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "presence_penalty": sampling["presence_penalty"],
        # vLLM / NIM request extensions (OpenAI client would put these in extra_body)
        "top_k": sampling["top_k"],
        "repetition_penalty": sampling["repetition_penalty"],
    }
    if has_video:
        payload["media_io_kwargs"] = ({"video": {"fps": float(s.reasoner_fps)}} if backend == "nim"
                                      else {"video": {"num_frames": -1, "fps": s.reasoner_fps}})
    if response_format:
        payload["response_format"] = response_format
    try:
        with client(s.reasoner_timeout_s, headers) as c:
            r = c.post(f"{api}/chat/completions", json=payload)
            if r.status_code in (400, 422) and any(k in payload for k in _EXTENSION_FIELDS):
                # some gateways reject vLLM request extensions -> retry once with plain OpenAI fields
                minimal = {k: v for k, v in payload.items() if k not in _EXTENSION_FIELDS}
                r = c.post(f"{api}/chat/completions", json=minimal)
    except httpx.HTTPError as e:
        raise BackendError(f"reasoner request failed: {type(e).__name__}: {e}", retryable=False) from e
    raise_for(r, "reasoner chat/completions")
    try:
        data = r.json()
        msg = data["choices"][0]["message"]
    except Exception as e:
        raise BackendError(f"unexpected reasoner response: {r.text[:300]}") from e
    content = msg.get("content") or ""
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, reasoning, {"model": data.get("model", model), "usage": data.get("usage")}


# ------------------------------------------------------------------ assess_clip
def assess_inputs(video_path: str, context: Optional[dict], s: Optional[Settings] = None) -> tuple[dict, str, str]:
    """Cache-key inputs, system prompt and user text for an assessment."""
    s = s or get_settings()
    text = build_assess_text(context, think=s.reasoner_think, include_labels=s.include_labels_in_prompt)
    inputs = {"video": cache.FileRef(str(video_path)), "system": CWI_SYSTEM_PROMPT, "prompt": text,
              "think": s.reasoner_think, "prompt_version": PROMPT_VERSION}
    return inputs, CWI_SYSTEM_PROMPT, text


def _video_alias(video_path: str) -> Optional[str]:
    try:
        return cache.alias_key("video", cache.file_sha256(video_path))
    except OSError:
        return None


def _finish(result: dict, *, source: str, t0: float, key: str, context: Optional[dict], **extra: Any) -> dict:
    c = normalize_context(context)
    out = {
        "verdict": result.get("verdict", "review"),
        "defects": result.get("defects", []),
        "root_cause": result.get("root_cause", ""),
        "wps_findings": result.get("wps_findings", []),
        "recommended_action": result.get("recommended_action", ""),
        "reasoning": result.get("reasoning", ""),
        "code_basis": result.get("code_basis", ""),
        "source": source,
        "latency_ms": int(round((time.perf_counter() - t0) * 1000)),
        "clip_id": c.get("clip_id"),
        "cache_key": key,
    }
    for k in ("model", "backend", "warnings", "attempts", "origin_source", "origin_latency_ms", "cache_match",
              "error", "raw_output", "usage"):
        if k in result and k not in extra:
            out[k] = result[k]
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _from_cache(rec: dict, *, t0: float, key: str, context: Optional[dict], match: str) -> dict:
    return _finish(rec, source=SOURCE_CACHE, t0=t0, key=rec.get("cache_key", key), context=context,
                   origin_source=rec.get("source"), origin_latency_ms=rec.get("latency_ms"), cache_match=match)


def assess_clip(video_path: str, context: Optional[dict] = None, *, backend: Optional[str] = None) -> dict:
    """Weld-quality assessment of one clip.

    Returns ``{verdict: accept|reject|review, defects: [{type, confidence, t_s}], root_cause,
    wps_findings: [{param, status, note}], recommended_action, reasoning, code_basis,
    source: cosmos3-local|nim|cache|heuristic-fallback, latency_ms, clip_id, cache_key, ...}``.
    """
    t0 = time.perf_counter()
    s = get_settings()
    inputs, system, text = assess_inputs(video_path, context, s)
    key = cache.make_key(FN_ASSESS, inputs)
    b = backend or resolve_backend("reasoner", s)
    alias = _video_alias(video_path)

    if cache.reading_enabled():
        rec = cache.get_json(FN_ASSESS, key)
        if rec:
            return _from_cache(rec, t0=t0, key=key, context=context, match="exact")

    error = None
    if b in ("local", "nim"):
        if not Path(video_path).exists():
            error = f"video not found: {video_path}"
        else:
            try:
                res = _assess_remote(video_path, system, text, b, s)
                out = _finish(res, source=backend_source(b), t0=t0, key=key, context=context, backend=b)
                if cache.writing_enabled():
                    cache.put_json(FN_ASSESS, key, {**out, "context": context, "prompt": text},
                                   aliases=[alias] if alias else [])
                return out
            except (BackendError, AssessmentError) as e:
                error = str(e)
                if not s.fallback:
                    raise

    # replay / fallback path
    if alias and s.cache_mode != "off":
        k2 = cache.resolve_alias(FN_ASSESS, alias)
        rec = cache.get_json(FN_ASSESS, k2) if k2 else None
        if rec:
            return _from_cache(rec, t0=t0, key=key, context=context, match="same-video")
    res = heuristics.assess(context)
    return _finish(res, source=SOURCE_HEURISTIC, t0=t0, key=key, context=context, backend=b, error=error)


def _assess_remote(video_path: str, system: str, text: str, backend: str, s: Settings) -> dict:
    vurl = _video_url(video_path, backend, s)
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": _user_content(text, vurl)}]
    attempts: list[dict] = []
    last_err: Optional[Exception] = None
    sampling = SAMPLING_REASONING if s.reasoner_think else SAMPLING_DIRECT
    n_attempts = 1 + max(0, s.reasoner_retries)
    for i in range(n_attempts):
        guided = i == 2  # third try: no <think>, JSON-schema guided decoding
        msgs = messages
        fmt = None
        samp = sampling
        if i == 1 and attempts and attempts[-1].get("content"):
            msgs = messages + [
                {"role": "assistant", "content": attempts[-1].get("content", "")[:6000]},
                {"role": "user", "content": f"Your previous answer could not be used ({attempts[-1]['error']}). "
                                            "Reply again with ONLY the JSON object in the requested schema, "
                                            "after a short <think> block."},
            ]
        if guided:
            no_think_text = text.split("\n\nAnswer the question using the following format:")[0]
            msgs = [messages[0], {"role": "user", "content": _user_content(no_think_text, vurl)}]
            fmt = {"type": "json_schema", "json_schema": {"name": "weld_assessment", "schema": ASSESS_JSON_SCHEMA}}
            samp = SAMPLING_DIRECT
        try:
            content, reasoning, info = _post_chat(backend, s, msgs, sampling=samp, has_video=True, response_format=fmt)
        except BackendError as e:
            last_err = e
            attempts.append({"attempt": i + 1, "error": str(e)[:300], "content": ""})
            if guided and e.status in (400, 422):
                break  # server does not support response_format json_schema
            if e.retryable and i + 1 < n_attempts:
                time.sleep(min(2.0 * (i + 1), 6.0))
                continue
            raise
        try:
            result, warnings = parse_assessment(content, reasoning)
            result.update({"model": info["model"], "usage": info.get("usage"), "warnings": warnings,
                           "attempts": i + 1, "raw_output": content[:8000]})
            if guided and not result.get("reasoning"):
                result["reasoning"] = "(guided JSON decoding; no reasoning trace requested)"
            return result
        except AssessmentError as e:
            last_err = e
            attempts.append({"attempt": i + 1, "error": str(e), "content": content})
    raise AssessmentError(f"no valid assessment after {n_attempts} attempts: {last_err}")


# ------------------------------------------------------------------ chat / explain
def _normalise_messages(messages: list, video_url: Optional[str]) -> list:
    msgs = [dict(m) for m in messages]
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": CWI_SYSTEM_PROMPT})
    if video_url:
        for m in reversed(msgs):
            if m.get("role") == "user":
                content = m.get("content")
                parts = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
                m["content"] = [{"type": "video_url", "video_url": {"url": video_url}}] + [
                    p for p in parts if p.get("type") != "video_url"]
                break
    return msgs


def chat_meta(messages: list, video_path: Optional[str] = None, *, backend: Optional[str] = None,
              think: bool = False) -> dict:
    """Like :func:`chat` but returns ``{text, reasoning, source, latency_ms, ...}``."""
    t0 = time.perf_counter()
    s = get_settings()
    inputs = {"messages": messages, "video": cache.FileRef(video_path) if video_path else None, "think": think,
              "system": CWI_SYSTEM_PROMPT}
    key = cache.make_key(FN_CHAT, inputs)
    b = backend or resolve_backend("reasoner", s)
    ms = lambda: int(round((time.perf_counter() - t0) * 1000))  # noqa: E731

    if cache.reading_enabled():
        rec = cache.get_json(FN_CHAT, key)
        if rec:
            return {**rec, "source": SOURCE_CACHE, "origin_source": rec.get("source"), "latency_ms": ms()}
    error = None
    if b in ("local", "nim"):
        try:
            vurl = _video_url(video_path, b, s) if video_path else None
            msgs = _normalise_messages(messages, vurl)
            if think and msgs and msgs[-1].get("role") == "user":
                from .prompts import THINK_SUFFIX
                last = msgs[-1]
                if isinstance(last["content"], list):
                    last["content"] = last["content"] + [{"type": "text", "text": THINK_SUFFIX}]
                else:
                    last["content"] = f"{last['content']}\n\n{THINK_SUFFIX}"
            content, reasoning, info = _post_chat(b, s, msgs, sampling=SAMPLING_REASONING if think else SAMPLING_DIRECT,
                                                  has_video=bool(video_path))
            think_txt, answer = split_think(content)
            out = {"text": answer or content, "reasoning": reasoning or think_txt, "source": backend_source(b),
                   "latency_ms": ms(), "model": info["model"], "cache_key": key}
            if cache.writing_enabled():
                cache.put_json(FN_CHAT, key, out)
            return out
        except BackendError as e:
            error = str(e)
            if not s.fallback:
                raise
    # replay fallback: reuse a cached assessment of the same video if we have one
    text = ("[Cosmos Reasoner offline - replay mode] No cached answer for this exact question. "
            "Run the Reasoner on the GPU pod (worldmodels/serve/reasoner.sh) or warm the cache.")
    if video_path and Path(video_path).exists():
        alias = _video_alias(video_path)
        k2 = cache.resolve_alias(FN_ASSESS, alias) if alias else None
        rec = cache.get_json(FN_ASSESS, k2) if k2 else None
        if rec:
            d = ", ".join(f"{x['type']} ({x['confidence']:.0%})" for x in rec.get("defects", [])) or "none"
            text = (f"[cached Cosmos assessment of this clip] Verdict: {rec.get('verdict')}. Defects: {d}. "
                    f"Root cause: {rec.get('root_cause')} Recommended action: {rec.get('recommended_action')}")
            return {"text": text, "reasoning": rec.get("reasoning", ""), "source": SOURCE_CACHE,
                    "origin_source": rec.get("source"), "latency_ms": ms(), "cache_match": "same-video-assessment"}
    return {"text": text, "reasoning": "", "source": SOURCE_HEURISTIC, "latency_ms": ms(), "error": error}


def chat(messages: list, video_path: Optional[str] = None) -> str:
    """Free-form Cosmos3 Reasoner chat (OpenAI-style ``messages``); returns the answer text
    with any ``<think>`` block removed. Use :func:`chat_meta` for source/latency/reasoning."""
    return chat_meta(messages, video_path)["text"]


def explain_signals_meta(context: Optional[dict], *, backend: Optional[str] = None) -> dict:
    t0 = time.perf_counter()
    s = get_settings()
    text = build_explain_text(context)
    key = cache.make_key(FN_EXPLAIN, {"prompt": text, "system": CWI_SYSTEM_PROMPT, "v": PROMPT_VERSION})
    b = backend or resolve_backend("reasoner", s)
    ms = lambda: int(round((time.perf_counter() - t0) * 1000))  # noqa: E731
    if cache.reading_enabled():
        rec = cache.get_json(FN_EXPLAIN, key)
        if rec:
            return {**rec, "source": SOURCE_CACHE, "origin_source": rec.get("source"), "latency_ms": ms()}
    error = None
    if b in ("local", "nim"):
        try:
            msgs = [{"role": "system", "content": CWI_SYSTEM_PROMPT}, {"role": "user", "content": text}]
            content, reasoning, info = _post_chat(b, s, msgs, sampling=SAMPLING_DIRECT, has_video=False, max_tokens=1024)
            think_txt, answer = split_think(content)
            out = {"text": answer or content, "reasoning": reasoning or think_txt, "source": backend_source(b),
                   "latency_ms": ms(), "model": info["model"], "cache_key": key}
            if cache.writing_enabled():
                cache.put_json(FN_EXPLAIN, key, out)
            return out
        except BackendError as e:
            error = str(e)
            if not s.fallback:
                raise
    return {"text": heuristics.explain(context), "reasoning": "", "source": SOURCE_HEURISTIC,
            "latency_ms": ms(), "error": error, "cache_key": key}


def explain_signals(context: Optional[dict]) -> str:
    """Plain-language explanation of a clip's sensor stats / WPS excursions."""
    return explain_signals_meta(context)["text"]


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2, default=str)
