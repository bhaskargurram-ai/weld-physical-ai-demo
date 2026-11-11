"""Robust parsing of Cosmos3 Reasoner output.

The Reasoner answers ``<think>...</think>`` + final answer when asked to reason
(prompt guide), may wrap JSON in markdown fences, and may add prose around it.
Some vLLM builds with a reasoning parser move the trace into
``message.reasoning_content`` / ``message.reasoning`` instead.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .prompts import DEFECT_TYPES

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def split_think(text: str) -> tuple[str, str]:
    """Return ``(reasoning, answer)`` from a raw completion."""
    text = text or ""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip(), (text[: m.start()] + text[m.end():]).strip()
    # unterminated / truncated <think> or a closing tag only
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        return pre.replace("<think>", "").strip(), post.strip()
    if text.lstrip().lower().startswith("<think>"):
        return text.lstrip()[7:].strip(), ""
    return "", text.strip()


def extract_json(text: str) -> Optional[Any]:
    """First JSON object (or array) found in ``text``; tolerant of fences/prose."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        for opener in ("{", "["):
            start = cand.find(opener)
            while start != -1:
                try:
                    obj, _ = dec.raw_decode(cand[start:])
                    if isinstance(obj, (dict, list)):
                        return obj
                except json.JSONDecodeError:
                    fixed = _repair(cand[start:])
                    if fixed is not None:
                        return fixed
                start = cand.find(opener, start + 1)
    return None


def _repair(fragment: str) -> Optional[Any]:
    """Cheap repairs: trailing commas, single quotes, python literals, truncation."""
    s = fragment
    end = s.rfind("}")
    if end != -1:
        s = s[: end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    for attempt in (s, s.replace("'", '"')):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None


_VERDICT_MAP = {
    "accept": "accept", "accepted": "accept", "acceptable": "accept", "pass": "accept", "passed": "accept",
    "ok": "accept", "good": "accept", "conforming": "accept",
    "reject": "reject", "rejected": "reject", "rejectable": "reject", "fail": "reject", "failed": "reject",
    "repair": "reject", "nonconforming": "reject", "non-conforming": "reject", "unacceptable": "reject",
    "review": "review", "inconclusive": "review", "uncertain": "review", "borderline": "review",
    "further review": "review", "needs review": "review", "investigate": "review", "hold": "review",
}

_DEFECT_SYNONYMS = {
    "porosity": "porosity", "pore": "porosity", "pores": "porosity", "gas porosity": "porosity",
    "blowhole": "porosity", "blow hole": "porosity", "pinhole": "porosity", "pinholes": "porosity",
    "wormhole": "porosity", "cluster porosity": "porosity",
    "lack of fusion": "lack_of_fusion", "lack_of_fusion": "lack_of_fusion", "lof": "lack_of_fusion",
    "incomplete fusion": "lack_of_fusion", "cold lap": "lack_of_fusion", "cold lap/lack of fusion": "lack_of_fusion",
    "incomplete penetration": "lack_of_fusion", "lack of penetration": "lack_of_fusion",
    "undercut": "undercut", "under cut": "undercut",
    "burn through": "burn_through", "burn-through": "burn_through", "burn_through": "burn_through",
    "burnthrough": "burn_through", "melt through": "burn_through", "excessive penetration": "burn_through",
    "arc instability": "arc_instability", "arc_instability": "arc_instability", "unstable arc": "arc_instability",
    "spatter": "arc_instability", "excessive spatter": "arc_instability", "erratic arc": "arc_instability",
    "crack": "crack", "cracks": "crack", "cracking": "crack", "hot crack": "crack", "cold crack": "crack",
    "crater crack": "crack", "centerline crack": "crack", "hydrogen crack": "crack",
}


def normalize_verdict(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    key = v.strip().lower().rstrip(".")
    if key in _VERDICT_MAP:
        return _VERDICT_MAP[key]
    for k, val in _VERDICT_MAP.items():
        if k in key:
            return val
    return None


def normalize_defect_type(t: Any) -> str:
    s = str(t or "").strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    if s.replace(" ", "_") in DEFECT_TYPES:
        return s.replace(" ", "_")
    if s in _DEFECT_SYNONYMS:
        return _DEFECT_SYNONYMS[s]
    for k, v in _DEFECT_SYNONYMS.items():
        if k in s:
            return v
    return s.replace(" ", "_") or "unknown"


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        if m:
            # "mm:ss.ff" timestamps
            mm = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", x)
            if mm:
                return float(mm.group(1)) * 60 + float(mm.group(2))
            return float(m.group(0))
    return None


class AssessmentError(ValueError):
    pass


def normalize_assessment(obj: Any) -> tuple[dict, list[str]]:
    """Validate + coerce a parsed Reasoner answer into the contract shape.

    Raises ``AssessmentError`` if it is unusable (no recognisable verdict).
    Returns ``(result, warnings)``.
    """
    warnings: list[str] = []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise AssessmentError("answer is not a JSON object")
    lower = {str(k).lower(): v for k, v in obj.items()}
    verdict = normalize_verdict(lower.get("verdict") or lower.get("decision") or lower.get("result"))
    if verdict is None:
        raise AssessmentError(f"missing/invalid verdict: {lower.get('verdict')!r}")

    defects_in = lower.get("defects") or lower.get("defects_found") or []
    if isinstance(defects_in, dict):
        defects_in = [defects_in]
    if isinstance(defects_in, str):
        defects_in = [] if defects_in.strip().lower() in ("", "none", "no defects") else [{"type": defects_in}]
    defects = []
    for d in defects_in if isinstance(defects_in, list) else []:
        if isinstance(d, str):
            d = {"type": d}
        if not isinstance(d, dict):
            continue
        dl = {str(k).lower(): v for k, v in d.items()}
        dtype = normalize_defect_type(dl.get("type") or dl.get("defect") or dl.get("name"))
        if dtype in ("none", "no_defect", "no_defects", "unknown", ""):
            continue
        conf = _num(dl.get("confidence", dl.get("probability", dl.get("score"))))
        if conf is None:
            conf = 0.5
            warnings.append(f"defect {dtype}: missing confidence, set 0.5")
        if conf > 1.0:
            conf = conf / 100.0 if conf <= 100 else 1.0
        conf = min(1.0, max(0.0, conf))
        t_s = _num(dl.get("t_s", dl.get("time_s", dl.get("t", dl.get("time", dl.get("timestamp"))))))
        if dtype not in DEFECT_TYPES:
            warnings.append(f"non-standard defect type {dtype!r}")
        defects.append({"type": dtype, "confidence": round(conf, 3), "t_s": None if t_s is None else round(t_s, 2)})

    findings_in = lower.get("wps_findings") or lower.get("wps_compliance") or []
    if isinstance(findings_in, (str, dict)):
        findings_in = [findings_in]
    findings = []
    for f in findings_in if isinstance(findings_in, list) else []:
        if isinstance(f, str):
            status = "violation" if re.search(r"violat|exceed|outside|below|above|out of", f, re.I) else "within"
            findings.append({"param": "general", "status": status, "note": f})
            continue
        if isinstance(f, dict):
            fl = {str(k).lower(): v for k, v in f.items()}
            status = str(fl.get("status", "")).lower()
            if status not in ("within", "violation", "marginal"):
                status = "violation" if re.search(r"violat|fail|out|exceed|non", status) else ("marginal" if "marg" in status else "within")
            findings.append({"param": str(fl.get("param") or fl.get("parameter") or "general"), "status": status,
                             "note": str(fl.get("note") or fl.get("detail") or fl.get("evidence") or "")})

    def _text(*keys: str) -> str:
        for k in keys:
            v = lower.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v:
                return "; ".join(str(x) for x in v)
        return ""

    root = _text("root_cause", "rootcause", "cause")
    action = _text("recommended_action", "action", "recommendation", "recommendations", "next_steps")
    if not root:
        warnings.append("missing root_cause")
    if not action:
        warnings.append("missing recommended_action")
    # consistency: a reject with no defects and no violations is suspicious but allowed
    if verdict == "accept" and any(d["confidence"] >= 0.7 and d["type"] in ("crack", "lack_of_fusion", "burn_through") for d in defects):
        warnings.append("accept verdict despite high-confidence rejectable defect")
    return {
        "verdict": verdict,
        "defects": defects,
        "root_cause": root,
        "wps_findings": findings,
        "recommended_action": action,
        "code_basis": _text("code_basis", "basis", "code_reference", "acceptance_criteria"),
    }, warnings


def parse_assessment(content: str, reasoning_field: str = "") -> tuple[dict, list[str]]:
    """Raw message content (+ optional separate reasoning) -> normalised result.

    Raises ``AssessmentError`` when no valid JSON verdict can be recovered.
    """
    think, answer = split_think(content or "")
    reasoning = (reasoning_field or "").strip() or think
    obj = extract_json(answer) if answer else None
    if obj is None:
        obj = extract_json(content or "")  # JSON inside the think block or untagged
    if obj is None:
        raise AssessmentError("no JSON object found in model output")
    result, warnings = normalize_assessment(obj)
    result["reasoning"] = reasoning or str(obj.get("reasoning", "")) if isinstance(obj, dict) else reasoning
    return result, warnings
