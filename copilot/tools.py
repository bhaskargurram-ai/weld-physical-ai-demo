"""Copilot tools: shared by every LLM backend (Claude, Cosmos Reasoner, scripted router).

Each tool returns a ToolOutput dict:
    {"result": <compact JSON for the LLM>, "cards": [<UI cards>], "ui": [<UI actions>]}
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .data import SIGNAL_INFO, SIGNALS, DataLayer, _defect_list
from .jobs import JobManager
from .models import CHANGE_TEMPLATES, WorldModels, parse_change

PROCESSES = ["GTAW", "GMAW", "FCAW", "SMAW", "SAW"]
PASS_TYPES = ["root", "hot", "fill", "cap"]
DEFECTS = ["porosity", "lack_of_fusion", "undercut", "burn_through", "arc_instability", "crack"]

_SEQ = {"type": "string", "description": "Weld pass sequence name, e.g. 'SP-1043_J07_P3_FCAW'. "
                                        "Loose references like 'SP-1043 J07 fill' are resolved."}

TOOL_SPECS: list[dict] = [
    {"name": "list_passes",
     "description": "List weld passes (one Mosaico sequence per pass) with metadata: spool, joint, pass type, "
                    "process, WPS, RT (radiography) result, defects and WPS-violation counts. Use filters to narrow.",
     "input_schema": {"type": "object", "properties": {
         "process": {"type": "string", "enum": PROCESSES},
         "spool_id": {"type": "string", "description": "e.g. SP-1043"},
         "joint_id": {"type": "string", "description": "e.g. J07"},
         "pass_type": {"type": "string", "enum": PASS_TYPES},
         "has_defect": {"type": "boolean"},
         "rt_result": {"type": "string", "enum": ["accept", "reject", "pending"]},
         "only_wps_violations": {"type": "boolean", "description": "Only passes that broke the WPS"},
         "since_days": {"type": "number", "description": "Only passes started within N days of the most recent pass"},
     }}},
    {"name": "find_anomalies",
     "description": "Search sensor streams across passes. With `param` + `gt`/`lt`, finds time windows where that "
                    "signal crossed the threshold. With no param, returns every WPS limit violation across passes.",
     "input_schema": {"type": "object", "properties": {
         "param": {"type": "string", "enum": SIGNALS},
         "gt": {"type": "number"}, "lt": {"type": "number"},
         "process": {"type": "string", "enum": PROCESSES},
         "spool_id": {"type": "string"},
     }}},
    {"name": "get_signals_summary",
     "description": "Statistics (mean/std/min/max) of arc current, voltage, travel speed, wire feed, gas flow, heat "
                    "input and interpass temperature for one pass, optionally within [t0, t1] seconds, compared with "
                    "the WPS limits. Shows a mini chart.",
     "input_schema": {"type": "object", "properties": {
         "sequence": _SEQ, "t0": {"type": "number"}, "t1": {"type": "number"}}, "required": ["sequence"]}},
    {"name": "wps_check",
     "description": "Check one pass against its WPS (Welding Procedure Specification): every out-of-limit interval "
                    "with value vs limits, plus labelled defect windows and the RT result.",
     "input_schema": {"type": "object", "properties": {"sequence": _SEQ}, "required": ["sequence"]}},
    {"name": "show_weld",
     "description": "Jump the dashboard's weld replay (torch camera video + synced signal plots) to a pass and time. "
                    "Use `event` (e.g. 'porosity', 'defect', 'violation') to jump to where that started.",
     "input_schema": {"type": "object", "properties": {
         "sequence": _SEQ, "t": {"type": "number", "description": "seconds from arc start"},
         "event": {"type": "string"},
         "view": {"type": "string", "enum": ["torch", "edge", "depth", "sim2real"]}}}},
    {"name": "assess_weld",
     "description": "Run the Cosmos 3 Reasoner (vision-language world model) on a 5 s torch-camera clip with sensor "
                    "context: returns verdict accept/reject/review, defects with confidence, root cause, WPS "
                    "findings, recommended action and reasoning trace.",
     "input_schema": {"type": "object", "properties": {
         "sequence": _SEQ, "clip_id": {"type": "string"},
         "t": {"type": "number", "description": "centre of the clip (s); default: first defect"}}}},
    {"name": "what_if",
     "description": "Counterfactual video with the Cosmos 3 Generator: re-imagine a pass clip under a process change "
                    "(e.g. 'travel speed -20%', 'gas loss', 'porosity', 'voltage +20%', 'interpass overheat'). "
                    "Long-running: starts a background job; the UI shows progress and then the video.",
     "input_schema": {"type": "object", "properties": {
         "sequence": _SEQ, "clip_id": {"type": "string"},
         "change": {"type": "string", "description": "the process change, free text or template key"},
         "t": {"type": "number"}}, "required": ["change"]}},
    {"name": "sim2real",
     "description": "Cosmos 3 Transfer: turn the simulator's control video (edge or depth) of a pass into photoreal "
                    "weld footage for training data. Long-running background job.",
     "input_schema": {"type": "object", "properties": {
         "sequence": _SEQ, "kind": {"type": "string", "enum": ["edge", "depth"]}, "t": {"type": "number"}},
         "required": ["sequence"]}},
    {"name": "dataset_stats",
     "description": "Curated training-dataset statistics (Cosmos Curator output): clip counts kept/rejected, by "
                    "defect class, reject reasons. Optional defect filter.",
     "input_schema": {"type": "object", "properties": {"defect": {"type": "string", "enum": DEFECTS + ["none"]}}}},
    {"name": "search_clips",
     "description": "Search curated clips by caption text / defect / sequence (keyword match over the manifest).",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"}, "limit": {"type": "integer"}, "kept_only": {"type": "boolean"}},
         "required": ["text"]}},
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def validate_input(name: str, args: Any) -> str | None:
    """Light schema validation (types/required/enums). Returns an error string or None."""
    spec = next((t for t in TOOL_SPECS if t["name"] == name), None)
    if spec is None:
        return f"unknown tool {name}"
    if not isinstance(args, dict):
        return "input must be an object"
    sch = spec["input_schema"]
    for r in sch.get("required", []):
        if args.get(r) in (None, ""):
            return f"missing required field '{r}'"
    for k, v in args.items():
        p = sch["properties"].get(k)
        if p is None or v is None:
            continue
        typ = p.get("type")
        ok = {"string": isinstance(v, str), "number": isinstance(v, (int, float)) and not isinstance(v, bool),
              "integer": isinstance(v, int) and not isinstance(v, bool), "boolean": isinstance(v, bool)}.get(typ, True)
        if not ok:
            return f"field '{k}' must be {typ}"
        if "enum" in p and v not in p["enum"]:
            return f"field '{k}' must be one of {p['enum']}"
    return None


class Toolbox:
    def __init__(self, data: DataLayer, wm: WorldModels, jobs: JobManager):
        self.data, self.wm, self.jobs = data, wm, jobs

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict, ctx: dict | None = None) -> dict:
        ctx = ctx if ctx is not None else {}
        args = {k: v for k, v in (args or {}).items() if v is not None}
        err = validate_input(name, args)
        if err:
            return {"result": {"error": err}, "cards": [], "ui": [], "is_error": True}
        try:
            out = getattr(self, f"t_{name}")(ctx=ctx, **args)
        except LookupError as e:
            return {"result": {"error": str(e)}, "cards": [], "ui": [], "is_error": True}
        except Exception as e:  # noqa: BLE001
            return {"result": {"error": f"{type(e).__name__}: {e}"}, "cards": [], "ui": [], "is_error": True}
        out.setdefault("cards", [])
        out.setdefault("ui", [])
        return out

    # ------------------------------------------------------------ helpers
    def resolve_sequence(self, ref: str | None, ctx: dict | None = None, prefer_defect: bool = True) -> str:
        rows = self.data.all_passes()
        names = {r["sequence_name"] for r in rows}
        ctx = ctx or {}
        if not ref:
            if ctx.get("last_sequence") in names:
                return ctx["last_sequence"]
            raise LookupError("no weld pass specified")
        ref = ref.strip()
        if ref in names:
            return ref
        m = _parse_ref(ref)
        cand = rows
        if m.get("spool"):
            cand = [r for r in cand if (r["spool_id"] or "").upper() == m["spool"]]
        if m.get("joint"):
            cand = [r for r in cand if (r["joint_id"] or "").upper() == m["joint"]]
        if m.get("pass_no"):
            cand = [r for r in cand if r["pass_no"] == m["pass_no"]]
        if m.get("pass_type"):
            cand = [r for r in cand if (r["pass_type"] or "").lower() == m["pass_type"]]
        if m.get("process"):
            cand = [r for r in cand if (r["process"] or "").upper() == m["process"]]
        if not m or not cand or len(cand) == len(rows):
            low = ref.lower()
            sub = [r for r in rows if low in r["sequence_name"].lower()]
            if sub:
                cand = sub
            elif ctx.get("last_sequence") in names and re.search(r"\b(that|this|same|it)\b", low):
                return ctx["last_sequence"]
            else:
                raise LookupError(f"no weld pass matches '{ref}'")
        if prefer_defect:
            cand = sorted(cand, key=lambda r: (not r["has_defect"], r["rt_result"] != "reject", r["pass_no"]))
        return cand[0]["sequence_name"]

    def _seq(self, sequence: str | None, ctx: dict) -> str:
        seq = self.resolve_sequence(sequence, ctx)
        ctx["last_sequence"] = seq
        return seq

    # ------------------------------------------------------------ tools
    def t_list_passes(self, ctx, process=None, spool_id=None, joint_id=None, pass_type=None, has_defect=None,
                      rt_result=None, only_wps_violations=False, since_days=None) -> dict:
        rows = self.data.list_passes(process=process, spool_id=spool_id, joint_id=joint_id, pass_type=pass_type,
                                     has_defect=has_defect, rt_result=rt_result, with_violations=True)
        if since_days:
            dates = [_dt(r.get("started_at")) for r in self.data.all_passes()]
            dates = [d for d in dates if d]
            if dates:
                cutoff = max(dates) - timedelta(days=float(since_days))
                rows = [r for r in rows if (_dt(r.get("started_at")) or cutoff) >= cutoff]
        if only_wps_violations:
            rows = [r for r in rows if r.get("violation_count")]
        slim = [{"sequence": r["sequence_name"], "spool": r["spool_id"], "joint": r["joint_id"],
                 "pass": f"P{r['pass_no']} {r['pass_type']}", "process": r["process"], "wps": r["wps_id"],
                 "rt": r["rt_result"] or "pending", "defects": r["defect_types"],
                 "wps_violations": r.get("violation_count", 0), "violated": r.get("violated_params", []),
                 "started": (r.get("started_at") or "")[:16].replace("T", " ")} for r in rows]
        if len(slim) == 1:
            ctx["last_sequence"] = slim[0]["sequence"]
        card = {"type": "table", "title": f"{len(slim)} weld passes", "row_action": "show_weld",
                "columns": [{"key": "sequence", "label": "Pass"}, {"key": "process", "label": "Proc"},
                            {"key": "rt", "label": "RT", "badge": True},
                            {"key": "wps_violations", "label": "WPS viol."},
                            {"key": "violated", "label": "Params"}, {"key": "defects", "label": "Defects"}],
                "rows": slim[:40]}
        return {"result": {"count": len(slim), "passes": slim[:40]}, "cards": [card]}

    def t_find_anomalies(self, ctx, param=None, gt=None, lt=None, process=None, spool_id=None) -> dict:
        rows = self.data.list_passes(process=process, spool_id=spool_id)
        hits = []
        if not param:
            for r in rows:
                for v in self.data.wps_violations(r["sequence_name"]):
                    hits.append({"sequence": r["sequence_name"], "param": v["param"], "t_start_s": v["t_start_s"],
                                 "t_end_s": v["t_end_s"], "value": v["value"], "limit_lo": v["limit_lo"],
                                 "limit_hi": v["limit_hi"], "rt": r["rt_result"] or "pending"})
            title = f"{len(hits)} WPS violations across {len({h['sequence'] for h in hits})} passes"
        else:
            if gt is None and lt is None:
                raise LookupError("give gt or lt with param")
            for r in rows:
                df = self.data.get_signals(r["sequence_name"])
                if param not in df.columns:
                    continue
                x = df[param].rolling(20, center=True, min_periods=1).median().values
                t = df["t_s"].values
                bad = np.ones(len(x), bool)
                if gt is not None:
                    bad &= x > gt
                if lt is not None:
                    bad &= x < lt
                idx = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
                for a, b in zip(idx[::2], idx[1::2]):
                    if t[b - 1] - t[a] < 0.2:
                        continue
                    seg = x[a:b]
                    hits.append({"sequence": r["sequence_name"], "param": param, "t_start_s": round(float(t[a]), 2),
                                 "t_end_s": round(float(t[b - 1]), 2),
                                 "value": round(float(seg.max() if gt is not None else seg.min()), 3),
                                 "rt": r["rt_result"] or "pending"})
            cond = " and ".join(x for x in ([f"> {gt}" if gt is not None else None, f"< {lt}" if lt is not None else None]) if x)
            title = f"{param} {cond}: {len(hits)} windows in {len({h['sequence'] for h in hits})} passes"
        if hits:
            ctx["last_sequence"] = hits[0]["sequence"]
        card = {"type": "table", "title": title, "row_action": "show_weld", "time_key": "t_start_s",
                "columns": [{"key": "sequence", "label": "Pass"}, {"key": "param", "label": "Signal"},
                            {"key": "t_start_s", "label": "From s"}, {"key": "t_end_s", "label": "To s"},
                            {"key": "value", "label": "Worst"}, {"key": "rt", "label": "RT", "badge": True}],
                "rows": hits[:60]}
        return {"result": {"count": len(hits), "title": title, "hits": hits[:60]}, "cards": [card]}

    def t_get_signals_summary(self, ctx, sequence, t0=None, t1=None) -> dict:
        seq = self._seq(sequence, ctx)
        df = self.data.get_signals(seq, t0, t1)
        lim = self.data.wps_limits(seq)
        stats = {}
        for p in SIGNALS:
            if p not in df.columns:
                continue
            s = df[p]
            lo, hi = (lim.get(p) or [None, None])
            out = 0.0
            if lo is not None:
                out += float((s < lo).mean())
            if hi is not None:
                out += float((s > hi).mean())
            stats[p] = {"mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                        "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
                        "wps": [lo, hi], "pct_out_of_wps": round(100 * out, 1), "unit": SIGNAL_INFO[p]["unit"]}
        a = float(df["t_s"].iloc[0]) if len(df) else 0
        b = float(df["t_s"].iloc[-1]) if len(df) else 0
        flagged = [p for p, s in stats.items() if s["pct_out_of_wps"] > 0]
        params = (flagged + [p for p in ("current_a", "voltage_v", "travel_speed_mmps", "gas_lpm") if p not in flagged])[:4]
        return {"result": {"sequence": seq, "t0": round(a, 2), "t1": round(b, 2), "stats": stats},
                "cards": [self.chart_card(seq, params, a, b, title=f"{seq} · {a:.1f}–{b:.1f}s")]}

    def t_wps_check(self, ctx, sequence) -> dict:
        seq = self._seq(sequence, ctx)
        viol = self.data.wps_violations(seq)
        meta = self.data.get_pass(seq) or {}
        wins = self.data.defect_windows(seq)
        dur = meta.get("duration_s") or 0
        bad_s = sum(v["t_end_s"] - v["t_start_s"] for v in viol)
        res = {"sequence": seq, "wps_id": meta.get("wps_id"), "rt_result": meta.get("rt_result") or "pending",
               "violations": viol, "defect_windows": wins, "duration_s": dur,
               "compliant": not viol, "out_of_spec_seconds": round(bad_s, 2),
               "wps_limits": self.data.wps_limits(seq)}
        params = list(dict.fromkeys(v["param"] for v in viol))[:3] or ["current_a", "voltage_v", "gas_lpm"]
        cards = [{"type": "table", "title": f"{seq} vs {meta.get('wps_id')}: "
                                            f"{'COMPLIANT' if not viol else f'{len(viol)} violations'}",
                  "row_action": "show_weld", "time_key": "t_start_s", "default_sequence": seq,
                  "columns": [{"key": "param", "label": "Parameter"}, {"key": "t_start_s", "label": "From s"},
                              {"key": "t_end_s", "label": "To s"}, {"key": "value", "label": "Worst"},
                              {"key": "limit_lo", "label": "Min"}, {"key": "limit_hi", "label": "Max"}],
                  "rows": [{**v, "sequence": seq} for v in viol]}]
        if dur:
            cards.append(self.chart_card(seq, params, 0, dur, title=f"{seq} · full pass"))
        return {"result": res, "cards": cards}

    def t_show_weld(self, ctx, sequence=None, t=None, event=None, view=None) -> dict:
        if not sequence and event and ctx.get("last_sequence") is None:
            sequence = self._find_seq_with_event(event)
        seq = self._seq(sequence, ctx)
        if t is None and event:
            t = self._event_time(seq, event)
            if t is None and not sequence:
                other = self._find_seq_with_event(event)
                if other:
                    seq = other
                    ctx["last_sequence"] = seq
                    t = self._event_time(seq, event)
        t = float(t or 0.0)
        action = {"action": "show_weld", "sequence": seq, "t": round(t, 2)}
        if view:
            action["view"] = view
        card = {"type": "jump", "sequence": seq, "t": round(t, 2), "event": event,
                "title": f"Replay → {seq} @ {t:.1f}s" + (f" ({event})" if event else "")}
        return {"result": {"ok": True, "sequence": seq, "t": round(t, 2)}, "cards": [card], "ui": [action]}

    def t_assess_weld(self, ctx, sequence=None, clip_id=None, t=None) -> dict:
        seq = None if clip_id else self._seq(sequence, ctx)
        res = self.wm.assess(seq, clip_id, t)
        ctx["last_sequence"] = res["sequence_name"]
        card = {"type": "verdict", **res}
        slim = {k: res.get(k) for k in ("verdict", "defects", "root_cause", "wps_findings", "recommended_action",
                                         "source", "latency_s", "sequence_name", "clip_id", "t0_s", "t1_s",
                                         "rt_result", "note")}
        slim["reasoning"] = (res.get("reasoning") or "")[:1500]
        return {"result": slim, "cards": [card],
                "ui": [{"action": "show_weld", "sequence": res["sequence_name"], "t": res["t0_s"], "soft": True}]}

    def t_what_if(self, ctx, change, sequence=None, clip_id=None, t=None) -> dict:
        seq = None if clip_id else self._seq(sequence, ctx)
        key, text = parse_change(change)
        label = CHANGE_TEMPLATES.get(key or "", {}).get("label") or change
        job = self.jobs.submit("what_if", f"What-if: {label} · {seq or clip_id}",
                               {"sequence": seq, "clip_id": clip_id, "change": change, "t": t},
                               lambda progress: self.wm.what_if(seq, change, clip_id=clip_id, t=t, progress=progress))
        return {"result": {"job_id": job.id, "status": "started", "change": text, "sequence": seq,
                           "note": "Video renders in the background; the UI shows progress and the result."},
                "cards": [{"type": "job", "job_id": job.id, "kind": "what_if", "label": job.label}]}

    def t_sim2real(self, ctx, sequence, kind="edge", t=None) -> dict:
        seq = self._seq(sequence, ctx)
        job = self.jobs.submit("sim2real", f"Sim→Real ({kind}) · {seq}", {"sequence": seq, "kind": kind, "t": t},
                               lambda progress: self.wm.sim2real(seq, kind=kind, t=t, progress=progress))
        return {"result": {"job_id": job.id, "status": "started", "sequence": seq, "control": kind},
                "cards": [{"type": "job", "job_id": job.id, "kind": "sim2real", "label": job.label}]}

    def t_dataset_stats(self, ctx, defect=None) -> dict:
        st = self.data.dataset_stats(defect if defect != "none" else None)
        if not st["total"]:
            return {"result": {"error": "no curated manifest found (run curate first)"}, "cards": []}
        bars = sorted(({"label": k, "value": v["kept"], "total": v["total"]} for k, v in st["by_defect"].items()),
                      key=lambda b: -b["value"])
        card = {"type": "stats", "title": "Curated dataset" + (f" · {defect}" if defect else ""),
                "kpis": [{"label": "Clips", "value": st["total"]}, {"label": "Kept", "value": st["kept"]},
                         {"label": "Rejected", "value": st["rejected"]}, {"label": "Passes", "value": st["sequences"]}],
                "bars": bars, "reasons": st["reject_reasons"]}
        return {"result": st, "cards": [card]}

    def t_search_clips(self, ctx, text, limit=8, kept_only=False) -> dict:
        terms = [w for w in re.findall(r"[a-z0-9_\-]+", text.lower()) if len(w) > 2 and w not in _STOP]
        scored = []
        for r in self.data.manifest():
            if kept_only and not r.get("kept"):
                continue
            hay = " ".join([str(r.get("caption", "")), r.get("sequence_name", ""), r.get("clip_id", ""),
                            " ".join(_defect_list(r)), str(r.get("rt_result"))]).lower().replace("_", " ")
            score = sum(1 for w in terms if w.replace("_", " ") in hay or w.rstrip("s") in hay)
            if score:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], not x[1].get("kept")))
        hits = [r for _, r in scored[: int(limit or 8)]]
        rows = [{"clip_id": r["clip_id"], "sequence": r.get("sequence_name"), "t0_s": r.get("t0_s"),
                 "t1_s": r.get("t1_s"), "caption": r.get("caption"), "defects": _defect_list(r),
                 "kept": r.get("kept"), "video_url": self.data.clip_video(r)} for r in hits]
        return {"result": {"count": len(scored), "clips": [{k: v for k, v in x.items() if k != "video_url"} for x in rows]},
                "cards": [{"type": "clips", "title": f"{len(scored)} clips match “{text}”", "rows": rows}]}

    # ------------------------------------------------------------ cards
    def chart_card(self, seq: str, params: list[str], t0: float, t1: float, title: str = "") -> dict:
        df = self.data.get_signals(seq, t0, t1)
        step = max(1, len(df) // 300)
        d = df.iloc[::step]
        lim = self.data.wps_limits(seq)
        series = {p: [round(float(x), 3) for x in d[p].values] for p in params if p in d.columns}
        return {"type": "chart", "title": title, "sequence": seq, "t": [round(float(x), 2) for x in d["t_s"].values],
                "series": series, "wps": {p: lim.get(p) for p in series},
                "info": {p: SIGNAL_INFO[p] for p in series},
                "defects": self.data.defect_windows(seq)}

    def _event_time(self, seq: str, event: str) -> float | None:
        ev = event.lower().replace(" ", "_")
        for w in self.data.defect_windows(seq):
            if ev in ("defect", "defects", "start", "anomaly") or ev.rstrip("s") in (w["defect"] or ""):
                return float(w["t_start"])
        if "viol" in ev or "wps" in ev or ev in ("anomaly", "defect"):
            v = self.data.wps_violations(seq)
            if v:
                return float(v[0]["t_start_s"])
        for p in SIGNALS:
            if ev.split("_")[0] in p:
                vv = [x for x in self.data.wps_violations(seq) if x["param"] == p]
                if vv:
                    return float(vv[0]["t_start_s"])
        return None

    def _find_seq_with_event(self, event: str) -> str | None:
        for r in sorted(self.data.all_passes(), key=lambda r: r["rt_result"] != "reject"):
            if self._event_time(r["sequence_name"], event) is not None:
                return r["sequence_name"]
        return None


_STOP = {"the", "and", "for", "with", "show", "find", "clips", "clip", "that", "have", "any", "all", "from", "what"}


def _dt(s: str | None):
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def _parse_ref(text: str) -> dict:
    out: dict = {}
    t = text.upper().replace("_", " ")
    m = re.search(r"SP[- ]?(\d{3,5})", t)
    if m:
        out["spool"] = f"SP-{m.group(1)}"
    m = re.search(r"\bJ(?:OINT)?[-_ ]?(\d{1,3})\b", t)
    if m:
        out["joint"] = f"J{int(m.group(1)):02d}"
    m = re.search(r"(?:\bP|PASS[-_ ]?)(\d)\b", t)
    if m:
        out["pass_no"] = int(m.group(1))
    for pt in PASS_TYPES:
        if re.search(rf"\b{pt.upper()}\b", t):
            out["pass_type"] = pt
            break
    for pr in PROCESSES:
        if re.search(rf"\b{pr}\b", t):
            out["process"] = pr
    return out


def tool_result_text(out: dict, limit: int = 6000) -> str:
    s = json.dumps(out.get("result"), default=str)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"
