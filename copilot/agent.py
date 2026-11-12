"""Weld Copilot chat agent.

Backends (env COPILOT_LLM):
  claude   - Anthropic Messages API, streaming tool-use loop (model: COPILOT_MODEL, default claude-opus-5-5)
  cosmos   - Cosmos 3 Reasoner via its OpenAI-compatible endpoint (COSMOS_REASONER_URL); native function
             calling if the server supports it, otherwise a ReAct-style JSON tool protocol
  scripted - no LLM: keyword/intent router over the same tools (zero keys, deterministic demo)

Every backend yields the same UI event stream (dicts, sent to the browser as SSE):
  {"type": "text", "delta"} | {"type": "thinking", "delta"} | {"type": "tool_start", "id", "name", "input"}
  {"type": "tool_end", "id", "name", "ok", "summary"} | {"type": "card", "card"} | {"type": "ui", ...action}
  {"type": "done"} | {"type": "error", "message"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from . import config
from .tools import TOOL_SPECS, Toolbox, tool_result_text, validate_input

log = logging.getLogger("copilot.agent")

SYSTEM_PROMPT = """You are Weld Copilot, an expert welding engineer and AWS Certified Welding Inspector (CWI) \
assistant for an ASME fabrication shop that builds pressure vessels and process-piping skids (ASME Sec. VIII / \
Sec. IX, B31.3). The shop's robotic weld cell runs GTAW root passes and FCAW fill/cap passes on A106 Gr.B pipe.

You have tools over the shop's physical-AI stack: Mosaico (time-synced weld sensor data: arc current, voltage, \
wire feed, shielding gas, travel speed, heat input, interpass temperature, torch camera), the WPS limits, \
radiography (RT) results, the Cosmos Curator training-clip dataset, and NVIDIA Cosmos 3 world models (Reasoner for \
visual weld QA; Generator for what-if video; Transfer for sim-to-real).

How to answer:
- Be concise and specific, like a CWI writing an NCR: lead with the answer, then evidence.
- Always cite data: sequence names (e.g. SP-1043_J07_P3_FCAW), timestamps in seconds from arc start, measured \
values vs the WPS limits, RT result. Never invent numbers - get them from tools.
- Use show_weld to jump the replay to the moment you are talking about.
- For "why did it fail" questions: wps_check, then assess_weld, then explain the root cause with the defect's \
sensor signature and a corrective action.
- what_if and sim2real start background video jobs; tell the user it is rendering, don't wait for it.
- Use short markdown (bold, bullets, small tables). No preamble."""

SYSTEM_PROMPT_REAL = """You are Weld Copilot, a welding-engineering assistant with a Certified Welding Inspector's \
discipline. You work on REAL, public welding data (both CC BY 4.0, credited in the app):

- **AIMEN SAW** - Submerged Arc Welding Open Repository (PENELOPE, AIMEN; Zenodo 10.5281/zenodo.15083865). \
Coupons PP3-PP7: SA-516 Gr.70N pressure-vessel plate, 52 mm butt weld, 24 tandem-SAW beads each (DC lead + AC trail \
electrode; flux-shielded, so there is NO shielding-gas signal). Lead/trail current & voltage, wire speed, pyrometers, \
bead photos, and RT + UT NDT indications with their positions mapped to weld time. Sequence names like PP5_B02_SAW \
(coupon PP5, bead 2). rt_result: indications / no_indications.
- **BAM tandem-GMAW** (Chaurasia, Fabry, Pittner, Rethmeier; Zenodo 10.5281/zenodo.21824586). Workpieces like WID617 / \
WID630: S355J2+N, 30 mm V-groove, layers L1 (single-wire root) to L6 (tandem). Lead/trail current & voltage, wire \
feed, gas, thermocouples, a real HDR process-camera video. NOT NDT-inspected (rt_result not_inspected). Sequence \
names like WID617_L3_TGMAW (workpiece WID617, layer 3).
{sim_line}
Rules - follow them strictly:
- Only state what the tools return. Cite sequence, time in seconds from arc start, measured value vs window, NDT \
method and indication count. Never invent numbers.
- NEVER invent a defect cause. NDT tells you what and where, not why. Do not explain an indication with gas loss, \
heat input, travel speed or any other mechanism unless a tool result shows that signal changed inside the \
indication windows - and even then call it a correlation, not a proven cause. If the cause is unknown, say \
"the data does not show the cause".
- When the signals show no anomaly (ndt_summary signal_check `signature: false`), say so plainly, with the numbers: \
real defects often leave no trace in the electrical signals. That is a finding, not a failure.
- The settings window is the published nominal setting ±10 % (`wps_basis`) - a demo assumption, not a qualified WPS. \
Call excursions "outside the published settings", not code violations.
- BAM passes were not inspected: never call them sound or defect-free. A bead photo shows only the surface; RT/UT \
indications are internal.
- Results whose source contains "mock" or "heuristic" are rule-based stand-ins, not Cosmos 3 - say so. Generated \
what-if / transfer videos are synthetic.
- Tools: ndt_summary for NDT questions, show_weld to move the replay, show_photo for bead photos, \
get_signals_summary(params=[...]) to compare lead vs trail electrode, list_passes(only_wps_violations) for \
excursions. what_if and sim2real start background jobs; say it is rendering, don't wait.
- Short markdown (bold, bullets, small tables). Lead with the answer. No preamble."""

SIM_LINE = ("- Passes marked **SIMULATED** (source.synthetic) come from the WeldCell simulator: never present them as "
            "real welds, and keep their numbers separate from the real datasets.\n")


def system_prompt(data) -> str:
    """Real-data prompt when the loaded passes include real ones, else the original simulator prompt."""
    try:
        rows = data.all_passes()
    except Exception:  # noqa: BLE001
        rows = []
    if not any(not r.get("synthetic", True) for r in rows):
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT_REAL.format(sim_line=SIM_LINE if any(r.get("synthetic") for r in rows) else "")

_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ctx: dict = field(default_factory=dict)            # last_sequence etc.
    claude_messages: list = field(default_factory=list)
    oai_messages: list = field(default_factory=list)
    created: float = field(default_factory=time.time)


def _summary(name: str, out: dict) -> str:
    r = out.get("result") or {}
    if out.get("is_error"):
        return r.get("error", "error")
    if name == "list_passes":
        return f"{r.get('count')} passes"
    if name == "find_anomalies":
        return r.get("title") or f"{r.get('count')} hits"
    if name == "wps_check":
        return f"{len(r.get('violations', []))} violations · RT {r.get('rt_result')}"
    if name == "assess_weld":
        return f"{(r.get('verdict') or '').upper()} · {r.get('source')} · {r.get('latency_s')}s"
    if name in ("what_if", "sim2real"):
        return f"job {r.get('job_id')} started"
    if name == "show_weld":
        return f"{r.get('sequence')} @ {r.get('t')}s"
    if name == "dataset_stats":
        return f"{r.get('kept')}/{r.get('total')} clips kept"
    if name == "search_clips":
        return f"{r.get('count')} clips"
    if name == "get_signals_summary":
        return f"{r.get('sequence')} {r.get('t0')}–{r.get('t1')}s"
    if name == "show_photo":
        return f"{len(r.get('photos') or [])} photo(s) · {r.get('sequence')}"
    if name == "ndt_summary":
        if r.get("sequence"):
            chk = r.get("signal_check") or {}
            return (f"{sum((r.get('counts') or {}).values())} indications · "
                    f"{'signal change' if chk.get('signature') else 'no signal change' if chk else 'no check'}")
        return f"{r.get('with_indications')}/{r.get('inspected')} inspected passes with indications"
    return "ok"


class BaseAgent:
    name = "base"
    label = "Agent"

    def __init__(self, toolbox: Toolbox):
        self.tb = toolbox

    async def run_tool(self, session: Session, name: str, args: dict, tool_id: str | None = None):
        """Execute a tool; yields UI events; final yielded item is ("__result__", out)."""
        tool_id = tool_id or uuid.uuid4().hex[:8]
        yield {"type": "tool_start", "id": tool_id, "name": name, "input": args}
        out = await asyncio.to_thread(self.tb.call, name, args, session.ctx)
        for c in out.get("cards", []):
            yield {"type": "card", "card": c}
        for a in out.get("ui", []):
            yield {"type": "ui", **a}
        yield {"type": "tool_end", "id": tool_id, "name": name, "ok": not out.get("is_error"),
               "summary": _summary(name, out)}
        yield ("__result__", out)

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:  # pragma: no cover - abstract
        raise NotImplementedError
        yield {}


# =============================================================================== scripted
async def _stream_text(text: str, delay: float = 0.01) -> AsyncIterator[dict]:
    for chunk in re.findall(r"\S+\s*|\s+", text):
        yield {"type": "text", "delta": chunk}
        if delay:
            await asyncio.sleep(delay)


class ScriptedAgent(BaseAgent):
    """Deterministic intent router: same tools, templated narration. Works with zero API keys."""
    name = "scripted"
    label = "Scripted (no LLM)"

    def __init__(self, toolbox: Toolbox, delay: float | None = None):
        super().__init__(toolbox)
        self.delay = float(os.environ.get("COPILOT_SCRIPT_DELAY", "0.012")) if delay is None else delay

    def default_seq(self, ctx: dict) -> str:
        if ctx.get("last_sequence"):
            return ctx["last_sequence"]
        rows = self.tb.data.all_passes()
        for want in (lambda r: r["sequence_name"] == _DEFAULT_SEQ,
                     lambda r: r["has_defect"] and not r.get("synthetic"), lambda r: r["has_defect"], lambda r: True):
            hit = next((r for r in rows if want(r)), None)
            if hit:
                return hit["sequence_name"]
        return _DEFAULT_SEQ

    def route(self, text: str, ctx: dict) -> list[tuple[str, dict]]:
        """Map an utterance to a tool plan: [(tool, args), ...]."""
        t = text.lower()
        ref = _extract_ref(text)
        seq_arg = {"sequence": ref} if ref else {}
        defect = next((d for d in ("porosity", "lack_of_fusion", "slag_inclusion", "undercut", "burn_through",
                                   "arc_instability", "crack")
                       if d.replace("_", " ") in t or d.split("_")[0] in t.replace("-", " ")), None)
        if re.search(r"photo.?real|sim.?(to|2|→|->)\s?real|transfer|from the simulator|synthetic video", t):
            kind = "depth" if "depth" in t else "edge"
            return [("sim2real", {**(seq_arg or {"sequence": self.default_seq(ctx)}), "kind": kind})]
        if re.search(r"what if|what-if|what would|simulate|counterfactual", t):
            change = re.sub(r"^.*?(what if|what-if|what would happen if|simulate)\s*", "", text, flags=re.I)
            change = re.sub(r"\s+on (that|this|the) pass\??$", "", change, flags=re.I)
            if ref:   # "... on WID630 layer 4" -> keep only the change
                change = re.sub(r"\s+(on|for|of|in)\s+(PP|WID|SP|V)[-_ ]?\d.*$", "", change, flags=re.I)
            change = change.strip(" ?") or "travel speed -20%"
            return [("what_if", {**(seq_arg or {"sequence": self.default_seq(ctx)}), "change": change})]
        if re.search(r"\b(photo|picture|image)s?\b", t) and not re.search(r"photo.?real", t):
            plan = [("show_photo", seq_arg or {"sequence": self.default_seq(ctx)})]
            if re.search(r"inspect|assess|look at|grade|check|cosmos|reasoner", t):
                plan.append(("assess_weld", seq_arg or {}))
            return plan
        if re.search(r"lead.{0,20}trail|trail.{0,20}lead|tandem|both (wires|electrodes)|second electrode", t):
            if "voltage" in t:
                params = ["voltage_v", "voltage2_v"]
            elif "wire" in t or "wfs" in t:
                params = ["wfs_mpm", "wfs2_mpm"]
            elif "current" in t:
                params = ["current_a", "current2_a"]
            else:
                params = ["current_a", "current2_a", "voltage_v", "voltage2_v"]
            return [("get_signals_summary", {**(seq_arg or {"sequence": self.default_seq(ctx)}), "params": params})]
        if re.search(r"signals? (change|differ|show)|did the (arc|current|voltage)|electrical signature|"
                     r"signature|show up in the", t) and (ref or ctx.get("last_sequence")):
            plan = []
            if re.search(r"\bshow\b|jump|moment", t):
                plan.append(("show_weld", {**seq_arg, "event": defect or "defect"}))
            plan.append(("ndt_summary", seq_arg or {"sequence": ctx.get("last_sequence")}))
            return plan
        if re.search(r"\bndt\b|indication|radiograph|ultrason|\brt\b.*\but\b|x-?ray", t):
            if ref:
                return [("ndt_summary", seq_arg)]
            co = re.search(r"\b(pp\d{1,2}|wid\d{3})\b", t)
            return [("ndt_summary", {"coupon": co.group(1).upper()} if co else {})]
        if re.search(r"how many|curated|dataset|training (clips|data)|clips? (do|for)", t):
            plan = [("dataset_stats", {"defect": defect} if defect else {})]
            if defect:
                plan.append(("search_clips", {"text": defect.replace("_", " "), "limit": 4, "kept_only": True}))
            return plan
        if re.search(r"search|find clips|clips (with|showing)", t):
            return [("search_clips", {"text": text})]
        if re.search(r"\bwhy\b|root cause|fail(ed)? rt|rejected|what went wrong", t):
            return [("wps_check", seq_arg or {"sequence": self.default_seq(ctx)}),
                    ("assess_weld", seq_arg or {}), ("show_weld", {"event": "defect"})]
        if re.search(r"show me|jump|go to|play|moment|when did", t):
            ev = defect or ("violation" if "viol" in t or "wps" in t else "defect")
            args = {"event": ev}
            m = re.search(r"(\d+(?:\.\d+)?)\s*s(ec)?\b", t)
            if m:
                args = {"t": float(m.group(1))}
            return [("show_weld", {**seq_arg, **args})]
        if re.search(r"assess|inspect|grade|\bqa\b|accept|reasoner|cosmos.*look", t):
            return [("assess_weld", seq_arg)]
        m = re.search(r"(current|voltage|travel|gas|wire|heat|interpass)\w*.*?(above|over|>|below|under|<|greater|less)\s*"
                      r"(?:than\s*)?(\d+(?:\.\d+)?)", t)
        if m:
            p = {"current": "current_a", "voltage": "voltage_v", "travel": "travel_speed_mmps", "gas": "gas_lpm",
                 "wire": "wfs_mpm", "heat": "heat_input_kj_mm", "interpass": "interpass_c"}[m.group(1)]
            key = "gt" if m.group(2) in ("above", "over", ">", "greater") else "lt"
            return [("find_anomalies", {"param": p, key: float(m.group(3))})]
        if re.search(r"wps|violat|out of (spec|procedure)|broke|non.?conform|outside (their |the |its )?"
                     r"(published )?(settings|window|limits|range)|published settings", t):
            if ref:
                return [("wps_check", seq_arg)]
            args = {"only_wps_violations": True}
            if "week" in t:
                args["since_days"] = 7
            elif "today" in t:
                args["since_days"] = 1
            return [("list_passes", args)]
        if re.search(r"summar|stats|average|mean|signal", t) and (ref or ctx.get("last_sequence")):
            return [("get_signals_summary", seq_arg or {"sequence": ctx.get("last_sequence")})]
        if re.search(r"list|passes|spool|joint|welds|\brt\b|reject|coupon|workpiece|beads|layers", t):
            args: dict[str, Any] = {}
            sp = re.search(r"\b(sp|v)[-_ ]?(\d{3,5})", t)
            if sp:
                args["spool_id"] = f"{sp.group(1).upper()}-{sp.group(2)}"
            co = re.search(r"\b(pp\d{1,2}|wid\d{3})\b", t)
            if co:
                args["spool_id"] = co.group(1).upper()
            for pr in ("gtaw", "fcaw", "smaw", "gmaw", "tgmaw", "saw"):
                if re.search(rf"\b{pr}\b", t):
                    args["process"] = pr.upper()
            if "aimen" in t:
                args["source"] = "aimen_saw"
            elif "bam" in t:
                args["source"] = "bam_tgmaw"
            elif re.search(r"simulat|synthetic", t):
                args["source"] = "sim"
            if "reject" in t or "fail" in t:
                args["rt_result"] = "reject"
            if "defect" in t:
                args["has_defect"] = True
            return [("list_passes", args)]
        return []

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        plan = self.route(text, session.ctx)
        if not plan:
            async for e in _stream_text(_help_text(self.tb.data), self.delay):
                yield e
            yield {"type": "done"}
            return
        results: list[tuple[str, dict, dict]] = []
        for name, args in plan:
            out = None
            async for ev in self.run_tool(session, name, args):
                if isinstance(ev, tuple):
                    out = ev[1]
                else:
                    yield ev
            results.append((name, args, out or {}))
            if out and out.get("is_error"):
                break
        async for e in _stream_text(self.narrate(text, results, session.ctx), self.delay):
            yield e
        yield {"type": "done"}

    # ---------------------------------------------------------------- narration
    def narrate(self, text: str, results: list[tuple[str, dict, dict]], ctx: dict) -> str:
        parts: list[str] = []
        by = {n: o.get("result") or {} for n, _, o in results}
        for name, _, out in results:
            if out.get("is_error"):
                return f"I couldn't run **{name}**: {(out.get('result') or {}).get('error')}."
        if "list_passes" in by:
            r = by["list_passes"]
            ps = r.get("passes", [])
            if not ps:
                return "No passes match that filter."
            viol = [p for p in ps if p["wps_violations"]]
            rej = [p for p in ps if p["rt"] == "reject"]
            real = [p for p in ps if not p.get("simulated", True)]
            if real and viol and len(viol) == len(ps):
                parts.append(f"**{len(viol)} pass{'es' if len(viol) != 1 else ''} ran outside their published settings:**")
                for p in viol[:8]:
                    sim = " · *SIMULATED*" if p.get("simulated") else ""
                    parts.append(f"- `{p['sequence']}` — {', '.join(_pname(x) for x in p['violated'])} outside the "
                                 f"window · {p.get('ndt') or p['rt']}{sim}")
                basis = "; ".join(r.get("wps_basis") or []) or "published nominal settings ±10 %"
                parts.append(f"\nWindow = {basis}. An excursion is a deviation from the published settings, not "
                             "by itself a defect: compare with the NDT column.")
            elif real:
                n_ind = sum(1 for p in ps if p["rt"] == "indications")
                parts.append(f"**{len(ps)} passes** match — {n_ind} with NDT indications, {len(viol)} outside their "
                             "published settings.")
                for p in ps[:8]:
                    sim = " · *SIMULATED*" if p.get("simulated") else ""
                    parts.append(f"- `{p['sequence']}` · {p['process']} {p['pass']} · {p.get('ndt') or p['rt']}"
                                 f"{' · ' + str(p['wps_violations']) + ' excursion(s)' if p['wps_violations'] else ''}{sim}")
            elif viol and len(viol) == len(ps):
                parts.append(f"**{len(viol)} passes broke their WPS** in this window; "
                             f"**{sum(p['rt'] == 'reject' for p in viol)}** of them were rejected at RT.")
                for p in viol[:6]:
                    rt = "RT **reject**" if p["rt"] == "reject" else f"RT {p['rt']}"
                    defs = ", ".join(d.replace("_", " ") for d in p["defects"])
                    parts.append(f"- `{p['sequence']}` — {', '.join(_pname(x) for x in p['violated'])} out of limits"
                                 f" · {rt}{' · ' + defs if defs else ''}")
                worst = next((p for p in viol if p["rt"] == "reject" and "gas_lpm" in p["violated"]), viol[0])
                parts.append(f"\nWorth a look first: **{worst['sequence']}** ({worst['pass']}). "
                             "Ask me *why it failed RT*.")
            else:
                parts.append(f"**{len(ps)} passes** match — {len(rej)} RT reject, {len(viol)} with WPS violations.")
                for p in ps[:8]:
                    parts.append(f"- `{p['sequence']}` · {p['process']} {p['pass']} · RT {p['rt']}"
                                 f"{' · ' + str(p['wps_violations']) + ' viol.' if p['wps_violations'] else ''}")
        if "find_anomalies" in by:
            r = by["find_anomalies"]
            parts.append(f"**{r.get('title')}**.")
            for h in r.get("hits", [])[:5]:
                parts.append(f"- `{h['sequence']}` {h['param']} = {h['value']} from {h['t_start_s']}s to {h['t_end_s']}s")
        if "ndt_summary" in by:
            parts.append(_narrate_ndt(by["ndt_summary"]))
        if "show_photo" in by:
            r = by["show_photo"]
            if r.get("error"):
                parts.append(r["error"])
            else:
                parts.append(f"The bead photo of **{r['sequence']}** is in the replay panel ({r.get('caption')}) — "
                             "scroll to zoom, drag to pan, thumbnails below.")
                if r.get("ndt_indications"):
                    parts.append(f"NDT found **{r['ndt_indications']} indication(s)** in this bead; RT/UT indications are "
                                 "internal, so they are usually not visible on the bead surface.")
        if "wps_check" in by:
            r = by["wps_check"]
            seq = r["sequence"]
            if r.get("wps_basis") and r["violations"]:
                parts.append(f"**{seq}** left its published settings {len(r['violations'])}×, "
                             f"{r['out_of_spec_seconds']}s in total (window: {r['wps_basis']}):")
                for v in r["violations"][:5]:
                    lim = (f"{v['limit_lo']}–{v['limit_hi']}" if v.get("limit_lo") is not None else f"≤ {v['limit_hi']}")
                    parts.append(f"- **{_pname(v['param'])}** {v['value']} (window {lim}) · "
                                 f"{v['t_start_s']:.1f}–{v['t_end_s']:.1f}s")
            elif r.get("wps_basis"):
                parts.append(f"**{seq}** stayed inside its published settings for the whole pass "
                             f"(NDT: {r['rt_result'].replace('_', ' ')}).")
            elif r["violations"]:
                parts.append(f"**{seq}** (RT **{r['rt_result']}**) broke **{r['wps_id']}** "
                             f"{len(r['violations'])}×, {r['out_of_spec_seconds']}s out of spec in total:")
                for v in r["violations"][:5]:
                    lim = (f"{v['limit_lo']}–{v['limit_hi']}" if v.get("limit_lo") is not None else f"≤ {v['limit_hi']}")
                    parts.append(f"- **{_pname(v['param'])}** {v['value']} (WPS {lim}) · "
                                 f"{v['t_start_s']:.1f}–{v['t_end_s']:.1f}s")
            else:
                parts.append(f"**{seq}** stayed inside **{r['wps_id']}** for the whole pass (RT {r['rt_result']}).")
        if "assess_weld" in by:
            a = by["assess_weld"]
            ds = ", ".join(f"{d['type'].replace('_', ' ')} ({round(100 * d['confidence'])}%)" for d in a.get("defects", []))
            src = str(a.get("source") or "")
            who = ("Cosmos 3 Reasoner" if (src.startswith("cosmos3") or src in ("nim", "cache"))
                   else "Rule-based check (Cosmos 3 offline)")
            what = {"photo": "the bead photo", "signals": "the signals"}.get(a.get("input_kind"), "clip")
            span = f" {a.get('t0_s')}–{a.get('t1_s')}s" if what == "clip" else ""
            parts.append(f"\n**{who}:** {(a.get('verdict') or '').upper()}"
                         f"{' — ' + ds if ds else ''} on {what}{span}.")
            if a.get("ndt_indications"):
                parts.append("**NDT ground truth:** " + ", ".join(f"{k.replace('_', ' ')} ×{v}"
                                                                 for k, v in a["ndt_indications"].items()))
            parts.append(f"**{'Cause' if a.get('ndt_indications') is not None else 'Root cause'}:** {a.get('root_cause')}")
            parts.append(f"**Action:** {a.get('recommended_action')}")
        if "show_weld" in by and "assess_weld" not in by:
            r = by["show_weld"]
            if "ndt_summary" in by:
                parts.insert(0, f"Replay is at **{r['sequence']} @ {r['t']:.1f}s** — the NDT strip under the plots "
                                "marks every indication window.\n")
            else:
                parts.append(f"Replay is at **{r['sequence']} @ {r['t']:.1f}s** — watch the camera and the "
                             "signal plots: the shaded band is the settings window, red marks where it was left.")
        elif "show_weld" in by:
            r = by["show_weld"]
            parts.append(f"\nI've jumped the replay to **{r['t']:.1f}s**, where it starts.")
        if "what_if" in by:
            r = by["what_if"]
            parts.append(f"Rendering a **what-if** on `{r.get('sequence')}` with the Cosmos 3 Generator: "
                         f"*{r.get('change')}* The video lands in the card above when done (job `{r.get('job_id')}`). "
                         "It is a generated, synthetic video — not a measurement.")
        if "sim2real" in by:
            r = by["sim2real"]
            parts.append(f"Running **Cosmos 3 Transfer** on the simulator's `{r.get('control')}` control video of "
                         f"`{r.get('sequence')}` → photoreal footage. Same weld, same timing, real-looking pixels: "
                         "every simulated pass becomes labelled training video.")
        if "dataset_stats" in by:
            r = by["dataset_stats"]
            if r.get("error"):
                parts.append(r["error"])
            else:
                d = results[[n for n, _, _ in results].index("dataset_stats")][1].get("defect")
                if d:
                    parts.append(f"We have **{r['kept']} curated {d.replace('_', ' ')} clips** kept for training "
                                 f"({r['total']} before filtering, {r['rejected']} rejected) from {r['sequences']} "
                                 f"pass{'es' if r['sequences'] != 1 else ''}.")
                else:
                    parts.append(f"Curated dataset: **{r['kept']} of {r['total']} clips kept** from {r['sequences']} passes.")
                    top = sorted(r["by_defect"].items(), key=lambda kv: -kv[1]["kept"])[:6]
                    parts.append("By class: " + ", ".join(f"{k.replace('_', ' ')} {v['kept']}" for k, v in top) + ".")
                if d and r["kept"] < 50:
                    parts.append("That's thin for a defect classifier — I can **synthesize more with Cosmos** "
                                 "(Generate tab → *Porosity synthesis*).")
        if "search_clips" in by and "dataset_stats" not in by:
            parts.append(f"Found **{by['search_clips'].get('count')} clips**; top matches below.")
        if "get_signals_summary" in by and by["get_signals_summary"].get("lead_vs_trail") and any(
                "params" in a for n, a, _ in results if n == "get_signals_summary"):
            r = by["get_signals_summary"]
            parts.append(f"**Lead vs trail electrode — {r['sequence']}** (means over the steady arc, "
                         f"{r['t0']:.0f}–{r['t1']:.0f} s minus start/stop transients):")
            parts.append("| | Lead | Trail | Δ trail |\n|---|---:|---:|---:|")
            for p, v in r["lead_vs_trail"].items():
                d = (v["trail"] - v["lead"]) / v["lead"] * 100 if v["lead"] else 0
                nd = {"A": 0, "V": 1}.get(v["unit"], 2)
                parts.append(f"| {_pname(p).split(' (')[0]} | {v['lead']:.{nd}f} {v['unit']} | {v['trail']:.{nd}f} {v['unit']} | {d:+.0f} % |")
            out_ = [p for p, st in r["stats"].items() if st["pct_out_of_wps"]]
            parts.append("\nBoth wires stay inside the published settings on the steady arc." if not out_ else
                         "\nOutside the published settings: " + ", ".join(f"{_pname(p)} {r['stats'][p]['pct_out_of_wps']}% of samples" for p in out_) + ".")
        elif "get_signals_summary" in by:
            r = by["get_signals_summary"]
            parts.append(f"**{r['sequence']}** {r['t0']}–{r['t1']}s:")
            for p, s in r["stats"].items():
                flag = f" · **{s['pct_out_of_wps']}% out of WPS**" if s["pct_out_of_wps"] else ""
                parts.append(f"- {_pname(p)}: mean {s['mean']} {s['unit']} (min {s['min']}, max {s['max']}){flag}")
            for p, note in (r.get("missing_signals") or {}).items():
                parts.append(f"- {_pname(p)}: {note}")
        return "\n".join(parts) if parts else "Done."


def _narrate_ndt(r: dict) -> str:
    if r.get("sequence"):
        seq, counts = r["sequence"], r.get("counts") or {}
        if not r.get("inspected"):
            return (f"**{seq}** was **not NDT-inspected** — there is no ground truth on internal defects for this "
                    "pass, so the data can't say whether it is sound.")
        if not counts:
            return f"**{seq}**: {'/'.join(r.get('methods') or ['NDT'])} found **no indications** in this pass."
        inds = r.get("indications") or []
        ts = [i["t_start"] for i in inds if i.get("t_start") is not None]
        xs = [i.get("x_start_mm") for i in inds if i.get("x_start_mm") is not None]
        where = f" between {min(ts):.1f} and {max(ts):.1f} s" if ts else ""
        where += f" (x ≈ {min(xs):.0f}–{max(xs):.0f} mm)" if xs else ""
        out = [f"**{seq}** — {'/'.join(r.get('methods') or ['NDT'])}: "
               + ", ".join(f"**{k.replace('_', ' ')} ×{v}**" for k, v in counts.items()) + where + "."]
        chk = r.get("signal_check") or {}
        if chk.get("params"):
            rows = []
            for p, c in chk["params"].items():
                rows.append(f"| {_pname(p)} | {c['mean_in']} | {c['mean_clean']} | "
                            f"{c['windows_outside_band']}/{c['windows_checked']} |")
            n = chk.get("n_windows")
            if chk.get("signature"):
                sig = [p for p, c in chk["params"].items() if c["signature"]]
                out.append(f"\n**Did the arc signals change? Partly** — {', '.join(_pname(p) for p in sig)} differ inside "
                           f"the {n} indication windows. That is a correlation, not a proven cause.")
            else:
                c0 = chk["params"].get("current_a") or next(iter(chk["params"].values()))
                out.append(f"\n**Did the arc signals change? No.** Inside the {n} indication windows the lead current "
                           f"averages {c0['mean_in']} {c0['unit']} vs {c0['mean_clean']} {c0['unit']} in the clean weld, "
                           "and no signal leaves its normal band more often than chance.")
            out.append("\n| Signal | in windows | clean weld | windows outside 1–99 % band |\n|---|---:|---:|---:|")
            out.extend(rows)
            if not chk.get("signature"):
                out.append("\nThis real defect left **no electrical signature** — a current/voltage monitor alone would "
                           "not have caught it, and the data does not show its cause.")
        if r.get("note"):
            out.append(f"\n_{r['note']}_")
        return "\n".join(out)
    by_type = r.get("indications_by_type") or {}
    head = (f"**{r.get('with_indications')} of {r.get('inspected')} inspected passes have NDT indications**"
            + (f" on {r['coupon']}" if r.get("coupon") else ""))
    if by_type:
        head += ": " + ", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]))
    out = [head + "."]
    if r.get("materials"):
        out[0] += f" Material: {', '.join(r['materials'])}."
    for x in (r.get("rows") or [])[:8]:
        sim = " · *SIMULATED*" if x.get("simulated") else ""
        out.append(f"- `{x['sequence']}` — {x['ndt']}{sim}")
    if r.get("not_inspected"):
        out.append(f"\n{r['not_inspected']} passes (BAM tandem-GMAW) were **not inspected** — no NDT ground truth there.")
    if r.get("simulated_excluded"):
        out.append(f"{r['simulated_excluded']} *SIMULATED* pass(es) from the WeldCell simulator are left out — not real NDT.")
    out.append("\nAsk *“did the arc signals change?”* on one of them to check for an electrical signature.")
    return "\n".join(out)


_DEFAULT_SEQ = "SP-1043_J07_P3_FCAW"
_HELP_REAL = ("I'm the Weld Copilot, working on real AIMEN SAW and BAM tandem-GMAW data. Try: **Which SA-516 vessel "
              "passes had NDT indications?**, **Show porosity on PP5 bead 2 — did the arc signals change?**, **Compare "
              "lead vs trail current on WID617 layer 3**, **Which passes ran outside their published settings?**, "
              "**Inspect the bead photo of PP5 bead 2**, or **What if travel speed dropped 20 % on WID630 layer 4?**")


def _help_text(data) -> str:
    try:
        real = any(not r.get("synthetic", True) for r in data.all_passes())
    except Exception:  # noqa: BLE001
        real = False
    return _HELP_REAL if real else _HELP


_HELP = ("I'm the Weld Copilot. Try: **Which welds this week broke the WPS?**, **Why did SP-1043 J07 fill pass "
         "fail RT?**, **Show me the moment porosity started**, **What if travel speed dropped 20% on that pass?**, "
         "**Make photoreal training video from the simulator**, or **How many curated clips do we have for porosity?**")


def _pname(p: str) -> str:
    from .data import SIGNAL_INFO
    i = SIGNAL_INFO.get(p)
    return f"{i['label']} ({i['unit']})" if i else p


def fit_to_sdk(fn: Any, kw: dict) -> dict:
    """Keep the request identical on the wire when the installed `anthropic` SDK predates a parameter.

    Request fields the SDK method does not declare (e.g. `fallbacks` on older releases) move into
    `extra_body`; `betas` becomes the `anthropic-beta` header. Newer SDKs get the kwargs unchanged.
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    kw = dict(kw)
    if "betas" in kw and "betas" not in params:
        betas = kw.pop("betas")
        headers = dict(kw.get("extra_headers") or {})
        headers["anthropic-beta"] = ",".join([*filter(None, [headers.get("anthropic-beta")]), *betas])
        kw["extra_headers"] = headers
    unknown = [k for k in kw if k not in params and k not in ("extra_headers", "extra_body")]
    if unknown:
        extra = dict(kw.get("extra_body") or {})
        for k in unknown:
            extra[k] = kw.pop(k)
        kw["extra_body"] = extra
        log.info("anthropic SDK %s lacks %s; sending them in extra_body", _sdk_version(), ", ".join(unknown))
    return kw


def _sdk_version() -> str:
    try:
        import anthropic
        return anthropic.__version__
    except Exception:  # noqa: BLE001
        return "?"


def _extract_ref(text: str) -> str | None:
    m = re.search(r"\b(?:PP\d{1,2}|WID[-_ ]?\d{3})(?:[_ ]+(?:B\d{1,2}|bead\s*\d{1,2}|L\d|layer\s*\d))?(?:_(?:SAW|TGMAW|GMAW))?",
                  text, re.I)
    if m:
        return m.group(0)
    m = re.search(r"(?:SP|V)[-_ ]?\d{3,5}(?:[_ ]+(?:J|CS)\d{1,3})?(?:[_ ]+P\d)?(?:[_ ]+(?:GTAW|FCAW|SMAW|GMAW|SAW))?", text, re.I)
    if not m:
        return None
    ref = m.group(0)
    rest = text[m.end(): m.end() + 30].lower()
    for pt in ("root", "hot", "fill", "cap"):
        if pt in rest or pt in ref.lower():
            ref += f" {pt}"
            break
    return ref


# =============================================================================== claude
class ClaudeAgent(BaseAgent):
    """Anthropic Messages API, manual streaming tool-use loop (we own the loop so tool calls render live)."""
    name = "claude"

    def __init__(self, toolbox: Toolbox, client: Any = None, model: str | None = None, max_turns: int = 8):
        super().__init__(toolbox)
        self.model = model or os.environ.get("COPILOT_MODEL", config.CLAUDE_DEFAULT_MODEL)
        self.effort = os.environ.get("COPILOT_EFFORT", "medium")
        self.max_turns = max_turns
        self._client = client
        self.label = f"Claude · {self.model}"

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        return self._client

    def tools(self) -> list[dict]:
        return [{**t, "eager_input_streaming": True} for t in TOOL_SPECS]

    def system(self) -> list[dict]:
        names = [r["sequence_name"] for r in self.tb.data.all_passes()]
        inventory = f"Passes in Mosaico ({len(names)}): " + ", ".join(names[:80])
        return [{"type": "text", "text": system_prompt(self.tb.data), "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": inventory}]

    def request_kwargs(self, messages: list) -> dict:
        kw: dict[str, Any] = dict(model=self.model, max_tokens=16000, system=self.system(), tools=self.tools(),
                                  messages=messages, thinking={"type": "adaptive", "display": "summarized"},
                                  output_config={"effort": self.effort})
        if self.model in _FALLBACK_MODELS:
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["fallbacks"] = "default"
        return kw

    def stream_kwargs(self, messages: list) -> dict:
        return fit_to_sdk(self.client.beta.messages.stream, self.request_kwargs(messages))

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        import anthropic
        msgs = session.claude_messages
        checkpoint = len(msgs)
        msgs.append({"role": "user", "content": text})
        try:
            json_retries = 0
            for _turn in range(self.max_turns):
                try:
                    async with self.client.beta.messages.stream(**self.stream_kwargs(msgs)) as stream:
                        async for event in stream:
                            et = getattr(event, "type", None)
                            if et == "text":
                                yield {"type": "text", "delta": event.text}
                            elif et == "thinking":
                                yield {"type": "thinking", "delta": event.thinking}
                        final = await stream.get_final_message()
                    json_retries = 0
                except ValueError:
                    # tool-input JSON the SDK could not parse at all (eager input streaming); re-issue the turn
                    json_retries += 1
                    if json_retries > 2:
                        raise
                    continue
                stop = getattr(final, "stop_reason", None)
                if stop == "refusal":
                    yield {"type": "text", "delta": "\n\n_The model declined this request._"}
                    msgs.append({"role": "assistant", "content": final.content})
                    break
                if stop == "pause_turn":
                    msgs.append({"role": "assistant", "content": final.content})
                    continue
                tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
                msgs.append({"role": "assistant", "content": final.content})
                if not tool_uses:
                    break
                if stop == "max_tokens":
                    raise RuntimeError("tool input truncated at max_tokens")
                results = []
                for b in tool_uses:
                    err = validate_input(b.name, b.input)
                    if err:
                        results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                        "content": json.dumps({"INVALID_INPUT": err})})
                        continue
                    out = {}
                    async for ev in self.run_tool(session, b.name, dict(b.input), b.id):
                        if isinstance(ev, tuple):
                            out = ev[1]
                        else:
                            yield ev
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": tool_result_text(out),
                                    **({"is_error": True} if out.get("is_error") else {})})
                msgs.append({"role": "user", "content": results})
            yield {"type": "done"}
        except anthropic.AuthenticationError:
            del msgs[checkpoint:]
            yield {"type": "error", "message": "Anthropic authentication failed - set ANTHROPIC_API_KEY "
                                               "(or run with COPILOT_LLM=scripted)."}
        except anthropic.RateLimitError:
            del msgs[checkpoint:]
            yield {"type": "error", "message": "Claude API rate limit - retry in a moment."}
        except anthropic.APIStatusError as e:
            del msgs[checkpoint:]
            yield {"type": "error", "message": f"Claude API error {e.status_code}: {e.message}"}
        except anthropic.APIConnectionError:
            del msgs[checkpoint:]
            yield {"type": "error", "message": "Cannot reach the Claude API (network)."}
        except Exception as e:  # noqa: BLE001
            del msgs[checkpoint:]
            log.exception("claude agent failed")
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


# =============================================================================== cosmos reasoner
REACT_PROTOCOL = """You can call tools. To call one, reply with ONLY a JSON object on one line:
{"tool": "<name>", "args": {...}}
When you have enough information, reply with ONLY: {"final": "<markdown answer for the user>"}
Tools:
"""


class CosmosAgent(BaseAgent):
    """Cosmos 3 Reasoner over its OpenAI-compatible vLLM endpoint."""
    name = "cosmos"

    def __init__(self, toolbox: Toolbox, client: Any = None, model: str | None = None, max_turns: int = 6):
        super().__init__(toolbox)
        self.model = model or os.environ.get("COSMOS_REASONER_MODEL", "nvidia/Cosmos3-Nano")
        self.base_url = os.environ.get("COSMOS_REASONER_URL", "http://localhost:8000/v1")
        self._client = client
        self.max_turns = max_turns
        self.mode = os.environ.get("COPILOT_COSMOS_TOOLS", "auto")  # auto | native | react
        self.label = f"Cosmos 3 Reasoner · {self.model}"

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=os.environ.get("COSMOS_API_KEY", "EMPTY"))
        return self._client

    def oai_tools(self) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in TOOL_SPECS]

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        msgs = session.oai_messages
        if not msgs:
            msgs.append({"role": "system", "content": system_prompt(self.tb.data)})
        checkpoint = len(msgs)
        msgs.append({"role": "user", "content": text})
        try:
            if self.mode in ("auto", "native"):
                try:
                    async for ev in self._native(session, msgs):
                        yield ev
                    yield {"type": "done"}
                    return
                except Exception as e:  # noqa: BLE001
                    if self.mode == "native" or not _looks_like_no_tools(e):
                        raise
                    log.info("reasoner has no function calling (%s) - switching to ReAct protocol", e)
                    self.mode = "react"
                    del msgs[checkpoint + 1:]
            async for ev in self._react(session, msgs):
                yield ev
            yield {"type": "done"}
        except Exception as e:  # noqa: BLE001
            del msgs[checkpoint:]
            log.exception("cosmos agent failed")
            yield {"type": "error", "message": f"Cosmos Reasoner error ({self.base_url}): {type(e).__name__}: {e}"}

    async def _native(self, session: Session, msgs: list) -> AsyncIterator[dict]:
        for _ in range(self.max_turns):
            stream = await self.client.chat.completions.create(model=self.model, messages=msgs, tools=self.oai_tools(),
                                                               tool_choice="auto", stream=True, max_tokens=2048)
            content, calls = "", {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                if getattr(d, "content", None):
                    content += d.content
                    yield {"type": "text", "delta": d.content}
                for tc in getattr(d, "tool_calls", None) or []:
                    c = calls.setdefault(tc.index, {"id": tc.id or f"call_{tc.index}", "name": "", "args": ""})
                    if tc.function and tc.function.name:
                        c["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        c["args"] += tc.function.arguments
            if not calls:
                msgs.append({"role": "assistant", "content": content})
                return
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for c in calls.values()]})
            for c in calls.values():
                try:
                    args = json.loads(c["args"] or "{}")
                except ValueError:
                    args = None
                out = {"result": {"error": "invalid JSON arguments"}, "is_error": True}
                if isinstance(args, dict):
                    async for ev in self.run_tool(session, c["name"], args, c["id"]):
                        if isinstance(ev, tuple):
                            out = ev[1]
                        else:
                            yield ev
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result_text(out, 4000)})

    async def _react(self, session: Session, msgs: list) -> AsyncIterator[dict]:
        proto = REACT_PROTOCOL + "\n".join(
            f"- {t['name']}: {t['description']} args={json.dumps(t['input_schema'].get('properties', {}))}"
            for t in TOOL_SPECS)
        for _ in range(self.max_turns):
            req = [{"role": "system", "content": system_prompt(self.tb.data) + "\n\n" + proto}] + [m for m in msgs if m["role"] != "system"]
            resp = await self.client.chat.completions.create(model=self.model, messages=req, max_tokens=2048,
                                                             temperature=0.2)
            reply = (resp.choices[0].message.content or "").strip()
            obj = _first_json(reply)
            if obj and obj.get("tool"):
                msgs.append({"role": "assistant", "content": json.dumps(obj)})
                out = {}
                async for ev in self.run_tool(session, obj["tool"], obj.get("args") or {}):
                    if isinstance(ev, tuple):
                        out = ev[1]
                    else:
                        yield ev
                msgs.append({"role": "user", "content": f"Tool result for {obj['tool']}: {tool_result_text(out, 4000)}"})
                continue
            final = obj.get("final") if obj else reply
            msgs.append({"role": "assistant", "content": final})
            async for e in _stream_text(str(final), 0.005):
                yield e
            return
        yield {"type": "text", "delta": "_(stopped after too many tool calls)_"}


def _looks_like_no_tools(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("tool", "function", "auto", "400", "not supported", "enable-auto-tool-choice"))


def _first_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


# =============================================================================== factory
def make_agent(toolbox: Toolbox, backend: str | None = None) -> BaseAgent:
    backend = backend or config.llm_backend()
    if backend == "claude":
        return ClaudeAgent(toolbox)
    if backend == "cosmos":
        return CosmosAgent(toolbox)
    return ScriptedAgent(toolbox)
