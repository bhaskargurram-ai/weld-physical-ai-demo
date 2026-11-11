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

    def route(self, text: str, ctx: dict) -> list[tuple[str, dict]]:
        """Map an utterance to a tool plan: [(tool, args), ...]."""
        t = text.lower()
        ref = _extract_ref(text)
        seq_arg = {"sequence": ref} if ref else {}
        defect = next((d for d in ("porosity", "lack_of_fusion", "undercut", "burn_through", "arc_instability", "crack")
                       if d.replace("_", " ") in t or d.split("_")[0] in t.replace("-", " ")), None)
        if re.search(r"photo.?real|sim.?(to|2|→|->)\s?real|transfer|from the simulator|synthetic video", t):
            kind = "depth" if "depth" in t else "edge"
            return [("sim2real", {**(seq_arg or {"sequence": ctx.get("last_sequence") or _DEFAULT_SEQ}), "kind": kind})]
        if re.search(r"what if|what-if|what would|simulate|counterfactual", t):
            change = re.sub(r"^.*?(what if|what-if|what would happen if|simulate)\s*", "", text, flags=re.I)
            change = re.sub(r"\s+on (that|this|the) pass\??$", "", change, flags=re.I).strip(" ?") or "travel speed -20%"
            return [("what_if", {**(seq_arg or {"sequence": ctx.get("last_sequence") or _DEFAULT_SEQ}), "change": change})]
        if re.search(r"how many|curated|dataset|training (clips|data)|clips? (do|for)", t):
            plan = [("dataset_stats", {"defect": defect} if defect else {})]
            if defect:
                plan.append(("search_clips", {"text": defect.replace("_", " "), "limit": 4, "kept_only": True}))
            return plan
        if re.search(r"search|find clips|clips (with|showing)", t):
            return [("search_clips", {"text": text})]
        if re.search(r"\bwhy\b|root cause|fail(ed)? rt|rejected|what went wrong", t):
            return [("wps_check", seq_arg or {"sequence": ctx.get("last_sequence") or _DEFAULT_SEQ}),
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
        if re.search(r"wps|violat|out of (spec|procedure)|broke|non.?conform", t):
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
        if re.search(r"list|passes|spool|joint|welds|rt|reject", t):
            args: dict[str, Any] = {}
            sp = re.search(r"sp[-_ ]?(\d{3,5})", t)
            if sp:
                args["spool_id"] = f"SP-{sp.group(1)}"
            for pr in ("gtaw", "fcaw", "smaw", "gmaw", "saw"):
                if re.search(rf"\b{pr}\b", t):
                    args["process"] = pr.upper()
            if "reject" in t or "fail" in t:
                args["rt_result"] = "reject"
            if "defect" in t:
                args["has_defect"] = True
            return [("list_passes", args)]
        return []

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        plan = self.route(text, session.ctx)
        if not plan:
            async for e in _stream_text(_HELP, self.delay):
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
            if viol and len(viol) == len(ps):
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
        if "wps_check" in by:
            r = by["wps_check"]
            seq = r["sequence"]
            if r["violations"]:
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
            parts.append(f"\n**Cosmos 3 Reasoner:** {(a.get('verdict') or '').upper()}"
                         f"{' — ' + ds if ds else ''} on clip {a.get('t0_s')}–{a.get('t1_s')}s.")
            parts.append(f"**Root cause:** {a.get('root_cause')}")
            parts.append(f"**Action:** {a.get('recommended_action')}")
        if "show_weld" in by and "assess_weld" not in by:
            r = by["show_weld"]
            parts.append(f"Replay is at **{r['sequence']} @ {r['t']:.1f}s** — watch the torch camera and the "
                         "signal plots: the shaded band is the WPS window, red marks where it was left.")
        elif "show_weld" in by:
            r = by["show_weld"]
            parts.append(f"\nI've jumped the replay to **{r['t']:.1f}s**, where it starts.")
        if "what_if" in by:
            r = by["what_if"]
            parts.append(f"Rendering a **what-if** on `{r.get('sequence')}` with the Cosmos 3 Generator: "
                         f"*{r.get('change')}* The video lands in the card above when done (job `{r.get('job_id')}`).")
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
        if "get_signals_summary" in by:
            r = by["get_signals_summary"]
            parts.append(f"**{r['sequence']}** {r['t0']}–{r['t1']}s:")
            for p, s in r["stats"].items():
                flag = f" · **{s['pct_out_of_wps']}% out of WPS**" if s["pct_out_of_wps"] else ""
                parts.append(f"- {_pname(p)}: mean {s['mean']} {s['unit']} (min {s['min']}, max {s['max']}){flag}")
        return "\n".join(parts) if parts else "Done."


_DEFAULT_SEQ = "SP-1043_J07_P3_FCAW"
_HELP = ("I'm the Weld Copilot. Try: **Which welds this week broke the WPS?**, **Why did SP-1043 J07 fill pass "
         "fail RT?**, **Show me the moment porosity started**, **What if travel speed dropped 20% on that pass?**, "
         "**Make photoreal training video from the simulator**, or **How many curated clips do we have for porosity?**")


def _pname(p: str) -> str:
    from .data import SIGNAL_INFO
    i = SIGNAL_INFO.get(p)
    return f"{i['label']} ({i['unit']})" if i else p


def _extract_ref(text: str) -> str | None:
    m = re.search(r"SP[-_ ]?\d{3,5}(?:[_ ]+J\d{1,3})?(?:[_ ]+P\d)?(?:[_ ]+(?:GTAW|FCAW|SMAW|GMAW|SAW))?", text, re.I)
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
        return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": inventory}]

    def request_kwargs(self, messages: list) -> dict:
        kw: dict[str, Any] = dict(model=self.model, max_tokens=16000, system=self.system(), tools=self.tools(),
                                  messages=messages, thinking={"type": "adaptive", "display": "summarized"},
                                  output_config={"effort": self.effort})
        if self.model in _FALLBACK_MODELS:
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["fallbacks"] = "default"
        return kw

    async def run(self, session: Session, text: str) -> AsyncIterator[dict]:
        import anthropic
        msgs = session.claude_messages
        checkpoint = len(msgs)
        msgs.append({"role": "user", "content": text})
        try:
            json_retries = 0
            for _turn in range(self.max_turns):
                try:
                    async with self.client.beta.messages.stream(**self.request_kwargs(msgs)) as stream:
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
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
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
            req = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + proto}] + [m for m in msgs if m["role"] != "system"]
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
