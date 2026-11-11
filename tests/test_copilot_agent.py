"""Weld Copilot agent tests: scripted intent routing, Claude tool loop (mocked client), Cosmos ReAct (mocked)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from copilot.agent import ClaudeAgent, CosmosAgent, ScriptedAgent, Session, _first_json
from copilot.data import DataLayer
from copilot.jobs import JobManager
from copilot.models import WorldModels
from copilot.tools import TOOL_SPECS, Toolbox, validate_input

DEMO = [
    ("Which welds this week broke the WPS?", ["list_passes"]),
    ("Why did SP-1043 J07 fill pass fail RT?", ["wps_check", "assess_weld", "show_weld"]),
    ("Show me the moment porosity started", ["show_weld"]),
    ("What if travel speed dropped 20% on that pass?", ["what_if"]),
    ("Make photoreal training video from the simulator", ["sim2real"]),
    ("How many curated clips do we have for porosity?", ["dataset_stats", "search_clips"]),
]


@pytest.fixture(scope="module")
def toolbox(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("copilot_agent_fx")
    from copilot.fixtures import make_dataset
    make_dataset(root, videos=False, workers=1)
    for k, v in {"DATA_DIR": str(root), "COPILOT_DATA": "local", "COPILOT_MOCK_DELAY": "0",
                 "MOSAICO_OFFLINE": "1", "COSMOS_BACKEND": "replay"}.items():
        mp.setenv(k, v)
    data = DataLayer()
    yield Toolbox(data, WorldModels(data), JobManager())
    mp.undo()


def collect(agen) -> list:
    async def run():
        return [e async for e in agen]
    return asyncio.run(run())


# ---------------------------------------------------------------- scripted router
@pytest.mark.parametrize("text,tools", DEMO)
def test_scripted_routes_demo_prompts(toolbox, text, tools):
    plan = ScriptedAgent(toolbox, delay=0).route(text, {"last_sequence": "SP-1043_J07_P3_FCAW"})
    assert [t for t, _ in plan] == tools


def test_scripted_route_args(toolbox):
    a = ScriptedAgent(toolbox, delay=0)
    plan = a.route("Which welds this week broke the WPS?", {})
    assert plan[0][1] == {"only_wps_violations": True, "since_days": 7}
    plan = a.route("What if travel speed dropped 20% on that pass?", {"last_sequence": "SP-1042_J05_P2_FCAW"})
    assert plan[0][1]["sequence"] == "SP-1042_J05_P2_FCAW" and "travel speed dropped 20%" in plan[0][1]["change"]
    plan = a.route("find passes where gas below 10", {})
    assert plan == [("find_anomalies", {"param": "gas_lpm", "lt": 10.0})]
    plan = a.route("list FCAW passes on SP-1043", {})
    assert plan == [("list_passes", {"spool_id": "SP-1043", "process": "FCAW"})]
    assert a.route("hello there", {}) == []


def test_scripted_full_conversation(toolbox):
    agent = ScriptedAgent(toolbox, delay=0)
    s = Session()
    ev = collect(agent.run(s, "Why did SP-1043 J07 fill pass fail RT?"))
    text = "".join(e.get("delta", "") for e in ev if e["type"] == "text")
    assert "Shielding gas" in text and "REJECT" in text
    assert s.ctx["last_sequence"] == "SP-1043_J07_P3_FCAW"
    ev = collect(agent.run(s, "Show me the moment porosity started"))
    ui = [e for e in ev if e["type"] == "ui"]
    assert ui[0]["sequence"] == "SP-1043_J07_P3_FCAW" and abs(ui[0]["t"] - 11.2) < 0.1
    ev = collect(agent.run(s, "zzz"))
    assert "Try" in "".join(e.get("delta", "") for e in ev if e["type"] == "text")


def test_sequence_resolution(toolbox):
    r = toolbox.resolve_sequence
    assert r("SP-1043_J07_P3_FCAW") == "SP-1043_J07_P3_FCAW"
    assert r("SP-1043 J07 fill") == "SP-1043_J07_P3_FCAW"
    assert r("sp1042 j5 pass 2") == "SP-1042_J05_P2_FCAW"
    assert r(None, {"last_sequence": "SP-1041_J03_P1_GTAW"}) == "SP-1041_J03_P1_GTAW"
    with pytest.raises(LookupError):
        r("SP-7777")


def test_find_anomalies_threshold(toolbox):
    out = toolbox.call("find_anomalies", {"param": "gas_lpm", "lt": 10})
    seqs = {h["sequence"] for h in out["result"]["hits"]}
    assert seqs == {"SP-1043_J07_P3_FCAW"}
    out = toolbox.call("find_anomalies", {})
    assert out["result"]["count"] >= 5


def test_validate_input():
    assert validate_input("wps_check", {}) == "missing required field 'sequence'"
    assert validate_input("find_anomalies", {"gt": "high"}) == "field 'gt' must be number"
    assert validate_input("nope", {}) == "unknown tool nope"
    assert validate_input("list_passes", {"has_defect": True}) is None


# ---------------------------------------------------------------- Claude backend (mocked client)
class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()

    async def get_final_message(self):
        return self.final


class FakeClaude:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: getattr(o, "__dict__", str(o)))))
        events, final = self.turns.pop(0)
        return FakeStream(events, final)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def test_claude_tool_loop(toolbox):
    turns = [
        ([SimpleNamespace(type="thinking", thinking="Check the WPS first."), SimpleNamespace(type="text", text="Checking. ")],
         SimpleNamespace(stop_reason="tool_use", content=[_text("Checking. "),
                                                          _tool("tu_1", "wps_check", {"sequence": "SP-1043 J07 fill"}),
                                                          _tool("tu_2", "list_passes", {"process": "TIG"})])),
        ([SimpleNamespace(type="text", text="Gas fell to 5.7 L/min at 11.3 s.")],
         SimpleNamespace(stop_reason="end_turn", content=[_text("Gas fell to 5.7 L/min at 11.3 s.")])),
    ]
    fake = FakeClaude(turns)
    agent = ClaudeAgent(toolbox, client=fake)
    s = Session()
    ev = collect(agent.run(s, "Why did J07 fail?"))
    types = [e["type"] for e in ev]
    assert types[0] == "thinking" and types[-1] == "done"
    assert any(e["type"] == "tool_start" and e["name"] == "wps_check" for e in ev)
    assert any(e["type"] == "card" and e["card"]["type"] == "table" for e in ev)
    assert "Gas fell" in "".join(e.get("delta", "") for e in ev if e["type"] == "text")
    # request shape
    first = fake.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"]["type"] == "adaptive"
    assert first["output_config"]["effort"] in ("low", "medium", "high", "xhigh", "max")
    assert all(t.get("eager_input_streaming") for t in first["tools"])
    assert {t["name"] for t in first["tools"]} == {t["name"] for t in TOOL_SPECS}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    # second request carries tool results for both tool_use ids, invalid input flagged as error
    results = fake.calls[1]["messages"][-1]["content"]
    by_id = {r["tool_use_id"]: r for r in results}
    assert set(by_id) == {"tu_1", "tu_2"}
    assert "SP-1043_J07_P3_FCAW" in by_id["tu_1"]["content"] and not by_id["tu_1"].get("is_error")
    assert by_id["tu_2"]["is_error"] is True
    assert len(s.claude_messages) == 4  # user, assistant(tool_use), user(tool_result), assistant
    assert s.ctx["last_sequence"] == "SP-1043_J07_P3_FCAW"


def test_claude_refusal_and_error_rollback(toolbox):
    fake = FakeClaude([([], SimpleNamespace(stop_reason="refusal", content=[]))])
    s = Session()
    ev = collect(ClaudeAgent(toolbox, client=fake).run(s, "x"))
    assert "declined" in "".join(e.get("delta", "") for e in ev if e["type"] == "text")

    class Boom(FakeClaude):
        def _stream(self, **kw):
            raise RuntimeError("network down")
    s2 = Session()
    ev = collect(ClaudeAgent(toolbox, client=Boom([])).run(s2, "x"))
    assert ev[-1]["type"] == "error" and "network down" in ev[-1]["message"]
    assert s2.claude_messages == []


def test_claude_model_override_without_fallbacks(toolbox, monkeypatch):
    monkeypatch.setenv("COPILOT_MODEL", "claude-haiku-4-5")
    kw = ClaudeAgent(toolbox, client=object()).request_kwargs([])
    assert kw["model"] == "claude-haiku-4-5" and "fallbacks" not in kw


# ---------------------------------------------------------------- Cosmos Reasoner backend (ReAct, mocked)
class FakeOAI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        r = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def test_cosmos_react_protocol(toolbox, monkeypatch):
    monkeypatch.setenv("COPILOT_COSMOS_TOOLS", "react")
    fake = FakeOAI(['```json\n{"tool": "wps_check", "args": {"sequence": "SP-1043_J07_P3_FCAW"}}\n```',
                    '{"final": "**Porosity**: gas 5.7 L/min vs WPS 18-25."}'])
    agent = CosmosAgent(toolbox, client=fake)
    ev = collect(agent.run(Session(), "why did it fail"))
    assert any(e["type"] == "tool_start" and e["name"] == "wps_check" for e in ev)
    assert "Porosity" in "".join(e.get("delta", "") for e in ev if e["type"] == "text")
    assert ev[-1]["type"] == "done"


def test_first_json():
    assert _first_json('Sure: {"tool": "x", "args": {"a": {"b": 1}}} done') == {"tool": "x", "args": {"a": {"b": 1}}}
    assert _first_json("no json here") is None
