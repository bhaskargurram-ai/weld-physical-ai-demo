"""Weld Copilot API tests on contract-shaped fixture data (no video, no GPU, no API keys)."""
from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("copilot_fx")
    from copilot.fixtures import make_dataset
    make_dataset(root, videos=False, workers=1)
    for k, v in {"DATA_DIR": str(root), "COPILOT_DATA": "local", "COPILOT_LLM": "scripted", "MOSAICO_OFFLINE": "1",
                 "COPILOT_MOCK_DELAY": "0", "COPILOT_SCRIPT_DELAY": "0", "COSMOS_BACKEND": "replay"}.items():
        mp.setenv(k, v)
    mp.delenv("COPILOT_FIXTURES", raising=False)
    from copilot.app import create_app
    app = create_app()
    with TestClient(app) as c:
        c.app_ref = app
        yield c
    mp.undo()


def sse_events(text: str) -> list[dict]:
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("data:"):
            out.append(json.loads(line[5:].strip()))
    return out


def test_health_pipeline_strip(client):
    h = client.get("/api/health").json()
    ids = [s["id"] for s in h["stages"]]
    assert ids == ["collect", "mosaico", "curator", "cosmos", "copilot"]
    assert h["stages"][0]["metric"] == 24
    assert h["stages"][1]["status"] == "offline"
    assert h["llm"]["backend"] == "scripted"
    assert h["mode"]


def test_list_passes_and_filters(client):
    rows = client.get("/api/passes").json()["passes"]
    assert len(rows) == 24
    r = next(x for x in rows if x["sequence_name"] == "SP-1043_J07_P3_FCAW")
    assert r["rt_result"] == "reject" and r["defect_types"] == ["porosity"]
    assert r["violation_count"] >= 1 and "gas_lpm" in r["violated_params"]
    gtaw = client.get("/api/passes", params={"process": "GTAW"}).json()["passes"]
    assert gtaw and all(x["process"] == "GTAW" for x in gtaw)
    rej = client.get("/api/passes", params={"rt_result": "reject"}).json()["passes"]
    assert {x["rt_result"] for x in rej} == {"reject"}


def test_pass_detail_and_signals(client):
    d = client.get("/api/passes/SP-1043_J07_P3_FCAW").json()
    assert d["wps"]["gas_lpm"] == [18.0, 25.0]
    gas = [v for v in d["violations"] if v["param"] == "gas_lpm"]
    assert gas and 10.5 < gas[0]["t_start_s"] < 12.5 and gas[0]["value"] < 8
    assert d["defect_windows"][0]["defect"] == "porosity"
    assert any(e["type"] == "defect_on" and abs(e["t_s"] - 11.2) < 0.05 for e in d["events"])
    s = client.get("/api/passes/SP-1043_J07_P3_FCAW/signals", params={"hz": 10}).json()
    assert {"current_a", "voltage_v", "travel_speed_mmps", "gas_lpm", "heat_input_kj_mm", "interpass_c"} <= set(s["series"])
    assert len(s["t"]) == len(s["series"]["gas_lpm"]) and 250 < len(s["t"]) < 350
    assert client.get("/api/passes/NOPE").status_code == 404


def test_dataset_endpoint(client):
    d = client.get("/api/dataset").json()
    assert d["stats"]["total"] == d["count"] > 100
    assert d["stats"]["kept"] + d["stats"]["rejected"] == d["stats"]["total"]
    assert "porosity" in d["stats"]["by_defect"]
    assert d["card"].startswith("# ")
    por = client.get("/api/dataset", params={"defect": "porosity"}).json()
    assert por["count"] >= 1 and all("porosity" in r["defects"] for r in por["rows"])


def test_tool_endpoint_and_validation(client):
    r = client.post("/api/tools/wps_check", json={"sequence": "SP-1043 J07 fill"})
    assert r.status_code == 200
    out = r.json()
    assert out["result"]["sequence"] == "SP-1043_J07_P3_FCAW"
    assert out["cards"][0]["type"] == "table"
    bad = client.post("/api/tools/list_passes", json={"process": "TIG"})
    assert bad.status_code == 400 and "must be one of" in bad.json()["result"]["error"]
    nf = client.post("/api/tools/wps_check", json={"sequence": "SP-9999 J99"})
    assert nf.status_code == 400


def test_assess_tool_returns_verdict_card(client):
    out = client.post("/api/tools/assess_weld", json={"sequence": "SP-1043_J07_P3_FCAW"}).json()
    v = out["cards"][0]
    assert v["type"] == "verdict" and v["verdict"] in ("reject", "review")
    assert v["defects"][0]["type"] == "porosity"
    assert v["source"] and v["latency_s"] >= 0
    assert out["ui"][0]["action"] == "show_weld"


def test_chat_sse_scripted_flow(client):
    r = client.post("/api/chat", json={"message": "Which welds this week broke the WPS?"})
    ev = sse_events(r.text)
    sid = ev[0]["session_id"]
    assert any(e.get("type") == "tool_start" and e["name"] == "list_passes" for e in ev)
    assert any(e.get("type") == "card" and e["card"]["type"] == "table" for e in ev)
    text = "".join(e.get("delta", "") for e in ev if e.get("type") == "text")
    assert "SP-1043_J07_P3_FCAW" in text and "broke their WPS" in text
    assert ev[-1]["type"] == "done"
    r2 = client.post("/api/chat", json={"message": "Why did SP-1043 J07 fill pass fail RT?", "session_id": sid})
    ev2 = sse_events(r2.text)
    names = [e["name"] for e in ev2 if e.get("type") == "tool_start"]
    assert names == ["wps_check", "assess_weld", "show_weld"]
    ui = [e for e in ev2 if e.get("type") == "ui"]
    assert ui and ui[-1]["sequence"] == "SP-1043_J07_P3_FCAW" and abs(ui[-1]["t"] - 11.2) < 0.1
    # follow-up uses the conversation context ("that pass")
    r3 = client.post("/api/chat", json={"message": "Show me the moment porosity started", "session_id": sid})
    ui3 = [e for e in sse_events(r3.text) if e.get("type") == "ui"]
    assert ui3[0]["sequence"] == "SP-1043_J07_P3_FCAW"


def test_generate_job_progress_sse(client, monkeypatch):
    wm = client.app_ref.state.wm

    def fake_what_if(seq, change, clip_id=None, t=None, seed=0, prompt=None, progress=None):
        for i in range(3):
            progress(0.2 + 0.25 * i, f"step {i}")
            time.sleep(0.05)
        return {"kind": "what_if", "sequence_name": seq, "video_url": "/files/x.mp4", "source": "test",
                "latency_s": 0.1, "t0_s": 0, "t1_s": 5, "label": change}
    monkeypatch.setattr(wm, "what_if", fake_what_if)
    j = client.post("/api/generate", json={"kind": "what_if", "sequence": "SP-1043_J07_P3_FCAW",
                                           "change": "travel_speed_-20"}).json()
    assert j["status"] in ("queued", "running")
    ev = sse_events(client.get(f"/api/jobs/{j['id']}/events").text)
    assert ev[-1]["status"] == "done" and ev[-1]["result"]["source"] == "test"
    assert any(0 < e["progress"] < 1 for e in ev)
    gal = client.get("/api/gallery").json()["items"]
    assert gal and gal[0]["id"] == j["id"]


def test_prompt_and_templates(client):
    t = client.get("/api/templates").json()["templates"]
    assert {x["key"] for x in t} >= {"travel_speed_-20", "gas_loss", "porosity"}
    p = client.post("/api/prompt", json={"sequence": "SP-1043_J07_P3_FCAW", "change": "gas_loss"}).json()
    assert "Shielding gas" in json.loads(p["prompt"])["action"]


def test_files_route_is_sandboxed(client):
    assert client.get("/files/../../etc/passwd").status_code == 404
    assert client.get("/files/abs/etc/passwd").status_code == 404
    assert client.get("/files/generated/SP-1043_J07_P3_FCAW/arc.parquet").status_code == 403


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Weld Copilot" in r.text
    assert client.get("/static/app.js").status_code == 200
