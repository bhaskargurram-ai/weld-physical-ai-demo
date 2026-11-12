"""Weld Copilot on the real-data contract (docs/REAL_DATA.md): AIMEN SAW (photos, NDT) + BAM tandem-GMAW (video,
not inspected) + one SIMULATED pass, built by copilot.fixtures_real. No GPU, keys or network."""
from __future__ import annotations

import asyncio
import json
import shutil

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REAL_DEMO = [
    ("Which SA-516 vessel passes had NDT indications?", ["ndt_summary"]),
    ("Show porosity on PP5 bead 2 — did the arc signals change?", ["show_weld", "ndt_summary"]),
    ("Compare lead vs trail current on WID617 layer 3", ["get_signals_summary"]),
    ("Which passes ran outside their published settings?", ["list_passes"]),
    ("Inspect the bead photo of PP5 bead 2", ["show_photo", "assess_weld"]),
    ("What if travel speed dropped 20 % on WID630 layer 4?", ["what_if"]),
]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    from copilot.fixtures_real import BAM_FPS, make_dataset, render_bam_video
    r = tmp_path_factory.mktemp("copilot_real_fx")
    make_dataset(r, videos=False, with_sim=True)
    if shutil.which("ffmpeg"):                      # one short (3 s) grey process-camera clip
        d = r / "generated" / "WID617_L3_TGMAW"
        render_bam_video(d, 3.0, True, "WID617 L3")
        meta = json.loads((d / "meta.json").read_text())
        meta["video"] = {"file": "torch_cam.mp4", "kind": "real_process_camera", "fps": BAM_FPS, "t_offset_s": 0.0}
        (d / "meta.json").write_text(json.dumps(meta))
    return r


@pytest.fixture(scope="module")
def client(root):
    mp = pytest.MonkeyPatch()
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


def sse(text: str) -> list[dict]:
    return [json.loads(x[5:].strip()) for x in text.replace("\r\n", "\n").split("\n") if x.startswith("data:")]


def no_nan(obj) -> bool:
    return "NaN" not in json.dumps(obj) and "Infinity" not in json.dumps(obj)


# ---------------------------------------------------------------- data layer / API
def test_passes_grouping_fields_and_filters(client):
    r = client.get("/api/passes").json()
    rows = {x["sequence_name"]: x for x in r["passes"]}
    assert len(rows) == 9 and r["data"]["real"] and r["data"]["simulated"] == 1
    p = rows["PP5_B02_SAW"]
    assert p["source_id"] == "aimen_saw" and p["license"] == "CC BY 4.0" and not p["synthetic"]
    assert p["rt_result"] == "indications" and p["defect_counts"] == {"porosity": 14} and p["pass_label"] == "Bead 2"
    assert p["has_photo"] and not p["has_video"] and p["electrodes"] == 2 and p["ndt_inspected"]
    assert rows["PP4_B05_SAW"]["rt_result"] == "no_indications"
    assert rows["WID617_L3_TGMAW"]["rt_result"] == "not_inspected" and not rows["WID617_L3_TGMAW"]["ndt_inspected"]
    assert rows["SP-1043_J07_P3_FCAW"]["synthetic"] is True
    q = lambda **kw: {x["sequence_name"] for x in client.get("/api/passes", params=kw).json()["passes"]}  # noqa: E731
    assert q(source="bam_tgmaw") == {s for s in rows if s.startswith("WID")}
    assert q(source="sim") == {"SP-1043_J07_P3_FCAW"}
    assert q(defect="lack_of_fusion") == {"PP5_B01_SAW"}
    assert q(rt_result="indications") == {"PP5_B01_SAW", "PP5_B02_SAW"}
    assert no_nan(r)


def test_saw_pass_detail_photo_ndt_and_signals(client):
    d = client.get("/api/passes/PP5_B02_SAW").json()
    assert d["source"]["doi"] == "10.5281/zenodo.15083865" and "Tapia" in d["source"]["authors"]
    assert d["media"]["photos"][0]["url"].endswith(".jpg") and "AIMEN, CC BY 4.0" in d["media"]["photo_caption"]
    assert "torch" not in d["media"]
    assert len(d["ndt"]["indications"]) == 14 and d["ndt"]["methods"] == ["RT", "UT"]
    assert len(d["defect_windows"]) == 14 and d["defect_windows"][0]["x_start_mm"] is not None
    assert "flux" in d["signal_notes"]["gas_lpm"] and "±10 %" in d["wps_basis"]
    assert client.get(d["media"]["photos"][0]["url"]).status_code == 200
    s = client.get("/api/passes/PP5_B02_SAW/signals", params={"hz": 10}).json()
    assert "gas_lpm" not in s["series"] and {"current2_a", "voltage2_v"} <= set(s["series"])
    assert s["electrodes"] == 2 and s["notes"]["gas_lpm"] and "current_a" in s["wps"]
    assert no_nan(d) and no_nan(s)


def test_bam_pass_detail_video_caption_and_missing_trail(client):
    d = client.get("/api/passes/WID617_L3_TGMAW").json()
    assert d["source"]["id"] == "bam_tgmaw" and d["pass"]["rt_result"] == "not_inspected"
    if shutil.which("ffmpeg"):
        v = d["media"]["video"]
        assert d["media"]["torch"] and v["real"] and v["caption"].startswith("Real process camera — BAM, CC BY 4.0")
    root = client.get("/api/passes/WID617_L1_TGMAW/signals").json()   # L1 root = single wire: no trail channels
    assert "current2_a" not in root["series"] and "gas_lpm" in root["series"]


def test_sources_endpoint_carries_full_attribution(client):
    src = {x["id"]: x for x in client.get("/api/sources").json()["sources"]}
    for sid, doi in (("aimen_saw", "10.5281/zenodo.15083865"), ("bam_tgmaw", "10.5281/zenodo.21824586")):
        x = src[sid]
        assert x["doi"] == doi and x["license"] == "CC BY 4.0" and x["authors"] and x["title"] and x["loaded"]
        assert "creativecommons.org/licenses/by/4.0" in x["license_url"]
        assert "Modified" in x["attribution"] and doi in x["attribution"] and ".." not in x["attribution"]
    assert src["sim"]["synthetic"] and src["aimen_saw"]["indications"] == 15


def test_minimal_pass_dir_without_optional_streams(root, client):
    """Only meta.json + arc.parquet + events.jsonl: everything else optional."""
    src = root / "generated" / "PP4_B05_SAW"
    dst = root / "generated" / "PP4_B06_SAW"
    dst.mkdir()
    for f in ("arc.parquet", "events.jsonl"):
        shutil.copy(src / f, dst / f)
    meta = json.loads((src / "meta.json").read_text())
    meta.update({"sequence_name": "PP4_B06_SAW", "pass_no": 6, "photos": ["photos/missing.jpg"]})
    (dst / "meta.json").write_text(json.dumps(meta))
    client.post("/api/refresh")
    try:
        d = client.get("/api/passes/PP4_B06_SAW").json()
        assert "photos" not in d["media"] and d["signal_notes"]["travel_speed_mmps"]
        s = client.get("/api/passes/PP4_B06_SAW/signals").json()
        assert "travel_speed_mmps" not in s["series"] and "heat_input_kj_mm" not in s["series"]
        out = client.post("/api/tools/get_signals_summary", json={"sequence": "PP4 B06"}).json()
        assert "current_a" in out["result"]["stats"]
    finally:
        shutil.rmtree(dst)
        client.post("/api/refresh")


# ---------------------------------------------------------------- tools
def test_ndt_summary_tool_is_honest_about_signals(client):
    allp = client.post("/api/tools/ndt_summary", json={}).json()["result"]
    assert allp["with_indications"] == 2 and allp["not_inspected"] == 4
    assert allp["indications_by_type"]["porosity"] >= 14
    one = client.post("/api/tools/ndt_summary", json={"sequence": "PP5 bead 2"}).json()
    r = one["result"]
    assert r["sequence"] == "PP5_B02_SAW" and r["counts"] == {"porosity": 14}
    chk = r["signal_check"]
    assert chk["n_windows"] == 14 and chk["signature"] is False
    assert abs(chk["params"]["current_a"]["delta_pct"]) < 2
    assert one["cards"][0]["type"] == "table" and one["cards"][-1]["signal_check"]["signature"] is False
    co = client.post("/api/tools/ndt_summary", json={"coupon": "PP5"}).json()["result"]
    assert co["inspected"] == 3 and co["with_indications"] == 2


def test_show_photo_and_photo_assessment(client):
    out = client.post("/api/tools/show_photo", json={"sequence": "PP5 B02"}).json()
    assert out["cards"][0]["type"] == "photo" and out["ui"][0]["view"] == "photo"
    bam = client.post("/api/tools/show_photo", json={"sequence": "WID617 L3"}).json()["result"]
    assert bam["photos"] == [] and "no bead photo" in bam["error"]
    a = client.post("/api/tools/assess_weld", json={"sequence": "PP5_B02_SAW"}).json()
    v = a["cards"][0]
    assert v["input_kind"] == "photo" and v["image_url"].endswith(".jpg")
    assert v["ndt_indications"] == {"porosity": 14} and a["ui"][0]["view"] == "photo"
    # no invented cause: without a signal signature the rule check must not blame gas / heat input / speed
    rc = v["root_cause"].lower()
    assert "not determinable" in rc and not any(w in rc for w in ("gas", "heat input", "travel speed", "contamination"))
    if "mock" in v["source"] or "heuristic" in v["source"]:
        assert v["verdict"] == "review"


def test_lead_vs_trail_summary_and_wps_basis(client):
    out = client.post("/api/tools/get_signals_summary",
                      json={"sequence": "WID617 layer 3", "params": ["current_a", "current2_a"]}).json()
    r = out["result"]
    assert r["sequence"] == "WID617_L3_TGMAW" and set(out["cards"][0]["series"]) == {"current_a", "current2_a"}
    assert r["lead_vs_trail"]["current_a"]["lead"] > r["lead_vs_trail"]["current_a"]["trail"] > 200
    bad = client.post("/api/tools/get_signals_summary", json={"sequence": "WID617_L3_TGMAW", "params": ["nope"]})
    assert bad.status_code == 400
    lp = client.post("/api/tools/list_passes", json={"only_wps_violations": True}).json()["result"]
    seqs = {p["sequence"] for p in lp["passes"]}
    assert {"PP5_B03_SAW", "WID630_L4_TGMAW"} <= seqs and lp["wps_basis"]
    assert next(p for p in lp["passes"] if p["sequence"] == "PP5_B03_SAW")["ndt"] == "NDT clear"


def test_reference_parsing(client):
    tb = client.app_ref.state.toolbox
    for ref, seq in [("PP5 B02", "PP5_B02_SAW"), ("PP5 bead 2", "PP5_B02_SAW"), ("pp5_b02", "PP5_B02_SAW"),
                     ("WID617 L3", "WID617_L3_TGMAW"), ("WID617 layer 3", "WID617_L3_TGMAW"),
                     ("WID630 layer 4", "WID630_L4_TGMAW"), ("SP-1043 J07 fill", "SP-1043_J07_P3_FCAW")]:
        assert tb.resolve_sequence(ref) == seq, ref
    assert tb.resolve_sequence("layer 2", {"last_sequence": "WID617_L3_TGMAW"}) == "WID617_L2_TGMAW"
    assert tb.resolve_sequence("bead 3", {"last_sequence": "PP5_B02_SAW"}) == "PP5_B03_SAW"
    with pytest.raises(LookupError):
        tb.resolve_sequence("PP9 bead 1")


# ---------------------------------------------------------------- agent
@pytest.mark.parametrize("text,tools", REAL_DEMO)
def test_scripted_routes_real_demo_prompts(client, text, tools):
    from copilot.agent import ScriptedAgent
    plan = ScriptedAgent(client.app_ref.state.toolbox, delay=0).route(text, {})
    assert [t for t, _ in plan] == tools


def test_scripted_real_route_args(client):
    from copilot.agent import ScriptedAgent
    a = ScriptedAgent(client.app_ref.state.toolbox, delay=0)
    assert a.route("Compare lead vs trail current on WID617 layer 3", {})[0][1] == {
        "sequence": "WID617 layer 3", "params": ["current_a", "current2_a"]}
    wi = a.route("What if travel speed dropped 20 % on WID630 layer 4?", {})[0][1]
    assert wi["sequence"] == "WID630 layer 4" and wi["change"] == "travel speed dropped 20 %"
    assert a.route("Which passes ran outside their published settings?", {})[0][1] == {"only_wps_violations": True}


def test_chat_porosity_signal_check_says_no_signature(client):
    ev = sse(client.post("/api/chat", json={"message": "Show porosity on PP5 bead 2 — did the arc signals change?"}).text)
    ui = [e for e in ev if e.get("type") == "ui"]
    assert ui and ui[0]["sequence"] == "PP5_B02_SAW" and abs(ui[0]["t"] - 12.0) < 0.05
    text = "".join(e.get("delta", "") for e in ev if e.get("type") == "text")
    assert "Did the arc signals change? No." in text and "no electrical signature" in text
    assert ev[-1]["type"] == "done"


def test_chat_ndt_question_and_suggestions(client):
    sug = client.get("/api/suggestions").json()
    assert [d["label"] for d in sug["demo"]] == [t for t, _ in REAL_DEMO]
    text = "".join(e.get("delta", "") for e in sse(client.post(
        "/api/chat", json={"message": REAL_DEMO[0][0]}).text) if e.get("type") == "text")
    assert "2 of 3 inspected passes have NDT indications" in text and "not inspected" in text
    assert "SIMULATED" in text   # the simulator pass with RT reject is marked, never passed off as real


def test_system_prompt_forbids_invented_causes(client):
    from copilot.agent import SYSTEM_PROMPT, system_prompt
    p = system_prompt(client.app_ref.state.data)
    assert "NEVER invent a defect cause" in p and "no anomaly" in p and "SIMULATED" in p
    assert "10.5281/zenodo.15083865" in p and "not_inspected" in p
    assert p != SYSTEM_PROMPT


def test_health_and_flow_describe_real_sources(client):
    h = client.get("/api/health").json()
    assert h["stages"][0]["name"] == "Real data" and "AIMEN SAW" in h["stages"][0]["detail"]
    f = client.get("/api/flow").json()
    assert f["data"]["real"] and {s["id"] for s in f["sources"]} == {"aimen_saw", "bam_tgmaw"}
    assert f["finding"]["passes_checked"] == 2 and f["finding"]["passes_with_signature"] == 0
    assert f["example"]["sequence"] == "PP5_B02_SAW" and f["example"]["signal_check"]["signature"] is False
    assert f["real"]["not_inspected"] == 4


# ---------------------------------------------------------------- worldmodels image hook
def test_assess_image_uses_whatever_hook_worldmodels_exposes():
    from copilot.models import _assess_image
    seen = {}

    def kw(video_path, context, *, image_path=None):
        seen["kw"] = (video_path, image_path)
        return {"verdict": "accept"}

    def positional(video_path, context=None):
        seen["pos"] = video_path
        return {"verdict": "review"}
    assert _assess_image(kw, "/x/b.jpg", {})["verdict"] == "accept" and seen["kw"] == (None, "/x/b.jpg")
    assert _assess_image(positional, "/x/b.jpg", {})["verdict"] == "review" and seen["pos"] == "/x/b.jpg"


def test_photo_what_if_mock_renders_from_still(client, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed")
    wm = client.app_ref.state.wm
    import copilot.models as m
    monkeypatch.setattr(m, "_try_import", lambda name: None)       # no worldmodels: copilot mock path
    res = wm.what_if("PP5_B02_SAW", "travel speed -20%", progress=lambda *_: None)
    assert res["source_image_url"].endswith(".jpg") and res["video_url"].endswith(".mp4") and "mock" in res["source"]
    asyncio.run(asyncio.sleep(0))
