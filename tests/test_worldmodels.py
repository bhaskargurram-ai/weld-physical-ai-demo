"""Tests for worldmodels/: reasoner JSON flow against a fake OpenAI-compatible server,
generator / transfer request shapes, cache + replay behaviour, prompt building.

No GPU and no network: every HTTP call goes through ``worldmodels.http.set_transport``.
"""
from __future__ import annotations

import base64
import json
import re
from email.parser import BytesParser
from email.policy import default as email_default
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest

from worldmodels import cache, config, generate, heuristics, parsing, prompts, reason, transfer
from worldmodels.http import set_transport
from worldmodels.video import frames_for_seconds, probe, write_mp4


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COSMOS_CACHE_DIR", str(tmp_path / "cache"))
    for k in ("COSMOS_BACKEND", "COSMOS_REASONER_BACKEND", "COSMOS_GENERATOR_BACKEND", "NVIDIA_API_KEY", "NGC_API_KEY",
              "COSMOS_CACHE", "COSMOS_REASONER_MODEL", "COSMOS_FRAMEWORK_DIR", "COSMOS_ANOMALYGEN_DIR"):
        monkeypatch.delenv(k, raising=False)
    config.clear_probe_cache()
    reason._MODEL_MEMO.clear()
    yield
    set_transport(None)
    config.clear_probe_cache()


def _make_video(path: Path, n: int = 24, size=(320, 180), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    frames = [np.full((size[1], size[0], 3), (i * 7 + seed * 40) % 255, np.uint8) for i in range(n)]
    frames[0][:10, :10] = rng.integers(0, 255, (10, 10, 3))
    return write_mp4(frames, path, fps=24)


@pytest.fixture
def clip(tmp_path) -> Path:
    return _make_video(tmp_path / "clips" / "SP-1043_J07_P2_FCAW_c003.mp4")


@pytest.fixture
def sidecar() -> dict:
    return {
        "clip_id": "SP-1043_J07_P2_FCAW_c003",
        "sequence_name": "SP-1043_J07_P2_FCAW",
        "t0_s": 10.0, "t1_s": 15.0,
        "meta": {"spool_id": "SP-1043", "joint_id": "J07", "pass_no": 2, "pass_type": "fill", "process": "FCAW",
                 "position": "1G", "material": "A106 Gr.B", "pipe_od_mm": 168.3, "wall_mm": 7.11,
                 "wps_id": "WPS-FCAW-FILL-02", "rt_result": "reject",
                 "wps": {"wps_id": "WPS-FCAW-FILL-02", "process": "FCAW", "current_a": [180, 240], "voltage_v": [23, 27],
                         "travel_speed_mmps": [4.0, 7.0], "wfs_mpm": [7.0, 10.0], "gas_lpm": [18, 25],
                         "max_interpass_c": 250, "heat_input_kj_mm": [0.7, 1.5]}},
        "signal_stats": {"current_a": {"mean": 205.1, "std": 6.2, "min": 190.0, "max": 221.0},
                         "gas_lpm": {"mean": 9.4, "std": 4.1, "min": 5.2, "max": 21.0}},
        "wps_violations": [{"param": "gas_lpm", "t_start_s": 11.2, "t_end_s": 13.9, "value": 6.1, "limit_lo": 18, "limit_hi": 25}],
        "defects_in_window": [{"defect": "porosity", "t_start": 11.5, "t_end": 13.8, "severity": 0.8}],
        "rt_result": "reject",
    }


GOOD_JSON = {
    "verdict": "Reject",
    "defects": [{"type": "Gas porosity", "confidence": 85, "t_s": "1.6"}],
    "root_cause": "Shielding gas fell to ~6 L/min (WPS 18-25), atmospheric contamination of the pool.",
    "wps_findings": [{"param": "gas_lpm", "status": "violation", "note": "6.1 L/min < 18"},
                     {"param": "current_a", "status": "within", "note": "205 A"}],
    "recommended_action": "Grind out porous length, check gas flow, re-weld and RT.",
    "code_basis": "B31.3 Table 341.3.2 surface porosity",
}


def _chat_response(content: str, model: str = "nvidia/Cosmos3-Nano", **msg_extra) -> httpx.Response:
    return httpx.Response(200, json={"id": "x", "model": model, "object": "chat.completion",
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **msg_extra},
                                                  "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 50}})


class FakeCosmos:
    """Minimal OpenAI-compatible reasoner + vLLM-Omni + NIM fake."""

    def __init__(self, chat_replies=None, video_bytes: bytes | None = None):
        self.chat_replies = list(chat_replies or [])
        self.requests: list[httpx.Request] = []
        self.video_bytes = video_bytes

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "nvidia/Cosmos3-Nano"}]})
        if path.endswith("/chat/completions"):
            reply = self.chat_replies.pop(0) if self.chat_replies else "<think>ok</think>" + json.dumps(GOOD_JSON)
            return reply if isinstance(reply, httpx.Response) else _chat_response(reply)
        if path.endswith("/videos/sync"):
            return httpx.Response(200, content=self.video_bytes, headers={"content-type": "video/mp4"})
        if path.endswith("/v1/infer") or "/cosmos/" in path:
            return httpx.Response(200, json={"b64_video": base64.b64encode(self.video_bytes).decode(), "action": None})
        return httpx.Response(404, text="404 page not found")

    def chat_payloads(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/chat/completions")]


def form_fields(request: httpx.Request) -> tuple[dict, dict]:
    """(fields, files) of a urlencoded or multipart request."""
    ctype = request.headers.get("content-type", "")
    body = request.content if request.content else request.read()
    if ctype.startswith("application/x-www-form-urlencoded"):
        return {k: v[0] for k, v in parse_qs(body.decode()).items()}, {}
    msg = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if part.get_filename():
            files[name] = (part.get_filename(), part.get_content_type(), part.get_payload(decode=True))
        else:
            fields[name] = part.get_payload(decode=True).decode()
    return fields, files


@pytest.fixture
def fake_mp4(tmp_path) -> bytes:
    return _make_video(tmp_path / "fake_gen.mp4", n=12, size=(320, 192), seed=3).read_bytes()


# ----------------------------------------------------------------------------- parsing
def test_split_think_and_extract_json():
    raw = "<think>\nGas dropped at 1.6 s.\n</think>\n\n```json\n" + json.dumps(GOOD_JSON) + "\n```"
    think, answer = parsing.split_think(raw)
    assert think.startswith("Gas dropped")
    assert parsing.extract_json(answer)["verdict"] == "Reject"
    # prose + trailing comma + python literals
    obj = parsing.extract_json('Here you go: {"verdict": "accept", "defects": [], "ok": True,}')
    assert obj == {"verdict": "accept", "defects": [], "ok": True}
    # reasoning-parser style: closing tag only
    t, a = parsing.split_think("reasoning text</think>{\"verdict\":\"review\"}")
    assert t == "reasoning text" and json.loads(a)["verdict"] == "review"


def test_normalize_assessment_coerces_fields():
    res, warnings = parsing.parse_assessment("<think>why</think>" + json.dumps(GOOD_JSON))
    assert res["verdict"] == "reject"
    assert res["defects"] == [{"type": "porosity", "confidence": 0.85, "t_s": 1.6}]
    assert res["wps_findings"][0] == {"param": "gas_lpm", "status": "violation", "note": "6.1 L/min < 18"}
    assert res["reasoning"] == "why"
    assert not [w for w in warnings if "verdict" in w]
    for raw, want in (("PASS", "accept"), ("Rejectable per B31.3", "reject"), ("needs further review", "review")):
        assert parsing.normalize_verdict(raw) == want
    assert parsing.normalize_defect_type("Incomplete fusion (cold lap)") == "lack_of_fusion"
    assert parsing.normalize_defect_type("burn-through") == "burn_through"
    with pytest.raises(parsing.AssessmentError):
        parsing.parse_assessment('{"defects": []}')
    with pytest.raises(parsing.AssessmentError):
        parsing.parse_assessment("I cannot see the weld clearly.")


# ----------------------------------------------------------------------------- prompts
def test_assess_prompt_contains_context_but_not_labels(sidecar):
    text = prompts.build_assess_text(sidecar, think=True, include_labels=False)
    assert "WPS WPS-FCAW-FILL-02" in text and "shielding gas flow 18-25 L/min" in text
    assert "mean 9.4" in text                                  # signal stats
    assert "6.1 L/min (WPS 18-25 L/min) from t=1.2 s" in text  # violation, relative to clip start
    assert "Radiography" not in text and "porosity\", \"t_start" not in text   # no ground-truth leak
    assert text.rstrip().endswith("Write your final answer immediately after the </think> tag.")
    assert '"verdict": "accept" | "reject" | "review"' in text
    with_labels = prompts.build_assess_text(sidecar, include_labels=True)
    assert "Radiography (RT) result for the joint: reject" in with_labels
    assert "ASME Section IX" in prompts.CWI_SYSTEM_PROMPT and "B31.3" in prompts.CWI_SYSTEM_PROMPT


def test_context_normalisation_accepts_manifest_row():
    row = {"clip_id": "c1", "sequence_name": "SP-1_J01_P1_GTAW", "caption": "TIG root",
           "sensor_summary": {"current_a": {"mean": 100}, "process": "GTAW", "wps_id": "WPS-GTAW-ROOT-01"},
           "wps_violations": [], "defects": [{"defect": "burn_through", "severity": 0.9}], "rt_result": "reject"}
    c = prompts.normalize_context(row)
    assert c["signal_stats"] == {"current_a": {"mean": 100}} and c["defects"][0]["defect"] == "burn_through"
    assert c["process"] == "GTAW" and c["wps"]["current_a"] == (90, 130)
    c2 = prompts.normalize_context({"meta": {"wps_id": "WPS-GTAW-ROOT-01"}})
    assert c2["wps"]["current_a"] == (90, 130)              # filled from collect.schema.WPS_LIBRARY


def test_weld_scene_prompt_is_cosmos_json_style():
    p = prompts.weld_scene_prompt("travel speed drops 20%", defect="undercut", seconds=5, width=832, height=480)
    for k in ("subjects", "background_setting", "lighting", "aesthetics", "cinematography", "style_medium", "actions",
              "temporal_caption", "resolution", "duration", "fps"):
        assert k in p
    assert p["resolution"] == {"W": 832, "H": 480} and p["duration"] == "5s"
    assert "undercut" in p["subjects"][0]["appearance_details"]
    assert prompts.prompt_query_text(prompts.compact(p)) == "travel speed drops 20%"
    assert len(prompts.negative_prompt("video")) > 1000 and prompts.negative_prompt("video").startswith("{")


# ----------------------------------------------------------------------------- reasoner flow
def test_assess_clip_local_roundtrip_and_cache(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(["<think>\nAt 1.6 s pores appear while gas is 6 L/min.\n</think>\n" + json.dumps(GOOD_JSON)])
    set_transport(httpx.MockTransport(fake))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["verdict"] == "reject" and res["source"] == "cosmos3-local"
    assert res["defects"][0]["type"] == "porosity" and res["reasoning"].startswith("At 1.6 s")
    assert isinstance(res["latency_ms"], int) and res["clip_id"] == sidecar["clip_id"]
    for k in ("verdict", "defects", "root_cause", "wps_findings", "recommended_action", "reasoning"):
        assert k in res

    payload = fake.chat_payloads()[0]
    assert payload["model"] == "nvidia/Cosmos3-Nano"
    assert payload["messages"][0]["role"] == "system" and "Welding Inspector" in payload["messages"][0]["content"]
    user = payload["messages"][1]["content"]
    assert user[0]["type"] == "video_url" and user[0]["video_url"]["url"].startswith("file://")   # media first
    assert user[1]["type"] == "text"
    assert payload["media_io_kwargs"] == {"video": {"num_frames": -1, "fps": 4.0}}
    assert payload["temperature"] == 0.6 and payload["top_k"] == 20                            # reasoning preset

    # second call: exact cache hit, no new request
    n = len(fake.requests)
    again = reason.assess_clip(str(clip), sidecar)
    assert again["source"] == "cache" and again["origin_source"] == "cosmos3-local" and again["cache_match"] == "exact"
    assert len(fake.requests) == n

    # replay with a different context still finds the Cosmos answer for the same video
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    other = reason.assess_clip(str(clip), {"clip_id": sidecar["clip_id"], "caption": "different"})
    assert other["source"] == "cache" and other["cache_match"] == "same-video" and other["verdict"] == "reject"


def test_assess_clip_retries_on_bad_json(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(["<think>hmm</think> The weld looks porous.", "<think>ok</think>" + json.dumps(GOOD_JSON)])
    set_transport(httpx.MockTransport(fake))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["source"] == "cosmos3-local" and res["attempts"] == 2
    second = fake.chat_payloads()[1]["messages"]
    assert second[-2]["role"] == "assistant" and "could not be used" in second[-1]["content"]


def test_assess_clip_guided_third_attempt_then_heuristic(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(["nope", "still nope", json.dumps({"verdict": "review", "defects": [], "root_cause": "x",
                                                          "wps_findings": [], "recommended_action": "y"})])
    set_transport(httpx.MockTransport(fake))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["verdict"] == "review" and res["attempts"] == 3
    third = fake.chat_payloads()[2]
    assert third["response_format"]["type"] == "json_schema" and "<think>" not in third["messages"][1]["content"][1]["text"]

    # all attempts unusable -> labelled heuristic fallback with the error
    fake2 = FakeCosmos(["a", "b", "c"])
    set_transport(httpx.MockTransport(fake2))
    clip2 = _make_video(clip.with_name("other.mp4"), seed=9)
    res2 = reason.assess_clip(str(clip2), sidecar)
    assert res2["source"] == "heuristic-fallback" and "no valid assessment" in res2["error"]


def test_reasoner_400_on_extensions_retries_plain(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos([httpx.Response(400, json={"error": "unknown field top_k"}), "<think>t</think>" + json.dumps(GOOD_JSON)])
    set_transport(httpx.MockTransport(fake))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["source"] == "cosmos3-local"
    p1, p2 = fake.chat_payloads()
    assert "top_k" in p1 and "top_k" not in p2 and "media_io_kwargs" not in p2


def test_reasoning_content_field_is_surfaced(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos([_chat_response(json.dumps(GOOD_JSON), reasoning_content="trace from reasoning parser")])
    set_transport(httpx.MockTransport(fake))
    assert reason.assess_clip(str(clip), sidecar)["reasoning"] == "trace from reasoning parser"


def test_nim_backend_uses_base64_and_auth(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    fake = FakeCosmos()
    set_transport(httpx.MockTransport(fake))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["source"] == "nim"
    req = [r for r in fake.requests if r.url.path.endswith("/chat/completions")][0]
    assert req.url.host == "integrate.api.nvidia.com" and req.headers["authorization"] == "Bearer nvapi-test"
    body = json.loads(req.content)
    assert body["model"] == "nvidia/cosmos3-nano-reasoner"
    assert body["messages"][1]["content"][0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert body["media_io_kwargs"] == {"video": {"fps": 4.0}}


def test_connection_error_falls_back_to_heuristic(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")

    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    set_transport(httpx.MockTransport(boom))
    res = reason.assess_clip(str(clip), sidecar)
    assert res["source"] == "heuristic-fallback" and res["verdict"] == "reject" and "refused" in res["error"]
    monkeypatch.setenv("COSMOS_FALLBACK", "0")
    from worldmodels.http import BackendError
    with pytest.raises(BackendError):
        reason.assess_clip(str(clip), sidecar)


def test_heuristic_rules(sidecar):
    r = heuristics.assess(sidecar)
    assert r["verdict"] == "reject" and r["defects"][0]["type"] == "porosity"
    assert r["defects"][0]["t_s"] == 1.5 and any(f["param"] == "gas_lpm" and f["status"] == "violation" for f in r["wps_findings"])
    assert "gas" in r["root_cause"].lower()
    clean = {**sidecar, "wps_violations": [], "defects_in_window": [], "rt_result": "accept"}
    assert heuristics.assess(clean)["verdict"] == "accept"
    only_violation = {**sidecar, "defects_in_window": [], "rt_result": None}
    rv = heuristics.assess(only_violation)
    assert rv["verdict"] == "review" and rv["defects"][0]["type"] == "porosity" and rv["defects"][0]["confidence"] < 0.5
    assert "EXCURSION gas_lpm" in heuristics.explain(sidecar)


def test_chat_and_explain(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(["<think>x</think>The gas flow dropped below the WPS minimum.", "- gas low -> porosity"])
    set_transport(httpx.MockTransport(fake))
    txt = reason.chat([{"role": "user", "content": "Why did J07 fail RT?"}], video_path=str(clip))
    assert txt == "The gas flow dropped below the WPS minimum."
    sent = fake.chat_payloads()[0]["messages"]
    assert sent[0]["role"] == "system" and sent[-1]["content"][0]["type"] == "video_url"
    assert reason.explain_signals(sidecar) == "- gas low -> porosity"
    # replay: explain falls back to the rule-based text, chat to cached text
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    assert reason.explain_signals_meta(sidecar)["source"] == "cache"
    assert "heuristic-fallback" in reason.explain_signals({**sidecar, "clip_id": "z", "signal_stats": {"gas_lpm": {"mean": 3.0, "min": 1.0, "max": 5.0}}})
    meta = reason.chat_meta([{"role": "user", "content": "new question"}])
    assert meta["source"] == "heuristic-fallback" and "replay mode" in meta["text"]


def test_chat_replay_reuses_cached_assessment(monkeypatch, clip, sidecar):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    set_transport(httpx.MockTransport(FakeCosmos()))
    reason.assess_clip(str(clip), sidecar)
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    meta = reason.chat_meta([{"role": "user", "content": "What is wrong with this weld?"}], str(clip))
    assert meta["source"] == "cache" and "Verdict: reject" in meta["text"]


# ----------------------------------------------------------------------------- cache
def test_cache_key_tracks_content_not_path(tmp_path):
    a = _make_video(tmp_path / "a.mp4", seed=1)
    b = tmp_path / "copy" / "b.mp4"
    b.parent.mkdir()
    b.write_bytes(a.read_bytes())
    c = _make_video(tmp_path / "c.mp4", seed=2)
    k = lambda p: cache.make_key("f", {"video": cache.FileRef(str(p)), "x": 1.0})  # noqa: E731
    assert k(a) == k(b) and k(a) != k(c)
    assert cache.make_key("f", {"x": 1}) != cache.make_key("g", {"x": 1})


def test_auto_backend_resolution(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.port)
        if request.url.port == 8000:
            return httpx.Response(200, json={"data": [{"id": "nvidia/Cosmos3-Nano"}]})
        raise httpx.ConnectError("down", request=request)

    set_transport(httpx.MockTransport(handler))
    assert config.resolve_backend("reasoner") == "local"
    assert config.resolve_backend("generator") == "replay"
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    config.clear_probe_cache()
    assert config.resolve_backend("generator") == "nim"
    monkeypatch.setenv("COSMOS_GENERATOR_BACKEND", "replay")
    assert config.resolve_backend("generator") == "replay"


# ----------------------------------------------------------------------------- generator
def test_what_if_local_form_cache_and_replay(monkeypatch, tmp_path, fake_mp4):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(video_bytes=fake_mp4)
    set_transport(httpx.MockTransport(fake))
    meta = generate.what_if_meta("travel speed drops 20% on the fill pass", seconds=5, seed=7, preset="live")
    assert meta["source"] == "cosmos3-local" and Path(meta["path"]).read_bytes() == fake_mp4
    req = [r for r in fake.requests if r.url.path.endswith("/videos/sync")][0]
    assert str(req.url) == "http://localhost:8001/v1/videos/sync" and req.headers["accept"] == "video/mp4"
    fields, files = form_fields(req)
    assert fields["size"] == "832x480" and fields["num_frames"] == "121" and fields["fps"] == "24"
    assert fields["num_inference_steps"] == "35" and fields["guidance_scale"] == "6.0" and fields["flow_shift"] == "10.0"
    assert fields["seed"] == "7" and not files
    ep = json.loads(fields["extra_params"])
    assert ep == {"use_resolution_template": False, "use_duration_template": False, "guardrails": False}
    prompt = json.loads(fields["prompt"])
    assert prompt["resolution"] == {"W": 832, "H": 480} and "travel speed drops 20%" in prompt["temporal_caption"]
    assert json.loads(fields["negative_prompt"])["subjects"]
    side = generate.read_sidecar(meta["path"])
    assert side["source"] == "cosmos3-local" and "latency_ms" in side

    # exact hit
    again = generate.what_if_meta("travel speed drops 20% on the fill pass", seconds=5, seed=7, preset="live")
    assert again["source"] == "cache" and again["cache_match"] == "exact"
    # replay: different preset -> same-prompt alias; reworded -> nearest-prompt
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    alias = generate.what_if_meta("travel speed drops 20% on the fill pass", seconds=5, seed=7, preset="showcase")
    assert alias["source"] == "cache" and alias["cache_match"] == "same-prompt-and-inputs"
    near = generate.what_if_meta("what if the travel speed drops 20 percent", seconds=5)
    assert near["source"] == "cache" and near["cache_match"] == "nearest-prompt"
    # miss -> labelled placeholder that is a real, playable mp4
    ph = generate.what_if_meta("argon cylinder runs empty on a stainless root", seconds=3)
    assert ph["source"] == "heuristic-fallback" and ph["placeholder"] is True
    info = probe(ph["path"])
    assert info.codec == "h264" and info.frames >= 48
    assert generate.what_if("argon cylinder runs empty on a stainless root", seconds=3) == ph["path"]


def test_what_if_image_and_video_conditioning(monkeypatch, tmp_path, fake_mp4, clip):
    from PIL import Image

    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(video_bytes=fake_mp4)
    set_transport(httpx.MockTransport(fake))
    img = tmp_path / "frame.png"
    Image.fromarray(np.zeros((480, 832, 3), np.uint8)).save(img)
    m1 = generate.what_if_meta("gas flow lost", image_path=str(img), preset="showcase")
    fields, files = form_fields(fake.requests[-1])
    assert m1["mode"] == "image2video" and files["input_reference"][1] == "image/png" and fields["size"] == "1280x720"
    # replay with a different conditioning frame (copilot grabs its own) -> same-prompt alias
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    img2 = tmp_path / "frame2.png"
    Image.fromarray(np.full((480, 832, 3), 50, np.uint8)).save(img2)
    r = generate.what_if_meta("gas flow lost", image_path=str(img2), preset="live")
    assert r["source"] == "cache" and r["cache_match"] == "same-prompt" and r["path"] == m1["path"]
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    m2 = generate.what_if_meta("continue the pass", video_path=str(clip), seconds=2.5)
    fields, files = form_fields(fake.requests[-1])
    assert m2["mode"] == "video2video" and files["input_reference"][1] == "video/mp4"
    assert fields["num_frames"] == str(frames_for_seconds(2.5, 24)) == "61"
    ep = json.loads(fields["extra_params"])
    assert ep["condition_frame_indexes_vision"] == [0, 1] and ep["condition_video_keep"] == "last"


def test_what_if_nim_json_body(monkeypatch, tmp_path, fake_mp4):
    from PIL import Image

    monkeypatch.setenv("COSMOS_BACKEND", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("COSMOS_TRANSCODE_H264", "0")
    fake = FakeCosmos(video_bytes=fake_mp4)
    set_transport(httpx.MockTransport(fake))
    img = tmp_path / "f.png"
    Image.fromarray(np.zeros((192, 320, 3), np.uint8)).save(img)
    meta = generate.what_if_meta("porosity", image_path=str(img), preset="live")
    assert meta["source"] == "nim"
    req = fake.requests[-1]
    assert str(req.url) == config.NIM_GENERATOR_URL and req.headers["authorization"] == "Bearer nvapi-test"
    body = json.loads(req.content)
    assert body["model_mode"] == "image2video" and body["resolution"] == "480" and body["num_frames"] == 121
    assert isinstance(body["fps"], float) and isinstance(body["num_inference_steps"], int)
    assert body["input_reference"].startswith("data:image/png;base64,")
    assert "size" not in body and "extra_params" not in body              # strict NIM schema
    assert generate.nim_resolution_key(1280, 720) == "720" and generate.nim_resolution_key(960, 960) == "720_1_1"


def test_nim_plain_404_is_reported(monkeypatch, fake_mp4):
    monkeypatch.setenv("COSMOS_BACKEND", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    set_transport(httpx.MockTransport(lambda r: httpx.Response(404, text="404 page not found")))
    meta = generate.what_if_meta("anything at all", seconds=2)
    assert meta["placeholder"] and "not provisioned" in meta["error"]


# ----------------------------------------------------------------------------- transfer
def test_sim2real_omni_request(monkeypatch, tmp_path, fake_mp4):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    monkeypatch.setenv("COSMOS_TRANSFER_BACKEND", "omni")
    ctrl = tmp_path / "gen" / "SEQ" / "control"
    edge = _make_video(ctrl / "edge.mp4", n=120, size=(320, 180))
    depth = _make_video(ctrl / "depth.mp4", n=120, size=(320, 180), seed=4)
    fake = FakeCosmos(video_bytes=fake_mp4)
    set_transport(httpx.MockTransport(fake))
    out = transfer.sim2real(str(edge), "edge", "photoreal FCAW fill pass", seed=3)
    assert Path(out).exists() and transfer.controls_for_pass(ctrl.parent) == {"edge": str(edge), "depth": str(depth)}
    fields, files = form_fields(fake.requests[-1])
    ep = json.loads(fields["extra_params"])
    assert ep["edge"] == {"control_path": str(edge.resolve()), "preset_edge_threshold": "medium"}
    assert ep["resolution"] == "480" and fields["size"] == "832x480"
    assert ep["max_frames"] == 117 and fields["num_frames"] == "117"        # 120 control frames -> 4k+1
    assert ep["control_guidance"] == 1.5 and ep["num_conditional_frames"] == 1 and ep["guardrails"] is False
    assert fields["guidance_scale"] == "3.0" and json.loads(fields["prompt"])["context"].endswith("control video exactly")

    transfer.sim2real_multi({"edge": str(edge), "depth": str(depth)}, "photoreal FCAW fill pass")
    fields, _ = form_fields(fake.requests[-1])
    ep = json.loads(fields["extra_params"])
    assert set(ep) >= {"edge", "depth"} and "control_weight" not in ep["depth"]

    # replay: placeholder uses the control video as background, labelled
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    ph = transfer.sim2real_meta({"depth": str(depth)}, "a new prompt never generated")
    assert ph["placeholder"] and ph["source"] == "heuristic-fallback" and probe(ph["path"]).frames > 0
    hit = transfer.sim2real_meta({"edge": str(edge)}, "photoreal FCAW fill pass", 3)
    assert hit["source"] == "cache"


def test_transfer_framework_spec_and_nim_body(tmp_path):
    edge = _make_video(tmp_path / "edge.mp4", n=48)
    depth = _make_video(tmp_path / "depth.mp4", n=48, seed=2)
    req = transfer.build_transfer_request({"edge": str(edge), "depth": str(depth)}, "photoreal weld", preset="showcase",
                                          weights={"edge": 0.3, "depth": 0.7})
    spec = transfer.framework_spec(req, "weld_edge_depth_x", tmp_path / "prompt.json")
    # same field set as cosmos3-inference-transfer/specs/edge.json
    for k in ("name", "model_mode", "resolution", "aspect_ratio", "num_frames", "fps", "shift", "num_steps", "seed",
              "num_video_frames_per_chunk", "num_conditional_frames", "num_first_chunk_conditional_frames",
              "share_vision_temporal_positions", "negative_metadata_mode", "negative_prompt_keep_metadata", "guidance",
              "control_guidance", "negative_prompt_file", "prompt_path", "emphasize_control_in_prompt"):
        assert k in spec
    assert spec["model_mode"] == "video2video" and spec["resolution"] == "720" and spec["aspect_ratio"] == "16,9"
    assert spec["num_steps"] == 50 and spec["edge"]["weight"] == 0.3 and spec["depth"]["control_path"] == str(depth.resolve())
    assert Path(spec["negative_prompt_file"]).exists()
    body = transfer.nim_transfer_body(req)
    assert body["transfer"]["edge"]["video"].startswith("data:video/mp4;base64,") and "preset_edge_threshold" not in body["transfer"]["edge"]
    assert body["resolution"] == "720" and body["num_frames"] == 45
    with pytest.raises(ValueError):
        transfer.sim2real(str(edge), "multi", "x")


# ----------------------------------------------------------------------------- CLI / warm cache / anomaly
def test_cli_assess_replay(monkeypatch, capsys, clip, sidecar, tmp_path):
    from worldmodels.cli import main

    side = tmp_path / "side.json"
    side.write_text(json.dumps(sidecar))
    assert main(["--backend", "replay", "assess", str(clip), "--context", str(side)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "heuristic-fallback" and out["verdict"] == "reject"


def test_warm_cache_end_to_end(monkeypatch, tmp_path, fake_mp4, sidecar):
    from worldmodels.showcase import get_showcase, list_showcase, load_showcase, warm_cache

    data = Path(tmp_path / "data")
    clips = data / "curated" / "clips"
    c1 = _make_video(clips / "c1.mp4", seed=1)
    _make_video(clips / "c2.mp4", seed=2)
    (data / "export").mkdir(parents=True)
    (data / "export" / "c1.json").write_text(json.dumps({**sidecar, "clip_id": "c1"}))
    manifest = data / "curated" / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in [
        {"clip_id": "c1", "clip_uri": "clips/c1.mp4", "kept": True, "defects": [{"defect": "porosity"}]},
        {"clip_id": "c2", "clip_uri": "s3://raw/c2.mp4", "kept": True, "defects": []},
        {"clip_id": "c3", "clip_uri": "clips/c3.mp4", "kept": False},
    ]) + "\n")
    ctrl = data / "generated" / "SP-1043_J07_P2_FCAW" / "control"
    _make_video(ctrl / "edge.mp4", n=30)
    _make_video(ctrl / "depth.mp4", n=30, seed=5)
    (ctrl.parent / "meta.json").write_text(json.dumps({"defects": [{"defect": "porosity"}]}))

    monkeypatch.setenv("COSMOS_BACKEND", "local")
    monkeypatch.setenv("COSMOS_TRANSFER_BACKEND", "omni")
    fake = FakeCosmos(video_bytes=fake_mp4)
    set_transport(httpx.MockTransport(fake))
    rep = warm_cache(manifest, preset="smoke", log=lambda m: None)
    assert [a["clip_id"] for a in rep["assess"]] == ["c1", "c2"]
    assert all(a["source"] == "cosmos3-local" for a in rep["assess"]), rep
    assert set(rep["what_if"]) == {"baseline-fcaw-fill", "travel_speed_-20", "gas_loss", "porosity", "voltage_+20",
                                   "interpass_hot", "root-burn-through", "continue-clip"}
    assert set(rep["transfer"]) == {"sim2real-edge", "sim2real-depth", "sim2real-edge-depth-porosity"}
    assert not rep["errors"], rep["errors"]
    assert rep["what_if"]["gas_loss"]["image"].endswith("c1_first.png")   # auto:porosity -> c1

    # the copilot, in replay mode with no GPU, gets the same Cosmos outputs back
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    res = reason.assess_clip(str(c1), {**sidecar, "clip_id": "c1"})
    assert res["source"] == "cache" and res["origin_source"] == "cosmos3-local"
    assert get_showcase("sim2real-edge")["source"] == "cosmos3-local"
    assert Path(list_showcase()["what_if"]["root-burn-through"]["path"]).exists()
    st = cache.stats()["functions"]
    assert st["assess_clip"]["json"] == 2 and st["what_if"]["media"] == 8 and st["sim2real"]["media"] == 3
    # a copilot-style request (template text, its own frame) replays the showcase clip
    from PIL import Image
    own = tmp_path / "copilot_frame.png"
    Image.fromarray(np.zeros((192, 320, 3), np.uint8)).save(own)
    tmpl = load_showcase()["what_if"][2]
    hit = generate.what_if_meta(tmpl["prompt"], image_path=str(own), seconds=5, seed=0,
                                scene={"process": "FCAW", "pass_type": "fill", "defect": None, "view": "torch"})
    assert hit["source"] == "cache" and hit["cache_match"] == "same-prompt"
    near = generate.what_if_meta("what if the shielding gas flow collapses?", seconds=5)
    assert near["source"] == "cache" and near["cache_match"] == "nearest-prompt" and "gas" in near["query_text"].lower()


def test_warm_cache_dry_run_without_manifest(tmp_path):
    from worldmodels.showcase import warm_cache

    rep = warm_cache(tmp_path / "missing.jsonl", dry_run=True, log=lambda m: None)
    assert rep["errors"] and not rep["assess"]


def test_anomaly_placeholder(tmp_path):
    from PIL import Image

    from worldmodels import anomaly

    img = np.full((240, 320, 3), 90, np.uint8)
    img[110:130, :] = np.random.default_rng(0).integers(60, 200, (20, 320, 3))   # textured "bead"
    p = tmp_path / "bead.png"
    Image.fromarray(img).save(p)
    meta = anomaly.synth_defect_meta(str(p), "porosity", seed=1)
    assert meta["source"] == "heuristic-fallback" and meta["anomaly_type"] == "metal_surface+MT_Blowhole"
    out = np.asarray(Image.open(meta["path"]))
    assert out.shape == img.shape and not np.array_equal(out, img)
    mask = np.asarray(Image.open(meta["mask"]))
    ys = np.nonzero(mask)[0]
    assert ys.size and 90 < ys.mean() < 150                                    # mask on the bead band
    with pytest.raises(ValueError):
        anomaly.synth_defect(str(p), "spatter-free")


def test_healthcheck_runs_offline(monkeypatch, capsys):
    from worldmodels.serve import healthcheck

    set_transport(httpx.MockTransport(lambda r: httpx.Response(503)))
    rc = healthcheck.main([])
    out = capsys.readouterr().out
    assert "Reasoner" in out and "Generator" in out and rc in (0, 1)


def test_frame_math():
    assert frames_for_seconds(5, 24) == 121 and frames_for_seconds(7.875, 24) == 189 and frames_for_seconds(2.5, 24) == 61
    from worldmodels.video import frames_4k1
    assert frames_4k1(121) == 121 and frames_4k1(120) == 117
    assert re.match(r"^\d+x\d+$", "x".join(map(str, generate.resolve_size("720", generate.PRESETS["live"]))))


def test_generator_readiness_uses_health_not_models(monkeypatch):
    """vLLM-Omni answers /v1/models with 200 + [] while still loading; only /health means ready."""
    from worldmodels import config as cfgmod
    monkeypatch.setenv("COSMOS_GENERATOR_URL", "http://localhost:8001")
    monkeypatch.setenv("COSMOS_REASONER_URL", "http://localhost:8000/v1")
    s = cfgmod.get_settings()
    assert cfgmod.health_url("generator", s) == "http://localhost:8001/health"
    assert cfgmod.health_url("reasoner", s) == "http://localhost:8000/v1/models"
    monkeypatch.setenv("COSMOS_GENERATOR_URL", "http://localhost:8001/v1")
    assert cfgmod.health_url("generator", cfgmod.get_settings()) == "http://localhost:8001/health"
