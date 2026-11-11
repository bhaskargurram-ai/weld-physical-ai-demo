"""runpod/launch.py against a fake Runpod API v2 (no network)."""
from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launch", ROOT / "runpod" / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class FakeResp(io.BytesIO):
    def __iter__(self):
        return iter(self.getvalue().splitlines(keepends=True))


class FakeApi:
    """Routes urllib requests to canned Runpod API v2 behaviour and records every call."""

    def __init__(self, unavailable=("NVIDIA H100 80GB HBM3",), volumes=None, log_lines=None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.unavailable = unavailable if callable(unavailable) else set(unavailable)
        self.volumes = list(volumes or [])
        self.pod_status = "RUNNING"
        self.log_lines = log_lines or ["WELDWORLD 00:00:01 START run=x", "WELDWORLD 00:10:00 DONE rc=0"]

    def __call__(self, req, timeout=None):
        method, url = req.get_method(), req.full_url
        path = url.split("/v2", 1)[1].split("?")[0]
        body = json.loads(req.data) if req.data else None
        self.calls.append((method, path, body))
        assert req.headers["Authorization"] == "Bearer rp-key"

        def ok(obj, code=200):
            return FakeResp(json.dumps(obj).encode())

        def err(code, msg):
            raise urllib.error.HTTPError(url, code, msg, {}, io.BytesIO(json.dumps({"error": msg}).encode()))

        if (method, path) == ("GET", "/network-volumes"):
            return ok({"networkVolumes": self.volumes, "pagination": {"nextCursor": None, "hasNextPage": False}})
        if (method, path) == ("GET", "/catalog/datacenters"):
            return ok({"dataCenters": [
                {"id": "US-TX-3", "networkVolumeTypes": ["STANDARD"],
                 "gpuAvailability": [{"id": "NVIDIA H100 80GB HBM3", "availability": "LOW"}]},
                {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"],
                 "gpuAvailability": [{"id": "NVIDIA H100 80GB HBM3", "availability": "HIGH"}]},
                {"id": "NO-VOL", "networkVolumeTypes": [],
                 "gpuAvailability": [{"id": "NVIDIA H100 80GB HBM3", "availability": "HIGH"}]},
            ]})
        if (method, path) == ("POST", "/network-volumes"):
            n = sum(1 for m, p, _ in self.calls if (m, p) == ("POST", "/network-volumes"))
            v = {"id": f"vol_{n}", "name": body["name"], "size": body["size"], "dataCenter": body["dataCenter"]}
            self.volumes.append(v)
            return ok(v, 201)
        if (method, path) == ("POST", "/pods"):
            if callable(self.unavailable):
                if self.unavailable(body):
                    err(400, "There are no longer any instances available with the requested specifications.")
            elif body["gpu"]["id"] in self.unavailable:
                err(400, "There are no instances currently available")
            return ok({"id": "pod_1", "cost": 5.98, **body}, 201)
        if method == "GET" and path == "/pods/pod_1/logs":
            lines = []
            for i, t in enumerate(self.log_lines):
                lines += [f"id: 2026-11-11T00:00:0{i}Z", "data: " + json.dumps({"source": "container", "line": t}), ""]
            return FakeResp(("\n".join(lines) + "\n").encode())
        if method == "GET" and path == "/pods/pod_1":
            return ok({"id": "pod_1", "status": self.pod_status})
        if method == "DELETE" and path == "/pods/pod_1":
            self.pod_status = "TERMINATED"
            return FakeResp(b"")
        if method == "DELETE" and path.startswith("/network-volumes/"):
            self.volumes = [v for v in self.volumes if v["id"] != path.rsplit("/", 1)[1]]
            return FakeResp(b"")
        err(404, f"unexpected {method} {path}")


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "STATE_FILE", tmp_path / "state.json")
    for k in ("RUNPOD_API_KEY", "HF_TOKEN", "GITHUB_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / ".env"
    f.write_text(
        "RUNPOD_API_KEY=rp-key\nHF_TOKEN=hf_x   # read token\nANTHROPIC_API_KEY='sk-ant-x'\n"
        "GPU_TYPE=NVIDIA H100 80GB HBM3,NVIDIA H100 NVL  # in order\nGPU_COUNT=2\nRUN_MODE=job\n"
    )
    return f


def test_parse_env_file_handles_comments_quotes_spaces(env_file):
    cfg = launch.parse_env_file(env_file)
    assert cfg["HF_TOKEN"] == "hf_x"
    assert cfg["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert launch.gpu_types(cfg)[:2] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]
    assert launch.gpu_types(cfg).count("NVIDIA H100 NVL") == 1          # fallbacks de-duplicated


def test_example_env_parses_and_has_required_keys():
    cfg = launch.parse_env_file(ROOT / ".env.example")
    for k in ("RUNPOD_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "GPU_TYPE", "VOLUME_NAME", "RUN_MODE"):
        assert k in cfg
    assert cfg["GPU_TYPE"] == "NVIDIA H100 80GB HBM3"
    # empty keys followed by a comment stay empty
    assert cfg["RUNPOD_API_KEY"] == cfg["HF_TOKEN"] == cfg["COPILOT_LLM"] == ""
    assert cfg["SERVE_HOURS"] == "2" and cfg["DELETE_VOLUME_AT_END"] == "0"


def test_pick_datacenter_prefers_stock_and_volume_support():
    data = [{"id": "A", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "G1", "availability": "LOW"}]},
            {"id": "B", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "G2", "availability": "HIGH"}]},
            {"id": "C", "networkVolumeTypes": [], "gpuAvailability": [{"id": "G1", "availability": "HIGH"}]}]
    assert launch.pick_datacenter(data, ["G1", "G2"]) == "A"      # first GPU type wins over more stock
    assert launch.pick_datacenter(data, ["G2"]) == "B"
    assert launch.pick_datacenter(data, ["G1"], forced="X") == "X"
    with pytest.raises(SystemExit):
        launch.pick_datacenter(data, ["G9"])


def test_pod_body_never_forwards_account_key(env_file):
    cfg = launch.load_config(env_file)
    body = launch.pod_body(cfg, "NVIDIA H100 80GB HBM3", "vol_1", "EU-RO-1")
    assert "RUNPOD_API_KEY" not in body["env"]
    assert body["env"]["HF_TOKEN"] == "hf_x" and body["env"]["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert body["mounts"] == {"network": [{"volumeId": "vol_1", "path": "/workspace"}]}
    assert body["gpu"] == {"id": "NVIDIA H100 80GB HBM3", "count": 2, "minCudaVersion": "13.0"}
    assert body["dataCenterIds"] == ["EU-RO-1"] and "7860/http" in body["ports"]
    assert body["cmd"][:2] == ["bash", "-lc"] and "pod_entry.sh" in body["cmd"][2]


def test_full_run_creates_volume_falls_back_gpu_follows_logs_and_terminates(env_file, monkeypatch, capsys):
    api = FakeApi()
    monkeypatch.setattr(launch.urllib.request, "urlopen", api)
    rc = launch.main(["run", "--env-file", str(env_file)])
    assert rc == 0
    seq = [(m, p) for m, p, _ in api.calls]
    assert ("POST", "/network-volumes") in seq
    vol_body = next(b for m, p, b in api.calls if (m, p) == ("POST", "/network-volumes"))
    assert vol_body["dataCenter"] == "EU-RO-1"                      # HIGH stock + volume support
    pods = [b for m, p, b in api.calls if (m, p) == ("POST", "/pods")]
    assert [b["gpu"]["id"] for b in pods] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]
    assert ("DELETE", "/pods/pod_1") in seq                          # terminated at the end
    assert not any(m == "DELETE" and p.startswith("/network-volumes") for m, p in seq)  # volume kept
    assert json.loads(launch.STATE_FILE.read_text())["pod_id"] is None
    assert "DONE rc=0" in capsys.readouterr().out


def test_failed_run_propagates_rc_and_can_delete_volume(env_file, monkeypatch):
    env_file.write_text(env_file.read_text() + "DELETE_VOLUME_AT_END=1\n")
    api = FakeApi(unavailable=(), volumes=[{"id": "vol_9", "name": "weldworld", "size": 300, "dataCenter": "US-TX-3"}],
                  log_lines=["WELDWORLD 00:00:01 stage=bootstrap rc=1", "WELDWORLD 00:00:02 DONE rc=10"])
    monkeypatch.setattr(launch.urllib.request, "urlopen", api)
    assert launch.main(["run", "--env-file", str(env_file)]) == 10
    seq = [(m, p) for m, p, _ in api.calls]
    assert ("POST", "/network-volumes") not in seq                   # reused existing volume
    assert next(b for m, p, b in api.calls if (m, p) == ("POST", "/pods"))["dataCenterIds"] == ["US-TX-3"]
    assert ("DELETE", "/network-volumes/vol_9") in seq


def test_dry_run_masks_secrets(env_file, monkeypatch, capsys):
    monkeypatch.setattr(launch.urllib.request, "urlopen", FakeApi())
    assert launch.main(["run", "--env-file", str(env_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "hf_x" not in out and "sk-ant-x" not in out and '"HF_TOKEN": "***"' in out


def test_bad_image_stops_immediately(env_file, monkeypatch):
    api = FakeApi(unavailable=())
    orig = api.__call__

    def bad_image(req, timeout=None):
        if req.get_method() == "POST" and req.full_url.endswith("/pods"):
            api.calls.append(("POST", "/pods", json.loads(req.data)))
            raise urllib.error.HTTPError(req.full_url, 422, "x", {}, io.BytesIO(
                b'{"detail":"Container image \\"x\\" was not found on the registry."}'))
        return orig(req, timeout)

    monkeypatch.setattr(launch.urllib.request, "urlopen", bad_image)
    with pytest.raises(SystemExit, match="container image"):
        launch.main(["run", "--env-file", str(env_file)])
    assert sum(1 for m, p, _ in api.calls if (m, p) == ("POST", "/pods")) == 1   # no retry on other GPU types


def test_default_image_tag_is_a_published_release():
    cfg = launch.parse_env_file(ROOT / ".env.example")
    assert cfg["IMAGE"] == launch.DEFAULTS["IMAGE"] == "runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404"


def test_attempt_order_prefers_cuda13_then_two_gpus():
    cfg = dict(launch.DEFAULTS, GPU_TYPE="G1", GPU_FALLBACK_TYPES="G2")
    assert launch.pod_attempts(cfg) == [("G1", 2, "13.0"), ("G2", 2, "13.0"), ("G1", 1, "13.0"), ("G2", 1, "13.0"),
                                        ("G1", 2, "12.8"), ("G2", 2, "12.8"), ("G1", 1, "12.8"), ("G2", 1, "12.8")]
    cfg.update(ALLOW_SINGLE_GPU="0", ALLOW_CUDA_FALLBACK="0")
    assert launch.pod_attempts(cfg) == [("G1", 2, "13.0"), ("G2", 2, "13.0")]


def test_pod_body_adapts_env_to_a100_and_cuda12(env_file):
    cfg = launch.load_config(env_file)
    env = launch.pod_body(cfg, "NVIDIA A100-SXM4-80GB", "v", "dc", count=1, min_cuda="12.8")["env"]
    assert env["GENERATOR_PRECISION"] == "bf16" and env["OFFLOAD"] == "1" and env["CURATE_BACKEND"] == "lite"
    env = launch.pod_body(cfg, "NVIDIA H100 80GB HBM3", "v", "dc")["env"]
    assert "GENERATOR_PRECISION" not in env and env["CURATE_BACKEND"] == "curator-next"


def test_empty_volume_is_moved_when_its_datacenter_is_sold_out(env_file, monkeypatch):
    # the user's case: volume created in US-CA-2 by an earlier attempt, no pod ever ran, nothing free there
    launch.STATE_FILE.write_text(json.dumps({"volume_id": "vol_old"}))
    api = FakeApi(unavailable=lambda b: b["dataCenterIds"] == ["US-CA-2"],
                  volumes=[{"id": "vol_old", "name": "weldworld", "size": 300, "dataCenter": "US-CA-2"}])
    monkeypatch.setattr(launch.urllib.request, "urlopen", api)
    assert launch.main(["run", "--env-file", str(env_file)]) == 0
    seq = [(m, p) for m, p, _ in api.calls]
    assert ("DELETE", "/network-volumes/vol_old") in seq
    new_vol = next(b for m, p, b in api.calls if (m, p) == ("POST", "/network-volumes"))
    assert new_vol["dataCenter"] == "EU-RO-1"
    ok_pod = [b for m, p, b in api.calls if (m, p) == ("POST", "/pods")][-1]
    assert ok_pod["dataCenterIds"] == ["EU-RO-1"] and ok_pod["gpu"]["count"] == 2
    st = json.loads(launch.STATE_FILE.read_text())
    assert st["used_volumes"] == [st["volume_id"]] and st["volume_id"] != "vol_old" and st["data_center"] == "EU-RO-1"


def test_volume_with_data_is_never_moved(env_file, monkeypatch):
    launch.STATE_FILE.write_text(json.dumps({"volume_id": "vol_old", "used_volumes": ["vol_old"]}))
    api = FakeApi(unavailable=lambda b: True,
                  volumes=[{"id": "vol_old", "name": "weldworld", "size": 300, "dataCenter": "US-CA-2"}])
    monkeypatch.setattr(launch.urllib.request, "urlopen", api)
    with pytest.raises(SystemExit, match="already holds data"):
        launch.main(["run", "--env-file", str(env_file)])
    assert not any(m == "DELETE" for m, _, _ in api.calls)


def test_untracked_volume_is_never_moved(env_file, monkeypatch):
    api = FakeApi(unavailable=lambda b: True,   # no state file: volume may hold someone else's data
                  volumes=[{"id": "vol_x", "name": "weldworld", "size": 300, "dataCenter": "US-CA-2"}])
    monkeypatch.setattr(launch.urllib.request, "urlopen", api)
    with pytest.raises(SystemExit):
        launch.main(["run", "--env-file", str(env_file)])
    assert not any(m == "DELETE" for m, _, _ in api.calls)
