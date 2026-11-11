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
        self.unavailable = set(unavailable)
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
            v = {"id": "vol_1", "name": body["name"], "size": body["size"], "dataCenter": body["dataCenter"]}
            self.volumes.append(v)
            return ok(v, 201)
        if (method, path) == ("POST", "/pods"):
            if body["gpu"]["id"] in self.unavailable:
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
    assert launch.gpu_types(cfg) == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]


def test_example_env_parses_and_has_required_keys():
    cfg = launch.parse_env_file(ROOT / ".env.example")
    for k in ("RUNPOD_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "GPU_TYPE", "VOLUME_NAME", "RUN_MODE"):
        assert k in cfg
    assert cfg["GPU_TYPE"] == "NVIDIA H100 80GB HBM3"


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
