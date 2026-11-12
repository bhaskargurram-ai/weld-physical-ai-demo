#!/usr/bin/env python3
"""One-command Runpod run for the WeldWorld demo (runs on your laptop, stdlib only).

    cp .env.example .env            # fill in RUNPOD_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY (+ GITHUB_TOKEN)
    python3 runpod/launch.py        # = run: volume -> pod -> pipeline -> follow logs -> terminate

Commands
    run        create/reuse the network volume, create the GPU pod, follow its logs, terminate at the end
    status     show the pod/volume recorded in .runpod_state.json
    availability  which data centers list stock for your GPU types right now
    logs       re-attach to the running pod's log stream (terminates it when the run finishes)
    down       terminate the pod now (add --delete-volume to also delete the storage)

The pod runs runpod/pod_entry.sh and terminates itself when the code ends (and after MAX_HOURS
whatever happens); this launcher terminates it too, so nothing keeps billing if either side dies.
Uses the Runpod REST API v2 (https://api.runpod.io/v2).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".runpod_state.json"
MOUNT = "/workspace"
COPILOT_PORT = 7860

# Values forwarded from .env into the pod's environment. RUNPOD_API_KEY is deliberately NOT here:
# Runpod injects a pod-scoped key, which pod_entry.sh uses to terminate the pod.
POD_ENV_KEYS = [
    "HF_TOKEN", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "GITHUB_TOKEN", "REPO_URL", "REPO_REF",
    "RUN_MODE", "SERVE_HOURS", "MAX_HOURS", "N_PASSES", "CURATE_BACKEND", "COSMOS_GEN_PRESET",
    "COPILOT_LLM", "COPILOT_MODEL", "HF_USER",
    "DATA_SOURCE", "REAL_SOURCES", "AIMEN_COUPONS", "BAM_WIDS", "WARM_LIMIT", "WARM_PHOTO_LIMIT",
]
DEFAULTS = {
    "REPO_URL": "https://github.com/bhaskargurram-ai/weld-physical-ai-demo",
    "REPO_REF": "main",
    "RUN_MODE": "job",
    "SERVE_HOURS": "2",
    "MAX_HOURS": "6",
    "N_PASSES": "24",
    "CURATE_BACKEND": "curator-next",
    "COSMOS_GEN_PRESET": "live",
    "GPU_TYPE": "NVIDIA H100 80GB HBM3",
    "GPU_COUNT": "2",
    "CLOUD": "SECURE",
    "MIN_CUDA_VERSION": "13.0",
    "IMAGE": "runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404",
    "CONTAINER_DISK_GB": "80",
    "VOLUME_NAME": "weldworld",
    "VOLUME_SIZE_GB": "150",
    "DATA_CENTER": "",
    "DELETE_VOLUME_AT_END": "0",
    # automatic fallbacks when the preferred GPU is sold out (each can be turned off with 0 / empty)
    "GPU_FALLBACK_TYPES": "NVIDIA H100 NVL,NVIDIA H200,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe",
    "ALLOW_SINGLE_GPU": "1",
    "ALLOW_CUDA_FALLBACK": "1",
    "AUTO_RELOCATE_VOLUME": "1",
    "POD_NAME": "weldworld",
}
# Data centers with Runpod's S3-compatible API for network volumes (docs.runpod.io/storage/s3-api, 2026-11):
# runpod/download.py can only fetch results from a volume in one of these, so new volumes prefer them.
S3_DATACENTERS = {"EU-CZ-1", "EU-RO-1", "EUR-IS-1", "EUR-NO-1", "US-CA-2", "US-GA-2", "US-IL-1", "US-KS-2",
                  "US-MD-1", "US-MO-1", "US-MO-2", "US-NC-1", "US-NC-2", "US-NE-1", "US-WA-1"}
AVAIL_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

# Pod start command: start Runpod's own services (SSH/Jupyter) if the image has them, fetch the repo
# onto the volume (token never written to git config), then hand over to pod_entry.sh. If anything
# here fails the pod terminates itself instead of idling on a GPU.
BOOT_SCRIPT = r"""
set -uo pipefail
say() { echo "WELDWORLD $(date -u +%H:%M:%S) $*"; }
die() {
  say "boot failed: $*"
  curl -sS -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H 'Content-Type: application/json' \
    --data '{"action":"terminate"}' >/dev/null 2>&1
  say "DONE rc=2"; sleep infinity
}
[ -x /start.sh ] && { /start.sh >/var/log/runpod-start.log 2>&1 & }
REPO=/workspace/weld-physical-ai-demo
URL="${REPO_URL}"
[ -n "${GITHUB_TOKEN:-}" ] && URL="${URL/https:\/\//https://x-access-token:${GITHUB_TOKEN}@}"
mkdir -p "$REPO" && cd "$REPO" || die "no /workspace (is the network volume mounted?)"
[ -d .git ] || git init -q || die "git init"
say "fetching ${REPO_URL} @ ${REPO_REF}"
git fetch -q --depth 1 "$URL" "${REPO_REF}" || die "git fetch failed (private repo? set GITHUB_TOKEN)"
git checkout -qf FETCH_HEAD || die "git checkout"
exec bash runpod/pod_entry.sh
"""


# ============================================================================ env
def parse_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; supports quotes, `export`, and trailing `# comments` (values may contain spaces)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        val = val.strip()
        if val[:1] in ("'", '"'):
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end > 0 else val[1:]
        elif val.startswith("#"):
            val = ""  # `KEY=   # comment` means empty
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            out[key] = val
    return out


def load_config(env_path: Path) -> dict[str, str]:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in parse_env_file(env_path).items() if v != ""})
    # real environment variables win over the file (handy for CI / one-off overrides)
    for k in set(cfg) | set(POD_ENV_KEYS) | {"RUNPOD_API_KEY"}:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


# ============================================================================ API
class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class Runpod:
    def __init__(self, api_key: str, base: str = API, opener=None):
        self.key = api_key
        self.base = base.rstrip("/")
        self._open = opener or urllib.request.urlopen

    def _req(self, method: str, path: str, body: dict | None = None, params: dict | None = None,
             stream: bool = False, headers: dict | None = None, timeout: float = 60):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(body).encode() if body is not None else None
        h = {"Authorization": f"Bearer {self.key}", "Accept": "application/json",
             "User-Agent": "weldworld-launcher/1.0"}
        if data is not None:
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        for attempt in range(6):
            req = urllib.request.Request(url, data=data, method=method, headers=h)
            try:
                resp = self._open(req, timeout=timeout)
                if stream:
                    return resp
                raw = resp.read().decode() or "{}"
                return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code in (429, 502, 503, 504) and attempt < 5:
                    wait = float(e.headers.get("Retry-After") or 2 ** attempt)
                    time.sleep(min(wait, 30))
                    continue
                raise ApiError(e.code, text) from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise ApiError(0, str(e)) from None
        raise ApiError(0, "unreachable")

    # --- volumes
    def list_volumes(self) -> list[dict]:
        out, cursor = [], None
        while True:
            r = self._req("GET", "/network-volumes", params={"cursor": cursor} if cursor else None)
            out += r.get("networkVolumes") or []
            cursor = (r.get("pagination") or {}).get("nextCursor")
            if not cursor:
                return out

    def create_volume(self, name: str, size_gb: int, data_center: str) -> dict:
        return self._req("POST", "/network-volumes", {"name": name, "size": size_gb, "dataCenter": data_center})

    def delete_volume(self, vid: str) -> None:
        self._req("DELETE", f"/network-volumes/{vid}")

    def datacenters(self) -> list[dict]:
        return self._req("GET", "/catalog/datacenters", params={"include": "GPU_AVAILABILITY"}).get("dataCenters") or []

    # --- pods
    def create_pod(self, body: dict) -> dict:
        return self._req("POST", "/pods", body)

    def get_pod(self, pid: str) -> dict:
        return self._req("GET", f"/pods/{pid}")

    def terminate_pod(self, pid: str) -> None:
        try:
            self._req("DELETE", f"/pods/{pid}")
        except ApiError as e:
            if e.status != 404:
                raise

    def logs(self, pid: str, last_id: str | None = None):
        headers = {"Accept": "text/event-stream"}
        if last_id:
            headers["Last-Event-ID"] = last_id
        return self._req("GET", f"/pods/{pid}/logs", params=None if last_id else {"tail": 200},
                         stream=True, headers=headers, timeout=120)


# ============================================================================ planning
def gpu_types(cfg: dict) -> list[str]:
    """GPU_TYPE (preferred, comma list) followed by GPU_FALLBACK_TYPES, without duplicates."""
    out: list[str] = []
    for g in (cfg.get("GPU_TYPE", "") + "," + cfg.get("GPU_FALLBACK_TYPES", "")).split(","):
        g = g.strip()
        if g and g not in out:
            out.append(g)
    return out


def pod_attempts(cfg: dict) -> list[tuple[str, int, str]]:
    """(gpu, count, min_cuda) in priority order: keep CUDA 13 (real Cosmos Curator) before keeping 2 GPUs,
    keep 2 GPUs before falling back to CUDA 12.8 (Curator -> CPU 'lite' curation)."""
    cuda = cfg.get("MIN_CUDA_VERSION", "")
    cudas = [cuda]
    if cfg.get("ALLOW_CUDA_FALLBACK") == "1" and cuda and float(cuda) > 12.8:
        cudas.append("12.8")
    count = int(cfg["GPU_COUNT"])
    counts = [count] + ([1] if cfg.get("ALLOW_SINGLE_GPU") == "1" and count > 1 else [])
    return [(g, n, c) for c in cudas for n in counts for g in gpu_types(cfg)]


def pick_datacenter(dcs: list[dict], gpus: list[str], forced: str = "", prefer=S3_DATACENTERS) -> str:
    """Best data center for the first GPU type that has stock and that supports network volumes; data centers in
    ``prefer`` (S3 API for runpod/download.py) win over any without it."""
    if forced:
        return forced
    best, best_key = None, (-1, -1, -1)
    for dc in dcs:
        if not dc.get("networkVolumeTypes"):
            continue
        avail = {g.get("id"): AVAIL_SCORE.get(g.get("availability", "NONE"), 0)
                 for g in dc.get("gpuAvailability") or []}
        for rank, gpu in enumerate(gpus):
            score = avail.get(gpu, 0)
            if score:
                key = (dc["id"] in prefer, len(gpus) - rank, score)  # S3 API, earlier GPU types, more stock
                if key > best_key:
                    best, best_key = dc["id"], key
                break
    if not best:
        raise SystemExit(f"No data center with network volumes currently lists stock for {gpus}. "
                         "Try another GPU_TYPE (comma-separated list) or set DATA_CENTER.")
    return best


def pod_env(cfg: dict) -> dict[str, str]:
    env = {k: cfg[k] for k in POD_ENV_KEYS if cfg.get(k)}
    env["WORKSPACE"] = MOUNT
    env["COPILOT_PORT"] = str(COPILOT_PORT)
    return env


def pod_body(cfg: dict, gpu: str, volume_id: str, data_center: str,
             count: int | None = None, min_cuda: str | None = None) -> dict:
    count = int(cfg["GPU_COUNT"]) if count is None else count
    min_cuda = cfg.get("MIN_CUDA_VERSION", "") if min_cuda is None else min_cuda
    gpu_spec: dict = {"id": gpu, "count": count}
    if min_cuda:
        gpu_spec["minCudaVersion"] = min_cuda
    env = pod_env(cfg)
    env["GPU_MODEL"] = gpu
    if "A100" in gpu:  # Ampere has no FP8: keep the generator in BF16 (offloaded when it shares one GPU)
        env["GENERATOR_PRECISION"] = "bf16"
        if count == 1:
            env["OFFLOAD"] = "1"
    if min_cuda and float(min_cuda) < 13 and env.get("CURATE_BACKEND", "").startswith("curator"):
        env["CURATE_BACKEND"] = "lite"  # Cosmos Curator needs a CUDA 13 driver
    return {
        "name": cfg["POD_NAME"],
        "image": cfg["IMAGE"],
        "cmd": ["bash", "-lc", BOOT_SCRIPT],
        "env": env,
        "ports": [f"{COPILOT_PORT}/http", "22/tcp"],
        "disk": int(cfg["CONTAINER_DISK_GB"]),
        "cloud": cfg["CLOUD"],
        "dataCenterIds": [data_center],
        "gpu": gpu_spec,
        "mounts": {"network": [{"volumeId": volume_id, "path": MOUNT}]},
    }


# ============================================================================ state
def save_state(**kw) -> dict:
    st = load_state()
    st.update(kw)
    STATE_FILE.write_text(json.dumps(st, indent=2))
    return st


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def log(msg: str) -> None:
    print(f"\033[36m[launch {time.strftime('%H:%M:%S')}]\033[0m {msg}", flush=True)


# ============================================================================ steps
def ensure_volume(rp: Runpod, cfg: dict) -> dict:
    name = cfg["VOLUME_NAME"]
    for v in rp.list_volumes():
        if v.get("name") == name:
            log(f"reusing network volume '{name}' ({v['id']}, {v.get('size')} GB, {v.get('dataCenter')})")
            if v.get("dataCenter") not in S3_DATACENTERS:
                log(f"  note: {v.get('dataCenter')} has no Runpod S3 API, so runpod/download.py cannot fetch results "
                    "from this volume. To move it: python runpod/launch.py down --delete-volume, then run again.")
            v["_known"] = load_state().get("volume_id") == v["id"]  # tracked by this checkout's state file
            return v
    dc = pick_datacenter(rp.datacenters(), gpu_types(cfg), cfg.get("DATA_CENTER", ""))
    log(f"creating network volume '{name}' ({cfg['VOLUME_SIZE_GB']} GB) in {dc}")
    v = rp.create_volume(name, int(cfg["VOLUME_SIZE_GB"]), dc)
    v.setdefault("dataCenter", dc)
    v["_known"] = True
    return v


def _short(gpu: str) -> str:
    return gpu.replace("NVIDIA ", "").replace(" 80GB HBM3", " SXM")


def try_pod_in(rp: Runpod, cfg: dict, vol: dict, quiet: bool = False) -> dict | None:
    """Try every (gpu, count, cuda) fallback in the volume's data center. Returns the pod or None."""
    for gpu, count, cuda in pod_attempts(cfg):
        label = f"{count}x {_short(gpu)}, CUDA>={cuda or 'any'} in {vol['dataCenter']}"
        try:
            pod = rp.create_pod(pod_body(cfg, gpu, vol["id"], vol["dataCenter"], count, cuda))
        except ApiError as e:
            if e.status == 422 and "image" in e.body.lower():
                raise SystemExit(f"Runpod rejected the container image: {e}\n"
                                 "Set IMAGE in .env to an existing tag, e.g. "
                                 "runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404 "
                                 "(list: https://hub.docker.com/r/runpod/pytorch/tags)") from None
            if e.status in (400, 409, 503) or "available" in e.body.lower():
                if not quiet:
                    log(f"  - {label}: none free")
                continue
            raise SystemExit(f"Pod creation failed ({label}): {e}") from None
        notes = []
        if count < int(cfg["GPU_COUNT"]):
            notes.append("single-GPU budget mode")
        if cuda and float(cuda) < 13:
            notes.append("CUDA 12.8 host -> Cosmos Curator replaced by CPU curation")
        log(f"pod {pod['id']} created: {label} (${pod.get('cost', '?')}/h)" + (f"  [{'; '.join(notes)}]" if notes else ""))
        return pod
    return None


def ranked_datacenters(rp: Runpod, gpus: list[str], exclude: str = "") -> list[str]:
    """Data centers with network-volume support, best stock for the earliest GPU types first."""
    ranked = []
    for dc in rp.datacenters():
        if dc.get("id") == exclude or not dc.get("networkVolumeTypes"):
            continue
        avail = {g.get("id"): AVAIL_SCORE.get(g.get("availability", "NONE"), 0) for g in dc.get("gpuAvailability") or []}
        score = max(((len(gpus) - i) * 10 + avail.get(g, 0) for i, g in enumerate(gpus) if avail.get(g)), default=0)
        if score:
            ranked.append((score + (1000 if dc["id"] in S3_DATACENTERS else 0), dc["id"]))
    return [d for _, d in sorted(ranked, reverse=True)]


def delete_volume_when_free(rp: Runpod, vid: str) -> bool:
    for _ in range(20):  # 409 while a pod still holds the mount
        try:
            rp.delete_volume(vid)
            return True
        except ApiError as e:
            if e.status == 404:
                return True
            if e.status != 409:
                log(f"  could not delete volume {vid}: {e}")
                return False
            time.sleep(15)
    return False


def create_pod(rp: Runpod, cfg: dict, vol: dict, move_volume: bool = False,
               wait_minutes: float = 0) -> tuple[dict, dict]:
    """Create the pod next to the volume. If nothing fits there, move the volume to a data center with
    stock when that is safe (never used) or explicitly allowed (--move-volume), else retry until
    --wait minutes have passed. Returns (pod, volume)."""
    deadline = time.time() + wait_minutes * 60
    log(f"looking for a GPU pod in {vol['dataCenter']} (preferred {cfg['GPU_COUNT']}x {_short(gpu_types(cfg)[0])}, "
        f"CUDA>={cfg.get('MIN_CUDA_VERSION') or 'any'})")
    rounds = 0
    while True:
        pod = try_pod_in(rp, cfg, vol, quiet=rounds > 0)
        if pod:
            return pod, vol
        rounds += 1
        never_used = bool(vol.get("_known")) and vol["id"] not in (load_state().get("used_volumes") or [])
        can_move = (cfg.get("AUTO_RELOCATE_VOLUME") == "1" and not cfg.get("DATA_CENTER")
                    and (never_used or move_volume))
        if can_move:
            for dc in ranked_datacenters(rp, gpu_types(cfg), exclude=vol["dataCenter"])[:4]:
                if never_used:
                    log(f"volume '{vol.get('name')}' is empty -> moving it to {dc}, which lists stock")
                else:
                    log(f"--move-volume: deleting '{vol.get('name')}' in {vol['dataCenter']} (cached weights/envs; "
                        f"the next setup re-downloads them, ~30 min) and recreating it in {dc}")
                if not delete_volume_when_free(rp, vol["id"]):
                    break
                save_state(volume_id=None)
                vol = rp.create_volume(cfg["VOLUME_NAME"], int(cfg["VOLUME_SIZE_GB"]), dc)
                vol.setdefault("dataCenter", dc)
                vol.setdefault("name", cfg["VOLUME_NAME"])
                vol["_known"] = True
                never_used = True
                save_state(volume_id=vol["id"], data_center=dc)
                pod = try_pod_in(rp, cfg, vol)
                if pod:
                    return pod, vol
        if time.time() < deadline:
            log(f"nothing free in {vol['dataCenter']} — retrying every 2 min until "
                f"{time.strftime('%H:%M', time.localtime(deadline))} (Ctrl-C to stop)")
            time.sleep(min(120, max(5, deadline - time.time())))
            continue
        if not can_move and not never_used:
            raise SystemExit(
                f"No GPU pod free in {vol['dataCenter']}, where volume '{vol.get('name')}' lives (it holds cached "
                "weights/envs from an earlier run, so it was not moved automatically). Options:\n"
                "  python runpod/launch.py availability        # see which data centers have GPUs now\n"
                "  python runpod/launch.py --wait 30           # keep retrying here for 30 minutes\n"
                "  python runpod/launch.py --move-volume       # recreate the volume where GPUs are free "
                "(re-downloads ~30 min of setup)")
        raise SystemExit("No GPU capacity found for any GPU type / data center right now. Runpod stock changes by "
                         "the minute: retry with --wait 30, or widen GPU_FALLBACK_TYPES. "
                         f"(volume '{cfg['VOLUME_NAME']}' is in {vol['dataCenter']})")


def availability_table(rp: Runpod, cfg: dict, vol_dc: str = "") -> str:
    gpus = gpu_types(cfg)
    rows = []
    for dc in rp.datacenters():
        if not dc.get("networkVolumeTypes"):
            continue
        avail = {g.get("id"): g.get("availability", "NONE") for g in dc.get("gpuAvailability") or []}
        cells = [avail.get(g, "-") for g in gpus]
        if any(c in ("LOW", "MEDIUM", "HIGH") for c in cells) or dc["id"] == vol_dc:
            rows.append((dc["id"] + (" *" if dc["id"] == vol_dc else ""), cells))
    rows.sort(key=lambda r: -sum(AVAIL_SCORE.get(c, 0) for c in r[1]))
    w = max([len(r[0]) for r in rows] + [11])
    head = "data center".ljust(w) + "  " + "  ".join(_short(g)[:12].ljust(12) for g in gpus)
    lines = [head, "-" * len(head)]
    lines += [n.ljust(w) + "  " + "  ".join(c.ljust(12) for c in cells) for n, cells in rows]
    lines.append("(* = where your volume is; stock per GPU type, not per count or CUDA version)")
    return "\n".join(lines)


class Follower:
    """Streams pod logs (SSE) and polls status until the run reports DONE or the pod stops."""

    def __init__(self, rp: Runpod, pod_id: str, deadline: float):
        self.rp, self.pod_id, self.deadline = rp, pod_id, deadline
        self.done = threading.Event()
        self.rc: int | None = None
        self.reason = ""
        self.last_id: str | None = None

    def _stream(self) -> None:
        backoff = 5
        while not self.done.is_set():
            try:
                resp = self.rp.logs(self.pod_id, self.last_id)
                backoff = 5
                for raw in resp:
                    if self.done.is_set():
                        return
                    line = raw.decode(errors="replace").rstrip("\n")
                    if line.startswith("id:"):
                        self.last_id = line[3:].strip()
                    elif line.startswith("data:"):
                        self._handle(line[5:].strip())
            except ApiError as e:
                if e.status == 404:
                    self._finish("pod is gone", None)
                    return
            except Exception:  # noqa: BLE001 — network blips: reconnect and resume via Last-Event-ID
                pass
            self.done.wait(backoff)
            backoff = min(backoff * 2, 60)

    def _handle(self, payload: str) -> None:
        try:
            ev = json.loads(payload)
            text = ev.get("line", "")
            src = ev.get("source", "")
        except ValueError:
            text, src = payload, ""
        if src == "system":
            self._system(text)
            return
        if text.startswith("WELDWORLD"):
            print(f"\033[1m{text}\033[0m", flush=True)
            m = re.search(r"COPILOT_URL=(\S+)", text)
            if m:
                log(f"Weld Copilot is live: {m.group(1)}")
            m = re.search(r"\bDONE rc=(\d+)", text)
            if m:
                self._finish("run finished", int(m.group(1)))
        else:
            print(f"  {text}", flush=True)

    _PULL = re.compile(r"^[0-9a-f]{12} (Extracting|Downloading|Waiting|Verifying Checksum|Download complete|"
                       r"Pulling fs layer|Already exists|Pull complete)")

    def _system(self, text: str) -> None:
        """Runpod system log: fold Docker's per-layer pull chatter into a progress line every 30 s."""
        m = self._PULL.match(text)
        if m:
            if m.group(1) == "Pull complete":
                self.layers_done = getattr(self, "layers_done", 0) + 1
            now = time.time()
            if now - getattr(self, "_last_pull_note", 0) > 30:
                self._last_pull_note = now
                print(f"\033[2m  [runpod] pulling container image... {getattr(self, 'layers_done', 0)} layers done\033[0m",
                      flush=True)
            return
        if text == getattr(self, "_last_system", None):
            return
        self._last_system = text
        print(f"\033[2m  [runpod] {text}\033[0m", flush=True)

    def _poll(self) -> None:
        while not self.done.wait(30):
            if time.time() > self.deadline:
                self._finish("MAX_HOURS deadline reached (launcher)", None)
                return
            try:
                st = self.rp.get_pod(self.pod_id).get("status")
            except ApiError as e:
                st = "TERMINATED" if e.status == 404 else None
            if st in ("EXITED", "ERROR", "TERMINATED"):
                self._finish(f"pod status {st}", None)

    def _finish(self, reason: str, rc: int | None) -> None:
        if not self.done.is_set():
            self.reason, self.rc = reason, rc
            self.done.set()

    def run(self) -> None:
        for fn in (self._stream, self._poll):
            threading.Thread(target=fn, daemon=True).start()
        while not self.done.wait(1):
            pass


def teardown(rp: Runpod, cfg: dict, pod_id: str | None, delete_volume: bool) -> None:
    if pod_id:
        log(f"terminating pod {pod_id}")
        try:
            rp.terminate_pod(pod_id)
        except ApiError as e:
            log(f"  terminate failed: {e} — check https://www.runpod.io/console/pods")
        save_state(pod_id=None)
    if delete_volume:
        vid = load_state().get("volume_id")
        if vid:
            log(f"deleting network volume {vid}")
            for _ in range(20):  # 409 while the terminated pod still holds the mount
                try:
                    rp.delete_volume(vid)
                    save_state(volume_id=None)
                    break
                except ApiError as e:
                    if e.status == 404:
                        save_state(volume_id=None)
                        break
                    if e.status != 409:
                        log(f"  delete failed: {e}")
                        break
                    time.sleep(15)
    else:
        log(f"network volume '{cfg['VOLUME_NAME']}' kept (models + results under {MOUNT}/results). "
            "Delete it with: python3 runpod/launch.py down --delete-volume")


def follow_and_teardown(rp: Runpod, cfg: dict, pod_id: str, detach: bool) -> int:
    if detach:
        log("detached: the pod terminates itself when the run ends or after MAX_HOURS.")
        return 0
    deadline = time.time() + float(cfg["MAX_HOURS"]) * 3600 + 900
    f = Follower(rp, pod_id, deadline)
    try:
        f.run()
    except KeyboardInterrupt:
        # Ctrl-C only stops *watching*: the pod keeps running and terminates itself when the run ends
        # (or after MAX_HOURS). Stopping it is an explicit `down`.
        log("Ctrl-C: stopped following the logs; the pod keeps running and terminates itself when the run ends.")
        log("  re-attach: python runpod/launch.py logs     stop it now: python runpod/launch.py down")
        return 130
    teardown(rp, cfg, pod_id, cfg.get("DELETE_VOLUME_AT_END") == "1")
    log(f"finished: {f.reason}" + (f" (rc={f.rc})" if f.rc is not None else ""))
    return f.rc if f.rc is not None else 1


# ============================================================================ commands
def cmd_run(rp: Runpod, cfg: dict, args) -> int:
    missing = [k for k in ("RUNPOD_API_KEY", "HF_TOKEN") if not cfg.get(k)]
    if missing and not args.dry_run:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")
    if not cfg.get("ANTHROPIC_API_KEY"):
        log("ANTHROPIC_API_KEY empty -> copilot chat uses Cosmos 3 Reasoner / scripted mode")
    if not cfg.get("GITHUB_TOKEN"):
        log("GITHUB_TOKEN empty -> the pod can only fetch the repo if it is public")
    st = load_state()
    if st.get("pod_id") and not args.force:
        try:
            status = rp.get_pod(st["pod_id"]).get("status")
        except ApiError:
            status = None
        if status and status != "TERMINATED":
            raise SystemExit(f"Pod {st['pod_id']} is still {status}. Use `logs` to follow it, `down` to stop it, "
                             "or --force to start another.")
    if args.dry_run:
        body = pod_body(cfg, gpu_types(cfg)[0], "<volume-id>", cfg.get("DATA_CENTER") or "<dc>")
        body["env"] = {k: ("***" if re.search(r"TOKEN|KEY", k) else v) for k, v in body["env"].items()}
        print(json.dumps(body, indent=2))
        return 0
    vol = ensure_volume(rp, cfg)
    save_state(volume_id=vol["id"], volume_name=cfg["VOLUME_NAME"], data_center=vol["dataCenter"])
    pod, vol = create_pod(rp, cfg, vol, move_volume=args.move_volume, wait_minutes=args.wait)
    used = sorted(set(load_state().get("used_volumes") or []) | {vol["id"]})
    save_state(pod_id=pod["id"], started=time.strftime("%Y-%m-%dT%H:%M:%S"), run_mode=cfg["RUN_MODE"],
               volume_id=vol["id"], data_center=vol["dataCenter"], used_volumes=used)
    log(f"console: https://www.runpod.io/console/pods  ·  copilot (serve mode): "
        f"https://{pod['id']}-{COPILOT_PORT}.proxy.runpod.net")
    log(f"mode={cfg['RUN_MODE']}  max={cfg['MAX_HOURS']}h  — Ctrl-C only stops watching; "
        "`python runpod/launch.py down` stops the pod")
    return follow_and_teardown(rp, cfg, pod["id"], args.detach)


def cmd_logs(rp: Runpod, cfg: dict, args) -> int:
    pid = load_state().get("pod_id")
    if not pid:
        raise SystemExit("No running pod recorded in .runpod_state.json")
    return follow_and_teardown(rp, cfg, pid, False)


def cmd_status(rp: Runpod, cfg: dict, args) -> int:
    st = load_state()
    print(json.dumps(st, indent=2) if st else "no state")
    if st.get("pod_id"):
        try:
            p = rp.get_pod(st["pod_id"])
            print(f"pod {p.get('id')}: {p.get('status')}  ${p.get('cost')}/h  dc={p.get('dataCenterId')}")
        except ApiError as e:
            print(f"pod {st['pod_id']}: {e}")
    for v in rp.list_volumes():
        if v.get("name") == cfg["VOLUME_NAME"]:
            print(f"volume {v['id']}: {v.get('size')} GB in {v.get('dataCenter')}")
    return 0


def cmd_availability(rp: Runpod, cfg: dict, args) -> int:
    print(availability_table(rp, cfg, load_state().get("data_center") or ""))
    return 0


def cmd_down(rp: Runpod, cfg: dict, args) -> int:
    teardown(rp, cfg, load_state().get("pod_id"), args.delete_volume)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "status", "logs", "down", "availability"])
    p.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    p.add_argument("--mode", choices=["job", "serve"], help="override RUN_MODE")
    p.add_argument("--detach", action="store_true", help="create the pod and exit (it still self-terminates)")
    p.add_argument("--delete-volume", action="store_true", help="down: also delete the network volume")
    p.add_argument("--force", action="store_true", help="run: start even if a previous pod is recorded")
    p.add_argument("--dry-run", action="store_true", help="run: print the pod request and exit")
    p.add_argument("--wait", type=float, default=0, metavar="MIN",
                   help="run: keep retrying for MIN minutes when no GPU is free")
    p.add_argument("--move-volume", action="store_true",
                   help="run: if the volume's data center is sold out, recreate the volume where GPUs are free "
                        "(its cached weights/envs are re-downloaded)")
    args = p.parse_args(argv)

    env_path = Path(args.env_file)
    if not env_path.exists() and args.command == "run" and not args.dry_run:
        raise SystemExit(f"{env_path} not found. Run: cp .env.example .env  and fill in your keys.")
    cfg = load_config(env_path)
    if args.mode:
        cfg["RUN_MODE"] = args.mode
    if not cfg.get("RUNPOD_API_KEY") and not args.dry_run:
        raise SystemExit("RUNPOD_API_KEY is empty (.env)")
    rp = Runpod(cfg.get("RUNPOD_API_KEY", ""))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # = detach, see above
    return {"run": cmd_run, "status": cmd_status, "logs": cmd_logs, "down": cmd_down,
            "availability": cmd_availability}[args.command](rp, cfg, args)


if __name__ == "__main__":
    sys.exit(main())
