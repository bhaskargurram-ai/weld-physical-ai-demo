"""Print the status of every Cosmos 3 endpoint the demo can use.

    python -m worldmodels.serve.healthcheck            # table
    python -m worldmodels.serve.healthcheck --json
    python -m worldmodels.serve.healthcheck --deep     # also a 1-token reasoner chat
    python -m worldmodels.serve.healthcheck --nim-probe  # also POST {} to the hosted generator route

Exit code 0 if at least one usable path (local, nim or a non-empty replay cache) exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Any

import httpx

from .. import cache
from ..config import clear_probe_cache, get_settings, resolve_backend


def _get(url: str, headers: dict | None = None, timeout: float = 5.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = httpx.get(url, headers=headers or {}, timeout=timeout)
        out: dict[str, Any] = {"ok": r.status_code < 300, "status": r.status_code,
                               "ms": int((time.perf_counter() - t0) * 1000)}
        try:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                out["models"] = [m.get("id") for m in data["data"]][:10]
        except Exception:
            out["body"] = r.text[:120]
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def gpu_status() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        i, name, used, total, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "mem_used_mib": int(used), "mem_total_mib": int(total), "util_pct": int(util)})
    return gpus


def collect(deep: bool = False, nim_probe: bool = False) -> dict:
    s = get_settings()
    clear_probe_cache()
    rep: dict[str, Any] = {"settings": {"COSMOS_BACKEND": s.backend, "reasoner_url": s.reasoner_api,
                                        "generator_url": s.generator_api, "cache_dir": str(s.cache_dir),
                                        "nvidia_api_key": bool(s.nvidia_api_key)}}
    rep["reasoner_local"] = _get(f"{s.reasoner_api}/models")
    rep["generator_local"] = _get(f"{s.generator_api}/models")
    if deep and rep["reasoner_local"].get("ok"):
        try:
            t0 = time.perf_counter()
            model = (rep["reasoner_local"].get("models") or [s.reasoner_model])[0]
            r = httpx.post(f"{s.reasoner_api}/chat/completions", timeout=120,
                           json={"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Say OK."}]})
            rep["reasoner_local"]["chat"] = {"status": r.status_code, "ms": int((time.perf_counter() - t0) * 1000),
                                             "text": r.json()["choices"][0]["message"]["content"][:40] if r.status_code == 200 else r.text[:120]}
        except Exception as e:
            rep["reasoner_local"]["chat"] = {"error": str(e)[:200]}
    if s.nvidia_api_key:
        h = {"Authorization": f"Bearer {s.nvidia_api_key}"}
        rep["reasoner_nim"] = _get(f"{s.nim_reasoner_url.rstrip('/')}/models", h)
        if rep["reasoner_nim"].get("models") is not None:
            rep["reasoner_nim"]["has_cosmos3_reasoner"] = any(
                "cosmos3" in str(m) for m in rep["reasoner_nim"].get("models", []))
        if nim_probe:
            # schema-invalid {} -> 422 means the route exists; plain-text 404 means not provisioned
            try:
                r = httpx.post(s.nim_generator_url, json={}, headers=h, timeout=30)
                rep["generator_nim"] = {"status": r.status_code, "route_exists": r.status_code != 404,
                                        "body": r.text[:120]}
            except Exception as e:
                rep["generator_nim"] = {"ok": False, "error": str(e)[:200]}
    else:
        rep["reasoner_nim"] = {"ok": False, "error": "NVIDIA_API_KEY not set"}
    from ..anomaly import anomalygen_available
    from ..transfer import framework_available
    rep["framework"] = {"dir": s.framework_dir or None, "ok": framework_available(s)}
    rep["anomalygen"] = {"dir": s.anomalygen_dir or None, "ok": anomalygen_available(s)}
    rep["cache"] = cache.stats()
    rep["gpus"] = gpu_status()
    rep["resolved"] = {"reasoner": resolve_backend("reasoner", s), "generator": resolve_backend("generator", s)}
    return rep


def _mark(ok: Any) -> str:
    return "UP  " if ok else "down"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--nim-probe", action="store_true")
    a = ap.parse_args(argv)
    rep = collect(deep=a.deep, nim_probe=a.nim_probe)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(f"COSMOS_BACKEND={rep['settings']['COSMOS_BACKEND']}  -> reasoner: {rep['resolved']['reasoner']}, "
              f"generator: {rep['resolved']['generator']}")
        rl, gl = rep["reasoner_local"], rep["generator_local"]
        print(f"[{_mark(rl.get('ok'))}] Reasoner  (vLLM)      {rep['settings']['reasoner_url']}  "
              f"{rl.get('models') or rl.get('error') or rl.get('status')}  {rl.get('chat', '')}")
        print(f"[{_mark(gl.get('ok'))}] Generator (vLLM-Omni) {rep['settings']['generator_url']}  "
              f"{gl.get('models') or gl.get('error') or gl.get('status')}")
        rn = rep.get("reasoner_nim", {})
        print(f"[{_mark(rn.get('ok'))}] Reasoner  (NIM API)   {rn.get('error') or ('cosmos3 listed' if rn.get('has_cosmos3_reasoner') else rn.get('status'))}")
        if "generator_nim" in rep:
            g = rep["generator_nim"]
            print(f"[{_mark(g.get('route_exists'))}] Generator (NIM API)   HTTP {g.get('status')} {g.get('body', '')[:60]!r}")
        print(f"[{_mark(rep['framework']['ok'])}] cosmos-framework      {rep['framework']['dir']}")
        print(f"[{_mark(rep['anomalygen']['ok'])}] cdf-anomalygen        {rep['anomalygen']['dir']}")
        fns = rep["cache"].get("functions", {})
        print(f"[{_mark(bool(fns))}] Replay cache          {rep['cache']['root']}  "
              + ", ".join(f"{k}: {v['json'] or v['media']}" for k, v in fns.items()))
        for g in rep["gpus"]:
            print(f"       GPU{g['index']} {g['name']}: {g['mem_used_mib']}/{g['mem_total_mib']} MiB, {g['util_pct']}% util")
    usable = rep["reasoner_local"].get("ok") or rep["generator_local"].get("ok") or rep.get("reasoner_nim", {}).get("ok") \
        or bool(rep["cache"].get("functions"))
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
