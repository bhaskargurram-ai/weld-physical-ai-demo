#!/usr/bin/env python3
"""WeldWorld end-to-end pipeline runner.

    collect -> ingest -> export -> curate -> warm-cache

Each stage is the package CLI owned by another part of the repo (see docs/PLAN.md); this script
only sequences them, wires paths/env, times them and prints a summary. Examples::

    python scripts/run_pipeline.py                         # real data (AIMEN SAW + BAM tandem GMAW), auto-detect mosaicod
    python scripts/run_pipeline.py --offline --replay      # no services, no GPU (CPU rehearsal)
    python scripts/run_pipeline.py --data-source sim       # the WeldCell simulator instead of real data
    BAM_WIDS=all AIMEN_COUPONS=PP5 python scripts/run_pipeline.py --only collect
    python scripts/run_pipeline.py --only curate --backend curator-next
    python scripts/run_pipeline.py --from export --dry-run

DATA_SOURCE=real|sim|all (default real, docs/REAL_DATA.md) picks the collect stage: ``real`` runs
``python -m collect.datasets.fetch`` (REAL_SOURCES, AIMEN_COUPONS, BAM_WIDS), ``sim`` the simulator,
``all`` both into the same DATA_DIR. Default DATA_DIR: data/real (real, all) or data (sim).

Environment: repo ``.env`` (copy of configs/demo.env.example) is loaded for variables that are not
already set. Flags win over both.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
STAGES = ["collect", "ingest", "export", "curate", "warm-cache"]

# --------------------------------------------------------------------------- console
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


BOLD, DIM, GREEN, YELLOW, RED, CYAN = (lambda s: _c("1", s)), (lambda s: _c("2", s)), \
    (lambda s: _c("32", s)), (lambda s: _c("33", s)), (lambda s: _c("31", s)), (lambda s: _c("36", s))


def banner(args: argparse.Namespace, env: dict) -> None:
    line = "─" * 78
    print(CYAN(line))
    print(BOLD("  WeldWorld pipeline") + DIM("  · physical AI for pipe-spool & pressure-vessel welding"))
    print(DIM(f"  source={args.data_source}  data={args.data_dir}  mosaico={'offline' if env.get('MOSAICO_OFFLINE') == '1' else env.get('MOSAICO_HOST', 'localhost') + ':' + env.get('MOSAICO_PORT', '6726')}"
              f"  curate={args.backend}  cosmos={env.get('COSMOS_BACKEND', 'auto')}"))
    print(CYAN(line))


def fmt_dt(s: float) -> str:
    return f"{s:5.1f}s" if s < 120 else f"{s / 60:4.1f}m"


# --------------------------------------------------------------------------- env
def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE, '#' comments (also inline after whitespace), no overwrite."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if val[:1] in ("'", '"') and val.count(val[0]) >= 2:
            val = val[1:val.index(val[0], 1)]
        else:
            for marker in (" #", "\t#"):
                if marker in val:
                    val = val[: val.index(marker)]
            val = val.strip()
        if val and not os.environ.get(key):
            os.environ[key] = val


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- CLI introspection
_HELP: dict[str, str] = {}


def module_present(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def module_help(py: str, mod: str) -> str:
    if mod not in _HELP:
        try:
            r = subprocess.run([py, "-m", mod, "--help"], cwd=REPO, capture_output=True, text=True, timeout=120)
            _HELP[mod] = r.stdout + r.stderr
        except (subprocess.TimeoutExpired, OSError):
            _HELP[mod] = ""
    return _HELP[mod]


def first_flag(py: str, mod: str, *candidates: str) -> Optional[str]:
    """Return the first of ``candidates`` that the module's --help mentions (CLIs evolve; be tolerant)."""
    h = module_help(py, mod)
    for c in candidates:
        if c in h:
            return c
    return None


def opt(py: str, mod: str, value, *flags: str) -> list[str]:
    f = first_flag(py, mod, *flags)
    return [f, str(value)] if f else []


# --------------------------------------------------------------------------- stages
@dataclass
class Stage:
    name: str
    title: str
    module: str                       # first importable candidate is used
    # (args, py, module) -> argv after -m module, or a list of (module, argv) sub-commands run in order
    build: Callable[[argparse.Namespace, str, str], list]
    count: Callable[[argparse.Namespace], str]                   # artifact summary after the stage
    alternatives: tuple[str, ...] = ()
    status: str = "pending"
    seconds: float = 0.0
    note: str = ""
    extra_env: dict = field(default_factory=dict)


def _n_dirs(p: Path) -> int:
    return sum(1 for x in p.iterdir() if x.is_dir() and (x / "meta.json").exists()) if p.is_dir() else 0


def _n_passes_by_source(p: Path) -> str:
    if not p.is_dir():
        return "0 passes"
    cnt: dict[str, int] = {}
    for x in p.iterdir():
        m = x / "meta.json"
        if not m.exists():
            continue
        try:
            meta = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        src = meta.get("source")
        sid = src.get("id") if isinstance(src, dict) else ("weldcell_sim" if meta.get("simulator") else "unknown")
        cnt[sid] = cnt.get(sid, 0) + 1
    return f"{sum(cnt.values())} passes" + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(cnt.items()))})" if cnt else "")


def _n_files(p: Path, pat: str) -> int:
    return sum(1 for _ in p.glob(pat)) if p.is_dir() else 0


def _manifest_stats(p: Path) -> str:
    if not p.is_file():
        return "no manifest"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    kept = sum(1 for r in rows if r.get("kept", True))
    img = sum(1 for r in rows if r.get("media_type") == "image")
    dfx = sum(1 for r in rows if r.get("defects") or r.get("rt_result") == "reject")
    return f"{len(rows)} items ({len(rows) - img} clips, {img} photos; {kept} kept, {dfx} with NDT/RT defect labels)"


def make_stages(a: argparse.Namespace) -> list[Stage]:
    d = Path(a.data_dir)
    gen, exp, cur = d / "generated", d / "export", d / "curated"

    def b_sim(a, py, m):
        return (["--out", str(gen)]
                + opt(py, m, a.n_passes, "--n-passes", "--passes", "--num-passes", "--n", "--count")
                + opt(py, m, a.seed, "--seed"))

    def b_real(a, py, m):
        argv = ["--sources", a.real_sources, "--data-dir", str(d)]
        if a.aimen_coupons:
            argv += opt(py, m, a.aimen_coupons, "--aimen-coupons", "--limit-coupons")
        if a.bam_wids:
            argv += opt(py, m, a.bam_wids, "--bam-wids")
        return argv

    def b_collect(a, py, m):
        cmds = []
        # simulator first: it clears existing pass dirs in its --out before writing (DATA_SOURCE=all shares one dir)
        if a.data_source in ("sim", "all"):
            cmds.append(("collect.simulate", b_sim(a, py, "collect.simulate")))
        if a.data_source in ("real", "all"):
            cmds.append(("collect.datasets.fetch", b_real(a, py, "collect.datasets.fetch")))
        return cmds

    def b_ingest(a, py, m):
        flag = first_flag(py, m, "--input", "--in", "--src")
        base = [flag, str(gen)] if flag else [str(gen)]
        return base + (["--no-replace"] if a.no_replace and first_flag(py, m, "--no-replace") else [])

    def b_export(a, py, m):
        return (["--out", str(exp)]
                + opt(py, m, a.clip_s, "--clip-s", "--clip-seconds")
                + opt(py, m, a.data_source, "--data-source")
                + (["--offline"] if os.environ.get("MOSAICO_OFFLINE") == "1" and first_flag(py, m, "--offline") else []))

    def b_curate(a, py, m):
        return ["--input", str(exp), "--out", str(cur), "--backend", a.backend]

    def b_warm(a, py, m):
        # worldmodels.cli warm-cache: Reasoner on kept clips + showcase what-if / sim2real generations
        return ["warm-cache", "--manifest", str(cur / "manifest.jsonl"), "--limit", str(a.warm_limit),
                "--photo-limit", str(a.warm_photo_limit), "--preset", a.warm_preset]

    return [
        Stage("collect", {"real": f"Fetch + convert real datasets ({a.real_sources}) -> generated/",
                          "sim": "WeldCell simulator -> generated/",
                          "all": f"Real datasets ({a.real_sources}) + WeldCell simulator -> generated/"}[a.data_source],
              "collect.datasets.fetch" if a.data_source != "sim" else "collect.simulate", b_collect,
              lambda a: _n_passes_by_source(gen)),
        Stage("ingest", "Ingest passes into Mosaico", "mosaico_io.ingest", b_ingest,
              lambda a: f"{_n_dirs(gen)} sequences"),
        Stage("export", "Query + export 5 s clips / photo windows with sensor sidecars", "mosaico_io.export", b_export,
              lambda a: f"{_n_files(exp, '*.json')} windows: {_n_files(exp, '*.mp4')} clips, "
                        f"{_n_files(exp, 'photos/*/*.jpg')} bead photos"),
        Stage("curate", "Curate (cut / caption / embed / filter)", "curate.run", b_curate,
              lambda a: _manifest_stats(cur / "manifest.jsonl")),
        Stage("warm-cache", "Pre-compute Cosmos 3 verdicts & showcase videos", "worldmodels.cli", b_warm,
              lambda a: f"{_n_files(d / 'cache' / 'worldmodels', '**/*.json')} cached calls"),
    ]


def _commands(st: Stage, a: argparse.Namespace, py: str) -> tuple[list[list[str]], Optional[str]]:
    """argv lists for a stage (multi-command stages: collect with DATA_SOURCE=all); error text if a module is missing."""
    mod = next((m for m in (st.module, *st.alternatives) if module_present(m)), None)
    if mod is None:
        return [], f"module {st.module} not found"
    built = st.build(a, py, mod)
    if built and isinstance(built[0], tuple):
        cmds = []
        for m, argv in built:
            if not module_present(m):
                return [], f"module {m} not found"
            cmds.append([py, "-m", m, *argv])
        return cmds, None
    return [[py, "-m", mod, *built]], None


def run_stage(i: int, n: int, st: Stage, a: argparse.Namespace, py: str) -> None:
    print()
    print(BOLD(f"[{i}/{n}] {st.name.upper():<10}") + " " + st.title)
    cmds, err = _commands(st, a, py)
    if err:
        st.status, st.note = "missing", err
        print("  " + YELLOW(f"skip: {st.note}"))
        return
    for argv in cmds:
        print(DIM("  $ " + " ".join(shlex.quote(x) for x in argv)))
    if a.dry_run:
        st.status = "dry-run"
        return
    t0 = time.monotonic()
    env = {**os.environ, **st.extra_env, "PYTHONUNBUFFERED": "1"}
    rc = 0
    for argv in cmds:
        proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(DIM("  │ ") + line.rstrip("\n"))
        rc = proc.wait()
        if rc != 0:
            break
    st.seconds = time.monotonic() - t0
    if rc == 0:
        st.status = "ok"
        try:
            st.note = st.count(a)
        except Exception as e:  # summary only, never fail the run on it
            st.note = f"(count failed: {e})"
        print("  " + GREEN(f"✓ {st.name} {fmt_dt(st.seconds)}") + DIM(f"  {st.note}"))
    else:
        st.status, st.note = "failed", f"exit {rc}"
        print("  " + RED(f"✗ {st.name} failed (exit {rc}) after {fmt_dt(st.seconds)}"))


def summary(stages: list[Stage], total: float) -> None:
    print()
    print(BOLD("  Summary"))
    icon = {"ok": GREEN("✓"), "failed": RED("✗"), "missing": YELLOW("–"), "skipped": DIM("·"),
            "dry-run": DIM("?"), "pending": DIM("·")}
    for st in stages:
        print(f"   {icon.get(st.status, '?')} {st.name:<11} {st.status:<8} {fmt_dt(st.seconds) if st.seconds else '     '}  {DIM(st.note)}")
    print(DIM(f"   total {fmt_dt(total)}"))


# --------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-source", default=os.environ.get("DATA_SOURCE", "real"), choices=["real", "sim", "all"],
                   help="real public datasets (default), the simulator, or both")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR"),
                   help="default: data/real for real|all, data for sim")
    p.add_argument("--real-sources", default=os.environ.get("REAL_SOURCES", "aimen_saw,bam_tgmaw"),
                   help="comma list for collect.datasets.fetch")
    p.add_argument("--aimen-coupons", default=os.environ.get("AIMEN_COUPONS", ""),
                   help="AIMEN coupons, e.g. PP5,PP6 (default: all PP3-PP7)")
    p.add_argument("--bam-wids", default=os.environ.get("BAM_WIDS", "617,630"),
                   help="BAM workpieces, e.g. 617,630 or all")
    p.add_argument("--n-passes", type=int, default=int(os.environ.get("N_PASSES", 24)))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SIM_SEED", 7)))
    p.add_argument("--clip-s", type=float, default=5.0, help="export clip length (CLIP_SECONDS)")
    p.add_argument("--backend", default=os.environ.get("CURATE_BACKEND", "lite"),
                   choices=["lite", "curator-next", "curator-legacy"], help="curation backend")
    p.add_argument("--offline", action="store_true", help="MOSAICO_OFFLINE=1: skip ingest, read data/generated")
    p.add_argument("--replay", action="store_true", help="COSMOS_BACKEND=replay: no GPU / network for world models")
    p.add_argument("--strict", action="store_true", help="fail instead of falling back to offline when mosaicod is down")
    p.add_argument("--no-replace", action="store_true", help="skip sequences that already exist in Mosaico")
    p.add_argument("--warm-limit", type=int, default=int(os.environ.get("WARM_LIMIT", 12)),
                   help="max clips to pre-assess in warm-cache")
    p.add_argument("--warm-photo-limit", type=int, default=None,
                   help="max photo items (bead photos) to pre-assess in warm-cache (default WARM_PHOTO_LIMIT or --warm-limit)")
    p.add_argument("--warm-preset", default=os.environ.get("WARM_PRESET", "live"), choices=["smoke", "live", "showcase"],
                   help="generation preset for warm-cache showcase videos")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--only", choices=STAGES, nargs="+", help="run only these stages")
    sel.add_argument("--from", dest="start", choices=STAGES, help="start at this stage")
    p.add_argument("--skip", choices=STAGES, nargs="+", default=[], help="skip these stages")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--python", default=sys.executable, help="interpreter for the stage CLIs")
    a = p.parse_args(argv)

    if a.warm_photo_limit is None:
        a.warm_photo_limit = int(os.environ.get("WARM_PHOTO_LIMIT") or a.warm_limit)
    if not a.data_dir:
        a.data_dir = "data" if a.data_source == "sim" else "data/real"
    a.data_dir = str(Path(a.data_dir) if Path(a.data_dir).is_absolute() else (REPO / a.data_dir))
    os.environ["DATA_DIR"] = a.data_dir
    os.environ["DATA_SOURCE"] = a.data_source
    os.environ["REAL_SOURCES"] = a.real_sources
    if a.aimen_coupons:
        os.environ["AIMEN_COUPONS"] = a.aimen_coupons
    if a.bam_wids:
        os.environ["BAM_WIDS"] = a.bam_wids
    if a.offline:
        os.environ["MOSAICO_OFFLINE"] = "1"
    if a.replay:
        os.environ["COSMOS_BACKEND"] = "replay"
    os.environ.setdefault("PYTHONPATH", str(REPO))

    stages = make_stages(a)
    wanted = set(a.only) if a.only else set(STAGES[STAGES.index(a.start):] if a.start else STAGES)
    wanted -= set(a.skip)

    # Mosaico availability: fall back to offline mode (still a complete demo) unless --strict.
    host, port = os.environ.get("MOSAICO_HOST", "localhost"), int(os.environ.get("MOSAICO_PORT", 6726))
    if os.environ.get("MOSAICO_OFFLINE") != "1" and ({"ingest", "export"} & wanted) and not a.dry_run:
        if not port_open(host, port):
            if a.strict:
                print(RED(f"mosaicod not reachable at {host}:{port} (start it: make infra-up / runpod/start_all.sh infra)"))
                return 2
            print(YELLOW(f"! mosaicod not reachable at {host}:{port} -> MOSAICO_OFFLINE=1 (reading data/generated directly)"))
            os.environ["MOSAICO_OFFLINE"] = "1"
    if os.environ.get("MOSAICO_OFFLINE") == "1" and "ingest" in wanted:
        wanted.discard("ingest")
        next(s for s in stages if s.name == "ingest").note = "offline mode"
    if a.backend.startswith("curator") and "curate" in wanted and not a.dry_run:
        os.environ.setdefault("S3_ENDPOINT", "http://localhost:9000")

    banner(a, dict(os.environ))
    t0 = time.monotonic()
    todo = [s for s in stages if s.name in wanted]
    for s in stages:
        if s.name not in wanted:
            s.status = "skipped"
    rc = 0
    for i, st in enumerate(todo, 1):
        run_stage(i, len(todo), st, a, a.python)
        if st.status == "failed":
            rc = 1
            if not a.keep_going:
                break
    summary(stages, time.monotonic() - t0)
    if rc == 0 and not a.dry_run:
        print()
        print("  Next: " + BOLD("make copilot") + DIM(f"  -> http://localhost:{os.environ.get('COPILOT_PORT', '7860')}"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
