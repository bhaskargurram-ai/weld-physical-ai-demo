#!/usr/bin/env python3
"""WeldWorld end-to-end pipeline runner.

    collect -> ingest -> export -> curate -> warm-cache

Each stage is the package CLI owned by another part of the repo (see docs/PLAN.md); this script
only sequences them, wires paths/env, times them and prints a summary. Examples::

    python scripts/run_pipeline.py                         # everything, auto-detect mosaicod
    python scripts/run_pipeline.py --offline --replay      # no services, no GPU (CPU rehearsal)
    python scripts/run_pipeline.py --only curate --backend curator-next
    python scripts/run_pipeline.py --from export --dry-run

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
    print(DIM(f"  data={args.data_dir}  mosaico={'offline' if env.get('MOSAICO_OFFLINE') == '1' else env.get('MOSAICO_HOST', 'localhost') + ':' + env.get('MOSAICO_PORT', '6726')}"
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
    build: Callable[[argparse.Namespace, str, str], list[str]]   # (args, py, module) -> argv after -m module
    count: Callable[[argparse.Namespace], str]                   # artifact summary after the stage
    alternatives: tuple[str, ...] = ()
    status: str = "pending"
    seconds: float = 0.0
    note: str = ""
    extra_env: dict = field(default_factory=dict)


def _n_dirs(p: Path) -> int:
    return sum(1 for x in p.iterdir() if x.is_dir()) if p.is_dir() else 0


def _n_files(p: Path, pat: str) -> int:
    return sum(1 for _ in p.glob(pat)) if p.is_dir() else 0


def _manifest_stats(p: Path) -> str:
    if not p.is_file():
        return "no manifest"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    kept = sum(1 for r in rows if r.get("kept", True))
    rej = sum(1 for r in rows if (r.get("rt_result") == "reject"))
    return f"{len(rows)} clips ({kept} kept, {rej} from RT-rejected welds)"


def make_stages(a: argparse.Namespace) -> list[Stage]:
    d = Path(a.data_dir)
    gen, exp, cur = d / "generated", d / "export", d / "curated"

    def b_collect(a, py, m):
        return (["--out", str(gen)]
                + opt(py, m, a.n_passes, "--n-passes", "--passes", "--num-passes", "--n", "--count")
                + opt(py, m, a.seed, "--seed"))

    def b_ingest(a, py, m):
        flag = first_flag(py, m, "--input", "--in", "--src")
        base = [flag, str(gen)] if flag else [str(gen)]
        return base + (["--replace"] if a.replace and first_flag(py, m, "--replace") else [])

    def b_export(a, py, m):
        return (["--out", str(exp)]
                + opt(py, m, a.clip_s, "--clip-s", "--clip-seconds")
                + (["--offline"] if os.environ.get("MOSAICO_OFFLINE") == "1" and first_flag(py, m, "--offline") else []))

    def b_curate(a, py, m):
        return ["--input", str(exp), "--out", str(cur), "--backend", a.backend]

    def b_warm(a, py, m):
        # worldmodels.cli warm-cache: Reasoner on kept clips + showcase what-if / sim2real generations
        return ["warm-cache", "--manifest", str(cur / "manifest.jsonl"), "--limit", str(a.warm_limit),
                "--preset", a.warm_preset]

    return [
        Stage("collect", "WeldCell simulator -> data/generated", "collect.simulate", b_collect,
              lambda a: f"{_n_dirs(gen)} passes"),
        Stage("ingest", "Ingest passes into Mosaico", "mosaico_io.ingest", b_ingest,
              lambda a: f"{_n_dirs(gen)} sequences"),
        Stage("export", "Query + export 5 s clips with sensor sidecars", "mosaico_io.export", b_export,
              lambda a: f"{_n_files(exp, '*.mp4')} clips"),
        Stage("curate", "Curate (cut / caption / embed / filter)", "curate.run", b_curate,
              lambda a: _manifest_stats(cur / "manifest.jsonl")),
        Stage("warm-cache", "Pre-compute Cosmos 3 verdicts & showcase videos", "worldmodels.cli", b_warm,
              lambda a: f"{_n_files(d / 'cache' / 'worldmodels', '**/*.json')} cached calls"),
    ]


def run_stage(i: int, n: int, st: Stage, a: argparse.Namespace, py: str) -> None:
    mod = next((m for m in (st.module, *st.alternatives) if module_present(m)), None)
    print()
    print(BOLD(f"[{i}/{n}] {st.name.upper():<10}") + " " + st.title)
    if mod is None:
        st.status, st.note = "missing", f"module {st.module} not found"
        print("  " + YELLOW(f"skip: {st.note}"))
        return
    argv = [py, "-m", mod, *st.build(a, py, mod)]
    print(DIM("  $ " + " ".join(shlex.quote(x) for x in argv)))
    if a.dry_run:
        st.status = "dry-run"
        return
    t0 = time.monotonic()
    env = {**os.environ, **st.extra_env, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(DIM("  │ ") + line.rstrip("\n"))
    rc = proc.wait()
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
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--n-passes", type=int, default=int(os.environ.get("N_PASSES", 24)))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SIM_SEED", 7)))
    p.add_argument("--clip-s", type=float, default=5.0, help="export clip length (CLIP_SECONDS)")
    p.add_argument("--backend", default=os.environ.get("CURATE_BACKEND", "lite"),
                   choices=["lite", "curator-next", "curator-legacy"], help="curation backend")
    p.add_argument("--offline", action="store_true", help="MOSAICO_OFFLINE=1: skip ingest, read data/generated")
    p.add_argument("--replay", action="store_true", help="COSMOS_BACKEND=replay: no GPU / network for world models")
    p.add_argument("--strict", action="store_true", help="fail instead of falling back to offline when mosaicod is down")
    p.add_argument("--replace", action="store_true", help="re-ingest sequences that already exist in Mosaico")
    p.add_argument("--warm-limit", type=int, default=int(os.environ.get("WARM_LIMIT", 12)),
                   help="max clips to pre-assess in warm-cache")
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

    a.data_dir = str(Path(a.data_dir) if Path(a.data_dir).is_absolute() else (REPO / a.data_dir))
    os.environ["DATA_DIR"] = a.data_dir
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
