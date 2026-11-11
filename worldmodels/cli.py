"""Command line for the Cosmos 3 clients.

    python -m worldmodels.cli assess clip.mp4 --context clip.json
    python -m worldmodels.cli explain --context clip.json
    python -m worldmodels.cli chat "Why did J07 fail RT?" --video clip.mp4
    python -m worldmodels.cli whatif --prompt "travel speed drops 20%" --image frame.png --preset live
    python -m worldmodels.cli transfer data/generated/SEQ/control/edge.mp4 --kind edge --prompt "photoreal FCAW fill"
    python -m worldmodels.cli transfer data/generated/SEQ/control/edge.mp4 --kind multi --control depth=.../depth.mp4 --prompt ...
    python -m worldmodels.cli anomaly frame.png --defect porosity
    python -m worldmodels.cli warm-cache --manifest data/curated/manifest.jsonl
    python -m worldmodels.cli health
    python -m worldmodels.cli cache-stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _load_context(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _apply_backend(args) -> None:
    if getattr(args, "backend", None):
        os.environ["COSMOS_BACKEND"] = args.backend
        os.environ.pop("COSMOS_REASONER_BACKEND", None)
        os.environ.pop("COSMOS_GENERATOR_BACKEND", None)


def cmd_assess(args) -> int:
    from . import reason

    ctx = _load_context(args.context)
    if not args.context:
        side = Path(args.video).with_suffix(".json")
        if side.exists():
            ctx = json.loads(side.read_text())
    res = reason.assess_clip(args.video, ctx)
    if not args.full:
        res = {k: v for k, v in res.items() if k != "raw_output"}
    _print(res)
    return 0


def cmd_explain(args) -> int:
    from . import reason

    _print(reason.explain_signals_meta(_load_context(args.context)))
    return 0


def cmd_chat(args) -> int:
    from . import reason

    _print(reason.chat_meta([{"role": "user", "content": args.question}], args.video, think=args.think))
    return 0


def cmd_whatif(args) -> int:
    from . import generate

    scene = {"process": args.process, "pass_type": args.pass_type, "defect": args.defect, "view": args.view}
    knobs = {"preset": args.preset, "resolution": args.resolution, "num_frames": args.frames, "steps": args.steps,
             "fps": args.fps, "scene": scene, "wrap": not args.raw_prompt}
    knobs = {k: v for k, v in knobs.items() if v is not None}
    _print(generate.what_if_meta(args.prompt, image_path=args.image, video_path=args.video,
                                 seconds=args.seconds, seed=args.seed, **knobs))
    return 0


def cmd_transfer(args) -> int:
    from . import transfer

    controls = {}
    if args.kind == "multi":
        controls["edge"] = args.control_video
        for c in args.control or []:
            k, _, p = c.partition("=")
            controls[k] = p
        if len(controls) == 1:  # auto-pick sibling depth.mp4 from the simulator's control/ dir
            sib = Path(args.control_video).with_name("depth.mp4")
            if sib.exists():
                controls["depth"] = str(sib)
    else:
        controls[args.kind] = args.control_video
    weights = None
    if args.weight:
        weights = {k: float(v) for k, _, v in (w.partition("=") for w in args.weight)}
    scene = {"process": args.process, "pass_type": args.pass_type, "defect": args.defect, "view": "torch"}
    knobs = {k: v for k, v in {"preset": args.preset, "scene": scene, "wrap": not args.raw_prompt}.items() if v is not None}
    _print(transfer.sim2real_meta(controls, args.prompt, args.seed, weights=weights, **knobs))
    return 0


def cmd_anomaly(args) -> int:
    from . import anomaly

    _print(anomaly.synth_defect_meta(args.image, args.defect, args.mask, args.seed))
    return 0


def cmd_warm(args) -> int:
    from .showcase import warm_cache

    rep = warm_cache(args.manifest, showcase=args.showcase, limit=args.limit, do_assess=not args.skip_assess,
                     do_explain=not args.skip_explain, do_generate=not args.skip_generate,
                     do_transfer=not args.skip_transfer, preset=args.preset, dry_run=args.dry_run)
    summary = {"assessed": len(rep["assess"]), "what_if": len(rep["what_if"]), "transfer": len(rep["transfer"]),
               "errors": rep["errors"], "elapsed_s": rep.get("elapsed_s"), "report": rep.get("report_path"),
               "sources": sorted({a["source"] for a in rep["assess"]} | {v["source"] for v in rep["what_if"].values()}
                                 | {v["source"] for v in rep["transfer"].values()})}
    _print(summary)
    if any(src not in ("cosmos3-local", "nim", "cache") for src in summary["sources"]):
        print("WARNING: some results are heuristic/placeholder - the GPU backends were not reachable.", file=sys.stderr)
    return 0


def cmd_health(args) -> int:
    from .serve.healthcheck import main as hc_main

    return hc_main(["--json"] if args.json else [])


def cmd_stats(args) -> int:
    from . import cache

    _print(cache.stats())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m worldmodels.cli", description="Cosmos 3 world-model clients (WeldWorld)")
    ap.add_argument("--backend", choices=["auto", "local", "nim", "replay"], help="override COSMOS_BACKEND")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("assess", help="weld QA verdict for one clip (Cosmos3 Reasoner)")
    p.add_argument("video")
    p.add_argument("--context", help="export sidecar / manifest row JSON (default: <video>.json if present)")
    p.add_argument("--full", action="store_true", help="include raw model output")
    p.set_defaults(fn=cmd_assess)

    p = sub.add_parser("explain", help="plain-language explanation of a clip's signals")
    p.add_argument("--context", required=True)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("chat", help="free question to the Reasoner")
    p.add_argument("question")
    p.add_argument("--video")
    p.add_argument("--think", action="store_true")
    p.set_defaults(fn=cmd_chat)

    def gen_common(q):
        q.add_argument("--prompt", required=True)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--preset", choices=["smoke", "live", "showcase"])
        q.add_argument("--process", default="FCAW")
        q.add_argument("--pass-type", default="fill")
        q.add_argument("--defect")
        q.add_argument("--raw-prompt", action="store_true", help="send the prompt as-is (no weld-cell JSON wrapper)")

    p = sub.add_parser("whatif", help="what-if video (Cosmos3 Generator)")
    gen_common(p)
    p.add_argument("--image")
    p.add_argument("--video")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--resolution", help="WxH or 256/480/720")
    p.add_argument("--frames", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--view", default="torch", choices=["torch", "cell"])
    p.set_defaults(fn=cmd_whatif)

    p = sub.add_parser("transfer", help="sim2real (Cosmos3 Transfer) from a control video")
    p.add_argument("control_video")
    p.add_argument("--kind", default="edge", choices=["edge", "depth", "blur", "seg", "multi"])
    p.add_argument("--control", action="append", help="extra control for --kind multi, e.g. depth=path.mp4")
    p.add_argument("--weight", action="append", help="per-control weight for multi, e.g. edge=0.5")
    gen_common(p)
    p.set_defaults(fn=cmd_transfer)

    p = sub.add_parser("anomaly", help="paint a weld defect into a still (Cosmos AnomalyGen, best effort)")
    p.add_argument("image")
    p.add_argument("--defect", required=True)
    p.add_argument("--mask")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_anomaly)

    p = sub.add_parser("warm-cache", help="run Reasoner on kept clips + showcase generations (populates replay)")
    p.add_argument("--manifest", default="data/curated/manifest.jsonl")
    p.add_argument("--showcase", help="showcase spec (default configs/cosmos/showcase.json)")
    p.add_argument("--limit", type=int)
    p.add_argument("--preset", default="showcase", choices=["smoke", "live", "showcase"])
    p.add_argument("--skip-assess", action="store_true")
    p.add_argument("--skip-explain", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_warm)

    p = sub.add_parser("health", help="status of every Cosmos endpoint")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_health)

    p = sub.add_parser("cache-stats", help="what is in the replay cache")
    p.set_defaults(fn=cmd_stats)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _apply_backend(args)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
