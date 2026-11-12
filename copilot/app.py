"""Weld Copilot web app.

    uvicorn copilot.app:app --host 0.0.0.0 --port 7860
    COPILOT_FIXTURES=1 uvicorn copilot.app:app --port 7860     # self-contained mock data (no other modules)
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from . import config

# Windows takes MIME types from the registry, where .js is sometimes text/plain: browsers then refuse
# the UI scripts and the page stays empty. Pin the types the UI needs.
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".svg", "image/svg+xml"), (".mp4", "video/mp4"), (".json", "application/json")):
    mimetypes.add_type(_type, _ext)
from .agent import Session, make_agent
from .data import SIGNAL_INFO, SIGNALS, DataLayer, clean_json, mosaico_health
from .jobs import JobManager
from .models import CHANGE_TEMPLATES, WorldModels, build_prompt, cosmos_health, mode_label, parse_change
from .tools import TOOL_SPECS, Toolbox

logging.basicConfig(level=os.environ.get("COPILOT_LOG", "INFO"))
log = logging.getLogger("copilot.app")

DEMO_FLOW = [
    {"label": "Which welds this week broke the WPS?", "hint": "Mosaico query"},
    {"label": "Why did SP-1043 J07 fill pass fail RT?", "hint": "WPS + Cosmos Reasoner"},
    {"label": "Show me the moment porosity started", "hint": "Replay jump"},
    {"label": "What if travel speed dropped 20% on that pass?", "hint": "Cosmos Generator"},
    {"label": "Make photoreal training video from the simulator", "hint": "Cosmos Transfer"},
    {"label": "How many curated clips do we have for porosity?", "hint": "Curator dataset"},
]
# Real data (AIMEN SAW + BAM tandem-GMAW). Each chip names passes from the default conversion (PP3-PP7, WID617/630).
DEMO_FLOW_REAL = [
    {"label": "Which SA-516 vessel passes had NDT indications?", "hint": "RT/UT ground truth"},
    {"label": "Show porosity on PP5 bead 2 — did the arc signals change?", "hint": "Replay + honest signal check"},
    {"label": "Compare lead vs trail current on WID617 layer 3", "hint": "Tandem signals"},
    {"label": "Which passes ran outside their published settings?", "hint": "Mosaico query"},
    {"label": "Inspect the bead photo of PP5 bead 2", "hint": "Cosmos 3 Reasoner · photo"},
    {"label": "What if travel speed dropped 20 % on WID630 layer 4?", "hint": "Cosmos 3 Generator"},
]


def create_app() -> FastAPI:
    if config.fixtures_mode() == "real":
        from .fixtures_real import ensure_fixtures as ensure_real
        os.environ.setdefault("DATA_DIR", str(config.FIXTURE_REAL_DIR))
        os.environ.setdefault("MOSAICO_OFFLINE", "1")
        ensure_real(config.data_dir(), videos=os.environ.get("COPILOT_FIXTURE_VIDEO", "1") == "1")
    elif config.fixtures_enabled():
        from .fixtures import ensure_fixtures
        os.environ.setdefault("DATA_DIR", str(config.FIXTURE_DIR))   # so worldmodels/mosaico_io agree on paths
        os.environ.setdefault("MOSAICO_OFFLINE", "1")
        ensure_fixtures(config.data_dir(), videos=os.environ.get("COPILOT_FIXTURE_VIDEO", "1") == "1")
    elif config.data_dir() == config.REAL_DIR.resolve():
        os.environ.setdefault("DATA_DIR", str(config.REAL_DIR))      # real data: mosaico_io/worldmodels read the same dir

    app = FastAPI(title="Weld Copilot", version="1.0")
    data = DataLayer()
    jobs = JobManager()
    wm = WorldModels(data)
    tb = Toolbox(data, wm, jobs)
    sessions: dict[str, Session] = {}
    app.state.data, app.state.jobs, app.state.wm, app.state.toolbox = data, jobs, wm, tb
    app.state.agent = make_agent(tb)
    app.state.sessions = sessions

    def session(sid: str | None) -> Session:
        if sid and sid in sessions:
            return sessions[sid]
        s = Session(id=sid) if sid else Session()
        sessions[s.id] = s
        return s

    # -------------------------------------------------------------- pages & files
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

    @app.api_route("/files/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def files(path: str):
        if path.startswith("abs/"):
            p = Path("/" + path[4:]).resolve()
        else:
            p = (config.data_dir() / path).resolve()
        roots = [config.data_dir().resolve(), config.REPO_ROOT.resolve()]
        if not any(p.is_relative_to(r) for r in roots) or not p.is_file():
            raise HTTPException(404, "not found")
        if p.suffix.lower() not in (".mp4", ".webm", ".png", ".jpg", ".jpeg", ".json", ".md", ".gif", ".pdf"):
            raise HTTPException(403, "file type not served")
        return FileResponse(p)

    # -------------------------------------------------------------- "How it works" tab
    @app.get("/api/flow")
    def flow():
        from .flow import build_flow
        return build_flow(data, app.state.agent, cosmos_health(), mosaico_health())

    # -------------------------------------------------------------- health / pipeline strip
    @app.get("/api/health")
    def health():
        passes = data.all_passes()
        man = data.manifest()
        cos = cosmos_health()
        mos = mosaico_health()
        agent = app.state.agent
        gen_dir = config.generated_dir()
        dm = data.data_mode()
        return {
            "time": time.time(),
            "mode": cos.get("mode_label") or mode_label(),
            "llm": {"backend": agent.name, "label": agent.label},
            "data_dir": str(config.data_dir()),
            "fixtures": config.fixtures_enabled(),
            "data": dm,
            "stages": [
                {"id": "collect", "name": "Real data" if dm["real"] else "Collect", "status": "ok" if passes else "empty",
                 "detail": f"{len(passes)} passes" + (f" · {' + '.join(dm['short'])}" if dm["real"] else "")
                           + (" (fixtures)" if config.fixtures_enabled() else ""),
                 "metric": len(passes),
                 "sub": f"{sum(1 for d in gen_dir.iterdir() if d.is_dir()) if gen_dir.exists() else 0} pass dirs"},
                {"id": "mosaico", "name": "Mosaico", "status": mos["status"], "detail": mos["short"],
                 "sub": f"source: {data.last_source}"},
                {"id": "curator", "name": "Curator", "status": "ok" if man else "empty",
                 "detail": f"{sum(1 for r in man if r.get('kept'))}/{len(man)} clips kept" if man else "no manifest yet",
                 "metric": len(man), "sub": "Cosmos Curator"},
                {"id": "cosmos", "name": "Cosmos 3", "status": "ok" if cos.get("reasoner") and cos.get("generator")
                 else "warn",
                 "detail": (f"{cos.get('mode_label')} · reasoner + generator ready" if cos.get("reasoner") and cos.get("generator")
                            else f"{cos.get('mode_label')} · R {'✓' if cos.get('reasoner') else 'fallback'} · "
                                 f"G {'✓' if cos.get('generator') else 'fallback'}"),
                 "sub": "worldmodels" if cos.get("worldmodels") else "mock (worldmodels not installed)",
                 "raw": {k: v for k, v in cos.items() if k != "raw"}},
                {"id": "copilot", "name": "Copilot", "status": "ok", "detail": agent.label,
                 "sub": f"{len([j for j in jobs.list() if j.status == 'running'])} jobs running"},
            ],
        }

    # -------------------------------------------------------------- passes
    @app.get("/api/passes")
    def passes(process: str | None = None, spool_id: str | None = None, has_defect: bool | None = None,
               rt_result: str | None = None, source: str | None = None, defect: str | None = None):
        rows = data.list_passes(process=process, spool_id=spool_id, has_defect=has_defect, rt_result=rt_result,
                                source=source, defect=defect, with_violations=True)
        return clean_json({"passes": rows, "source": data.last_source, "data": data.data_mode()})

    @app.get("/api/passes/{seq}")
    def pass_detail(seq: str):
        meta = data.get_meta(seq)
        if not meta:
            raise HTTPException(404, f"unknown pass {seq}")
        row = data.get_pass(seq) or {}
        s2r = [g for g in jobs.gallery() if g.get("kind") == "sim2real" and g.get("sequence_name") == seq]
        return clean_json({"sequence_name": seq, "pass": row, "meta": meta, "wps": data.wps_limits(seq),
                           "violations": data.wps_violations(seq), "events": data.events(seq),
                           "defect_windows": data.defect_windows(seq), "media": data.media(seq),
                           "source": data.source(seq), "ndt": data.ndt(seq), "wps_basis": meta.get("wps_basis"),
                           "signal_notes": data.signal_notes(seq), "sim2real": s2r[0] if s2r else None})

    @app.get("/api/passes/{seq}/signals")
    def signals(seq: str, hz: float = Query(25.0, gt=0, le=100), t0: float | None = None, t1: float | None = None):
        try:
            df = data.get_signals(seq, t0, t1)
        except FileNotFoundError:
            raise HTTPException(404, f"no signals for {seq}")
        src_hz = 1.0 / max(float(df["t_s"].diff().median() or 0.01), 1e-3)
        step = max(1, int(round(src_hz / hz)))
        d = df.iloc[::step]
        series = {p: [None if v != v else round(float(v), 4) for v in d[p].values] for p in SIGNALS
                  if p in d.columns and d[p].notna().any()}
        meta = data.get_meta(seq)
        return clean_json({"sequence_name": seq, "t": [round(float(x), 3) for x in d["t_s"].values], "series": series,
                           "info": SIGNAL_INFO, "wps": data.wps_limits(seq), "wps_basis": meta.get("wps_basis"),
                           "notes": data.signal_notes(seq), "electrodes": int(meta.get("electrodes") or 1)})

    @app.get("/api/sources")
    def sources():
        return data.sources_summary()

    # -------------------------------------------------------------- dataset
    @app.get("/api/dataset")
    def dataset(defect: str | None = None, kept: bool | None = None, q: str | None = None, limit: int = 400):
        rows = data.manifest()
        st = data.dataset_stats()
        out = []
        for r in rows:
            ds = [d.get("defect") if isinstance(d, dict) else d for d in (r.get("defects") or [])]
            if defect and defect not in ds and not (defect == "none" and not ds):
                continue
            if kept is not None and bool(r.get("kept")) != kept:
                continue
            if q and q.lower() not in json.dumps(r).lower():
                continue
            out.append({**r, "video_url": data.clip_video(r), "defects": ds})
        return {"stats": st, "rows": out[:limit], "count": len(out), "card": data.dataset_card()}

    # -------------------------------------------------------------- generate studio
    @app.get("/api/templates")
    def templates():
        return {"templates": [{"key": k, "label": v["label"], "text": v["text"]} for k, v in CHANGE_TEMPLATES.items()]}

    @app.post("/api/prompt")
    def prompt(body: dict = Body(...)):
        seq = body.get("sequence")
        meta = data.get_meta(seq) if seq else {}
        key, text = parse_change(body.get("change", ""))
        if body.get("kind") == "sim2real":
            text = (f"Photorealistic close-up of a robotic {meta.get('process', 'FCAW')} weld on "
                    f"{meta.get('material', 'A106 Gr.B')} pipe: bright arc, molten orange pool, realistic spatter, "
                    "metallic bead with temper colours.")
        return {"prompt": json.dumps(json.loads(build_prompt(meta, text)), indent=2), "template": key}

    @app.post("/api/generate")
    def generate(body: dict = Body(...)):
        kind = body.get("kind", "what_if")
        args: dict[str, Any] = {k: body[k] for k in ("sequence", "t", "clip_id") if body.get(k) not in (None, "")}
        seed = int(body.get("seed") or 0)
        prompt_override = body.get("prompt") or None
        try:
            seq = tb.resolve_sequence(args.get("sequence")) if args.get("sequence") else None
        except LookupError as e:
            raise HTTPException(400, str(e))
        if kind == "sim2real":
            if not seq:
                raise HTTPException(400, "sequence required")
            control = body.get("control", "edge")
            job = jobs.submit("sim2real", f"Sim→Real ({control}) · {seq}", {**args, "kind": control},
                              lambda progress: wm.sim2real(seq, kind=control, t=args.get("t"), seed=seed,
                                                           prompt=prompt_override, progress=progress))
        else:
            change = body.get("change") or "travel_speed_-20"
            key, _ = parse_change(change)
            label = CHANGE_TEMPLATES.get(key or "", {}).get("label") or change
            job = jobs.submit("what_if", f"What-if: {label} · {seq or args.get('clip_id')}", {**args, "change": change},
                              lambda progress: wm.what_if(seq, change, clip_id=args.get("clip_id"), t=args.get("t"),
                                                          seed=seed, prompt=prompt_override, progress=progress))
        return job.public()

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.public() for j in jobs.list()]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "no such job")
        return j.public()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "no such job")

        async def gen():
            last = -1
            while True:
                if j.version != last:
                    last = j.version
                    yield {"event": "progress", "data": json.dumps(j.public())}
                if j.status in ("done", "error"):
                    return
                await asyncio.sleep(0.25)
        return EventSourceResponse(gen())

    @app.get("/api/gallery")
    def gallery():
        return {"items": jobs.gallery()}

    # -------------------------------------------------------------- tools & chat
    @app.get("/api/tools")
    def tools():
        return {"tools": TOOL_SPECS}

    @app.post("/api/tools/{name}")
    def call_tool(name: str, body: dict = Body(default={})):
        s = session(body.pop("session_id", None))
        out = tb.call(name, body, s.ctx)
        return JSONResponse(out, status_code=400 if out.get("is_error") else 200)

    @app.get("/api/suggestions")
    def suggestions():
        dm = data.data_mode()
        return {"demo": DEMO_FLOW_REAL if dm["real"] else DEMO_FLOW, "data": dm}

    @app.post("/api/chat")
    async def chat(body: dict = Body(...)):
        text = (body.get("message") or "").strip()
        if not text:
            raise HTTPException(400, "empty message")
        s = session(body.get("session_id"))
        agent = app.state.agent

        async def gen():
            yield {"event": "session", "data": json.dumps({"session_id": s.id, "backend": agent.name,
                                                           "label": agent.label})}
            try:
                async for ev in agent.run(s, text):
                    yield {"event": ev.get("type", "message"), "data": json.dumps(ev, default=str)}
            except Exception as e:  # noqa: BLE001
                log.exception("chat failed")
                yield {"event": "error", "data": json.dumps({"type": "error", "message": str(e)})}
        return EventSourceResponse(gen(), ping=10)

    @app.post("/api/chat/reset")
    def chat_reset(body: dict = Body(default={})):
        sessions.pop(body.get("session_id", ""), None)
        return {"ok": True}

    @app.post("/api/refresh")
    def refresh():
        data.refresh()
        return {"ok": True}

    return app


app = create_app()
