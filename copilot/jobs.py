"""Background jobs for long-running Cosmos generations (what-if / sim2real), with progress for SSE."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import config


@dataclass
class Job:
    id: str
    kind: str
    label: str
    params: dict
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued"
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    version: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round(time.time() - self.created, 1)
        return d


class JobManager:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="copilot-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, label: str, params: dict, fn: Callable[..., dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, label=label, params=params)
        with self._lock:
            self._jobs[job.id] = job

        def progress(p: float, msg: str) -> None:
            with self._lock:
                job.progress = max(job.progress, min(float(p), 0.99))
                job.message = msg
                job.updated = time.time()
                job.version += 1

        def run() -> None:
            with self._lock:
                job.status, job.message = "running", "Starting"
                job.version += 1
            try:
                res = fn(progress=progress)
                with self._lock:
                    job.result, job.status, job.progress, job.message = res, "done", 1.0, "Done"
                    job.version += 1
                self._persist(job)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                with self._lock:
                    job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "Failed"
                    job.version += 1

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: -j.created)

    def _persist(self, job: Job) -> None:
        if not job.result:
            return
        rec = {"id": job.id, "kind": job.kind, "label": job.label, "created": job.created, **job.result}
        with (config.state_dir() / "gallery.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def gallery() -> list[dict[str, Any]]:
        p = config.state_dir() / "gallery.jsonl"
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return sorted(rows, key=lambda r: -r.get("created", 0))
