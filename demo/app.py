"""Live-demo API: upload a video, run the same pipeline as the submission, fetch the results.

    uvicorn demo.app:app --host 0.0.0.0 --port 8000

Endpoints (all under /api):
    GET  /api/health                 -> {"status": "ok"}
    GET  /api/limits                 -> accepted upload size / duration
    POST /api/jobs      (multipart)  -> {"job_id": ...}
    GET  /api/jobs/{id}              -> {"status", "stage", "progress", "error"}
    GET  /api/jobs/{id}/result       -> {"video", "events", "risk"}

Jobs run one at a time (the demo server is CPU-only), so a second upload waits in the queue.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solution  # noqa: E402
from src.video import probe  # noqa: E402

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/demo-jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
MAX_DURATION_SEC = float(os.environ.get("MAX_DURATION_SEC", "150"))
CORS_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o]


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "waiting in queue"
    progress: float = 0.0
    error: str | None = None
    result: dict | None = field(default=None, repr=False)


app = FastAPI(title="Traffic event detection demo")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)


def _set(job: Job, **kw) -> None:
    with _lock:
        for k, v in kw.items():
            setattr(job, k, v)


def _run(job: Job, path: Path) -> None:
    try:
        info = probe(str(path))
        _set(job, status="running", stage="detecting events", progress=0.05)
        events = solution.detect_events(str(path))
        _set(job, stage="computing accident risk", progress=0.6)
        risk = _risk_curve(path, info, lambda p: _set(job, progress=0.6 + 0.4 * p))
        result = {
            "video": {"name": path.name, "duration": info.duration, "fps": info.fps,
                      "width": info.width, "height": info.height},
            "events": sorted(events),
            "risk": risk,
        }
        _set(job, status="done", stage="done", progress=1.0, result=result)
    except Exception as exc:  # the demo must never crash the server
        traceback.print_exc()
        _set(job, status="error", stage="failed", error=str(exc))
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def _risk_curve(path: Path, info, on_progress) -> list[list[float]]:
    """Stream frames through RiskEstimator exactly like run_submission.py, keeping ~5 points per second."""
    import cv2

    est = solution.RiskEstimator()
    est.reset({"video_id": path.name, "fps": info.fps, "width": info.width, "height": info.height,
               "n_frames": info.n_frames})
    keep_every = max(1, round(info.fps / 5))
    cap = cv2.VideoCapture(str(path))
    curve, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = float(est.step(frame, idx / info.fps))
        if idx % keep_every == 0:
            curve.append([round(idx / info.fps, 2), round(min(1.0, max(0.0, score)), 4)])
        if idx % 100 == 0 and info.n_frames:
            on_progress(idx / info.n_frames)
        idx += 1
    cap.release()
    return curve


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/limits")
def limits() -> dict:
    return {"max_upload_mb": MAX_UPLOAD_MB, "max_duration_sec": MAX_DURATION_SEC, "formats": [".mp4"]}


@app.post("/api/jobs")
async def create_job(file: UploadFile) -> dict:
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(400, "Only .mp4 files are accepted.")
    job = Job(id=uuid.uuid4().hex[:12])
    folder = JOBS_DIR / job.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "upload.mp4"
    size = 0
    with open(path, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            out.write(chunk)
    try:
        duration = probe(str(path)).duration
    except RuntimeError:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "Could not read the video.")
    if duration > MAX_DURATION_SEC:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"Video is {duration:.0f} s long; the limit is {MAX_DURATION_SEC:.0f} s.")
    with _lock:
        _jobs[job.id] = job
    _worker.submit(_run, job, path)
    return {"job_id": job.id}


def _get(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get(job_id)
    with _lock:
        return {k: v for k, v in asdict(job).items() if k != "result"}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    job = _get(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is {job.status}.")
    return job.result
