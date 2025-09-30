# hume/hume_client.py
from pathlib import Path
import time
import json
import requests
from typing import Dict, Any, Union
from config import settings
import os

API = "https://api.hume.ai/v0"
HEADERS = {"X-Hume-Api-Key": settings.HUME_API_KEY}

import mimetypes

def _guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or ("audio/wav" if path.suffix.lower()==".wav" else "video/mp4")

def start_job(file: Union[str, Path], models: Dict[str, Any]) -> str:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(file)

    with file.open("rb") as fh:
        files = {"file": (file.name, fh, _guess_mime(file))}
        data  = {"json": json.dumps({"models": models})}
        r = requests.post(f"{API}/batch/jobs", files=files, data=data, headers=HEADERS, timeout=max(60, settings.HTTP_TIMEOUT))
    r.raise_for_status()
    job = r.json()
    job_id = job.get("job_id") or job.get("id") or job.get("jobId")
    if not job_id:
        raise RuntimeError(f"Hume start_job missing job_id: {job}")
    
    print(f"📤 Sending to Hume: {file} ({os.path.getsize(file)} bytes)")
    print(f"Headers: {HEADERS}")
    return job_id


def wait_job(job_id: str, poll_s: float = 2.0, timeout_s: float = None) -> str:
    timeout_s = timeout_s or settings.HUME_JOB_TIMEOUT
    t0 = time.time()
    while True:
        r = requests.get(f"{API}/batch/jobs/{job_id}", headers=HEADERS, timeout=settings.HTTP_TIMEOUT)
        r.raise_for_status()
        state = r.json().get("state", "")
        if state in {"COMPLETED", "FAILED"}:
            return state
        if time.time() - t0 > timeout_s:
            return "FAILED"
        time.sleep(poll_s)


def get_results(job_id: str) -> Dict[str, Any]:
    r = requests.get(f"{API}/batch/jobs/{job_id}/predictions", headers=HEADERS, timeout=max(60, settings.HTTP_TIMEOUT))
    r.raise_for_status()
    return r.json()


def process_clip(file: Union[str, Path], models: Dict[str, Any] = None) -> Dict[str, Any]:
    models = models or settings.HUME_MODELS
    job_id = start_job(file, models)
    state = wait_job(job_id)
    if state != "COMPLETED":
        raise RuntimeError(f"Hume job failed: {job_id}")
    return get_results(job_id)
