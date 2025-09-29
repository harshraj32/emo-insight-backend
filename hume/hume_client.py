import requests, time, os, json
from pathlib import Path
from config import settings

API = "https://api.hume.ai/v0"
HEADERS = {"X-Hume-Api-Key": settings.HUME_API_KEY}

def start_job(file: Path, models: dict) -> str:
    with open(file, "rb") as f:
        resp = requests.post(f"{API}/batch/jobs",
                             headers=HEADERS,
                             files={"file": (file.name, f, "video/mp4")},
                             data={"json": json.dumps({"models": models})})
    resp.raise_for_status()
    return resp.json()["job_id"]

def wait_job(job_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        r = requests.get(f"{API}/batch/jobs/{job_id}", headers=HEADERS)
        r.raise_for_status()
        st = r.json()["state"]["status"]
        if st in ("COMPLETED", "FAILED"):
            return st
        time.sleep(2)
    raise TimeoutError("Job timeout")

def get_results(job_id):
    r = requests.get(f"{API}/batch/jobs/{job_id}/predictions", headers=HEADERS)
    r.raise_for_status()
    return r.json()

def process_clip(file: Path, models={"prosody": {}, "face": {}}):
    job_id = start_job(file, models)
    state = wait_job(job_id)
    if state == "COMPLETED":
        results = get_results(job_id)
        os.remove(file)  # DELETE CLIP AFTER SUCCESS
        return results
    else:
        return {"error": f"Job {job_id} failed: {state}"}
