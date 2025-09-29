# hume_ingest_clips.py (extended with summarization)

import os, re, json, time, mimetypes
from pathlib import Path
import requests
from dotenv import load_dotenv

# ---------------- Config ----------------
CLIPS_DIR = Path("clips")
OUT_JSON  = Path("hume_results.json")
SUMMARY_JSON = Path("hume_summary.json")

HUME_API_BASE = "https://api.hume.ai/v0"
START_JOB_URL = f"{HUME_API_BASE}/batch/jobs"
GET_JOB_URL   = lambda job_id: f"{HUME_API_BASE}/batch/jobs/{job_id}"
GET_PRED_URL  = lambda job_id: f"{HUME_API_BASE}/batch/jobs/{job_id}/predictions"

POLL_INTERVAL_SEC = 2
POLL_TIMEOUT_SEC  = 180
# ---------------------------------------

load_dotenv()
HUME_API_KEY = os.getenv("HUME_API_KEY")
if not HUME_API_KEY:
    raise SystemExit("❌ Set HUME_API_KEY in your .env")

HEADERS = {"X-Hume-Api-Key": HUME_API_KEY}

NAME_RE = re.compile(
    r"^(?P<speaker>.+)_(?P<ts>\d{8}-\d{6})_(?P<kind>audio|audio_silent|video)\.(?P<ext>wav|mp4)$"
)

def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or ("audio/wav" if path.suffix.lower()==".wav" else "video/mp4")

def start_hume_job(file_path: Path, models: dict) -> str:
    with open(file_path, "rb") as fh:
        files = {"file": (file_path.name, fh, guess_mime(file_path))}
        data  = {"json": json.dumps({"models": models})}
        resp = requests.post(START_JOB_URL, headers=HEADERS, files=files, data=data, timeout=60)
    resp.raise_for_status()
    job = resp.json()
    return job.get("job_id") or job.get("id") or job.get("jobId")

def wait_for_job(job_id: str, timeout_sec=POLL_TIMEOUT_SEC, interval_sec=POLL_INTERVAL_SEC) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        r = requests.get(GET_JOB_URL(job_id), headers=HEADERS, timeout=30)
        if r.status_code == 200:
            state = (r.json().get("state", {}).get("status") or "").upper()
            if state in {"COMPLETED", "FAILED"}:
                return state
        time.sleep(interval_sec)
    raise TimeoutError(f"Timed out waiting for job {job_id}")

def fetch_predictions(job_id: str) -> dict:
    r = requests.get(GET_PRED_URL(job_id), headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def build_clip_index(clips_dir: Path):
    index = {}
    for p in clips_dir.glob("*.*"):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        ts = d["ts"]
        speaker = d["speaker"]
        kind = d["kind"]
        node = index.setdefault(ts, {"speaker": speaker, "audio": None, "video": None})
        if kind == "audio_silent":
            continue
        if kind == "audio":
            node["audio"] = p
        elif kind == "video":
            node["video"] = p
    return index

def summarize_results(raw_results: dict) -> dict:
    """Turn raw Hume results into clean per-clip summaries."""
    summaries = {}
    for ts, entry in raw_results.items():
        speaker = entry.get("speaker")
        clip_summary = {"speaker": speaker, "audio": None, "video": None}

        # ---- Audio (prosody) ----
        audio = entry.get("audio", {})
        preds = audio.get("predictions", [])
        if preds:
            try:
                model = preds[0]["results"]["predictions"][0]["models"]["prosody"]
                # collect all text
                texts = []
                emo_scores = {}
                for gp in model.get("grouped_predictions", []):
                    for p in gp.get("predictions", []):
                        if "text" in p:
                            texts.append(p["text"])
                        emotions = p.get("emotions", [])
                        for emo in emotions:
                            name, score = emo["name"], emo["score"]
                            emo_scores.setdefault(name, []).append(score)
                # average emotions
                avg_scores = {k: sum(v)/len(v) for k,v in emo_scores.items()}
                top3 = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                clip_summary["audio"] = {
                    "text": " ".join(texts).strip(),
                    "top_emotions": top3
                }
            except Exception as e:
                clip_summary["audio"] = {"error": str(e)}

        # ---- Video (face) ----
        video = entry.get("video", {})
        preds = video.get("predictions", [])
        if preds:
            try:
                model = preds[0]["results"]["predictions"][0]["models"]["face"]
                emo_scores = {}
                for gp in model.get("grouped_predictions", []):
                    for p in gp.get("predictions", []):
                        emotions = p.get("emotions", [])
                        for emo in emotions:
                            name, score = emo["name"], emo["score"]
                            emo_scores.setdefault(name, []).append(score)
                avg_scores = {k: sum(v)/len(v) for k,v in emo_scores.items()}
                top3 = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                clip_summary["video"] = {
                    "top_emotions": top3
                }
            except Exception as e:
                clip_summary["video"] = {"error": str(e)}

        summaries[ts] = clip_summary
    return summaries
HISTORY_JSON = Path("hume_history.json")

def append_to_history(ts: str, clip_summary: dict):
    """Append one summarized clip entry to rolling history JSON file."""
    history = []
    if HISTORY_JSON.exists():
        try:
            history = json.load(open(HISTORY_JSON, "r"))
        except Exception:
            history = []
    history.append({ts: clip_summary})
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📝 Appended summary for {ts} → {HISTORY_JSON.resolve()}")

def main():
    if not CLIPS_DIR.exists():
        raise SystemExit(f"❌ Clips dir not found: {CLIPS_DIR.resolve()}")

    results = {}
    if OUT_JSON.exists():
        try:
            results = json.load(open(OUT_JSON, "r"))
        except Exception:
            results = {}

    index = build_clip_index(CLIPS_DIR)
    print(f"➡️ Found {len(index)} clip timestamps to process")

    for ts, bundle in sorted(index.items()):
        speaker = bundle["speaker"]
        audio_p = bundle["audio"]
        video_p = bundle["video"]

        results.setdefault(ts, {"speaker": speaker, "audio": None, "video": None})

        # --- Process Audio ---
        if audio_p and (not results[ts]["audio"]):
            print(f"🎙️  {ts} [{speaker}] → Prosody: {audio_p.name}")
            try:
                job_id = start_hume_job(audio_p, models={"prosody": {}})
                state = wait_for_job(job_id)
                if state == "COMPLETED":
                    results[ts]["audio"] = {"job_id": job_id, "predictions": fetch_predictions(job_id)}
                else:
                    results[ts]["audio"] = {"job_id": job_id, "error": f"State={state}"}
            except Exception as e:
                results[ts]["audio"] = {"error": str(e)}
            with open(OUT_JSON, "w") as f: json.dump(results, f, indent=2)

        # --- Process Video ---
        if video_p and (not results[ts]["video"]):
            print(f"🎥  {ts} [{speaker}] → Face: {video_p.name}")
            try:
                job_id = start_hume_job(video_p, models={"face": {}})
                state = wait_for_job(job_id)
                if state == "COMPLETED":
                    results[ts]["video"] = {"job_id": job_id, "predictions": fetch_predictions(job_id)}
                else:
                    results[ts]["video"] = {"job_id": job_id, "error": f"State={state}"}
            except Exception as e:
                results[ts]["video"] = {"error": str(e)}
            with open(OUT_JSON, "w") as f: json.dump(results, f, indent=2)

        # --- Summarize this clip immediately ---
        one_clip_summary = summarize_results({ts: results[ts]})[ts]
        append_to_history(ts, one_clip_summary)

    # also keep a global summary snapshot if needed
    summary = summarize_results(results)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📦 Saved full summary results → {SUMMARY_JSON.resolve()}")

if __name__ == "__main__":
    main()
