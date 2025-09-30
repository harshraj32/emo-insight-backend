from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------- Core Helpers ----------

def get_model_predictions(hume_obj: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
    """
    Yield all prediction dicts for a given model ("prosody", "face").
    Handles multiple entries in results.predictions[] safely.
    """
    preds = hume_obj.get("results", {}).get("predictions", [])
    if not isinstance(preds, list):
        return []

    out = []
    for p in preds:
        model_data = p.get("models", {}).get(model, {})
        for group in model_data.get("grouped_predictions", []) or []:
            out.extend(group.get("predictions", []) or [])
    return out


def extract_errors(hume_obj: Dict[str, Any]) -> List[str]:
    """
    Extract error messages from the Hume result object.
    """
    errors = []
    for err in hume_obj.get("results", {}).get("errors", []) or []:
        msg = err.get("message")
        if msg:
            errors.append(msg)
    return errors


# ---------- Prosody (Voice) ----------

def extract_transcript(hume_obj: Dict[str, Any]) -> str:
    """
    Build transcript by concatenating prosody prediction texts in order.
    """
    segs = []
    for pred in get_model_predictions(hume_obj, "prosody"):
        txt = pred.get("text")
        begin = pred.get("time", {}).get("begin", 0)
        if isinstance(txt, str) and txt.strip():
            segs.append((begin, txt.strip()))

    segs.sort(key=lambda x: x[0])
    return " ".join(t for _, t in segs)


def aggregate_emotions(hume_obj: Dict[str, Any], model: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Average scores per emotion name over all predictions for a model.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    for pred in get_model_predictions(hume_obj, model):
        for emo in pred.get("emotions", []) or []:
            name, score = emo.get("name"), emo.get("score")
            if isinstance(name, str) and isinstance(score, (int, float)):
                scores[name].append(score)

    averaged = [(n, sum(v)/len(v)) for n, v in scores.items() if v]
    averaged.sort(key=lambda x: x[1], reverse=True)
    return [{"name": n, "score": round(s, 6)} for n, s in averaged[:top_k]]


# ---------- Filename parsing ----------

def parse_participant_and_ts_from_filename(hume_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract participant name + timestamp from filename if present.
    Example: "Alice_20250930-204036_audio.wav"
    """
    filename = (hume_obj.get("source") or {}).get("filename")
    if not isinstance(filename, str):
        return None, None

    m = re.match(r"(.+?)_(\d{8}-\d{6})_(?:audio|video)\.(?:wav|mp4)$", filename)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ---------- Public Entry ----------

def summarize_hume_batch(
    audio_obj: Optional[Dict[str, Any]],
    video_obj: Optional[Dict[str, Any]],
    participant: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the final summary structure expected by the coach pipeline.
    Includes:
      - transcript (from prosody)
      - top 3 emotions (voice + face)
      - frame count (video)
      - error handling
    """

    # Fill participant/timestamp from filenames if missing
    if participant is None and audio_obj:
        participant, _ = parse_participant_and_ts_from_filename(audio_obj)
    if timestamp is None and audio_obj:
        _, timestamp = parse_participant_and_ts_from_filename(audio_obj)
    if participant is None and video_obj:
        participant, _ = parse_participant_and_ts_from_filename(video_obj)
    if timestamp is None and video_obj:
        _, timestamp = parse_participant_and_ts_from_filename(video_obj)

    pkey = participant or "unknown"
    out: Dict[str, Any] = {
        pkey: {"audio": {}, "video": {}, "timestamp": timestamp or ""}
    }

    # ---------- Audio ----------
    try:
        if audio_obj:
            errs = extract_errors(audio_obj)
            transcript = extract_transcript(audio_obj)
            top = aggregate_emotions(audio_obj, "prosody", top_k=3)

            if errs:
                out[pkey]["audio"] = {"status": "error", "errors": errs}
            elif not transcript and not top:
                out[pkey]["audio"] = {"status": "no_data"}
            else:
                out[pkey]["audio"] = {
                    "status": "ok",
                    "transcript": transcript,
                    "top_emotions": top,
                }
        else:
            out[pkey]["audio"] = {"status": "missing"}
    except Exception as e:
        out[pkey]["audio"] = {"status": "error", "error": str(e)}

    # ---------- Video ----------
    try:
        if video_obj:
            errs = extract_errors(video_obj)
            topv = aggregate_emotions(video_obj, "face", top_k=3)
            preds = get_model_predictions(video_obj, "face")
            frame_count = len(preds)

            if errs:
                out[pkey]["video"] = {"status": "error", "errors": errs}
            elif not topv and frame_count == 0:
                out[pkey]["video"] = {"status": "no_data"}
            else:
                out[pkey]["video"] = {
                    "status": "ok",
                    "frame_count": frame_count,
                    "top_emotions": topv,
                }
        else:
            out[pkey]["video"] = {"status": "missing"}
    except Exception as e:
        out[pkey]["video"] = {"status": "error", "error": str(e)}

    return out
