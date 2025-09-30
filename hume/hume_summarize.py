# hume_summarize.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------- I/O ----------

def load_hume_json_from_file(path: str) -> Dict[str, Any]:
    """
    Hume batch results we receive are typically a one-element list: [{source, results}]
    Return that single dict. Raise if shape is unexpected.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list) and raw:
        return raw[0]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unexpected Hume JSON shape in {path!r}")

# ---------- Core Helper ----------

def _get_grouped_predictions(hume_obj: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
    """
    Safely extract grouped_predictions for a given model ("prosody", "face").
    Returns [] if predictions are missing or empty.
    """
    try:
        preds = hume_obj.get("results", {}).get("predictions", [])
        if not isinstance(preds, list) or not preds:
            return []
        models = preds[0].get("models", {})
        return models.get(model, {}).get("grouped_predictions", []) or []
    except Exception:
        return []

# ---------- Helpers: Audio (Prosody) ----------

def _iter_prosody_text_segments(hume_obj: Dict[str, Any]):
    groups = _get_grouped_predictions(hume_obj, "prosody")
    for group in groups:
        for p in group.get("predictions", []) or []:
            yield {
                "text": p.get("text"),
                "begin": (p.get("time") or {}).get("begin"),
                "end": (p.get("time") or {}).get("end"),
                "confidence": p.get("confidence"),
            }

def extract_transcript_from_prosody(hume_obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a transcript by concatenating prosody text segments ordered by 'begin'.
    Also return the raw segments for downstream use.
    """
    segs = [s for s in _iter_prosody_text_segments(hume_obj) if isinstance(s.get("text"), str)]
    segs.sort(key=lambda s: (s.get("begin") if s.get("begin") is not None else 0.0))
    transcript = " ".join(s["text"].strip() for s in segs if s["text"].strip())
    return transcript, segs

def aggregate_emotions_from_prosody(hume_obj: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Average scores per emotion name over all prosody predictions.
    """
    groups = _get_grouped_predictions(hume_obj, "prosody")
    scores: Dict[str, List[float]] = defaultdict(list)
    for g in groups:
        for pred in g.get("predictions", []) or []:
            for emo in pred.get("emotions", []) or []:
                name = emo.get("name")
                score = emo.get("score")
                if isinstance(name, str) and isinstance(score, (int, float)):
                    scores[name].append(float(score))
    averaged = [(name, sum(vals) / len(vals)) for name, vals in scores.items() if vals]
    averaged.sort(key=lambda x: x[1], reverse=True)
    return [{"name": n, "score": round(s, 6)} for n, s in averaged[:top_k]]

# ---------- Helpers: Video (Face) ----------

def aggregate_emotions_from_face(hume_obj: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Average scores per emotion name over all face frames.
    """
    groups = _get_grouped_predictions(hume_obj, "face")
    scores: Dict[str, List[float]] = defaultdict(list)
    for g in groups:
        for pred in g.get("predictions", []) or []:
            for emo in pred.get("emotions", []) or []:
                name = emo.get("name")
                score = emo.get("score")
                if isinstance(name, str) and isinstance(score, (int, float)):
                    scores[name].append(float(score))
    averaged = [(name, sum(vals) / len(vals)) for name, vals in scores.items() if vals]
    averaged.sort(key=lambda x: x[1], reverse=True)
    return [{"name": n, "score": round(s, 6)} for n, s in averaged[:top_k]]

# ---------- Filename parsing ----------

def parse_participant_and_ts_from_filename(hume_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract "Participant_YYYYMMDD-HHMMSS_audio.wav" (or video.mp4) → (Participant, timestamp)
    """
    filename = (hume_obj.get("source") or {}).get("filename")
    if not isinstance(filename, str):
        return None, None
    m = re.match(r"(.+?)_(\d{8}-\d{6})_(?:audio|video)\.(?:wav|mp4)$", filename)
    if m:
        return m.group(1), m.group(2)
    return None, None

# ---------- Public entrypoint ----------

def summarize_hume_batch(
    audio_obj: Optional[Dict[str, Any]],
    video_obj: Optional[Dict[str, Any]],
    participant: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the final summary structure expected by your coach pipeline.
    """
    # Fill participant/timestamp from filenames if not given
    if participant is None and audio_obj:
        participant, _ = parse_participant_and_ts_from_filename(audio_obj)
    if timestamp is None and audio_obj:
        _, timestamp = parse_participant_and_ts_from_filename(audio_obj)
    if participant is None and video_obj:
        participant, _ = parse_participant_and_ts_from_filename(video_obj)
    if timestamp is None and video_obj:
        _, timestamp = parse_participant_and_ts_from_filename(video_obj)

    pkey = participant or "unknown"
    out: Dict[str, Any] = {pkey: {"audio": {}, "video": {}, "timestamp": timestamp or ""}}

    # Audio summary
    try:
        if audio_obj:
            transcript, segs = extract_transcript_from_prosody(audio_obj)
            top = aggregate_emotions_from_prosody(audio_obj)
            if not transcript and not top:
                out[pkey]["audio"] = {"status": "no_data"}
            else:
                out[pkey]["audio"] = {
                    "status": "ok",
                    "transcript": transcript,
                    "segments": segs,
                    "top_emotions": top,
                }
        else:
            out[pkey]["audio"] = {"status": "missing"}
    except Exception as e:
        out[pkey]["audio"] = {"status": "error", "error": str(e)}

    # Video summary
    try:
        if video_obj:
            topv = aggregate_emotions_from_face(video_obj)
            groups = _get_grouped_predictions(video_obj, "face")
            frame_count = sum(len(g.get("predictions", [])) for g in groups)
            if not topv and frame_count == 0:
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
