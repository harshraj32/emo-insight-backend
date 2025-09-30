# hume/hume_summarize.py
from typing import Dict, Any, List

def summarize(results: dict) -> dict:
    """
    Summarize Hume API results:
    - Audio (prosody): transcript + top emotions
    - Video (face): top emotions
    """
    summary = {"audio": {}, "video": {}}

    try:
        # Normalize results
        if isinstance(results, list) and results:
            result = results[0]
        elif isinstance(results, dict):
            result = results
        else:
            return summary

        # Predictions live here
        if "results" in result:
            predictions = result["results"].get("predictions", [])
        elif "predictions" in result:
            predictions = result["predictions"]
        else:
            predictions = []

        if not predictions:
            return summary

        models = predictions[0].get("models", {})

        # ---------- AUDIO (Prosody) ----------
        if "prosody" in models:
            prosody = models["prosody"]
            grouped_preds = prosody.get("grouped_predictions", [])

            all_emotions: Dict[str, List[float]] = {}
            transcripts: List[str] = []

            for gp in grouped_preds:
                for pred in gp.get("predictions", []):
                    # Collect transcript text if present
                    text = pred.get("text")
                    if text:
                        transcripts.append(text.strip())

                    # Collect emotion scores
                    for emo in pred.get("emotions", []):
                        name = emo.get("name")
                        score = emo.get("score")
                        if name is not None and score is not None:
                            all_emotions.setdefault(name, []).append(score)

            if all_emotions:
                avg_emotions = {
                    name: sum(scores) / len(scores)
                    for name, scores in all_emotions.items()
                }
                top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                summary["audio"]["top_emotions"] = top3

            if transcripts:
                summary["audio"]["transcript"] = " ".join(transcripts)

        # ---------- VIDEO (Face) ----------
        if "face" in models:
            face = models["face"]
            grouped_preds = face.get("grouped_predictions", [])

            all_emotions: Dict[str, List[float]] = {}

            for gp in grouped_preds:
                for pred in gp.get("predictions", []):
                    for emo in pred.get("emotions", []):
                        name = emo.get("name")
                        score = emo.get("score")
                        if name is not None and score is not None:
                            all_emotions.setdefault(name, []).append(score)

            if all_emotions:
                avg_emotions = {
                    name: sum(scores) / len(scores)
                    for name, scores in all_emotions.items()
                }
                top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                summary["video"]["top_emotions"] = top3

    except Exception as e:
        summary["error"] = f"summarize failed: {e}"

    return summary
