# hume/summarize.py

def summarize(results: dict) -> dict:
    summary = {"audio": {}, "video": {}}
    try:
        preds = results[0]["results"]["predictions"][0]["models"]
        if "prosody" in preds:
            emo = preds["prosody"]["grouped_predictions"][0]["predictions"][0]["emotions"]
            top3 = sorted([(e["name"], e["score"]) for e in emo], key=lambda x: -x[1])[:3]
            summary["audio"]["top_emotions"] = top3
        if "face" in preds:
            emo = preds["face"]["grouped_predictions"][0]["predictions"][0]["emotions"]
            top3 = sorted([(e["name"], e["score"]) for e in emo], key=lambda x: -x[1])[:3]
            summary["video"]["top_emotions"] = top3
    except Exception as e:
        summary["error"] = str(e)
    return summary
