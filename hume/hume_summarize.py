# hume/hume_summarize.py

def summarize(results: dict) -> dict:
    """
    Summarize Hume API results to extract top emotions and transcript text
    from audio (prosody) and top emotions from video (face).
    """
    summary = {"audio": {}, "video": {}}

    try:
        # Normalize results structure
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
        elif isinstance(results, dict):
            result = results
        else:
            raise ValueError(f"Unexpected results format: {type(results)}")

        # Find predictions
        if "results" in result:
            predictions = result["results"].get("predictions", [])
        elif "predictions" in result:
            predictions = result["predictions"]
        else:
            predictions = []

        if not predictions:
            return summary

        models = predictions[0].get("models", {})

        # ---- Prosody (Audio) ----
        if "prosody" in models:
            prosody = models["prosody"]
            grouped_preds = prosody.get("grouped_predictions", [])

            if grouped_preds:
                all_emotions = {}
                transcripts = []

                for gp in grouped_preds:
                    for pred in gp.get("predictions", []):
                        # Collect transcript text if available
                        text = pred.get("text")
                        if text:
                            transcripts.append(text)

                        # Collect emotion scores
                        for emo in pred.get("emotions", []):
                            name, score = emo["name"], emo["score"]
                            all_emotions.setdefault(name, []).append(score)

                # Average scores
                avg_emotions = {
                    name: sum(scores) / len(scores)
                    for name, scores in all_emotions.items()
                }

                # Top 3 emotions
                top3 = sorted(
                    avg_emotions.items(), key=lambda x: x[1], reverse=True
                )[:3]

                summary["audio"]["top_emotions"] = top3
                if transcripts:
                    summary["audio"]["transcript"] = " ".join(transcripts)

        # ---- Face (Video) ----
        if "face" in models:
            face = models["face"]
            grouped_preds = face.get("grouped_predictions", [])

            if grouped_preds:
                all_emotions = {}

                for gp in grouped_preds:
                    for pred in gp.get("predictions", []):
                        for emo in pred.get("emotions", []):
                            name, score = emo["name"], emo["score"]
                            all_emotions.setdefault(name, []).append(score)

                avg_emotions = {
                    name: sum(scores) / len(scores)
                    for name, scores in all_emotions.items()
                }

                top3 = sorted(
                    avg_emotions.items(), key=lambda x: x[1], reverse=True
                )[:3]

                summary["video"]["top_emotions"] = top3

    except Exception as e:
        summary["error"] = str(e)

    return summary
