# hume/hume_summarize.py
from typing import Dict, Any, List
import json

def summarize(results: dict) -> dict:
    """
    Summarize Hume API results:
    - Audio (prosody): transcript + top emotions
    - Video (face): top emotions
    """
    summary = {"audio": {}, "video": {}}

    try:
        # Debug: Print the structure we're receiving
        print(f"[DEBUG] Summarize input type: {type(results)}")
        if isinstance(results, list) and len(results) > 0:
            print(f"[DEBUG] First element keys: {results[0].keys() if isinstance(results[0], dict) else 'Not a dict'}")
        
        # Handle the actual Hume response structure
        if isinstance(results, list) and results:
            # Get the first result item
            result_item = results[0]
            
            # Navigate to predictions
            if "results" in result_item and isinstance(result_item["results"], dict):
                predictions = result_item["results"].get("predictions", [])
            else:
                print("[DEBUG] No 'results' key found in expected format")
                return summary
        elif isinstance(results, dict):
            # Sometimes Hume returns a dict directly
            if "predictions" in results:
                predictions = results["predictions"]
            elif "results" in results and "predictions" in results["results"]:
                predictions = results["results"]["predictions"]
            else:
                print("[DEBUG] No predictions found in dict format")
                return summary
        else:
            print(f"[DEBUG] Unexpected results format: {type(results)}")
            return summary

        if not predictions:
            print("[DEBUG] No predictions to process")
            return summary

        # Get the first prediction (there's usually only one per file)
        if predictions and len(predictions) > 0:
            prediction = predictions[0]
            models = prediction.get("models", {})
        else:
            print("[DEBUG] Empty predictions list")
            return summary

        # ---------- AUDIO (Prosody) ----------
        if "prosody" in models:
            print("[DEBUG] Processing prosody model")
            prosody = models["prosody"]
            grouped_preds = prosody.get("grouped_predictions", [])

            all_emotions = {}  # Use regular dict
            transcripts = []

            for gp in grouped_preds:
                for pred in gp.get("predictions", []):
                    # Collect transcript text if present
                    text = pred.get("text")
                    if text:
                        transcripts.append(text.strip())

                    # Collect emotion scores
                    emotions_list = pred.get("emotions", [])
                    for emo in emotions_list:
                        if isinstance(emo, dict):  # Make sure it's a dict
                            name = emo.get("name")
                            score = emo.get("score")
                            if name is not None and score is not None:
                                if name not in all_emotions:
                                    all_emotions[name] = []
                                all_emotions[name].append(float(score))

            if all_emotions:
                # Calculate averages
                avg_emotions = {}
                for name, scores in all_emotions.items():
                    if scores:  # Make sure we have scores
                        avg_emotions[name] = sum(scores) / len(scores)
                
                # Get top 3
                if avg_emotions:
                    top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    summary["audio"]["top_emotions"] = top3
                    print(f"[DEBUG] Audio top emotions: {top3[:1]}")  # Just show first

            if transcripts:
                summary["audio"]["transcript"] = " ".join(transcripts)
                print(f"[DEBUG] Audio transcript length: {len(summary['audio']['transcript'])}")

        # ---------- VIDEO (Face) ----------
        if "face" in models:
            print("[DEBUG] Processing face model")
            face = models["face"]
            grouped_preds = face.get("grouped_predictions", [])

            all_emotions = {}  # Use regular dict

            for gp in grouped_preds:
                for pred in gp.get("predictions", []):
                    # For face model, emotions are directly in each prediction
                    emotions_list = pred.get("emotions", [])
                    for emo in emotions_list:
                        if isinstance(emo, dict):  # Make sure it's a dict
                            name = emo.get("name")
                            score = emo.get("score")
                            if name is not None and score is not None:
                                if name not in all_emotions:
                                    all_emotions[name] = []
                                all_emotions[name].append(float(score))

            if all_emotions:
                # Calculate averages
                avg_emotions = {}
                for name, scores in all_emotions.items():
                    if scores:  # Make sure we have scores
                        avg_emotions[name] = sum(scores) / len(scores)
                
                # Get top 3
                if avg_emotions:
                    top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    summary["video"]["top_emotions"] = top3
                    print(f"[DEBUG] Video top emotions: {top3[:1]}")  # Just show first

    except Exception as e:
        error_msg = f"summarize failed: {str(e)}"
        summary["error"] = error_msg
        print(f"[DEBUG] {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Return error in both audio and video
        summary["audio"] = {"status": "error", "error": str(e)}
        summary["video"] = {"status": "error", "error": str(e)}

    return summary