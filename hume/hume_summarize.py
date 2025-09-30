# hume/hume_summarize.py

def summarize(results: dict) -> dict:
    """
    Summarize Hume API results to extract top emotions from audio and video.
    """
    summary = {"audio": {}, "video": {}}
    
    try:
        # Handle the nested structure of Hume results
        if isinstance(results, list) and len(results) > 0:
            # results is a list, get first element
            result = results[0]
        elif isinstance(results, dict):
            # results might already be the dict we need
            result = results
        else:
            raise ValueError(f"Unexpected results format: {type(results)}")
        
        # Navigate to predictions
        if "results" in result:
            predictions = result["results"].get("predictions", [])
        elif "predictions" in result:
            predictions = result["predictions"]
        else:
            raise ValueError("Cannot find predictions in results")
        
        if not predictions:
            return summary
        
        # Get the models from the first prediction
        models = predictions[0].get("models", {})
        
        # Process prosody (audio) if present
        if "prosody" in models:
            prosody = models["prosody"]
            grouped_preds = prosody.get("grouped_predictions", [])
            
            if grouped_preds:
                # Collect all emotions across all predictions
                all_emotions = {}
                for gp in grouped_preds:
                    for pred in gp.get("predictions", []):
                        emotions = pred.get("emotions", [])
                        for emo in emotions:
                            name = emo["name"]
                            score = emo["score"]
                            if name not in all_emotions:
                                all_emotions[name] = []
                            all_emotions[name].append(score)
                
                # Average the scores
                avg_emotions = {}
                for name, scores in all_emotions.items():
                    avg_emotions[name] = sum(scores) / len(scores)
                
                # Get top 3
                top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                summary["audio"]["top_emotions"] = top3
        
        # Process face (video) if present
        if "face" in models:
            face = models["face"]
            grouped_preds = face.get("grouped_predictions", [])
            
            if grouped_preds:
                # Collect all emotions across all predictions
                all_emotions = {}
                for gp in grouped_preds:
                    for pred in gp.get("predictions", []):
                        emotions = pred.get("emotions", [])
                        for emo in emotions:
                            name = emo["name"]
                            score = emo["score"]
                            if name not in all_emotions:
                                all_emotions[name] = []
                            all_emotions[name].append(score)
                
                # Average the scores
                avg_emotions = {}
                for name, scores in all_emotions.items():
                    avg_emotions[name] = sum(scores) / len(scores)
                
                # Get top 3
                top3 = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                summary["video"]["top_emotions"] = top3
                
    except Exception as e:
        summary["error"] = str(e)
    
    return summary