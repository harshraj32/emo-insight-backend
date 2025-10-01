import os
import json
from datetime import datetime
from pathlib import Path
from config import settings

STORAGE_DIR = Path(settings.CLIPS_DIR).parent / "session_data"

def ensure_session_dir(session_id: str) -> Path:
    """Create and return session storage directory."""
    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def save_emotion_trail(session_id: str, speaker: str, timestamp: str, emotions: dict):
    """
    Append emotion data to speaker's trail file.
    Format: One JSON object per line for easy streaming/parsing.
    """
    session_dir = ensure_session_dir(session_id)
    trail_file = session_dir / f"{speaker}_emotions.jsonl"
    
    entry = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "audio_emotions": emotions.get("audio", {}).get("top_emotions", []),
        "video_emotions": emotions.get("video", {}).get("top_emotions", []),
    }
    
    with open(trail_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_transcript_line(session_id: str, speaker: str, timestamp: str, text: str):
    """
    Append transcript line to session transcript file.
    """
    session_dir = ensure_session_dir(session_id)
    transcript_file = session_dir / "transcript.jsonl"
    
    entry = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "speaker": speaker,
        "text": text,
    }
    
    with open(transcript_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_recent_emotion_trail(session_id: str, speaker: str, limit: int = 10) -> list:
    """
    Load recent emotion entries for a speaker.
    """
    session_dir = STORAGE_DIR / session_id
    trail_file = session_dir / f"{speaker}_emotions.jsonl"
    
    if not trail_file.exists():
        return []
    
    with open(trail_file, "r") as f:
        lines = f.readlines()
    
    # Return last N entries
    recent = []
    for line in lines[-limit:]:
        try:
            recent.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return recent

def get_recent_transcript(session_id: str, limit: int = 20) -> list:
    """
    Load recent transcript lines.
    """
    session_dir = STORAGE_DIR / session_id
    transcript_file = session_dir / "transcript.jsonl"
    
    if not transcript_file.exists():
        return []
    
    with open(transcript_file, "r") as f:
        lines = f.readlines()
    
    # Return last N entries
    recent = []
    for line in lines[-limit:]:
        try:
            recent.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return recent

def get_last_emotion_state(session_id: str, speaker: str) -> dict:
    """
    Get the most recent emotion state for comparison.
    """
    trail = get_recent_emotion_trail(session_id, speaker, limit=1)
    return trail[0] if trail else {}

def has_emotion_changed(old_state: dict, new_emotions: dict, threshold: float = 0.1) -> bool:
    """
    Determine if emotions changed significantly.
    Returns True if:
    - Top emotion changed
    - Top emotion score changed by more than threshold
    - No previous state (first detection)
    """
    if not old_state:
        return True  # First detection
    
    # Get previous top emotions
    old_audio = old_state.get("audio_emotions", [])
    old_video = old_state.get("video_emotions", [])
    
    # Get current top emotions
    new_audio = new_emotions.get("audio", {}).get("top_emotions", [])
    new_video = new_emotions.get("video", {}).get("top_emotions", [])
    
    # Check audio emotions
    if new_audio and old_audio:
        old_top = old_audio[0] if old_audio else {}
        new_top = new_audio[0] if new_audio else {}
        
        # Different emotion name
        if old_top.get("name") != new_top.get("name"):
            return True
        
        # Significant score change
        old_score = old_top.get("score", 0)
        new_score = new_top.get("score", 0)
        if abs(old_score - new_score) > threshold:
            return True
    
    # Check video emotions
    if new_video and old_video:
        old_top = old_video[0] if old_video else {}
        new_top = new_video[0] if new_video else {}
        
        if old_top.get("name") != new_top.get("name"):
            return True
        
        old_score = old_top.get("score", 0)
        new_score = new_top.get("score", 0)
        if abs(old_score - new_score) > threshold:
            return True
    
    return False

def get_blended_emotion_label(emotions: list, threshold: float = 0.07) -> str:
    """
    Create a label for close emotions.
    If top 3 are within threshold, return blended label.
    """
    if not emotions or len(emotions) == 0:
        return "Neutral"
    
    top = emotions[0]
    close_emotions = [e for e in emotions if top["score"] - e["score"] <= threshold]
    
    if len(close_emotions) == 1:
        # Clear winner
        return top["name"]
    elif len(close_emotions) == 2:
        # Two close emotions
        return f"{close_emotions[0]['name']} + {close_emotions[1]['name']}"
    else:
        # Three or more close
        names = [e["name"] for e in close_emotions[:3]]
        return f"{names[0]} + {names[1]} + {names[2]}"