import requests
import logging
import json
import os
from config import settings

logger = logging.getLogger("emo-insight")

# Validate configuration
if not settings.RECALL_API_KEY:
    raise ValueError("RECALL_API_KEY is not set")

BASE = f"https://{settings.RECALL_REGION}.recall.ai/api/v1"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Token {settings.RECALL_API_KEY}",  # IMPORTANT: 'Token ' prefix
}

def start_bot(meeting_url: str, session_id: str):
    """
    Start the meeting bot with Recall.
    """
    
    if os.getenv("RENDER_EXTERNAL_URL"):
        render_url = os.getenv("RENDER_EXTERNAL_URL")
        render_url = render_url.replace("https://", "").replace("http://", "").replace("wss://", "").replace("ws://", "")
        ws_url = f"wss://{render_url}/ws?session_id={session_id}"
    else:
        ws_url = f"wss://emo-insight-backend.onrender.com/ws?session_id={session_id}"
    
    print(f"➡️ creating bot...")
    print(f"   Meeting URL: {meeting_url}")
    print(f"   WebSocket URL: {ws_url}")
    print(f"   Session ID: {session_id}")
    
    payload = {
        "meeting_url": meeting_url,
        "recording_config": {
            "video_mixed_layout": "gallery_view_v2",
            "video_separate_png": {},
            "audio_separate_raw": {},
            "transcript": {
                "provider": {"recallai_streaming": {}},
                "diarization": {"use_separate_streams_when_available": True},
            },
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": ws_url,
                    "events": [
                        # Media data events
                        "video_separate_png.data",
                        "audio_separate_raw.data",
                        "transcript.data",
                        "transcript.partial_data",
                        # Participant events (ONLY these are allowed for real-time endpoints)
                        "participant_events.join",
                        "participant_events.leave",
                        "participant_events.speech_on",
                        "participant_events.speech_off",
                        "participant_events.webcam_on",
                        "participant_events.webcam_off",
                    ],
                }
            ],
        },
    }
    
    try:
        r = requests.post(f"{BASE}/bot", headers=headers, json=payload, timeout=30)
        print("status:", r.status_code)
        print("resp:", r.text)
        
        if r.status_code == 400:
            print("❌ Bad Request Details:")
            try:
                error_json = r.json()
                print(json.dumps(error_json, indent=2))
            except:
                print(r.text)
            
        r.raise_for_status()
        bot = r.json()
        print("✅ bot created:", json.dumps(bot, indent=2))
        
        bot_id = bot.get("id") or bot.get("bot_id")
        if not bot_id:
            raise RuntimeError(f"No bot ID in response: {bot}")
        
        return bot_id
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
def stop_bot(bot_id: str) -> None:
    """
    Stop the bot - matching test file structure.
    """
    print(f"➡️ stopping bot {bot_id}...")
    try:
        r = requests.post(f"{BASE}/bot/{bot_id}/leave/", headers=headers, timeout=30)
        print("status:", r.status_code, "resp:", r.text)
        
        # If leave doesn't work, try stop
        if r.status_code == 404:
            r = requests.post(f"{BASE}/bot/{bot_id}/stop", headers=headers, timeout=30)
            print("stop status:", r.status_code, "resp:", r.text)
            
    except Exception as e:
        print("stop failed:", e)