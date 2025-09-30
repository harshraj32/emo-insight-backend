import requests
from config import settings

BASE = f"https://{settings.RECALL_REGION}.recall.ai/api/v1"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Token {settings.RECALL_API_KEY}",  # Must have "Token " prefix
}

def start_bot(meeting_url: str, session_id: str):
    """
    Start the meeting bot with Recall and point it to our websocket receiver,
    including the session_id so ws_receiver can pull rep + objective.
    """
    # Build WebSocket URL with session_id parameter
    ws_url = f"{settings.BACKEND_URL.replace('https://', 'wss://')}/ws?session_id={session_id}"

    payload = {
        "meeting_url": meeting_url,
        "recording_config": {
            "video_mixed_layout": "gallery_view_v2",
            "video_separate_png": {},  # 2fps PNG frames per participant
            "audio_separate_raw": {},  # 16kHz PCM S16LE mono per participant
            "transcript": {
                "provider": {"recallai_streaming": {}},
                "diarization": {"use_separate_streams_when_available": True}
            },
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": ws_url,
                    "events": [
                        "video_separate_png.data",
                        "audio_separate_raw.data",
                        "transcript.data",
                        "transcript.partial_data"
                    ],
                }
            ],
        },
    }

    print(f"➡️ Creating bot for session {session_id}...")
    r = requests.post(f"{BASE}/bot", json=payload, headers=HEADERS, timeout=30)
    
    if r.status_code != 200:
        print(f"⚠️ Recall API error: {r.status_code} - {r.text}")
    
    r.raise_for_status()
    data = r.json()
    
    # Handle different possible response formats from Recall API
    bot_id = data.get("id") or data.get("bot_id")
    if not bot_id:
        raise RuntimeError(f"Recall start_bot missing id field: {data}")
    
    print(f"✅ Bot created with ID: {bot_id}")
    return bot_id

def stop_bot(bot_id: str) -> None:
    """
    Stop/leave the bot from the meeting.
    """
    print(f"➡️ Stopping bot {bot_id}...")
    
    # Try the leave endpoint (some Recall API versions use this)
    r = requests.post(f"{BASE}/bot/{bot_id}/leave", headers=HEADERS, timeout=30)
    
    # If leave doesn't work, try stop endpoint
    if r.status_code == 404:
        r = requests.post(f"{BASE}/bot/{bot_id}/stop", headers=HEADERS, timeout=30)
    
    if r.status_code not in (200, 202, 204):
        print(f"⚠️ Recall stop_bot warning: {r.status_code} - {r.text}")
        # Don't raise error on stop failure as bot might already be stopped
    else:
        print(f"✅ Bot {bot_id} stopped successfully")