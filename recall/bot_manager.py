import requests
from config import settings

BASE = f"https://{settings.RECALL_REGION}.recall.ai/api/v1"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Token {settings.RECALL_API_KEY}",
}

def start_bot(meeting_url: str, session_id: str):
    """
    Start the meeting bot with Recall and point it to our websocket receiver,
    including the session_id so ws_receiver can pull rep + objective.
    """
    ws_url = f"{settings.BACKEND_URL.replace('https://', 'wss://')}/ws?session_id={session_id}"

    payload = {
        "meeting_url": meeting_url,
        "recording_config": {
            "video_mixed_layout": "gallery_view_v2",
            "video_separate_png": {},
            "audio_separate_raw": {},
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

    r = requests.post(f"{BASE}/bot/", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    bot_id = data.get("id") or data.get("bot_id")
    if not bot_id:
        raise RuntimeError(f"Recall start_bot missing id field: {data}")
    return bot_id

def stop_bot(bot_id: str) -> None:
    r = requests.post(f"{BASE}/bot/{bot_id}/stop/", headers=HEADERS, timeout=30)
    if r.status_code not in (200, 202, 204):
        raise RuntimeError(f"Recall stop_bot failed: {r.status_code} {r.text}")
