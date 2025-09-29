import json
import requests
from config import settings

API = f"https://{settings.RECALL_REGION}.recall.ai/api/v1"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Token {settings.RECALL_API_KEY}",
}

def start_bot():
    payload = {
        "meeting_url": settings.TEST_MEETING_URL,
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
                    "url": settings.WS_RECEIVER_URL,
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
    r = requests.post(f"{API}/bot", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def stop_bot(bot_id: str):
    r = requests.post(f"{API}/bot/{bot_id}/leave/", headers=headers, timeout=30)
    r.raise_for_status()
    return True
