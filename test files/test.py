import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# --- env vars you must set ---
RECALL_API_KEY = os.getenv("RECALL_API_KEY")   # e.g. sk_live_...
TEST_MEETING_URL = os.getenv("TEST_MEETING_URL")  # e.g. https://meet.google.com/abc-defg-hij
WS_RECEIVER_URL = os.getenv("WS_RECEIVER_URL")   # e.g. wss://<sub>.ngrok-free.app
RECALL_REGION = os.getenv("RECALL_REGION", "us-east-1")  # us-east-1 or us-west-2

if not RECALL_API_KEY or not TEST_MEETING_URL or not WS_RECEIVER_URL:
    raise SystemExit("Set RECALL_API_KEY, TEST_MEETING_URL, WS_RECEIVER_URL in .env")

API = f"https://{RECALL_REGION}.recall.ai/api/v1"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Token {RECALL_API_KEY}",  # IMPORTANT: 'Token ' prefix
}


def start_bot():
    payload = {
        "meeting_url": TEST_MEETING_URL,
        "recording_config": {
            "video_mixed_layout": "gallery_view_v2",
            "video_separate_png": {},   # 2fps PNG frames
            "audio_separate_raw": {},   # 16kHz PCM S16LE mono, per participant
            "transcript": {
                "provider": {"recallai_streaming": {}},
                "diarization": {"use_separate_streams_when_available": True},
            },
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": WS_RECEIVER_URL,
                    "events": [
                        "video_separate_png.data",
                        "audio_separate_raw.data",
                        "transcript.data",
                        "transcript.partial_data",
                    ],
                }
            ],
        },
    }

    print("➡️ creating bot...")
    r = requests.post(f"{API}/bot", headers=headers, json=payload, timeout=30)
    print("status:", r.status_code)
    print("resp:", r.text)
    r.raise_for_status()
    bot = r.json()
    print("✅ bot created:", json.dumps(bot, indent=2))
    return bot["id"]


def stop_bot(bot_id):
    print(f"➡️ stopping bot {bot_id}...")
    r = requests.post(f"{API}/bot/{bot_id}/leave/", headers=headers, timeout=30)
    print("status:", r.status_code, "resp:", r.text)


if __name__ == "__main__":
    try:
        bot_id = start_bot()
        print("\n🤖 If your Meet requires approval, ADMIT the bot.")
        print("   Leave this running while you talk / show faces / speak.")
        print("   Watch the ws_receiver.py terminal for frames + transcripts + audio events.\n")
        input("Press ENTER to stop the bot...")
    finally:
        try:
            if "bot_id" in locals():
                stop_bot(bot_id)
        except Exception as e:
            print("stop failed:", e)
