import requests
import logging
import json
from config import settings

logger = logging.getLogger("emo-insight")

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
    # Fix WebSocket URL construction
    if settings.RENDER_EXTERNAL_URL:
        # If we have RENDER_EXTERNAL_URL, use it directly (no https:// prefix)
        ws_url = f"wss://{settings.RENDER_EXTERNAL_URL}/ws?session_id={session_id}"
    else:
        # Fallback to BACKEND_URL
        backend_host = settings.BACKEND_URL.replace("https://", "").replace("http://", "")
        ws_url = f"wss://{backend_host}/ws?session_id={session_id}"
    
    logger.info(f"🔧 Building bot configuration")
    logger.info(f"📡 WebSocket URL: {ws_url}")
    logger.info(f"🔗 Meeting URL: {meeting_url}")

    payload = {
        "meeting_url": meeting_url,
        "bot_name": f"SalesBuddy Bot",  # Add a bot name
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

    logger.info(f"➡️ Sending request to Recall API: {BASE}/bot")
    logger.debug(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        r = requests.post(f"{BASE}/bot", json=payload, headers=HEADERS, timeout=30)
        
        logger.info(f"📨 Recall API response: Status {r.status_code}")
        
        if r.status_code == 400:
            logger.error(f"❌ Bad Request - Response: {r.text}")
            error_data = r.json() if r.text else {}
            logger.error(f"❌ Error details: {json.dumps(error_data, indent=2)}")
            
            # Common 400 errors and solutions
            if "meeting_url" in str(error_data).lower():
                raise ValueError(f"Invalid meeting URL format: {meeting_url}")
            elif "websocket" in str(error_data).lower():
                raise ValueError(f"Invalid WebSocket URL: {ws_url}")
            else:
                raise ValueError(f"Recall API rejected request: {error_data}")
        
        elif r.status_code != 200:
            logger.error(f"❌ Recall API error: {r.status_code} - {r.text}")
        
        r.raise_for_status()
        data = r.json()
        
        logger.debug(f"📋 Response data: {json.dumps(data, indent=2)}")
        
        # Handle different possible response formats from Recall API
        bot_id = data.get("id") or data.get("bot_id")
        if not bot_id:
            logger.error(f"❌ No bot_id in response: {data}")
            raise RuntimeError(f"Recall start_bot missing id field: {data}")
        
        logger.info(f"✅ Bot created successfully with ID: {bot_id}")
        
        return bot_id
        
    except requests.exceptions.Timeout:
        logger.error("⏱️ Recall API request timed out")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Network error calling Recall API: {e}")
        raise
    except Exception as e:
        logger.exception(f"💥 Unexpected error starting bot")
        raise

def stop_bot(bot_id: str) -> None:
    """
    Stop/leave the bot from the meeting.
    """
    logger.info(f"🛑 Attempting to stop bot {bot_id}")
    
    # Try the leave endpoint first
    leave_url = f"{BASE}/bot/{bot_id}/leave"
    logger.debug(f"📡 Trying leave endpoint: {leave_url}")
    
    try:
        r = requests.post(leave_url, headers=HEADERS, timeout=30)
        logger.info(f"📨 Leave response: Status {r.status_code}")
        
        # If leave doesn't work, try stop endpoint
        if r.status_code == 404:
            logger.info("🔄 Leave endpoint returned 404, trying stop endpoint")
            stop_url = f"{BASE}/bot/{bot_id}/stop"
            r = requests.post(stop_url, headers=HEADERS, timeout=30)
            logger.info(f"📨 Stop response: Status {r.status_code}")
            
            if r.status_code == 404:
                logger.warning(f"⚠️ Bot {bot_id} not found - might already be stopped")
                return
        
        if r.status_code not in (200, 202, 204):
            logger.warning(f"⚠️ Unexpected stop response: {r.status_code} - {r.text}")
        else:
            logger.info(f"✅ Bot {bot_id} stopped successfully")
            
    except requests.exceptions.Timeout:
        logger.error("⏱️ Stop bot request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Network error stopping bot: {e}")
    except Exception as e:
        logger.exception(f"💥 Unexpected error stopping bot")