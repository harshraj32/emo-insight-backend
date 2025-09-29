import asyncio
import time
import uuid
from typing import Dict, Any
import logging
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from recall import ws_receiver
from fastapi import WebSocket
from config import settings
from recall import bot_manager
from hume.hume_client import process_clip  # imported to keep parity with your earlier file refs
from hume.hume_summarize import summarize
from affina.coach import coach_feedback

import event_bus

app = FastAPI(title="SalesBuddy Backend", version="1.0.0")
socket_manager = SocketManager(app, cors_allowed_origins="*")

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Logging setup =====
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("emo-insight")

app = FastAPI(title="SalesBuddy Backend", version="1.0.0")
socket_manager = SocketManager(app, cors_allowed_origins="*")

# ===== Middleware to log every HTTP request =====
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"➡️ {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"⬅️ {request.method} {request.url} {response.status_code}")
    return response


# Single source of truth for sessions
sessions: Dict[str, Dict[str, Any]] = event_bus.sessions

# ====== Wire event bus emitters to Socket.IO events ======
def _emit_advice(session_id: str, advice: str):
    # target the session room so only the right client gets it
    socket_manager.emit("affina_advice", {"session_id": session_id, "advice": advice}, room=session_id)

def _emit_emotion(payload: Dict[str, Any]):
    sid = payload.get("session_id")
    socket_manager.emit("emotion_detected", payload, room=sid or None)

def _emit_log(session_id: str, logs: list):
    socket_manager.emit("log_update", {"session_id": session_id, "logs": logs}, room=session_id)

event_bus.emit_advice = _emit_advice
event_bus.emit_emotion = _emit_emotion
event_bus.emit_log = _emit_log

# ====== Socket.IO room join (frontend emits 'join_session') ======
@socket_manager.on("join_session")
async def join_session(sid, data):
    session_id = (data or {}).get("session_id")
    if session_id in sessions:
        await socket_manager.enter_room(sid, session_id)
        # push latest logs so UI shows history after joining
        logs = sessions[session_id].get("logs", [])[-10:]
        socket_manager.emit("log_update", {"session_id": session_id, "logs": logs}, room=session_id)
        print(f"Client {sid} joined session {session_id}")

# ====== API ======
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "time": time.time(),
        "version": app.version,
        "services": {
            "recall_webhooks_enabled": bool(settings.RECALL_API_KEY),
            "hume_key_present": bool(settings.HUME_API_KEY),
            "openai_key_present": bool(settings.OPENAI_API_KEY),
        }
    }

@app.post("/api/start-session")
def start_session(payload: Dict[str, Any] = Body(...)):
    """
    Body from frontend (unchanged):
      - user_name
      - meeting_url
      - meeting_objective
      - selected_emotions
    """
    user_name = (payload.get("user_name") or "").strip()
    meeting_url = (payload.get("meeting_url") or "").strip()
    meeting_objective = (payload.get("meeting_objective") or "").strip()
    selected_emotions = payload.get("selected_emotions", [])

    if not user_name or not meeting_url:
        return {"success": False, "error": "user_name and meeting_url required"}

    session_id = str(uuid.uuid4())

    # Create session record
    sessions[session_id] = {
        "user_name": user_name,
        "meeting_url": meeting_url,
        "objective": meeting_objective,
        "emotions": selected_emotions,
        "bot_id": None,
        "created_at": time.time(),
        "phase": "pleasantries",
        "logs": [],
        "recent_events": [],
        "last_hume_summary": {},
    }

    # Start Recall bot with session_id embedded in its WS URL
    try:
        bot_id = bot_manager.start_bot(meeting_url, session_id)
        sessions[session_id]["bot_id"] = bot_id
        sessions[session_id]["logs"].append(f"Session {session_id} created. Bot {bot_id} joining...")
        event_bus.emit_log(session_id, sessions[session_id]["logs"][-10:])
    except Exception as e:
        sessions[session_id]["logs"].append(f"Bot start error: {e}")
        return {"success": False, "error": f"Failed to start bot: {e}"}

    return {"success": True, "session_id": session_id, "bot_id": bot_id}

@app.post("/api/stop-session")
def stop_session(payload: Dict[str, Any] = Body(...)):
    session_id = payload.get("session_id")
    sess = sessions.get(session_id)
    if not sess:
        return {"success": False, "error": "invalid session_id"}

    try:
        if sess.get("bot_id"):
            bot_id = sess["bot_id"]
            bot_manager.stop_bot(bot_id)
            sess["logs"].append(f"Bot {bot_id} stopped.")
    except Exception as e:
        sess["logs"].append(f"Bot stop error: {e}")
    finally:
        sess["logs"].append("Session stopped by user.")
        event_bus.emit_log(session_id, sess["logs"][-10:])
        sessions.pop(session_id, None)

    return {"success": True, "message": "Session stopped"}


@app.websocket("/ws")
async def recall_ws(websocket: WebSocket):
    await ws_receiver.fastapi_handler(websocket)