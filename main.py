import asyncio
import time
import uuid
import logging
from typing import Dict, Any

from fastapi import FastAPI, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager

from config import settings
from recall import bot_manager
from hume.hume_client import process_clip
from hume.hume_summarize import summarize
from affina.coach import coach_feedback
import event_bus
from ws_receiver import fastapi_handler

# ===== Logging setup =====
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("emo-insight")

# ===== FastAPI + Socket.IO =====
app = FastAPI(title="SalesBuddy Backend", version="1.0.0")

# For Render deployment - allow all origins or specify your frontend URL
cors_origins = os.getenv("FRONTEND_ORIGIN", "*").split(",")
socket_manager = SocketManager(app, cors_allowed_origins=cors_origins)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Root route (health/debug) =====
@app.get("/")
def root():
    return {"status": "running", "message": "Emo Insight Backend is alive!"}

# ===== Middleware: log every HTTP request =====
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"➡️ {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"⬅️ {request.method} {request.url} {response.status_code}")
    return response

# ===== Shared sessions =====
sessions: Dict[str, Dict[str, Any]] = event_bus.sessions

# ===== Wire event bus emitters to Socket.IO =====
async def _emit_advice(session_id: str, advice: str):
    await socket_manager.emit("affina_advice", {"session_id": session_id, "advice": advice}, room=session_id)

async def _emit_emotion(payload: Dict[str, Any]):
    sid = payload.get("session_id")
    await socket_manager.emit("emotion_detected", payload, room=sid or None)

async def _emit_log(session_id: str, logs: list):
    await socket_manager.emit("log_update", {"session_id": session_id, "logs": logs}, room=session_id)

event_bus.emit_advice = _emit_advice
event_bus.emit_emotion = _emit_emotion
event_bus.emit_log = _emit_log

# ===== WebSocket route for Recall.ai =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await fastapi_handler(websocket)

# ===== Socket.IO Events =====
@socket_manager.on("join_session")
async def join_session(sid, data):
    session_id = (data or {}).get("session_id")
    if session_id in sessions:
        await socket_manager.enter_room(sid, session_id)
        logs = sessions[session_id].get("logs", [])[-10:]
        await socket_manager.emit("log_update", {"session_id": session_id, "logs": logs}, room=session_id)
        logger.info(f"🔗 Client {sid} joined session {session_id}")
    else:
        logger.warning(f"⚠️ join_session: Unknown session {session_id}")

@socket_manager.on("recall_event")
async def recall_event(sid, data):
    """Relay Recall.ai events to logs."""
    session_id = (data or {}).get("session_id")
    if not session_id or session_id not in sessions:
        logger.warning(f"⚠️ recall_event: Unknown session {session_id}")
        return

    sessions[session_id]["logs"].append(f"Recall event: {data.get('event')}")
    await event_bus.emit_log(session_id, sessions[session_id]["logs"][-10:])
    logger.debug(f"📡 Recall event for {session_id}: {data.get('event')}")

# ===== API Routes =====
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
async def start_session(payload: Dict[str, Any] = Body(...)):
    user_name = (payload.get("user_name") or "").strip()
    meeting_url = (payload.get("meeting_url") or "").strip()
    meeting_objective = (payload.get("meeting_objective") or "").strip()
    selected_emotions = payload.get("selected_emotions", [])

    if not user_name or not meeting_url:
        return {"success": False, "error": "user_name and meeting_url required"}

    session_id = str(uuid.uuid4())

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

    try:
        bot_id = bot_manager.start_bot(meeting_url, session_id)
        sessions[session_id]["bot_id"] = bot_id
        sessions[session_id]["logs"].append(f"Session {session_id} created. Bot {bot_id} joining...")
        await event_bus.emit_log(session_id, sessions[session_id]["logs"][-10:])
        logger.info(f"🤖 Started Recall bot {bot_id} for session {session_id}")
    except Exception as e:
        sessions[session_id]["logs"].append(f"Bot start error: {e}")
        await event_bus.emit_log(session_id, sessions[session_id]["logs"][-10:])
        logger.exception("Bot start error")
        return {"success": False, "error": f"Failed to start bot: {e}"}

    return {"success": True, "session_id": session_id, "bot_id": bot_id}

@app.post("/api/stop-session")
async def stop_session(payload: Dict[str, Any] = Body(...)):
    session_id = payload.get("session_id")
    sess = sessions.get(session_id)
    if not sess:
        return {"success": False, "error": "invalid session_id"}

    try:
        if sess.get("bot_id"):
            bot_id = sess["bot_id"]
            bot_manager.stop_bot(bot_id)
            sess["logs"].append(f"Bot {bot_id} stopped.")
            logger.info(f"🛑 Bot {bot_id} stopped for session {session_id}")
    except Exception as e:
        sess["logs"].append(f"Bot stop error: {e}")
        logger.exception("Bot stop error")
    finally:
        sess["logs"].append("Session stopped by user.")
        await event_bus.emit_log(session_id, sess["logs"][-10:])
        sessions.pop(session_id, None)

    return {"success": True, "message": "Session stopped"}