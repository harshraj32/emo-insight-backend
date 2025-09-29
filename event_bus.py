from typing import Awaitable, Callable, Dict, Any

# UI emitters (wired in main.py at startup)
emit_advice: Callable[[str, str], Awaitable[None]] = lambda *_: None  # async
emit_emotion: Callable[[Dict[str, Any]], Awaitable[None]] = lambda *_: None  # async
emit_log: Callable[[str, list], Awaitable[None]] = lambda *_: None  # async

# Shared in-memory sessions store
# sessions[session_id] = {
#   "user_name": str,
#   "meeting_url": str,
#   "objective": str,
#   "emotions": list[str],
#   "bot_id": str,
#   "created_at": float,
#   "phase": str,
#   "logs": list[str],
#   "recent_events": list[...],
#   "last_hume_summary": dict
# }
sessions: Dict[str, Dict[str, Any]] = {}
