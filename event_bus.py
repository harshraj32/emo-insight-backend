from typing import Callable, Dict, Any

# UI emitters (wired in main.py at startup)
emit_advice: Callable[[str, str], None] = lambda *_: None
emit_emotion: Callable[[Dict[str, Any]], None] = lambda *_: None
emit_log: Callable[[str, list], None] = lambda *_: None

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
