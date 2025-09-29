# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
CLIPS_DIR = STORAGE_DIR / "clips"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
HISTORY_DIR = STORAGE_DIR / "history"

for d in (STORAGE_DIR, CLIPS_DIR, TRANSCRIPTS_DIR, HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# External service keys / config
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION")  # adjust to your region slug or base subdomain
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET")  # optional, if Recall signs webhooks

HUME_API_KEY = os.getenv("HUME_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Frontend origins
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://emo-insight-frontend.example")
ELECTRON_ORIGIN = os.getenv("ELECTRON_ORIGIN", "http://localhost")


# Safety/timeouts
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HUME_JOB_TIMEOUT = int(os.getenv("HUME_JOB_TIMEOUT", "600"))
