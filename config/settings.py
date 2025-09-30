# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Use mounted disk for storage if available (Render production)
if os.path.exists('/app/storage'):
    STORAGE_DIR = Path('/app/storage')
else:
    # Development: Use local storage
    BASE_DIR = Path(__file__).resolve().parents[1]
    STORAGE_DIR = BASE_DIR / "storage"

# Create subdirectories
CLIPS_DIR = STORAGE_DIR / "clips"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"  
HISTORY_DIR = STORAGE_DIR / "history"

# Create all necessary directories
for d in (STORAGE_DIR, CLIPS_DIR, TRANSCRIPTS_DIR, HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# External service keys
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION", "us-east-1")  # or us-west-2
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")

# Server config
PORT = int(os.getenv("PORT", "10000"))

# Fix the URL configuration
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")  # Just the domain, no protocol

# Construct proper backend URL
if RENDER_EXTERNAL_URL:
    # In production on Render
    BACKEND_URL = f"https://{RENDER_EXTERNAL_URL}"
else:
    # Fallback for development or manual config
    BACKEND_URL = os.getenv("BACKEND_URL", "https://emo-insight-backend.onrender.com")

# API Keys
HUME_API_KEY = os.getenv("HUME_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Frontend origins
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
ELECTRON_ORIGIN = os.getenv("ELECTRON_ORIGIN", "http://localhost")

# Hume models configuration
HUME_MODELS = {
    "prosody": {"granularity": "utterance"},
    "face": {"fps_pred": 3}
}

# Safety/timeouts
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HUME_JOB_TIMEOUT = int(os.getenv("HUME_JOB_TIMEOUT", "180"))

# Validate critical environment variables
if not RECALL_API_KEY:
    print("⚠️ WARNING: RECALL_API_KEY not set in environment")
if not HUME_API_KEY:
    print("⚠️ WARNING: HUME_API_KEY not set in environment")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set in environment")

print(f"🚀 Configuration loaded:")
print(f"  - Backend URL: {BACKEND_URL}")
print(f"  - Storage Dir: {STORAGE_DIR}")
print(f"  - Recall Region: {RECALL_REGION}")
print(f"  - Render URL: {RENDER_EXTERNAL_URL or 'Not on Render'}")