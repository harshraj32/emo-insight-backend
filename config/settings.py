# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Use mounted disk for storage if available, otherwise fallback
if os.path.exists('/app/storage'):
    # Production: Use Render's persistent disk
    STORAGE_DIR = Path('/app/storage')
else:
    # Development: Use local storage
    BASE_DIR = Path(__file__).resolve().parents[1]
    STORAGE_DIR = BASE_DIR / "storage"

CLIPS_DIR = STORAGE_DIR / "clips"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"  
HISTORY_DIR = STORAGE_DIR / "history"

# Create all necessary directories
for d in (STORAGE_DIR, CLIPS_DIR, TRANSCRIPTS_DIR, HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)


# External service keys / config
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION", "us-east-1")  # default to us-east-1
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")  # optional

# Server config - Render specific
PORT = int(os.getenv("PORT", "10000"))  # Render uses port 10000 by default

# Get Render URL or fallback to environment variable
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
if RENDER_EXTERNAL_URL:
    # Render provides this automatically
    BACKEND_URL = f"https://{RENDER_EXTERNAL_URL}"

# API Keys
HUME_API_KEY = os.getenv("HUME_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Frontend origins
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # Set your actual frontend URL
ELECTRON_ORIGIN = os.getenv("ELECTRON_ORIGIN", "http://localhost")

# Hume models configuration
HUME_MODELS = {
    "prosody": {},
    "face": {}
}

# Safety/timeouts
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HUME_JOB_TIMEOUT = int(os.getenv("HUME_JOB_TIMEOUT", "600"))

# Validate critical environment variables
if not RECALL_API_KEY:
    print("⚠️ WARNING: RECALL_API_KEY not set in environment")
if not HUME_API_KEY:
    print("⚠️ WARNING: HUME_API_KEY not set in environment")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set in environment")
    
print(f"🚀 Backend URL configured as: {BACKEND_URL}")