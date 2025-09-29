import os
from dotenv import load_dotenv

load_dotenv()

# Recall.ai
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION", "us-east-1")
TEST_MEETING_URL = os.getenv("TEST_MEETING_URL")
WS_RECEIVER_URL = os.getenv("WS_RECEIVER_URL")

# Hume.ai
HUME_API_KEY = os.getenv("HUME_API_KEY")

# Storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
CLIPS_DIR = os.path.join(STORAGE_DIR, "clips")
TRANSCRIPTS_DIR = os.path.join(STORAGE_DIR, "transcripts")
HISTORY_DIR = os.path.join(STORAGE_DIR, "history")

for d in [STORAGE_DIR, CLIPS_DIR, TRANSCRIPTS_DIR, HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)
