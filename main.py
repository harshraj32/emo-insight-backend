from fastapi import FastAPI, BackgroundTasks
from recall import bot_manager
from hume import hume_client, summarize
from affina.coach import coach_feedback
from config import settings
from pathlib import Path
import glob, json, os

app = FastAPI()
bot_id = None

@app.post("/start")
def start():
    global bot_id
    bot_id = bot_manager.start_bot()
    return {"bot_id": bot_id}

@app.post("/stop")
def stop():
    global bot_id
    if bot_id:
        bot_manager.stop_bot(bot_id)
        bot_id = None
    return {"status": "stopped"}

@app.post("/process_clips")
def process_clips():
    files = glob.glob(os.path.join(settings.CLIPS_DIR, "*.mp4"))
    out = {}
    for f in files:
        res = hume_client.process_clip(Path(f))
        summ = summarize.summarize(res)
        out[os.path.basename(f)] = summ
    return out

@app.post("/feedback")
def feedback(transcript_line: str):
    # Dummy: take last summary file
    files = glob.glob(os.path.join(settings.CLIPS_DIR, "*.mp4"))
    if not files:
        return {"error": "No clips"}
    res = hume_client.process_clip(Path(files[-1]))
    summ = summarize.summarize(res)
    fb = coach_feedback(summ, transcript_line)
    return {"feedback": fb, "summary": summ}
