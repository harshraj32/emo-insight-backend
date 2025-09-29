import asyncio
import base64
import datetime
import json
import os
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import WebSocket
from config import settings
from hume import hume_client
from hume.hume_summarize import summarize
from affina.coach import coach_feedback
import event_bus

executor = ThreadPoolExecutor(max_workers=4)

AUDIO_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FPS = 2
CLIP_LEN = 5.0

# buffer per participant (per session)
participant_data = defaultdict(
    lambda: {
        "frames": deque(),
        "audio_buffer": bytearray(),
        "last_clip_time": None,
        "start_time": None,
    }
)


async def create_clips_for_all(session_id, participants, start, end):
    """
    For a given 5s window, generate a clip for each speaker,
    send to Hume, summarize, and then generate one Affina feedback.
    """
    summaries = {}
    ts_str = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(settings.CLIPS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for speaker, data in participants.items():
        clip_name = f"{speaker}_{ts_str}"
        temp_dir = os.path.join(session_dir, f"tmp_{clip_name}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # save frames
            for i, (frame_data, frame_time) in enumerate(data["frames"]):
                if start <= frame_time <= end:
                    with open(os.path.join(temp_dir, f"frame_{i:04d}.png"), "wb") as f:
                        f.write(base64.b64decode(frame_data))

            # save audio.raw
            raw_path = os.path.join(temp_dir, "audio.raw")
            with open(raw_path, "wb") as f:
                f.write(bytes(data["audio_buffer"]))

            wav_path = os.path.join(temp_dir, "audio.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le",
                    "-ar", str(AUDIO_RATE),
                    "-ac", str(CHANNELS),
                    "-i", raw_path,
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )

            # combine → mp4
            mp4_out = os.path.join(session_dir, f"{clip_name}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(FPS),
                    "-i", os.path.join(temp_dir, "frame_%04d.png"),
                    "-i", wav_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    mp4_out,
                ],
                check=True,
                capture_output=True,
            )

            # ---- send to Hume ----
            results = hume_client.process_clip(
                Path(mp4_out), models={"prosody": {}, "face": {}}
            )
            summaries[speaker] = summarize(results)

            print(f"🎬 Processed clip for {speaker} in session {session_id}")

            # cleanup mp4 after processing
            os.remove(mp4_out)

        except Exception as e:
            summaries[speaker] = {"error": str(e)}
        finally:
            subprocess.run(["rm", "-rf", temp_dir])

    # ---- build combined context ----
    sess = event_bus.sessions.get(session_id, {})
    context = {
        "phase": sess.get("phase", "pitch"),
        "objective": sess.get("objective", ""),
        "emotions": sess.get("emotions", []),
        "summaries": summaries,   # 🔥 multi-speaker summary
    }
    feedback = coach_feedback(context, f"Clips {int(start)}–{int(end)}s")

    # stringify if dict (frontend shows string)
    advice_str = json.dumps(feedback) if isinstance(feedback, dict) else str(feedback)
    await event_bus.emit_advice(session_id, advice_str)

    # log
    logs = sess.setdefault("logs", [])
    logs.append(f"[{ts_str}] Processed multi-speaker window.")
    await event_bus.emit_log(session_id, logs[-10:])


def check_clips(session_id):
    now = time.time()
    sess_participants = {}
    for speaker, data in participant_data.items():
        if data["start_time"] is None:
            data["start_time"] = now
            data["last_clip_time"] = now
            continue
        if now - data["last_clip_time"] >= CLIP_LEN:
            start, end = data["last_clip_time"], now
            if data["frames"] or data["audio_buffer"]:
                sess_participants[speaker] = {
                    "frames": list(data["frames"]),
                    "audio_buffer": bytes(data["audio_buffer"]),
                }
            data["frames"].clear()
            data["audio_buffer"].clear()
            data["last_clip_time"] = now

    if sess_participants:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(executor, create_clips_for_all, session_id, sess_participants, start, end)


# 🔹 FastAPI WebSocket handler (instead of websockets.serve)
async def fastapi_handler(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")

    if not session_id or session_id not in event_bus.sessions:
        print("⚠️ ws_receiver: missing/unknown session_id. Closing.")
        await websocket.close()
        return

    print(f"✅ Recall connected for session {session_id}")

    async def clip_timer():
        while True:
            await asyncio.sleep(1)
            check_clips(session_id)

    clip_task = asyncio.create_task(clip_timer())
    try:
        while True:
            msg = await websocket.receive_json()
            evt_type = msg.get("event")
            payload = (msg.get("data") or {}).get("data") or {}
            speaker = payload.get("participant", {}).get("name") or "unknown"
            ts_now = time.time()

            if evt_type == "video_separate_png.data":
                buf = payload.get("buffer")
                if buf:
                    participant_data[speaker]["frames"].append((buf, ts_now))
            elif evt_type == "audio_separate_raw.data":
                buf = payload.get("buffer")
                if buf:
                    pcm = base64.b64decode(buf)
                    participant_data[speaker]["audio_buffer"].extend(pcm)
            elif evt_type.startswith("transcript"):
                sess = event_bus.sessions.get(session_id, {})
                logs = sess.setdefault("logs", [])
                text = payload.get("text") or payload.get("partial") or ""
                logs.append(f"[{int(ts_now)}] Transcript {speaker}: {text[:120]}")
                await event_bus.emit_log(session_id, logs[-10:])
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        clip_task.cancel()
        await websocket.close()
