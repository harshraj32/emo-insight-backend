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
        "last_audio_ts": None,
        "last_clip_time": None,
        "start_time": None,
    }
)
import json

def safe_summary(data):
    try:
        return json.loads(json.dumps(data))  # force JSON-safe
    except Exception as e:
        return {"error": f"serialization failed: {e}"}
    

def create_clips_for_all_sync(session_id, participants, start, end):
    """
    Synchronous function to create clips for all participants.
    This runs in the executor thread pool.
    """
    summaries = {}
    ts_str = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(settings.CLIPS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for speaker, data in participants.items():
        clip_name = f"{speaker}_{ts_str}".replace(" ", "_")
        temp_dir = os.path.join(session_dir, f"tmp_{clip_name}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # save frames
            frame_count = 0
            for frame_data, frame_time in data["frames"]:
                if start <= frame_time <= end:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.png")
                    with open(frame_path, "wb") as f:
                        f.write(base64.b64decode(frame_data))
                    frame_count += 1

            if frame_count == 0 and len(data["audio_buffer"]) == 0:
                print(f"📹 No data available for {speaker}")
                continue

            # save audio.raw
            raw_path = os.path.join(temp_dir, "audio.raw")
            with open(raw_path, "wb") as f:
                f.write(data["audio_buffer"])

            # convert to wav
            wav_path = os.path.join(temp_dir, "audio.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le",
                    "-ar", str(AUDIO_RATE),
                    "-ac", str(CHANNELS),
                    "-i", raw_path,
                    "-t", str(CLIP_LEN),
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )

            # if we have frames, create video
            if frame_count > 0:
                # create video from frames
                frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
                video_temp = os.path.join(temp_dir, "video_temp.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(FPS),
                        "-i", frame_pattern,
                        "-t", str(CLIP_LEN),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        video_temp,
                    ],
                    check=True,
                    capture_output=True,
                )

                # combine audio + video
                mp4_out = os.path.join(session_dir, f"{clip_name}.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", video_temp,
                        "-i", wav_path,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        mp4_out,
                    ],
                    check=True,
                    capture_output=True,
                )

                # send to Hume
                results = hume_client.process_clip(
                    Path(mp4_out), models={"prosody": {}, "face": {}}
                )
                summaries[str(speaker)] = safe_summary(summarize(results))

                print(f"🎬 Processed clip for {speaker} in session {session_id}")

                # cleanup mp4 after processing
                os.remove(mp4_out)
            else:
                # audio only
                results = hume_client.process_clip(
                    Path(wav_path), models={"prosody": {}}
                )
                summaries[str(speaker)] = safe_summary(summarize(results))
                print(f"🎵 Processed audio-only clip for {speaker}")

        except Exception as e:
            summaries[speaker] = {"error": str(e)}
            print(f"⚠️ Error processing {speaker}: {e}")
        finally:
            subprocess.run(["rm", "-rf", temp_dir])

    # return summaries to be processed asynchronously
    return summaries, ts_str, start, end

async def process_affina_feedback(session_id, summaries, ts_str):
    """
    Async function to process Affina feedback after clips are created.
    """
    sess = event_bus.sessions.get(session_id, {})
    context = {
        "phase": sess.get("phase", "pitch"),
        "objective": sess.get("objective", ""),
        "emotions": sess.get("emotions", []),
        "summaries": summaries,
    }
    
    # Emit emotion detection events for each speaker
    for speaker, summary in summaries.items():
        emotion_data = {
            "session_id": session_id,
            "speaker": speaker,
            "audio": summary.get("audio", {}),
            "video": summary.get("video", {})
        }
        await event_bus.emit_emotion(emotion_data)
    
    # Get transcript snippet from recent logs
    recent_transcript = ""
    logs = sess.get("logs", [])
    for log in logs[-5:]:
        if "Transcript" in log:
            recent_transcript = log
            break
    
    feedback = coach_feedback(context, recent_transcript or f"Clips at {ts_str}")

    # stringify if dict
    advice_str = json.dumps(feedback) if isinstance(feedback, dict) else str(feedback)
    await event_bus.emit_advice(session_id, advice_str)

    # log
    logs = sess.setdefault("logs", [])
    logs.append(f"[{ts_str}] Processed multi-speaker window.")
    await event_bus.emit_log(session_id, logs[-10:])

def check_clips(session_id):
    """
    Check if it's time to create clips for any participant.
    """
    now = time.time()
    sess_participants = {}
    
    for speaker, data in participant_data.items():
        if data["start_time"] is None:
            data["start_time"] = now
            data["last_clip_time"] = now
            continue
            
        if now - data["last_clip_time"] >= CLIP_LEN:
            start, end = data["last_clip_time"], now
            
            # collect relevant frames
            relevant_frames = [(f, t) for f, t in data["frames"] if start <= t <= end]
            
            if relevant_frames or len(data["audio_buffer"]) > 0:
                sess_participants[speaker] = {
                    "frames": list(relevant_frames),
                    "audio_buffer": bytes(data["audio_buffer"]),
                }
            
            # reset buffers
            data["frames"].clear()
            data["audio_buffer"].clear()
            data["last_clip_time"] = now
            data["last_audio_ts"] = None

    if sess_participants:
        # run synchronous clip creation in executor
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            executor, 
            create_clips_for_all_sync,
            session_id, sess_participants, start, end
        )
        
        # schedule async processing of results
        async def handle_results():
            summaries, ts_str, _, _ = await future
            if summaries:
                await process_affina_feedback(session_id, summaries, ts_str)
        
        asyncio.create_task(handle_results())

async def fastapi_handler(websocket: WebSocket):
    """
    FastAPI WebSocket handler for Recall.ai bot connection.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")

    if not session_id or session_id not in event_bus.sessions:
        print(f"⚠️ ws_receiver: missing/unknown session_id: {session_id}. Closing.")
        await websocket.close()
        return

    print(f"✅ Recall connected for session {session_id}")

    # log connection
    sess = event_bus.sessions.get(session_id, {})
    logs = sess.setdefault("logs", [])
    logs.append(f"WebSocket connected from Recall bot")
    await event_bus.emit_log(session_id, logs[-10:])

    async def clip_timer():
        while True:
            await asyncio.sleep(1)
            check_clips(session_id)

    clip_task = asyncio.create_task(clip_timer())
    
    try:
        while True:
            # receive message
            message = await websocket.receive_text()
            
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue
            
            evt_type = msg.get("event")
            payload = (msg.get("data") or {}).get("data") or {}
            participant = payload.get("participant", {})
            speaker = participant.get("name") or f"ID-{participant.get('id')}"
            ts_now = time.time()

            # create speaker-specific key for this session
            speaker_key = f"{session_id}_{speaker}"

            if evt_type == "video_separate_png.data":
                buf = payload.get("buffer")
                if buf:
                    participant_data[speaker_key]["frames"].append((buf, ts_now))
                    
            elif evt_type == "audio_separate_raw.data":
                buf = payload.get("buffer")
                rel_ts = payload.get("timestamp", {}).get("relative")
                
                if buf:
                    pcm = base64.b64decode(buf)
                    data = participant_data[speaker_key]
                    
                    # handle audio gaps
                    if rel_ts is not None and data["last_audio_ts"] is not None:
                        expected_samples = int((rel_ts - data["last_audio_ts"]) * AUDIO_RATE)
                        actual_samples = len(pcm) // SAMPLE_WIDTH
                        gap = expected_samples - actual_samples
                        if gap > 0:
                            # insert silence for missing samples
                            data["audio_buffer"].extend(b"\x00" * gap * SAMPLE_WIDTH)
                    
                    data["audio_buffer"].extend(pcm)
                    if rel_ts is not None:
                        data["last_audio_ts"] = rel_ts
                        
            elif evt_type in ["transcript.data", "transcript.partial_data"]:
                words = [w.get("text", "") for w in payload.get("words", [])]
                text = " ".join(words)
                
                if text:
                    partial = "(partial)" if evt_type == "transcript.partial_data" else ""
                    logs.append(f"[{int(ts_now)}] Transcript {speaker}{partial}: {text[:120]}")
                    await event_bus.emit_log(session_id, logs[-10:])
                    
                    # only show finalized transcripts in terminal
                    if evt_type == "transcript.data":
                        print(f"📝 {speaker}: {text}")
                        
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        logs.append(f"WebSocket error: {str(e)}")
        await event_bus.emit_log(session_id, logs[-10:])
    finally:
        clip_task.cancel()
        # cleanup participant data for this session
        keys_to_remove = [k for k in participant_data.keys() if k.startswith(f"{session_id}_")]
        for key in keys_to_remove:
            del participant_data[key]
        await websocket.close()