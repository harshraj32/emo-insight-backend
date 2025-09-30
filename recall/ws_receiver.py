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
import logging
logger = logging.getLogger(__name__)
from fastapi import WebSocket
from config import settings
from hume import hume_client
from affina.coach import coach_feedback
import event_bus
from hume.hume_summarize import load_hume_json_from_file, summarize_hume_batch

executor = ThreadPoolExecutor(max_workers=4)

AUDIO_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FPS = 2
CLIP_LEN = 5.0

# Buffer per participant (keyed by session_id:speaker)
participant_data = defaultdict(
    lambda: {
        "frames": deque(),
        "audio_buffer": bytearray(),
        "last_audio_ts": None,
        "last_clip_time": None,
        "start_time": None,
    }
)

def safe_summary(data):
    """Ensure data is JSON-serializable"""
    try:
        return json.loads(json.dumps(data))
    except Exception as e:
        return {"error": f"serialization failed: {e}"}

def create_clips_for_all_sync(session_id, participants_data, start, end):
    """
    Create and process clips for all participants in this time window.
    Returns summaries for each participant.
    """
    summaries = {}
    ts_str = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(settings.CLIPS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for speaker_name, data in participants_data.items():
        # Clean speaker name (remove any session prefix if accidentally included)
        clean_speaker = (
            speaker_name.replace(f"{session_id}_", "")
            if speaker_name.startswith(f"{session_id}_")
            else speaker_name
        )

        # Create unique clip names
        audio_clip_path = os.path.join(session_dir, f"{clean_speaker}_{ts_str}_audio.wav")
        video_clip_path = os.path.join(session_dir, f"{clean_speaker}_{ts_str}_video.mp4")
        temp_dir = os.path.join(session_dir, f"tmp_{clean_speaker}_{ts_str}")

        os.makedirs(temp_dir, exist_ok=True)

        try:
            audio_results = None
            video_results = None

            # Process audio if available
            if len(data["audio_buffer"]) > 0:
                raw_path = os.path.join(temp_dir, "audio.raw")
                with open(raw_path, "wb") as f:
                    f.write(data["audio_buffer"])

                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-f", "s16le",
                            "-ar", str(AUDIO_RATE),
                            "-ac", str(CHANNELS),
                            "-i", raw_path,
                            "-t", str(CLIP_LEN),
                            audio_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    if os.path.exists(audio_clip_path) and os.path.getsize(audio_clip_path) > 0:
                        audio_results = hume_client.process_clip(
                            Path(audio_clip_path),
                            models={"prosody": {"granularity": "utterance"}},
                        )
                        print(f"✅ Audio processed for {clean_speaker}: {os.path.getsize(audio_clip_path)} bytes")
                    else:
                        print(f"⚠️ Empty audio file for {clean_speaker}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ FFmpeg audio failed for {clean_speaker}: {e.stderr}")
                except Exception as e:
                    print(f"❌ Audio processing error for {clean_speaker}: {e}")

            # Process video if frames available
            frame_count = 0
            for frame_data, frame_time in data["frames"]:
                if start <= frame_time <= end:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.png")
                    with open(frame_path, "wb") as f:
                        f.write(base64.b64decode(frame_data))
                    frame_count += 1

            if frame_count > 0:
                frame_pattern = os.path.join(temp_dir, "frame_%04d.png")

                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-framerate", str(FPS),
                            "-i", frame_pattern,
                            "-t", str(CLIP_LEN),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            video_clip_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    if os.path.exists(video_clip_path) and os.path.getsize(video_clip_path) > 0:
                        video_results = hume_client.process_clip(
                            Path(video_clip_path),
                            models={"face": {"fps_pred": 3}},
                        )
                        print(f"✅ Video processed for {clean_speaker}: {frame_count} frames")
                    else:
                        print(f"⚠️ Empty video file for {clean_speaker}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ FFmpeg video failed for {clean_speaker}: {e.stderr}")
                except Exception as e:
                    print(f"❌ Video processing error for {clean_speaker}: {e}")
            logger.debug("Raw video_results from Hume (type=%s): %s", type(video_results), video_results)

            if isinstance(video_results, (dict, list)):
                try:
                    logger.debug("Raw video_results JSON dump:\n%s",
                                json.dumps(video_results, indent=2)[:4000])
                except Exception as e:
                    logger.debug("Could not JSON dump video_results: %s", e)
            
            if isinstance(audio_results, (dict, list)):
                try:
                    logger.debug("Raw video_results JSON dump:\n%s",
                                json.dumps(audio_results, indent=2)[:4000])
                except Exception as e:
                    logger.debug("Could not JSON dump video_results: %s", e)


            # 👉 NEW: build unified summary
            summary = summarize_hume_batch(
                audio_obj=audio_results,
                video_obj=video_results,
                participant=clean_speaker,
                timestamp=ts_str,
            )

            summaries[clean_speaker] = summary[clean_speaker]
            print(f"🎉 Summary created for {summaries}")
        except Exception as e:
            summaries[clean_speaker] = {
                "audio": {"status": "error", "error": str(e)},
                "video": {"status": "error", "error": str(e)},
                "timestamp": ts_str,
            }
            print(f"❌ Error processing {clean_speaker}: {e}")

        finally:
            if os.path.exists(temp_dir):
                subprocess.run(["rm", "-rf", temp_dir], capture_output=True)
    
    return summaries, ts_str
def check_and_create_clips(session_id):
    """
    Check if it's time to create clips for participants in this session.
    """
    now = time.time()
    participants_to_process = {}
    
    # Find all participants for this session
    session_prefix = f"{session_id}_"
    
    for key, data in list(participant_data.items()):
        if not key.startswith(session_prefix):
            continue
            
        # Extract speaker name from key
        speaker = key[len(session_prefix):]
        
        # Initialize timing if needed
        if data["start_time"] is None:
            data["start_time"] = now
            data["last_clip_time"] = now
            continue
        
        # Check if it's time to create a clip
        time_since_last = now - data["last_clip_time"]
        if time_since_last >= CLIP_LEN:
            clip_start = data["last_clip_time"]
            clip_end = now
            
            # Collect data for this time window
            relevant_frames = [(f, t) for f, t in data["frames"] if clip_start <= t <= clip_end]
            
            # Only process if we have data
            if relevant_frames or len(data["audio_buffer"]) > 0:
                participants_to_process[speaker] = {
                    "frames": list(relevant_frames),
                    "audio_buffer": bytes(data["audio_buffer"]),
                }
                print(f"📊 Queuing {speaker}: {len(relevant_frames)} frames, {len(data['audio_buffer'])} audio bytes")
            
            # Reset buffers for next clip
            data["frames"].clear()
            data["audio_buffer"].clear()
            data["last_clip_time"] = now
            data["last_audio_ts"] = None

    # Process clips if we have any
    if participants_to_process:
        print(f"🎬 Processing clips for {len(participants_to_process)} participants")
        
        # Run clip creation in executor
        loop = asyncio.get_running_loop()
        clip_start_time = now - CLIP_LEN
        
        # Create future for processing
        future = loop.run_in_executor(
            executor,
            create_clips_for_all_sync,
            session_id,
            participants_to_process,
            clip_start_time,
            now
        )
        
        # Handle results asynchronously
        async def process_results():
            try:
                summaries, ts_str = await future
                if summaries:
                    print(f"🎯 Got summaries for {len(summaries)} participants at {ts_str}")
                    await process_affina_feedback(session_id, summaries, ts_str)
            except Exception as e:
                print(f"❌ Error processing clips: {e}")
                import traceback
                traceback.print_exc()
        
        asyncio.create_task(process_results())


async def process_affina_feedback(session_id, summaries, ts_str):
    """
    Send summaries to Affina coach and emit results.
    """
    try:
        sess = event_bus.sessions.get(session_id)
        if not sess:
            print(f"⚠️ Session {session_id} not found for Affina processing")
            return

        # Debug: Print what we're sending to Affina
        print(f"\n[DEBUG] Sending to Affina:")
        print(f"  Session: {session_id}")
        print(f"  Summaries: {json.dumps(summaries, indent=2)}")

        # Prepare context for coach
        context = {
            "phase": sess.get("phase", "pitch"),
            "objective": sess.get("objective", ""),
            "emotions": sess.get("emotions", []),
            "summaries": summaries,
        }

        # Emit emotion events for each speaker
        for speaker, summary in summaries.items():
            audio_data = summary.get("audio", {})
            video_data = summary.get("video", {})

            # ✅ Updated to check for "ok"
            if audio_data.get("status") == "ok" or video_data.get("status") == "ok":
                emotion_data = {
                    "session_id": session_id,
                    "speaker": speaker,
                    "timestamp": ts_str,
                    "audio": audio_data,
                    "video": video_data,
                }
                print(f"[DEBUG] Emitting emotion for {speaker}: {emotion_data}")
                await event_bus.emit_emotion(emotion_data)

        # 👉 Get transcript from summaries, not logs
        recent_transcript = ""
        for speaker, summary in summaries.items():
            transcript = summary.get("audio", {}).get("transcript")
            if transcript:
                recent_transcript = transcript
                break  # just take the first available transcript

        if not recent_transcript:
            recent_transcript = f"[No transcript at {ts_str}]"

        print(f"[DEBUG] Recent transcript: {recent_transcript}")

        # Get coach feedback
        feedback = coach_feedback(context, recent_transcript)
        print(f"[DEBUG] Coach feedback received: {feedback}")

        # Emit advice
        if isinstance(feedback, dict):
            advice_message = feedback.get("recommendation", "Processing.")
        else:
            advice_message = str(feedback)

        await event_bus.emit_advice(session_id, advice_message)
        print(f"[DEBUG] Emitted advice: {advice_message}")

        # Update session logs
        sess["logs"].append(f"[{ts_str}] Processed {len(summaries)} speakers")
        sess["last_hume_summary"] = summaries
        sess["last_coach_feedback"] = feedback

        await event_bus.emit_log(session_id, sess["logs"][-10:])

    except Exception as e:
        print(f"❌ Error in Affina processing: {e}")
        import traceback
        traceback.print_exc()
        try:
            await event_bus.emit_advice(session_id, f"Coach temporarily unavailable: {str(e)}")
        except:
            pass

async def fastapi_handler(websocket: WebSocket):
    """
    Handle WebSocket connection from Recall bot.
    """
    await websocket.accept()
    
    # Get session_id from query params
    session_id = websocket.query_params.get("session_id")
    
    if not session_id:
        print("⚠️ WebSocket connected without session_id")
        await websocket.close(code=1008, reason="Missing session_id")
        return
    
    # Verify session exists
    sess = event_bus.sessions.get(session_id)
    if not sess:
        print(f"⚠️ Unknown session_id: {session_id}")
        await websocket.close(code=1008, reason="Unknown session")
        return
    
    print(f"✅ Recall bot connected for session {session_id}")
    sess["logs"].append("Recall bot connected")
    await event_bus.emit_log(session_id, sess["logs"][-10:])
    
    # Start clip processing timer
    async def clip_timer():
        while True:
            await asyncio.sleep(1.0)
            check_and_create_clips(session_id)
    
    clip_task = asyncio.create_task(clip_timer())
    
    try:
        while True:
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
            
            # Create unique key for this session+speaker
            participant_key = f"{session_id}_{speaker}"
            
            # Handle different event types
            if evt_type == "video_separate_png.data":
                buf_b64 = payload.get("buffer")
                if buf_b64:
                    participant_data[participant_key]["frames"].append((buf_b64, ts_now))
            
            elif evt_type == "audio_separate_raw.data":
                buf_b64 = payload.get("buffer", "")
                rel_ts  = payload.get("timestamp", {}).get("relative")

                if buf_b64 and rel_ts is not None:
                    pcm_bytes = base64.b64decode(buf_b64)

                    data = participant_data[participant_key]

                    # Gap fill exactly like in test.py
                    if data["last_audio_ts"] is not None:
                        expected_samples = int((rel_ts - data["last_audio_ts"]) * AUDIO_RATE)
                        gap = expected_samples - (len(pcm_bytes) // SAMPLE_WIDTH)
                        if gap > 0:
                            data["audio_buffer"].extend(b"\x00" * gap * SAMPLE_WIDTH)

                    data["audio_buffer"].extend(pcm_bytes)
                    data["last_audio_ts"] = rel_ts

            
            elif evt_type in ["transcript.data", "transcript.partial_data"]:
                words = [w.get("text", "") for w in payload.get("words", [])]
                text = " ".join(words).strip()
                
                if text:
                    is_partial = evt_type == "transcript.partial_data"
                    log_entry = f"Transcript {speaker}{'(partial)' if is_partial else ''}: {text[:150]}"
                    sess["logs"].append(log_entry)
                    
                    # Only emit non-partial transcripts
                    if not is_partial:
                        print(f"📝 {speaker}: {text}")
                        await event_bus.emit_log(session_id, sess["logs"][-10:])
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        sess["logs"].append(f"WebSocket error: {str(e)}")
        await event_bus.emit_log(session_id, sess["logs"][-10:])
    
    finally:
        clip_task.cancel()
        
        # Cleanup participant data for this session
        session_prefix = f"{session_id}_"
        keys_to_remove = [k for k in participant_data.keys() if k.startswith(session_prefix)]
        for key in keys_to_remove:
            del participant_data[key]
        
        print(f"🔌 WebSocket disconnected for session {session_id}")
        await websocket.close()